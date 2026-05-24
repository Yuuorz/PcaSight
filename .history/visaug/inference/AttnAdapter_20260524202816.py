"""
AGVP (Attention-Guided Visual Projection) adapter for LLaVA.
Upgraded from 1D-Vector Addition to Multi-Dimensional Subspace PCA Projection.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb


class AGVPAdapter(LlamaAttention):

    SYS_LEN = 35
    IMG_LEN = 576

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states)

        # ==========================================================
        # PCA Subspace Alignment Block (Prefill Only)
        # ==========================================================
        SYS_LEN = self.SYS_LEN
        IMG_LEN = self.IMG_LEN

        if kv_seq_len > SYS_LEN + IMG_LEN and q_len > SYS_LEN + IMG_LEN:
            text_start = SYS_LEN + IMG_LEN
            mode = getattr(self, '_agvp_mode', 'legacy')

            if mode == "subspace_pca":
                text_out = attn_output[:, :, text_start:, :]
                orig_norm = text_out.norm(dim=-1, keepdim=True)
                
                pca_k = getattr(self, '_pca_k', 3)
                scale = getattr(self, '_pca_scale', 1.0)
                zone_weight = getattr(self, '_zone_weight', 1.0)

                # 1. 抽取当前层专属的图片 Value 场特征矩阵: [B, H, 576, D]
                img_values = value_states[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
                
                # 2. 在线进行样本去中心化
                img_mean = img_values.mean(dim=2, keepdim=True)
                img_centered = img_values - img_mean

                # 3. 跨 Batch 和 Head 执行奇异值分解 (SVD)
                # 强制转为 fp32 计算保证底层求逆和 SVD 分解的数学收敛稳定性
                try:
                    _, _, V = torch.linalg.svd(img_centered.float(), full_matrices=False)
                    # V 的 shape 为 [B, H, D, D]，截取前 k 个基底向量组成视觉超平面
                    # W: [B, H, D, k]
                    W = V[:, :, :pca_k, :].transpose(-1, -2).to(text_out.dtype)
                except RuntimeError:
                    # 极端数值未收敛时的兜底：退回到一维平均质心扩展
                    W = F.normalize(img_values.mean(dim=2, keepdim=True), dim=-1).transpose(-1, -2)
                    W = W.expand(-1, -1, -1, pca_k)

                # 4. 计算文本 Token 在高维特征超平面上的正交投影分量
                # proj_subspace = text_out @ W @ W^T -> [B, H, text_len, D]
                proj_subspace = torch.matmul(torch.matmul(text_out, W), W.transpose(-1, -2))

                # 5. 纯 Pull 正交融合（移除了对系统提示方向的强排噪声项）
                text_out = text_out + (scale * zone_weight) * proj_subspace
                
                # 6. 锁死激活范数模长，防止深层自回归发散
                text_out = text_out * (orig_norm / (text_out.norm(dim=-1, keepdim=True) + 1e-8))

                attn_output = torch.cat([attn_output[:, :, :text_start, :], text_out], dim=2)

        # ---- End of PCA Block ----
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value