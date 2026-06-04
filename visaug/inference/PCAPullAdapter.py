"""
ClearSight Architectural Tool: Subspace PCA Adapter for LLaVA.
Performs Multi-Dimensional Subspace Orthogonal Pull Intervention.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb


class PCAPullAdapter(LlamaAttention):

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

        # if q_len > 1:
        #     print(f"[ClearSight Debug] LLaVA Layer {getattr(self, 'layer_idx', 'Active')} Forward Hooked via Subspace PCA!")
        
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
        # 核心修复：真正的 Subspace PCA 多维子空间正交投影算子 (无中心化版)
        # ==========================================================
        SYS_LEN = self.SYS_LEN
        IMG_LEN = self.IMG_LEN

        if kv_seq_len > SYS_LEN + IMG_LEN and q_len > SYS_LEN + IMG_LEN:
            text_start = SYS_LEN + IMG_LEN
            text_out = attn_output[:, :, text_start:, :]
            orig_norm = text_out.norm(dim=-1, keepdim=True)

            # 1. 动态截取从外部脚本传入的参数，增加默认防御值
            pca_k = getattr(self, '_pca_k', 3)
            scale = getattr(self, '_scale', getattr(self, '_pca_scale', 1.0)) # 兼容 S1-mini 参数名

            # 2. 提取全量视觉 token 场 [B, Heads, 576, Head_Dim]
            img_values = value_states[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
            
            # 3. 直接对绝对视觉特征场执行 SVD (绝对不减去 mean！)
            # 这样 V 的前 k 维将同时完美包裹“全局视觉质心”与“核心物体流形”
            _, _, V = torch.linalg.svd(img_values.float(), full_matrices=False)
            V = V.to(img_values.dtype) 
            
            # 提取前 k 个绝对主成分基础向量：[B, Heads, k, Head_Dim]
            U_k = V[:, :, :pca_k, :] 

            # 4. 计算文本残差在“绝对视觉子空间”上的正交投影分量
            proj_coords = torch.matmul(text_out, U_k.transpose(-1, -2))
            proj_subspace = torch.matmul(proj_coords, U_k)

            # 5. 执行多维正交引力 Pull (纯加法，抛弃 Push-sys 毒资产)
            text_modified = text_out + scale * proj_subspace

            # 6. 严格范数守恒球锁定 (Norm-Preservation锁)
            text_out = text_modified * (orig_norm / (text_modified.norm(dim=-1, keepdim=True) + 1e-8))

            # 7. 回填至残差流序列
            attn_output = torch.cat([attn_output[:, :, :text_start, :], text_out], dim=2)
        # ==========================================================

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value