"""
AGVP (Attention-Guided Visual Projection) adapter for LLaVA (LlamaAttention).
Drop-in replacement for AttnAdapter (VAF).

Instead of pre-softmax scaling, AGVP performs post-softmax geometric projection:
  1. Compute visual/system directions from attention-weighted value vectors
  2. Project text token representations: suppress system direction, enhance visual direction
  3. Norm preservation: only change direction, not magnitude
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

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states)

        # ---- AGVP: project away system, project toward visual ----
        SYS_LEN = self.SYS_LEN
        IMG_LEN = self.IMG_LEN

        if kv_seq_len > SYS_LEN + IMG_LEN:
            img_values = value_states[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
            sys_values = value_states[:, :, :SYS_LEN, :]

            if q_len > SYS_LEN + IMG_LEN:
                # Prefill: modify text tokens
                text_start = SYS_LEN + IMG_LEN

                text_to_img = attn_probs[:, :, text_start:, SYS_LEN:SYS_LEN + IMG_LEN]
                text_to_sys = attn_probs[:, :, text_start:, :SYS_LEN]

                d_vis = torch.matmul(text_to_img, img_values).mean(dim=2)
                d_sys = torch.matmul(text_to_sys, sys_values).mean(dim=2)
                d_vis_hat = F.normalize(d_vis, dim=-1)
                d_sys_hat = F.normalize(d_sys, dim=-1)

                # Cos-gate: decide whether to intervene based on direction separability
                cos_vs = (d_vis_hat * d_sys_hat).sum(dim=-1).mean()
                mode = getattr(self, '_agvp_mode', 'imb_cos_vis')
                cos_abs = abs(cos_vs.item())
                p_vis_val = text_to_img.sum(dim=-1).mean()
                p_sys_val = text_to_sys.sum(dim=-1).mean()
                imb = torch.clamp((p_sys_val - p_vis_val) / (p_sys_val + p_vis_val + 1e-8), min=0.0, max=1.0).item()

                # ---- Hybrid modes: cos sign selects operation ----
                if mode.startswith('hybrid'):
                    text_out = attn_output[:, :, text_start:, :]
                    orig_norm = text_out.norm(dim=-1, keepdim=True)
                    did_modify = False

                    if cos_vs < 0 and imb > 0:
                        # Rotation toward d_vis
                        if mode == 'hybrid_soft':
                            alpha = cos_abs * cos_abs * imb
                        else:
                            alpha = cos_abs
                        text_hat = F.normalize(text_out, dim=-1)
                        d_vis_exp = d_vis_hat.unsqueeze(2).expand_as(text_out)
                        new_dir = F.normalize((1 - alpha) * text_hat + alpha * d_vis_exp, dim=-1)
                        text_out = new_dir * orig_norm
                        did_modify = True
                    elif cos_vs > 0 and imb > 0:
                        # Projection: -sys +vis
                        if mode == 'hybrid_soft':
                            strength = cos_abs * cos_abs * imb
                        else:
                            strength = imb * cos_abs
                        proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
                        proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)
                        text_out = text_out - strength * proj_sys + strength * proj_vis
                        text_out = text_out * (orig_norm / (text_out.norm(dim=-1, keepdim=True) + 1e-8))
                        did_modify = True

                    if did_modify:
                        attn_output = torch.cat(
                            [attn_output[:, :, :text_start, :], text_out], dim=2
                        )

                # ---- proj_cospos: only project when cos > 0 ----
                elif mode == 'proj_cospos':
                    if cos_vs > 0 and imb > 0:
                        text_out = attn_output[:, :, text_start:, :]
                        orig_norm = text_out.norm(dim=-1, keepdim=True)
                        strength = imb * cos_abs
                        proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
                        proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)
                        text_out = text_out - strength * proj_sys + strength * proj_vis
                        text_out = text_out * (orig_norm / (text_out.norm(dim=-1, keepdim=True) + 1e-8))
                        attn_output = torch.cat(
                            [attn_output[:, :, :text_start, :], text_out], dim=2
                        )

                # ---- rotate_vis modes (cos < 0 only) ----
                elif mode.startswith('rotate_vis'):
                    if cos_vs < 0 and imb > 0:
                        text_out = attn_output[:, :, text_start:, :]
                        orig_norm = text_out.norm(dim=-1, keepdim=True)
                        if mode == 'rotate_vis_adaptive':
                            alpha = imb * cos_abs
                        elif mode == 'rotate_vis_adaptive_sq':
                            alpha = imb * cos_abs * cos_abs
                        else:
                            alpha = cos_abs
                        text_hat = F.normalize(text_out, dim=-1)
                        d_vis_exp = d_vis_hat.unsqueeze(2).expand_as(text_out)
                        new_dir = F.normalize((1 - alpha) * text_hat + alpha * d_vis_exp, dim=-1)
                        text_out = new_dir * orig_norm
                        attn_output = torch.cat(
                            [attn_output[:, :, :text_start, :], text_out], dim=2
                        )

                # ---- Legacy projection modes ----
                else:
                    gate_mode = getattr(self, '_agvp_cos_gate', 'neg')
                    do_intervene = (gate_mode == 'none') or \
                                   (gate_mode == 'neg' and cos_vs < 0) or \
                                   (gate_mode == 'pos' and cos_vs > 0)
                    if do_intervene:
                        text_out = attn_output[:, :, text_start:, :]
                        orig_norm = text_out.norm(dim=-1, keepdim=True)
                        proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
                        proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)
                        strength = imb * cos_abs
                        text_out = text_out - strength * proj_sys + strength * proj_vis
                        text_out = text_out * (orig_norm / (text_out.norm(dim=-1, keepdim=True) + 1e-8))
                        attn_output = torch.cat(
                            [attn_output[:, :, :text_start, :], text_out], dim=2
                        )
            # Decode phase: skip AGVP (prefill correction propagates through residual stream)
        # ----------------------------------------------------------

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

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
