"""
Model-specific forward modification for Intern-S1-mini (Qwen system architecture).
"""
import torch
import types
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

def _s1mini_pca_forward(self, hidden_states: torch.Tensor, position_embeddings: tuple, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    if kwargs.get("past_key_values", None) is not None:
        k, v = kwargs["past_key_values"].update(k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position")})

    kr = repeat_kv(k, self.num_key_value_groups)
    vr = repeat_kv(v, self.num_key_value_groups)
    aw = torch.matmul(q, kr.transpose(2, 3)) * self.scaling

    SYS_LEN, IMG_LEN = self._sys_len, self._img_len
    q_len, kv_len = q.shape[2], kr.shape[2]

    if kwargs.get("attention_mask", None) is not None: aw = aw + kwargs["attention_mask"][:, :, :, :kv_len]
    ap = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    ao = torch.matmul(ap, vr)

    # === Uncentered PCA 核心核心截取 ===
    if SYS_LEN + IMG_LEN > 0 and kv_len > SYS_LEN + IMG_LEN and q_len > SYS_LEN + IMG_LEN:
        ts = SYS_LEN + IMG_LEN
        text_tokens = ao[:, :, ts:, :]
        orig_norm = text_tokens.norm(dim=-1, keepdim=True)

        pca_k = getattr(self, '_pca_k', 8)
        scale = getattr(self, '_scale', 0.1)
        zone_weight = getattr(self, '_zone_weight', 1.0)
        img_values = vr[:, :, SYS_LEN:ts, :]

        try:
            _, _, V = torch.linalg.svd(img_values.float(), full_matrices=False)
            U_k = V[:, :, :pca_k, :].to(text_tokens.dtype)
        except RuntimeError:
            U_k = F.normalize(img_values.mean(dim=2, keepdim=True), dim=-1).transpose(-1, -2).expand(-1, -1, pca_k, -1).transpose(-1, -2)

        proj_coords = torch.matmul(text_tokens, U_k.transpose(-1, -2))
        proj_subspace = torch.matmul(proj_coords, U_k)

        text_modified = text_tokens + (scale * zone_weight) * proj_subspace
        text_tokens = text_modified * (orig_norm / (text_modified.norm(dim=-1, keepdim=True) + 1e-8))
        ao = torch.cat([ao[:, :, :ts, :], text_tokens], dim=2)

    ao = ao.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    return self.o_proj(ao), ap

def inject_s1mini_pca(model, args, target_layers="7-19"):
    model.config._attn_implementation = "eager"
    model.language_model.set_attn_implementation("eager")
    
    layers = list(range(int(target_layers.split("-")[0]), int(target_layers.split("-")[1]) + 1))
    print(f"=== Injecting Uncentered PCA Subspace into S1-mini ===")
    
    for i, layer in enumerate(model.language_model.layers):
        if i in layers:
            attn = layer.self_attn
            attn._sys_len, attn._img_len = 0, 0
            attn._pca_k = args.pca_k
            attn._scale = args.scale
            attn._zone_weight = 0.5 if i <= 16 else 1.0
            attn.forward = types.MethodType(_s1mini_pca_forward, attn)
    return model