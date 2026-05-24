"""
Visualize AGVP intervention metrics across ALL 36 layers of Intern-S1-mini (Qwen3).
Baseline analysis: no intervention, just collect per-layer attention metrics.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import math
import json
import types
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

MODEL_PATH = "/root/code/Intern-S1-mini"
IMG_PATH = "/root/code/ClearSight/data/coco/val2014/COCO_val2014_000000105156.jpg"
SAVE_DIR = "/root/code/ClearSight/outputs/visualize"

# ---- Collector ----
layer_metrics = {}


def _vis_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Standard attention forward that collects AGVP-relevant metrics."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states_r = repeat_kv(key_states, self.num_key_value_groups)
    value_states_r = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states_r.transpose(2, 3)) * self.scaling

    SYS_LEN = self._vis_sys_len
    IMG_LEN = self._vis_img_len
    q_len = query_states.shape[2]
    kv_len = key_states_r.shape[2]

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :kv_len]
        attn_weights = attn_weights + causal_mask

    attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_probs_drop = F.dropout(attn_probs, p=0.0, training=False)
    attn_output = torch.matmul(attn_probs_drop, value_states_r)

    # ---- Collect metrics (only during prefill) ----
    layer_id = self.layer_idx
    if SYS_LEN + IMG_LEN > 0 and q_len > SYS_LEN + IMG_LEN:
        text_start = SYS_LEN + IMG_LEN
        text_to_img = attn_probs[:, :, text_start:, SYS_LEN:SYS_LEN + IMG_LEN]
        text_to_sys = attn_probs[:, :, text_start:, :SYS_LEN]
        text_to_text = attn_probs[:, :, text_start:, text_start:]

        img_values = value_states_r[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
        sys_values = value_states_r[:, :, :SYS_LEN, :]

        p_vis = text_to_img.sum(dim=-1).mean().item()
        p_sys = text_to_sys.sum(dim=-1).mean().item()
        p_text = text_to_text.sum(dim=-1).mean().item()

        imbalance = max(0.0, min(1.0, (p_sys - p_vis) / (p_sys + p_vis + 1e-8)))

        d_vis = torch.matmul(text_to_img, img_values).mean(dim=2)
        d_sys = torch.matmul(text_to_sys, sys_values).mean(dim=2)
        d_vis_hat = F.normalize(d_vis, dim=-1)
        d_sys_hat = F.normalize(d_sys, dim=-1)

        cos_vis_sys = (d_vis_hat * d_sys_hat).sum(dim=-1).mean().item()

        sim_attn_output = torch.matmul(attn_probs, value_states_r)
        text_out = sim_attn_output[:, :, text_start:, :]
        orig_norm = text_out.norm(dim=-1).mean().item()

        proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
        proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)

        layer_metrics[layer_id] = {
            "p_vis": p_vis,
            "p_sys": p_sys,
            "p_text": p_text,
            "imbalance": imbalance,
            "cos_vis_sys": cos_vis_sys,
            "proj_sys_mag": proj_sys.norm(dim=-1).mean().item(),
            "proj_vis_mag": proj_vis.norm(dim=-1).mean().item(),
            "orig_norm": orig_norm,
        }

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_probs


def plot_metrics(metrics, save_path, title="S1-mini"):
    layers = sorted(metrics.keys())

    p_vis = [metrics[l]["p_vis"] for l in layers]
    p_sys = [metrics[l]["p_sys"] for l in layers]
    imbalance = [metrics[l]["imbalance"] for l in layers]
    cos_vis_sys = [metrics[l]["cos_vis_sys"] for l in layers]
    proj_sys_mag = [metrics[l]["proj_sys_mag"] for l in layers]
    proj_vis_mag = [metrics[l]["proj_vis_mag"] for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{title}: AGVP Layer Analysis ({len(layers)} layers)", fontsize=14, fontweight="bold")

    x = np.array(layers)

    ax = axes[0, 0]
    ax.bar(x - 0.2, p_vis, 0.4, label="p_vis (→img)", color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, p_sys, 0.4, label="p_sys (→sys)", color="coral", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention mass")
    ax.set_title("Text→Visual vs Text→System Attention")
    ax.legend(fontsize=9)
    ax.set_xticks(x[::2])

    ax = axes[0, 1]
    ax.plot(x, imbalance, "o-", color="purple", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(x, imbalance, alpha=0.15, color="purple")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Imbalance")
    ax.set_title("Imbalance = (p_sys - p_vis) / (p_sys + p_vis)")
    ax.set_xticks(x[::2])
    ax.set_ylim(-0.1, 1.0)

    ax = axes[1, 0]
    ax.plot(x, cos_vis_sys, "s-", color="green", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(d_vis, d_sys)")
    ax.set_title("Direction Separability (lower = more separable)")
    ax.set_xticks(x[::2])

    ax = axes[1, 1]
    ax.bar(x - 0.2, proj_sys_mag, 0.4, label="||proj_sys||", color="coral", alpha=0.8)
    ax.bar(x + 0.2, proj_vis_mag, 0.4, label="||proj_vis||", color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Projection magnitude")
    ax.set_title("AGVP Projection Strength")
    ax.legend(fontsize=9)
    ax.set_xticks(x[::2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


def main():
    print("Loading S1-mini model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
    )
    model.eval()

    # Set eager attention for all layers
    model.language_model.set_attn_implementation("eager")

    # Monkey-patch ALL layers
    num_layers = len(model.language_model.layers)
    print(f"Patching {num_layers} layers for metric collection...")
    for i in range(num_layers):
        attn = model.language_model.layers[i].self_attn
        attn._vis_sys_len = 0
        attn._vis_img_len = 0
        attn.forward = types.MethodType(_vis_attn_forward, attn)

    # Prepare input
    image = Image.open(IMG_PATH).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image in detail."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
        enable_thinking=False,
    )

    # Get boundaries
    img_mask = inputs["input_ids"][0] == model.config.image_token_id
    img_pos = img_mask.nonzero(as_tuple=True)[0]
    sys_len = img_pos[0].item()
    img_len = img_pos.shape[0]
    print(f"sys_len={sys_len}, img_len={img_len}")

    # Set boundaries on all layers
    for i in range(num_layers):
        attn = model.language_model.layers[i].self_attn
        attn._vis_sys_len = sys_len
        attn._vis_img_len = img_len

    device = torch.device("cuda:0")
    inputs = inputs.to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

    # Forward pass
    print(f"Running forward pass...")
    with torch.inference_mode():
        _ = model(**inputs, use_cache=False)

    # Save & plot
    os.makedirs(SAVE_DIR, exist_ok=True)

    json_path = os.path.join(SAVE_DIR, "agvp_s1mini_layer_analysis.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in layer_metrics.items()}, f, indent=2)
    print(f"Metrics saved to {json_path}")

    plot_path = os.path.join(SAVE_DIR, "agvp_s1mini_layer_analysis.png")
    plot_metrics(layer_metrics, plot_path, title="Intern-S1-mini")

    # Summary table
    print(f"\n{'Layer':>5} {'p_vis':>7} {'p_sys':>7} {'imbal':>7} {'cos(v,s)':>8} {'||proj_s||':>10} {'||proj_v||':>10}")
    print("-" * 60)
    for l in sorted(layer_metrics.keys()):
        m = layer_metrics[l]
        print(f"{l:>5} {m['p_vis']:>7.4f} {m['p_sys']:>7.4f} {m['imbalance']:>7.4f} {m['cos_vis_sys']:>8.4f} {m['proj_sys_mag']:>10.4f} {m['proj_vis_mag']:>10.4f}")


if __name__ == "__main__":
    main()
