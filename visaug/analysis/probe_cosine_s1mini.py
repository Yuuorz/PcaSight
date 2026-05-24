"""
Per-layer cosine similarity probe for Intern-S1-mini (Qwen3).
Runs a single forward pass, collects for every layer:
  1. p_vis / p_sys / p_text (attention mass distribution)
  2. imbalance score
  3. cos(d_vis, d_sys) — direction separability
  4. text→img cosine avg  — semantic alignment between text and visual tokens
  5. text→sys cosine avg  — alignment between text and system prompt tokens
  6. vis-sys cosine gap   = cos(text,img) - cos(text,sys)
  7. attn entropy         — how focused text→img attention is (normalized)

Uses wrapper pattern: calls original forward, then extracts metrics from outputs.
No qwen3 imports needed — avoids environment dependency.

Output: outputs/analysis/layer_cosine_s1mini.json + png
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

MODEL_PATH = "/root/code/Intern-S1-mini"
IMG_PATH = "/root/code/ClearSight/data/coco/val2014/COCO_val2014_000000105156.jpg"
SAVE_DIR = "/root/code/ClearSight/outputs/analysis"

layer_metrics = {}


def _make_wrapper_forward(orig_forward, layer_idx):
    """Wrap original forward to collect cosine metrics."""
    def wrapped_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                        past_key_values=None, cache_position=None, **kwargs):
        # Call original forward
        result = orig_forward(self, hidden_states, position_embeddings=position_embeddings,
                              attention_mask=attention_mask, past_key_values=past_key_values,
                              cache_position=cache_position, **kwargs)
        # Return is (attn_output, attn_weights) for eager attention
        attn_output, attn_probs = result

        bsz, q_len, _ = hidden_states.size()
        SYS_LEN = getattr(self, '_vis_sys_len', 0)
        IMG_LEN = getattr(self, '_vis_img_len', 0)

        if q_len <= SYS_LEN + IMG_LEN:
            return result

        with torch.no_grad():
            # Get value_states from the module
            v_proj = self.v_proj
            value_states = v_proj(hidden_states)
            # Determine shape for value_states
            head_dim = self.head_dim
            num_kv_heads = self.num_key_value_heads
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            # Repeat kv heads
            n_groups = self.num_key_value_groups
            if n_groups > 1:
                value_states = value_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_groups, q_len, head_dim).reshape(bsz, num_kv_heads * n_groups, q_len, head_dim)

            text_start = SYS_LEN + IMG_LEN
            text_to_img = attn_probs[:, :, text_start:, SYS_LEN:SYS_LEN + IMG_LEN]
            text_to_sys = attn_probs[:, :, text_start:, :SYS_LEN]
            text_to_text = attn_probs[:, :, text_start:, text_start:]

            img_values = value_states[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
            sys_values = value_states[:, :, :SYS_LEN, :]

            # Attention mass
            p_vis = text_to_img.sum(dim=-1).mean().item()
            p_sys = text_to_sys.sum(dim=-1).mean().item()
            p_text = text_to_text.sum(dim=-1).mean().item()

            imbalance = max(0.0, min(1.0, (p_sys - p_vis) / (p_sys + p_vis + 1e-8)))

            # Direction vectors
            d_vis = torch.matmul(text_to_img, img_values).mean(dim=2)
            d_sys = torch.matmul(text_to_sys, sys_values).mean(dim=2)
            d_vis_hat = F.normalize(d_vis, dim=-1)
            d_sys_hat = F.normalize(d_sys, dim=-1)
            cos_vis_sys = (d_vis_hat * d_sys_hat).sum(dim=-1).mean().item()

            # Text↔image cosine alignment
            text_out = attn_output[:, :, text_start:, :]
            text_hat = F.normalize(text_out, dim=-1)
            img_mean_hat = F.normalize(img_values.mean(dim=2), dim=-1).unsqueeze(2)
            sys_mean_hat = F.normalize(sys_values.mean(dim=2), dim=-1).unsqueeze(2)

            cos_ti = (text_hat * img_mean_hat).sum(dim=-1).mean().item()
            cos_ts = (text_hat * sys_mean_hat).sum(dim=-1).mean().item()
            cos_gap = cos_ti - cos_ts

            # Attention entropy
            entropy = -(text_to_img.float() * torch.log(text_to_img.float() + 1e-12)).sum(dim=-1).mean().item()
            entropy_norm = entropy / math.log(IMG_LEN) if IMG_LEN > 0 else 0.0

            # Projection magnitudes
            orig_norm = text_out.norm(dim=-1).mean().item()
            proj_sys_mag = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
            proj_vis_mag = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)

            layer_metrics[layer_idx] = {
                "p_vis": p_vis,
                "p_sys": p_sys,
                "p_text": p_text,
                "imbalance": imbalance,
                "cos_vis_sys": cos_vis_sys,
                "cos_text_img": cos_ti,
                "cos_text_sys": cos_ts,
                "cos_gap": cos_gap,
                "attn_entropy": entropy,
                "attn_entropy_norm": entropy_norm,
                "proj_sys_mag": proj_sys_mag.norm(dim=-1).mean().item(),
                "proj_vis_mag": proj_vis_mag.norm(dim=-1).mean().item(),
                "orig_norm": orig_norm,
            }

        return result

    return wrapped_forward


def plot_metrics(metrics, save_path, title="Intern-S1-mini"):
    layers = sorted(metrics.keys())
    p_vis = [metrics[l]["p_vis"] for l in layers]
    p_sys = [metrics[l]["p_sys"] for l in layers]
    imbalance = [metrics[l]["imbalance"] for l in layers]
    cos_vis_sys = [metrics[l]["cos_vis_sys"] for l in layers]
    cos_ti = [metrics[l]["cos_text_img"] for l in layers]
    cos_ts = [metrics[l]["cos_text_sys"] for l in layers]
    cos_gap = [metrics[l]["cos_gap"] for l in layers]
    entropy_norm = [metrics[l]["attn_entropy_norm"] for l in layers]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{title}: Layer-wise Cosine Diagnosis ({len(layers)} layers)", fontsize=14, fontweight="bold")
    x = np.array(layers)

    ax = axes[0, 0]
    ax.bar(x - 0.2, p_vis, 0.4, label="p_vis (→img)", color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, p_sys, 0.4, label="p_sys (→sys)", color="coral", alpha=0.8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Attention mass")
    ax.set_title("Text→Visual vs Text→System Attention"); ax.legend(fontsize=9)
    ax.set_xticks(x[::2])

    ax = axes[0, 1]
    ax.plot(x, imbalance, "o-", color="purple", linewidth=2)
    ax.fill_between(x, imbalance, alpha=0.15, color="purple")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Imbalance"); ax.set_title("Imbalance")
    ax.set_xticks(x[::2]); ax.set_ylim(-0.1, 1.0)

    ax = axes[0, 2]
    ax.plot(x, cos_vis_sys, "s-", color="green", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("cos(d_vis, d_sys)")
    ax.set_title("Direction Separability"); ax.set_xticks(x[::2])

    ax = axes[1, 0]
    ax.plot(x, cos_ti, "o-", color="steelblue", linewidth=2, label="cos(text, img)")
    ax.plot(x, cos_ts, "s-", color="coral", linewidth=2, label="cos(text, sys)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Layer"); ax.set_ylabel("Cosine similarity")
    ax.set_title("Text-Image vs Text-System Alignment"); ax.legend(fontsize=9)
    ax.set_xticks(x[::2])

    ax = axes[1, 1]
    ax.plot(x, cos_gap, "o-", color="darkorange", linewidth=2)
    ax.fill_between(x, cos_gap, alpha=0.15, color="darkorange")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("cos_gap")
    ax.set_title("Visual-System Alignment Gap"); ax.set_xticks(x[::2])

    ax = axes[1, 2]
    ax.plot(x, entropy_norm, "^-", color="brown", linewidth=2)
    ax.fill_between(x, entropy_norm, alpha=0.15, color="brown")
    ax.set_xlabel("Layer"); ax.set_ylabel("Norm. entropy")
    ax.set_title("Text→Image Attention Entropy"); ax.set_xticks(x[::2])

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
    model.language_model.set_attn_implementation("eager")

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

    device = torch.device("cuda:0")
    inputs = inputs.to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

    # Monkey-patch with wrapper
    num_layers = len(model.language_model.layers)
    print(f"Patching {num_layers} layers with wrapper...")
    for i in range(num_layers):
        attn = model.language_model.layers[i].self_attn
        orig_fwd = attn.forward
        attn._vis_sys_len = sys_len
        attn._vis_img_len = img_len
        attn.forward = types.MethodType(_make_wrapper_forward(orig_fwd, i), attn)

    print(f"Running forward pass...")
    with torch.inference_mode():
        _ = model(**inputs, use_cache=False)

    # Save & plot
    os.makedirs(SAVE_DIR, exist_ok=True)
    json_path = os.path.join(SAVE_DIR, "layer_cosine_s1mini.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in layer_metrics.items()}, f, indent=2)
    print(f"Metrics saved to {json_path}")

    plot_path = os.path.join(SAVE_DIR, "layer_cosine_s1mini.png")
    plot_metrics(layer_metrics, plot_path, title="Intern-S1-mini")

    print(f"\n{'Layer':>5} {'p_vis':>7} {'p_sys':>7} {'imbal':>7} {'cos(v,s)':>8} {'cos(t,i)':>8} {'cos(t,s)':>8} {'gap':>8} {'entropy':>7}")
    print("-" * 75)
    for l in sorted(layer_metrics.keys()):
        m = layer_metrics[l]
        print(f"{l:>5} {m['p_vis']:>7.4f} {m['p_sys']:>7.4f} {m['imbalance']:>7.4f} {m['cos_vis_sys']:>8.4f} {m['cos_text_img']:>8.4f} {m['cos_text_sys']:>8.4f} {m['cos_gap']:>8.4f} {m['attn_entropy_norm']:>7.4f}")


if __name__ == "__main__":
    main()
