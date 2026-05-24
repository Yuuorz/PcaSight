"""
Visualize AGVP intervention metrics across ALL 32 layers of LLaVA-v1.5-7b.
Runs a single forward pass, collects per-layer:
  1. p_vis / p_sys (attention mass to visual/system tokens)
  2. imbalance score
  3. direction separability (cosine between d_vis and d_sys)
  4. projection magnitude (how much AGVP would move the representation)

This guides the choice of which layers to apply AGVP.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../LLaVA"))

import math
import json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb

MODEL_PATH = "/root/code/llava-v1.5-7b"
IMG_PATH = "/root/code/ClearSight/data/coco/val2014/COCO_val2014_000000105156.jpg"
CONV_MODE = "vicuna_v1"
SAVE_DIR = "/root/code/ClearSight/outputs/visualize"

SYS_LEN = 35
IMG_LEN = 576

# ---- Collector ----
layer_metrics = {}


def _make_vis_forward(orig_forward, layer_idx):
    """Wrap original forward to collect AGVP metrics."""
    def vis_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        # Run original forward
        result = orig_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # force to get attn_weights
            use_cache=use_cache,
        )
        attn_output, attn_weights_out, present_kv = result

        bsz, q_len, _ = hidden_states.size()
        if q_len <= SYS_LEN + IMG_LEN:
            # Return with original output_attentions setting
            if not output_attentions:
                return attn_output, None, present_kv
            return result

        module = orig_forward.__self__

        with torch.no_grad():
            # Recompute value_states for direction extraction
            value_states = module.v_proj(hidden_states)
            value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            value_states_r = repeat_kv(value_states, module.num_key_value_groups)

            # attn_weights_out is (bsz, heads, q_len, kv_len)
            attn_probs = attn_weights_out

        text_start = SYS_LEN + IMG_LEN
        text_to_img = attn_probs[:, :, text_start:, SYS_LEN:SYS_LEN + IMG_LEN]
        text_to_sys = attn_probs[:, :, text_start:, :SYS_LEN]
        text_to_text = attn_probs[:, :, text_start:, text_start:]

        img_values = value_states_r[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
        sys_values = value_states_r[:, :, :SYS_LEN, :]

        # Attention mass
        p_vis = text_to_img.sum(dim=-1).mean().item()
        p_sys = text_to_sys.sum(dim=-1).mean().item()
        p_text = text_to_text.sum(dim=-1).mean().item()

        # Imbalance
        imbalance = max(0.0, min(1.0, (p_sys - p_vis) / (p_sys + p_vis + 1e-8)))

        # Direction vectors
        d_vis = torch.matmul(text_to_img, img_values).mean(dim=2)  # (bsz, heads, head_dim)
        d_sys = torch.matmul(text_to_sys, sys_values).mean(dim=2)
        d_vis_hat = F.normalize(d_vis, dim=-1)
        d_sys_hat = F.normalize(d_sys, dim=-1)

        # Direction separability (cosine between vis and sys directions)
        cos_vis_sys = (d_vis_hat * d_sys_hat).sum(dim=-1).mean().item()

        # Simulated projection magnitude
        sim_attn_output = torch.matmul(attn_probs, value_states_r)
        text_out = sim_attn_output[:, :, text_start:, :]
        orig_norm = text_out.norm(dim=-1).mean().item()

        proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
        proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)

        proj_sys_mag = proj_sys.norm(dim=-1).mean().item()
        proj_vis_mag = proj_vis.norm(dim=-1).mean().item()

        layer_metrics[layer_idx] = {
            "p_vis": p_vis,
            "p_sys": p_sys,
            "p_text": p_text,
            "imbalance": imbalance,
            "cos_vis_sys": cos_vis_sys,
            "proj_sys_mag": proj_sys_mag,
            "proj_vis_mag": proj_vis_mag,
            "orig_norm": orig_norm,
        }

        if not output_attentions:
            return attn_output, None, present_kv
        return result

    return vis_forward


def plot_metrics(metrics, save_path):
    layers = sorted(metrics.keys())
    n = len(layers)

    p_vis = [metrics[l]["p_vis"] for l in layers]
    p_sys = [metrics[l]["p_sys"] for l in layers]
    p_text = [metrics[l]["p_text"] for l in layers]
    imbalance = [metrics[l]["imbalance"] for l in layers]
    cos_vis_sys = [metrics[l]["cos_vis_sys"] for l in layers]
    proj_sys_mag = [metrics[l]["proj_sys_mag"] for l in layers]
    proj_vis_mag = [metrics[l]["proj_vis_mag"] for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("LLaVA-v1.5-7b: AGVP Layer Analysis (32 layers)", fontsize=14, fontweight="bold")

    x = np.array(layers)

    # Panel 1: Attention mass distribution
    ax = axes[0, 0]
    ax.bar(x - 0.2, p_vis, 0.4, label="p_vis (→img)", color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, p_sys, 0.4, label="p_sys (→sys)", color="coral", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention mass")
    ax.set_title("Text→Visual vs Text→System Attention")
    ax.legend(fontsize=9)
    ax.set_xticks(x[::2])

    # Panel 2: Imbalance score
    ax = axes[0, 1]
    ax.plot(x, imbalance, "o-", color="purple", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(x, imbalance, alpha=0.15, color="purple")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Imbalance")
    ax.set_title("Imbalance = (p_sys - p_vis) / (p_sys + p_vis)")
    ax.set_xticks(x[::2])
    ax.set_ylim(-0.1, 1.0)

    # Panel 3: Direction separability
    ax = axes[1, 0]
    ax.plot(x, cos_vis_sys, "s-", color="green", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(d_vis, d_sys)")
    ax.set_title("Direction Separability (lower = more separable)")
    ax.set_xticks(x[::2])

    # Panel 4: Projection magnitude
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
    disable_torch_init()

    print("Loading LLaVA model...")
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH, None, "llava-v1.5-7b")

    # Monkey-patch all 32 layers to collect metrics
    import types
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        vis_fwd = _make_vis_forward(attn.forward, i)
        attn.forward = types.MethodType(lambda self, *a, _f=vis_fwd, **kw: _f(*a, **kw), attn)

    # Prepare input
    qs = DEFAULT_IMAGE_TOKEN + '\n' + "Describe this image in detail."
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()

    image = Image.open(IMG_PATH).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    # Single forward pass (prefill only, no generation)
    print(f"Running forward pass (q_len will be ~{SYS_LEN + IMG_LEN + 20})...")
    with torch.inference_mode():
        _ = model(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            use_cache=False,
        )

    # Save & plot
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save raw metrics
    json_path = os.path.join(SAVE_DIR, "agvp_llava_layer_analysis.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in layer_metrics.items()}, f, indent=2)
    print(f"Metrics saved to {json_path}")

    # Plot
    plot_path = os.path.join(SAVE_DIR, "agvp_llava_layer_analysis.png")
    plot_metrics(layer_metrics, plot_path)

    # Print summary table
    print(f"\n{'Layer':>5} {'p_vis':>7} {'p_sys':>7} {'imbal':>7} {'cos(v,s)':>8} {'||proj_s||':>10} {'||proj_v||':>10}")
    print("-" * 60)
    for l in sorted(layer_metrics.keys()):
        m = layer_metrics[l]
        print(f"{l:>5} {m['p_vis']:>7.4f} {m['p_sys']:>7.4f} {m['imbalance']:>7.4f} {m['cos_vis_sys']:>8.4f} {m['proj_sys_mag']:>10.4f} {m['proj_vis_mag']:>10.4f}")


if __name__ == "__main__":
    main()
