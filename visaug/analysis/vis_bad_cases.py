"""
Bad case visualizer for S1-mini.
For each bad case from find_bad_cases.py, runs two forward passes
(baseline vs AGVP) and generates comparison plots:
  1. Attention heatmap (mid-layer)
  2. K/V norm comparison across layers
  3. Direction analysis (d_vis/d_sys)

Uses wrapper pattern: calls original forward, then extracts/modifies outputs.
No qwen3 imports needed -- avoids environment dependency.

Usage:
    python visaug/analysis/vis_bad_cases.py \
        --index outputs/analysis/bad_cases/bad_cases_index.json \
        --case-idx 0
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import json
import types
import copy
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_PATH = "/root/code/Intern-S1-mini"
IMG_FOLDER = "/root/code/ClearSight/data/coco/val2014"
OUTPUT_BASE = "/root/code/ClearSight/outputs/analysis/bad_cases"

_collector = {}


def _make_baseline_wrapper(orig_forward, layer_idx):
    """Wrapper that runs original forward and collects metrics.
    Matches Qwen3Attention forward signature (used by S1-mini LM).
    """
    def wrapped(self, hidden_states, attention_mask=None, output_attentions=None, **kwargs):
        # Force output_attentions to get attention probabilities for metrics
        result = orig_forward(hidden_states, attention_mask=attention_mask,
                              output_attentions=True, **kwargs)
        attn_output, attn_probs = result

        bsz, q_len, _ = hidden_states.size()
        SYS_LEN = getattr(self, '_vis_sys_len', 0)
        IMG_LEN = getattr(self, '_vis_img_len', 0)

        if q_len > SYS_LEN + IMG_LEN and SYS_LEN + IMG_LEN > 0:
            with torch.no_grad():
                text_start = SYS_LEN + IMG_LEN
                value_states = self.v_proj(hidden_states)
                head_dim = self.head_dim
                # Qwen3Attention doesn't store num_key_value_heads directly
                num_kv_heads = self.k_proj.out_features // head_dim
                num_heads = num_kv_heads * self.num_key_value_groups
                value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                n_groups = self.num_key_value_groups
                if n_groups > 1:
                    value_states = value_states[:, :, None, :, :].expand(
                        bsz, num_kv_heads, n_groups, q_len, head_dim
                    ).reshape(bsz, num_heads, q_len, head_dim)

                text_to_img = attn_probs[:, :, text_start:, SYS_LEN:SYS_LEN + IMG_LEN]
                text_to_sys = attn_probs[:, :, text_start:, :SYS_LEN]
                img_values = value_states[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
                sys_values = value_states[:, :, :SYS_LEN, :]

                p_vis = text_to_img.sum(dim=-1).mean().item()
                p_sys = text_to_sys.sum(dim=-1).mean().item()
                imb = max(0.0, min(1.0, (p_sys - p_vis) / (p_sys + p_vis + 1e-8)))

                d_vis = torch.matmul(text_to_img, img_values).mean(dim=2)
                d_sys = torch.matmul(text_to_sys, sys_values).mean(dim=2)
                d_vis_hat = F.normalize(d_vis, dim=-1)
                d_sys_hat = F.normalize(d_sys, dim=-1)
                cos_vs = (d_vis_hat * d_sys_hat).sum(dim=-1).mean().item()

                # Text↔image / system direction alignment (cos_gap)
                raw_out = torch.matmul(attn_probs, value_states)  # (bsz, heads, q_len, head_dim)
                text_raw = raw_out[:, :, text_start:, :]
                text_hat = F.normalize(text_raw, dim=-1)
                img_centroid = value_states[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :].mean(dim=2, keepdim=True).unsqueeze(2)
                sys_centroid = value_states[:, :, :SYS_LEN, :].mean(dim=2, keepdim=True).unsqueeze(2)
                img_centroid_hat = F.normalize(img_centroid, dim=-1)
                sys_centroid_hat = F.normalize(sys_centroid, dim=-1)
                cos_ti = (text_hat * img_centroid_hat).sum(dim=-1).mean().item()
                cos_ts = (text_hat * sys_centroid_hat).sum(dim=-1).mean().item()
                cos_gap_val = cos_ti - cos_ts

                _collector[layer_idx] = {
                    "attn_probs": attn_probs.cpu(),
                    "value_norm": value_states.norm(dim=-1).mean().item(),
                    "text_out": attn_output[:, text_start:, :].cpu(),
                    "p_vis": p_vis,
                    "p_sys": p_sys,
                    "imbalance": imb,
                    "cos_vis_sys": cos_vs,
                    "cos_text_img": cos_ti,
                    "cos_text_sys": cos_ts,
                    "cos_gap": cos_gap_val,
                    "d_vis_hat": d_vis_hat.cpu(),
                    "d_sys_hat": d_sys_hat.cpu(),
                }

        return result
    return wrapped


def _make_agvp_wrapper(orig_forward, layer_idx):
    """Wrapper that runs original forward, then applies AGVP projection.
    Matches Qwen3Attention forward signature (used by S1-mini LM).
    """
    def wrapped(self, hidden_states, attention_mask=None, output_attentions=None, **kwargs):
        # Force output_attentions to get attention probabilities for metrics
        result = orig_forward(hidden_states, attention_mask=attention_mask,
                              output_attentions=True, **kwargs)
        attn_output, attn_probs = result

        bsz, q_len, _ = hidden_states.size()
        SYS_LEN = getattr(self, '_vis_sys_len', 0)
        IMG_LEN = getattr(self, '_vis_img_len', 0)
        did_intervene = False

        if SYS_LEN + IMG_LEN > 0 and q_len > SYS_LEN + IMG_LEN:
            text_start = SYS_LEN + IMG_LEN

            with torch.no_grad():
                # Recompute value_states (same as original forward used)
                value_states = self.v_proj(hidden_states)
                head_dim = self.head_dim
                num_kv_heads = self.k_proj.out_features // head_dim
                num_heads = num_kv_heads * self.num_key_value_groups
                value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                n_groups = self.num_key_value_groups
                if n_groups > 1:
                    value_states = value_states[:, :, None, :, :].expand(
                        bsz, num_kv_heads, n_groups, q_len, head_dim
                    ).reshape(bsz, num_heads, q_len, head_dim)

                img_values = value_states[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
                sys_values = value_states[:, :, :SYS_LEN, :]

                text_to_img = attn_probs[:, :, text_start:, SYS_LEN:SYS_LEN + IMG_LEN]
                text_to_sys = attn_probs[:, :, text_start:, :SYS_LEN]

                d_vis = torch.matmul(text_to_img, img_values).mean(dim=2)
                d_sys = torch.matmul(text_to_sys, sys_values).mean(dim=2)
                d_vis_hat = F.normalize(d_vis, dim=-1)
                d_sys_hat = F.normalize(d_sys, dim=-1)

                cos_vs = (d_vis_hat * d_sys_hat).sum(dim=-1).mean()
                p_vis_val = text_to_img.sum(dim=-1).mean()
                p_sys_val = text_to_sys.sum(dim=-1).mean()
                imb = torch.clamp(
                    (p_sys_val - p_vis_val) / (p_sys_val + p_vis_val + 1e-8),
                    min=0.0, max=1.0,
                ).item()
                cos_abs = abs(cos_vs.item())

                # Recompute raw attention output (pre-o_proj) to modify
                raw_output = torch.matmul(attn_probs, value_states)  # (bsz, heads, q_len, head_dim)
                text_raw = raw_output[:, :, text_start:, :]
                orig_norm = text_raw.norm(dim=-1, keepdim=True)

                # Hybrid AGVP intervention (same as infer_pope_s1mini_agvp.py)
                if cos_vs < 0 and imb > 0:
                    alpha = cos_abs
                    text_hat = F.normalize(text_raw, dim=-1)
                    d_vis_exp = d_vis_hat.unsqueeze(2).expand_as(text_raw)
                    new_dir = F.normalize((1 - alpha) * text_hat + alpha * d_vis_exp, dim=-1)
                    text_raw = new_dir * orig_norm
                    did_intervene = True
                elif cos_vs > 0 and imb > 0:
                    strength = imb * cos_abs
                    proj_sys = (text_raw * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
                    proj_vis = (text_raw * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)
                    text_raw = text_raw - strength * proj_sys + strength * proj_vis
                    text_raw = text_raw * (orig_norm / (text_raw.norm(dim=-1, keepdim=True) + 1e-8))
                    did_intervene = True

                if did_intervene:
                    raw_output = torch.cat([raw_output[:, :, :text_start, :], text_raw], dim=2)
                    # Apply projection_layer to corrected output (same as original forward)
                    raw_output_t = raw_output.transpose(1, 2).contiguous()
                    raw_output_t = raw_output_t.reshape(bsz, q_len, -1).contiguous()
                    attn_output = self.o_proj(raw_output_t)

                _collector[layer_idx] = {
                    "attn_probs": attn_probs.cpu(),
                    "value_norm": value_states.norm(dim=-1).mean().item(),
                    "text_out": attn_output[:, text_start:, :].cpu(),
                    "p_vis": p_vis_val.item(),
                    "p_sys": p_sys_val.item(),
                    "imbalance": imb,
                    "cos_vis_sys": cos_vs.item(),
                    "d_vis_hat": d_vis_hat.cpu(),
                    "d_sys_hat": d_sys_hat.cpu(),
                    "did_intervene": did_intervene,
                }

                # Text↔image / system direction alignment (cos_gap) from current text_raw
                text_hat = F.normalize(text_raw, dim=-1)
                img_centroid = value_states[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :].mean(dim=2, keepdim=True).unsqueeze(2)
                sys_centroid = value_states[:, :, :SYS_LEN, :].mean(dim=2, keepdim=True).unsqueeze(2)
                img_centroid_hat = F.normalize(img_centroid, dim=-1)
                sys_centroid_hat = F.normalize(sys_centroid, dim=-1)
                cos_ti = (text_hat * img_centroid_hat).sum(dim=-1).mean().item()
                cos_ts = (text_hat * sys_centroid_hat).sum(dim=-1).mean().item()
                cos_gap_val = cos_ti - cos_ts
                _collector[layer_idx].update({
                    "cos_text_img": cos_ti,
                    "cos_text_sys": cos_ts,
                    "cos_gap": cos_gap_val,
                })

        # The original eager_attention_forward returns (attn_output_o, attn_weights_out)
        # where attn_output_o is the tensor after o_proj.
        # The decoder_layer expects this back and uses it directly.
        # Since we already applied o_proj above when we modified, we repack:
        if did_intervene and attn_output.dim() == 3:
            # Return in same format as original
            return attn_output, attn_probs

        return result
    return wrapped


# ──────────────────────────────────────────────
# Plotting functions (unchanged from original)
# ──────────────────────────────────────────────
def plot_attention_heatmap(baseline_data, agvp_data, layer_idx, save_path, sys_len, img_len):
    """Compare attention patterns at a specific layer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in zip(axes, [baseline_data, agvp_data],
                                ["Baseline", f"AGVP (layer {layer_idx})"]):
        attn = data[layer_idx]["attn_probs"]
        attn_mean = attn.mean(dim=1).squeeze(0)
        text_start = sys_len + img_len
        if attn_mean.shape[0] > text_start:
            attn_slice = attn_mean[text_start:, :].float().numpy()
            im = ax.imshow(attn_slice, aspect="auto", cmap="viridis", vmin=0)
            ax.set_xlabel("KV position")
            ax.set_ylabel("Text token")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            ax.axvline(x=sys_len, color="red", linestyle="--", linewidth=0.5, label="sys|img")
            ax.axvline(x=sys_len + img_len, color="blue", linestyle="--", linewidth=0.5, label="img|txt")
            if ax == axes[0]:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kv_norm(baseline_data, agvp_data, save_path):
    """Compare value norms across layers."""
    layers_b = sorted(baseline_data.keys())
    layers_a = sorted(agvp_data.keys())

    val_b = [baseline_data[l]["value_norm"] for l in layers_b]
    val_a = [agvp_data[l]["value_norm"] for l in layers_a]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(layers_b, val_b, "o-", color="steelblue", label="Baseline V")
    ax.plot(layers_a, val_a, "s--", color="coral", label="AGVP V")
    ax.set_xlabel("Layer")
    ax.set_ylabel("||V|| mean")
    ax.set_title("Value State Norm")
    ax.legend()
    ax.grid(True, alpha=0.15)

    # Second panel: value norm delta
    ax = axes[1]
    val_delta = [abs(a - b) for a, b in zip(val_a, val_b)]
    ax.bar(layers_b, val_delta, color="teal", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("|Δ||V|||")
    ax.set_title("Value Norm Change (|AGVP - Baseline|)")
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_direction_analysis(baseline_data, agvp_data, save_path):
    """Compare direction metrics: imbalance, cos(d_vis,d_sys), p_vis/p_sys."""
    layers = sorted(baseline_data.keys())

    imb_b = [baseline_data[l]["imbalance"] for l in layers]
    imb_a = [agvp_data[l]["imbalance"] for l in layers]
    cos_b = [baseline_data[l]["cos_vis_sys"] for l in layers]
    cos_a = [agvp_data[l]["cos_vis_sys"] for l in layers]
    pv_b = [baseline_data[l]["p_vis"] for l in layers]
    pv_a = [agvp_data[l]["p_vis"] for l in layers]
    ps_b = [baseline_data[l]["p_sys"] for l in layers]
    ps_a = [agvp_data[l]["p_sys"] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    x = np.array(layers)

    ax = axes[0]
    ax.plot(x, imb_b, "o-", color="purple", label="Baseline")
    ax.plot(x, imb_a, "s--", color="darkorange", label="AGVP")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Imbalance")
    ax.set_title("Attention Imbalance")
    ax.legend()
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    ax.plot(x, cos_b, "o-", color="green", label="Baseline")
    ax.plot(x, cos_a, "s--", color="darkorange", label="AGVP")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(d_vis, d_sys)")
    ax.set_title("Direction Separability")
    ax.legend()
    ax.grid(True, alpha=0.15)

    ax = axes[2]
    ax.plot(x, pv_b, "o-", color="steelblue", label="Baseline p_vis")
    ax.plot(x, ps_b, "o-", color="coral", label="Baseline p_sys")
    ax.plot(x, pv_a, "s--", color="steelblue", label="AGVP p_vis", alpha=0.7)
    ax.plot(x, ps_a, "s--", color="coral", label="AGVP p_sys", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention mass")
    ax.set_title("Attention Mass (p_vis / p_sys)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True,
                        help="Path to bad_cases_index.json")
    parser.add_argument("--case-idx", type=int, default=0,
                        help="Which bad case to visualize (0-based)")
    parser.add_argument("--vis-layer", type=int, default=15,
                        help="Which layer to visualize attention heatmap for")
    args = parser.parse_args()

    # Load bad case index
    with open(args.index, "r") as f:
        index_data = json.load(f)

    bad_cases = index_data["bad_cases"]
    if args.case_idx >= len(bad_cases):
        print(f"ERROR: case-idx {args.case_idx} >= {len(bad_cases)} bad cases")
        exit(1)

    bc = bad_cases[args.case_idx]
    qid = bc["question_id"]
    image_file = bc["image"]
    question = bc["question"]
    label = bc["label"]

    print(f"Visualizing bad case {args.case_idx}: qid={qid} label={label}")
    print(f"  Baseline: '{bc['baseline_answer']}'")
    print(f"  Method:   '{bc['method_answer']}'")

    # Output directory for this case
    case_dir = os.path.join(OUTPUT_BASE, f"case_{qid}")
    os.makedirs(case_dir, exist_ok=True)

    # Load model and processor
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
    )
    model.eval()
    model.language_model.set_attn_implementation("eager")

    # Prepare input
    image_path = os.path.join(IMG_FOLDER, image_file)
    image = Image.open(image_path).convert("RGB")
    qs = question if question.endswith(".") else question + "."
    qs = qs + " Please just answer yes or no."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": qs},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
        enable_thinking=False,
    )

    img_mask = inputs["input_ids"][0] == model.config.image_token_id
    img_pos = img_mask.nonzero(as_tuple=True)[0]
    sys_len = img_pos[0].item()
    img_len = img_pos.shape[0]
    print(f"sys_len={sys_len}, img_len={img_len}")

    device = torch.device("cuda:0")
    inputs = inputs.to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

    num_layers = len(model.language_model.layers)

    # ── Run Baseline ──
    print("Running baseline forward...")
    original_forwards = {}
    for i in range(num_layers):
        attn = model.language_model.layers[i].self_attn
        original_forwards[i] = attn.forward
        attn._vis_sys_len = sys_len
        attn._vis_img_len = img_len
        attn.forward = types.MethodType(_make_baseline_wrapper(original_forwards[i], i), attn)

    _collector.clear()
    with torch.inference_mode():
        _ = model(**inputs, use_cache=False)
    baseline_data = dict(_collector)

    # Restore
    for i in range(num_layers):
        model.language_model.layers[i].self_attn.forward = original_forwards[i]

    # Save baseline metrics
    baseline_metrics = {str(k): {kk: vv for kk, vv in v.items()
                                  if kk not in ("attn_probs", "text_out", "d_vis_hat", "d_sys_hat")}
                         for k, v in baseline_data.items()}
    with open(os.path.join(case_dir, "metrics_baseline.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    # ── Run AGVP ──
    print("Running AGVP forward...")
    for i in range(num_layers):
        attn = model.language_model.layers[i].self_attn
        attn._vis_sys_len = sys_len
        attn._vis_img_len = img_len
        attn.forward = types.MethodType(_make_agvp_wrapper(original_forwards[i], i), attn)

    _collector.clear()
    with torch.inference_mode():
        _ = model(**inputs, use_cache=False)
    agvp_data = dict(_collector)

    # Restore
    for i in range(num_layers):
        model.language_model.layers[i].self_attn.forward = original_forwards[i]

    agvp_metrics = {str(k): {kk: vv for kk, vv in v.items()
                              if kk not in ("attn_probs", "text_out", "d_vis_hat", "d_sys_hat")}
                     for k, v in agvp_data.items()}
    with open(os.path.join(case_dir, "metrics_agvp.json"), "w") as f:
        json.dump(agvp_metrics, f, indent=2)

    # ── Generate plots ──
    print("Generating plots...")

    # 1. Attention heatmap at vis_layer
    vis_layer = args.vis_layer
    if vis_layer in baseline_data and vis_layer in agvp_data:
        hp = os.path.join(case_dir, "attention_heatmap.png")
        plot_attention_heatmap(baseline_data, agvp_data, vis_layer, hp, sys_len, img_len)
        print(f"  Saved {hp}")
    else:
        print(f"  WARNING: vis_layer {vis_layer} not found in data")

    # 2. Value norm comparison
    kvp = os.path.join(case_dir, "kv_norm_comparison.png")
    plot_kv_norm(baseline_data, agvp_data, kvp)
    print(f"  Saved {kvp}")

    # 3. Direction analysis
    dp = os.path.join(case_dir, "direction_analysis.png")
    plot_direction_analysis(baseline_data, agvp_data, dp)
    print(f"  Saved {dp}")

    print(f"\nDone. All outputs in {case_dir}/")
