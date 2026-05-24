"""
Plot comparison of layer-wise cosine metrics between LLaVA and S1-mini.
Loads the two JSON files produced by probe_cosine_llava.py and probe_cosine_s1mini.py.
Output: outputs/analysis/layer_cosine_comparison.png
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = "/root/code/ClearSight/outputs/analysis"


def load_metrics(json_path):
    with open(json_path, "r") as f:
        raw = json.load(f)
    # Convert string keys back to int
    return {int(k): v for k, v in raw.items()}


def plot_comparison(llava_metrics, s1_metrics, save_path):
    metrics_names = [
        ("imbalance", "Imbalance", "purple"),
        ("cos_vis_sys", "cos(d_vis, d_sys)", "green"),
        ("cos_text_img", "cos(text, img)", "steelblue"),
        ("cos_text_sys", "cos(text, sys)", "coral"),
        ("cos_gap", "Vis-Sys Gap", "darkorange"),
        ("attn_entropy_norm", "Attn Entropy (norm)", "brown"),
    ]

    n_metrics = len(metrics_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics))
    fig.suptitle("Layer-wise Cosine Diagnosis: LLaVA vs S1-mini", fontsize=14, fontweight="bold")

    for ax, (key, label, color) in zip(axes, metrics_names):
        # LLaVA
        x_llava = sorted(llava_metrics.keys())
        y_llava = [llava_metrics[l][key] for l in x_llava]
        ax.plot(x_llava, y_llava, "o-", color=color, linewidth=2, label=f"LLaVA ({len(x_llava)} layers)", alpha=0.8)

        # S1-mini
        x_s1 = sorted(s1_metrics.keys())
        y_s1 = [s1_metrics[l][key] for l in x_s1]
        ax.plot(x_s1, y_s1, "s--", color=color, linewidth=2, label=f"S1-mini ({len(x_s1)} layers)", alpha=0.8)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_ylabel(label)
        ax.set_xlabel("Layer")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def main():
    llava_path = os.path.join(SAVE_DIR, "layer_cosine_llava.json")
    s1_path = os.path.join(SAVE_DIR, "layer_cosine_s1mini.json")

    if not os.path.exists(llava_path):
        print(f"WARNING: {llava_path} not found. Run probe_cosine_llava.py first.")
        return
    if not os.path.exists(s1_path):
        print(f"WARNING: {s1_path} not found. Run probe_cosine_s1mini.py first.")
        return

    llava_metrics = load_metrics(llava_path)
    s1_metrics = load_metrics(s1_path)

    print(f"LLaVA: {len(llava_metrics)} layers ({min(llava_metrics.keys())}-{max(llava_metrics.keys())})")
    print(f"S1-mini: {len(s1_metrics)} layers ({min(s1_metrics.keys())}-{max(s1_metrics.keys())})")

    save_path = os.path.join(SAVE_DIR, "layer_cosine_comparison.png")
    plot_comparison(llava_metrics, s1_metrics, save_path)


if __name__ == "__main__":
    main()
