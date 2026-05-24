"""
Supplement plots for Step 3 rationale.
Generates 3 evidence figures showing WHY adaptive routing is needed.

Usage:
    python visaug/analysis/supplement_plots.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ANALYSIS_DIR = "/root/code/ClearSight/outputs/analysis"
SAVE_DIR = os.path.join(ANALYSIS_DIR, "supplement")
os.makedirs(SAVE_DIR, exist_ok=True)

ZONE_COLORS = {"A": "#66c2a5", "B": "#fc8d62", "C": "#8da0cb", "D": "#e78ac3"}
ZONE_RANGES = {"A": (0, 7), "B": (7, 17), "C": (17, 31), "D": (31, 36)}
ZONE_NAMES = {
    "A": "L0-6  Early",
    "B": "L7-16 Sys-Dominant",
    "C": "L17-30 Entangled",
    "D": "L31-35 Late",
}


def load_probe(name):
    path = os.path.join(ANALYSIS_DIR, f"layer_cosine_{name}.json")
    data = json.load(open(path))
    return {int(k): v for k, v in data.items()}


def load_bad_case(case_id):
    base_path = os.path.join(ANALYSIS_DIR, "bad_cases",
                             f"case_{case_id}", "metrics_baseline.json")
    agvp_path = os.path.join(ANALYSIS_DIR, "bad_cases",
                             f"case_{case_id}", "metrics_agvp.json")
    base = json.load(open(base_path))
    agvp = json.load(open(agvp_path))
    return {int(k): v for k, v in base.items()}, \
        {int(k): v for k, v in agvp.items()}


def draw_zone_bg(ax, ylo, yhi, alpha=0.06):
    """Draw colored zone backgrounds."""
    for z, (lo, hi) in ZONE_RANGES.items():
        ax.axvspan(lo, hi - 0.5, alpha=alpha, color=ZONE_COLORS[z],
                   zorder=-1)
        mid = (lo + hi) / 2
        ax.text(mid, ylo + (yhi - ylo) * 0.02, z,
                transform=ax.get_xaxis_transform(),
                fontsize=11, fontweight="bold", color=ZONE_COLORS[z],
                ha="center", va="bottom", alpha=0.7)


# ──────────────────────────────────────────────
# Figure 1: Layer Profile with Zone Annotations
# ──────────────────────────────────────────────
def fig_layer_profile():
    """S1-mini 3-row profile: attention, cos_gap, cos_vis_sys.
    Zone-colored backgrounds + operation recommendation annotations.
    """
    s1 = load_probe("s1mini")
    layers = sorted(s1.keys())
    x = np.arange(len(layers))

    fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                             gridspec_kw={"height_ratios": [1, 1, 1]})
    fig.suptitle(
        "S1-mini Layer Profile: Three Zones, Three Different Behaviors",
        fontsize=14, fontweight="bold", y=1.01)

    # Row 1: attention mass
    ax = axes[0]
    p_vis = [s1[l]["p_vis"] for l in layers]
    p_sys = [s1[l]["p_sys"] for l in layers]
    ax.bar(x - 0.2, p_vis, 0.4, label="p_vis (→image)", color="steelblue", alpha=0.85)
    ax.bar(x + 0.2, p_sys, 0.4, label="p_sys (→system)", color="coral", alpha=0.85)
    draw_zone_bg(ax, 0, 1)
    ax.set_ylabel("Attention Mass")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.text(0.5, 1.02, "▲  L7-16: p_sys dominates (system bias)",
            transform=ax.transAxes, fontsize=9, ha="center", color="#d95f02")

    # Row 2: cos_gap
    ax = axes[1]
    cos_gap = [s1[l]["cos_gap"] for l in layers]
    colors = ["#d73027" if g < 0 else "#4575b4" for g in cos_gap]
    ax.bar(x, cos_gap, color=colors, alpha=0.8, width=0.7)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    draw_zone_bg(ax, -0.6, 0.2)
    ax.set_ylabel("cos_gap")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)

    # Annotation: L20 paradox
    l20_idx = layers.index(20)
    ax.annotate(
        f"L20: cos_gap={s1[20]['cos_gap']:.2f}\n(p_vis={s1[20]['p_vis']:.2f})",
        xy=(l20_idx, s1[20]["cos_gap"]),
        xytext=(l20_idx + 5, s1[20]["cos_gap"] - 0.25),
        fontsize=8, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5", alpha=0.9))

    # Zone annotations at bottom of row 2
    zone_notes = {
        "A": ("Low imb\nSkip", 0.06),
        "B": ("High imb\ncos_vs<0\n→ ROTATE", 0.50),
        "C": ("cos_vs>0\n→ PROJECT", 0.30),
        "D": ("Low imb\nSkip", 0.10),
    }
    for z, (lo, hi) in ZONE_RANGES.items():
        mid = (lo + hi) / 2
        txt = zone_notes[z][0]
        ax.text(mid, -0.55, txt, transform=ax.get_xaxis_transform(),
                fontsize=7.5, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=ZONE_COLORS[z], alpha=0.25))

    # Row 3: cos_vis_sys (direction separability)
    ax = axes[2]
    cos_vs = [s1[l]["cos_vis_sys"] for l in layers]
    ax.bar(x, cos_vs, color=["#d73027" if v < 0 else "#4575b4" for v in cos_vs],
           alpha=0.8, width=0.7)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(y=0.3, color="green", linestyle=":", alpha=0.3, linewidth=0.6)
    draw_zone_bg(ax, -0.6, 0.7)
    ax.set_ylabel("cos_vis_sys")
    ax.set_xlabel("Layer")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)

    # Threshold line annotation
    ax.text(len(layers) - 1, 0.33, "cos=0 (separable→entangled)",
            fontsize=7, color="green", ha="right", alpha=0.6)

    # Zone labels
    for z, (lo, hi) in ZONE_RANGES.items():
        mid = (lo + hi) / 2
        ax.text(mid, 0.65, ZONE_NAMES[z], transform=ax.get_xaxis_transform(),
                fontsize=8.5, ha="center", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=ZONE_COLORS[z], alpha=0.3))

    # Bottom summary bar
    ax.text(0.5, -0.25,
            "Key insight: cos_vis_sys<0 → rotation separates directions | "
            "cos_vis_sys>0 → rotation harms both directions → use projection instead",
            transform=ax.transAxes, fontsize=9, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "layer_profile_with_zones.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ──────────────────────────────────────────────
# Figure 2: The Cost of One-Size-Fits-All
# ──────────────────────────────────────────────
def fig_method_comparison():
    """Bar chart of all S1-mini methods colored by operation type.
    Shows bottom line: rotation degrades, projection matches baseline.
    """
    methods = [
        ("Baseline",     93.33, 93.07, "none",   "#666666"),
        ("VAF",          93.37, 93.31, "vaf",    "#66c2a5"),
        ("proj_cospos",  93.37, 93.10, "proj",   "#4575b4"),
        ("agvp_cospos",  93.43, 93.19, "proj",   "#4575b4"),
        ("hybrid_soft",  93.07, 92.74, "hybrid", "#fdae61"),
        ("AGVP L0-16",   92.07, 91.52, "rot",    "#d73027"),
        ("AGVR",         91.97, 91.44, "rot",    "#d73027"),
        ("hybrid",       91.90, 91.37, "rot",    "#d73027"),
        ("AGVP L0-35",   91.83, 91.25, "rot",    "#d73027"),
        ("rot_only",     91.80, 91.25, "rot",    "#d73027"),
    ]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.suptitle(
        "S1-mini POPE Random: The Cost of Wrong-Zone Intervention",
        fontsize=14, fontweight="bold")

    names = [m[0] for m in methods]
    accs = [m[1] for m in methods]
    f1s = [m[2] for m in methods]
    types = [m[3] for m in methods]
    base_colors = [m[4] for m in methods]

    x = np.arange(len(methods))
    w = 0.35

    # Baseline reference
    base_acc = methods[0][1]
    base_f1 = methods[0][2]
    ax.axhline(y=base_acc, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(len(methods) - 0.5, base_acc + 0.05, f"Baseline Acc={base_acc:.1f}%",
            fontsize=8, color="gray", ha="right")

    # Bars
    bars1 = ax.bar(x - w / 2, accs, w, label="Accuracy",
                   color=[c for c in base_colors], alpha=0.8, edgecolor="white", linewidth=0.3)
    bars2 = ax.bar(x + w / 2, f1s, w, label="F1 Score",
                   color=[c for c in base_colors], alpha=0.4, edgecolor="white", linewidth=0.3)

    # Operation type legend boxes at bottom
    op_legend = [
        ("none", "#666666", "No intervention"),
        ("vaf", "#66c2a5", "VAF (pre-softmax scale)"),
        ("proj", "#4575b4", "Projection (post-softmax)"),
        ("hybrid", "#fdae61", "Hybrid"),
        ("rot", "#d73027", "Rotation (AGVR/AGVP)"),
    ]
    leg_y = -0.18
    for i, (_, c, label) in enumerate(op_legend):
        ax.add_patch(plt.Rectangle((i * 0.2, leg_y), 0.04, 0.04,
                                    transform=ax.transAxes, color=c, alpha=0.8,
                                    clip_on=False))
        ax.text(i * 0.2 + 0.05, leg_y + 0.01, label,
                transform=ax.transAxes, fontsize=7.5, va="center")

    # Delta annotations
    rot_methods = [(i, m) for i, m in enumerate(methods) if m[3] == "rot"]
    avg_loss = np.mean([base_f1 - m[2] for _, m in rot_methods])
    min_rot_f1 = min(m[2] for _, m in rot_methods)
    ax.annotate(
        f"Rotation methods: avg F1 loss = {avg_loss:.2f} pts",
        xy=(rot_methods[0][0], rot_methods[0][1][2]),
        xytext=(rot_methods[-1][0], min_rot_f1 - 1.5),
        fontsize=9, color="#d73027", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0", alpha=0.9),
        arrowprops=dict(arrowstyle="<->", color="#d73027", lw=1.5))

    proj_only = [(i, m) for i, m in enumerate(methods) if m[3] == "proj"]
    max_proj_f1 = max(m[2] for _, m in proj_only)
    proj_mid = (proj_only[0][0] + proj_only[-1][0]) / 2
    ax.text(proj_mid, max_proj_f1 + 0.8,
            "Projection-only: matches baseline",
            fontsize=8, color="#4575b4", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f5ff", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5, rotation=25, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(89, 95)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.15)

    # Callout box
    ax.text(0.98, 0.98,
            "Bottom Line:\n"
            "• Rotation in cos<0 layers → −1.6 F1 pts\n"
            "• Projection in cos>0 layers → 0.0 F1 loss\n"
            "→ Adaptive routing is the answer",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "method_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ──────────────────────────────────────────────
# Figure 3: Zone-Level Bad Case Breakdown
# ──────────────────────────────────────────────
def fig_badcase_zone_breakdown():
    """Zone-level comparison: baseline vs AGVR cos_gap for each bad case.
    L7-16 (rotation works) vs L17-30 (rotation fails).
    """
    case_ids = [23, 35, 141, 167]
    case_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    # Compute zone averages for each case
    zones_to_show = [("B", "L7-16  (cos_vs<0 → rotation OK)"),
                     ("C", "L17-30  (cos_vs>0 → rotation harmful)")]
    results = {z: {"base": [], "agvp": []} for z, _ in zones_to_show}

    for cid in case_ids:
        base, agvp = load_bad_case(cid)
        for z, _ in zones_to_show:
            lo, hi = ZONE_RANGES[z]
            layers = [l for l in range(lo, hi) if l in base and l in agvp]
            base_cg = np.mean([base[l].get("cos_gap", 0) for l in layers])
            agvp_cg = np.mean([agvp[l].get("cos_gap", 0) for l in layers])
            results[z]["base"].append(base_cg)
            results[z]["agvp"].append(agvp_cg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(
        "Bad Case Mechanism: Rotation Helps L7-16, Harms (or Misses) L17-30",
        fontsize=13, fontweight="bold")

    for col, (z, title) in enumerate(zones_to_show):
        ax = axes[col]
        x = np.arange(len(case_ids))
        w = 0.3

        base_vals = results[z]["base"]
        agvp_vals = results[z]["agvp"]

        bars_base = ax.bar(x - w / 2, base_vals, w, label="Baseline",
                           color="gray", alpha=0.5, edgecolor="white")
        bars_agvp = ax.bar(x + w / 2, agvp_vals, w, label="AGVR",
                           color=case_colors, alpha=0.8, edgecolor="white")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Case {cid}" for cid in case_ids], fontsize=8)
        ax.set_ylabel("cos_gap (zone avg)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

        # Direction annotation
        for i in range(len(case_ids)):
            delta = agvp_vals[i] - base_vals[i]
            color = "green" if delta > 0 else "red"
            ax.annotate(f"{delta:+.3f}", (x[i], agvp_vals[i]),
                        fontsize=7, color=color, fontweight="bold",
                        xytext=(0, 6), textcoords="offset points", ha="center")

        # Zone summary
        mean_base = np.mean(base_vals)
        mean_agvp = np.mean(agvp_vals)
        ax.text(0.97, 0.05,
                f"Avg: {mean_base:.3f} → {mean_agvp:.3f} ({mean_agvp - mean_base:+.3f})",
                transform=ax.transAxes, fontsize=8.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        # What AGVR does in this zone
        if "rotation OK" in title:
            ax.text(0.5, -0.25, "AGVR rotation: cos_gap improves (moves toward +)",
                    transform=ax.transAxes, fontsize=8.5, ha="center", color="green")
        else:
            ax.text(0.5, -0.25,
                    "AGVR rotation: minimal effect — cos_vs>0 directions entangled",
                    transform=ax.transAxes, fontsize=8.5, ha="center", color="red")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "badcase_zone_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ──────────────────────────────────────────────
# Figure 5: Paper Figure — Phase Transition
# Story: S1-mini has a phase transition at L16-17, LLaVA doesn't.
#   Left = S1-mini, Right = LLaVA
#   Top row = p_vis + p_sys bars
#   Bottom row = cos_vis_sys line (THE key metric)
#   One sentence conclusion at bottom.
# ──────────────────────────────────────────────
def fig_paper_phase_transition():
    s1 = load_probe("s1mini")
    ll = load_probe("llava")
    s1_layers = sorted(s1.keys())
    ll_layers = sorted(ll.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [1, 1]})

    # ── Top-left: S1-mini attention ──
    ax = axes[0, 0]
    x1 = np.arange(len(s1_layers))
    p_vis1 = [s1[l]["p_vis"] for l in s1_layers]
    p_sys1 = [s1[l]["p_sys"] for l in s1_layers]
    ax.bar(x1 - 0.2, p_vis1, 0.4, label="to image", color="steelblue", alpha=0.85)
    ax.bar(x1 + 0.2, p_sys1, 0.4, label="to system", color="coral", alpha=0.85)
    for z, (lo, hi) in ZONE_RANGES.items():
        ax.axvspan(lo - 0.5, hi - 0.5, alpha=0.06, color=ZONE_COLORS[z], zorder=-1)
    ax.axvline(x=16.5, color="darkred", linestyle="--", linewidth=1.8)
    ax.set_title("S1-mini: Attention Mass", fontsize=11, fontweight="bold")
    ax.set_ylabel("Attention mass")
    ax.set_xticks(x1[::4]); ax.set_xticklabels([str(l) for l in s1_layers][::4], fontsize=7)
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    # ── Top-right: LLaVA attention ──
    ax = axes[0, 1]
    x2 = np.arange(len(ll_layers))
    p_vis2 = [ll[l]["p_vis"] for l in ll_layers]
    p_sys2 = [ll[l]["p_sys"] for l in ll_layers]
    ax.bar(x2 - 0.2, p_vis2, 0.4, label="to image", color="steelblue", alpha=0.85)
    ax.bar(x2 + 0.2, p_sys2, 0.4, label="to system", color="coral", alpha=0.85)
    ax.set_title("LLaVA-v1.5: Attention Mass", fontsize=11, fontweight="bold")
    ax.set_ylabel("Attention mass")
    ax.set_xticks(x2[::4]); ax.set_xticklabels([str(l) for l in ll_layers][::4], fontsize=7)
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    # ── Bottom-left: S1-mini cos_vis_sys (THE phase transition) ──
    ax = axes[1, 0]
    cos_vs1 = [s1[l]["cos_vis_sys"] for l in s1_layers]
    colors1 = ["#d73027" if v < 0 else "#4575b4" for v in cos_vs1]
    ax.bar(x1, cos_vs1, color=colors1, alpha=0.85, width=0.7)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=16.5, color="darkred", linestyle="--", linewidth=1.8)
    for z, (lo, hi) in ZONE_RANGES.items():
        ax.axvspan(lo - 0.5, hi - 0.5, alpha=0.06, color=ZONE_COLORS[z], zorder=-1)
    # Zone labels
    ax.text(3.5, 0.55, "A\nskip", fontsize=9, ha="center", fontweight="bold", color="#66c2a5")
    ax.text(12, -0.55, "B\ncos<0 → rotate", fontsize=9, ha="center", fontweight="bold", color="#fc8d62")
    ax.text(24, 0.55, "C\ncos>0 → project", fontsize=9, ha="center", fontweight="bold", color="#8da0cb")
    ax.text(33, 0.55, "D\nskip", fontsize=9, ha="center", fontweight="bold", color="#e78ac3")
    # L16-17 annotation
    ax.annotate("L16-17\nPhase Transition",
                xy=(16.5, 0), xytext=(20, -0.45),
                fontsize=10, color="darkred", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    ax.set_title("S1-mini: cos(d_vis, d_sys) — direction separability",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("cos_vis_sys  (red=separable, blue=entangled)")
    ax.set_xlabel("Layer")
    ax.set_xticks(x1[::4]); ax.set_xticklabels([str(l) for l in s1_layers][::4], fontsize=7)
    ax.set_ylim(-0.6, 0.7)

    # ── Bottom-right: LLaVA cos_vis_sys (all negative = no transition) ──
    ax = axes[1, 1]
    cos_vs2 = [ll[l]["cos_vis_sys"] for l in ll_layers]
    colors2 = ["#d73027" if v < 0 else "#4575b4" for v in cos_vs2]
    ax.bar(x2, cos_vs2, color=colors2, alpha=0.85, width=0.7)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(16, -0.48, "All layers cos<0\n→ no phase transition\n→ rotation is safe",
            fontsize=10, ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    ax.set_title("LLaVA-v1.5: cos(d_vis, d_sys) — always separable",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("cos_vis_sys  (all red = all separable)")
    ax.set_xlabel("Layer")
    ax.set_xticks(x2[::4]); ax.set_xticklabels([str(l) for l in ll_layers][::4], fontsize=7)
    ax.set_ylim(-0.6, 0.7)

    # ── Bottom caption ──
    fig.suptitle(
        "Why the same intervention works differently: "
        "S1-mini has a direction phase transition at L16-17; LLaVA doesn\'t.",
        fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, 0.01,
             "cos_vis_sys < 0 → directions are separable → rotation is effective.  "
             "cos_vis_sys > 0 → directions are entangled → rotation is harmful, use projection instead.",
             ha="center", fontsize=10, fontstyle="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = os.path.join(SAVE_DIR, "paper_figure_phase_transition.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Step 3 evidence figures...")
    fig_layer_profile()
    fig_method_comparison()
    fig_badcase_zone_breakdown()
    fig_paper_phase_transition()
    print(f"\nAll plots in {SAVE_DIR}/")
