import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
def data_prepare(data):
    """
    加载并处理存储的 pt 文件数据。
    """
    results = torch.load(data)
    vis_flows, attn_allocs = [], []

    for i, result in enumerate(results):
        vis_flow, attn_alloc = result
        vis_flows.append(vis_flow)
        attn_allocs.append(attn_alloc)

    vis_flows = np.array(vis_flows)
    attn_allocs = np.array(attn_allocs)

    vis_flows = vis_flows.mean(axis=0).transpose(1, 0)
    attn_allocs = attn_allocs.mean(axis=0).transpose(1, 0)

    return (vis_flows, attn_allocs)


def plot_vis_flow(data, path):
    """
    绘制并保存视觉流向 (Visual Flow) 显著性得分图。
    """
    # 与原版 analysis_plot.py 保持一致
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)

    num_layers = data.shape[1]
    x = np.arange(num_layers)
    colors = ['#bf1e2e', '#73bad6']
    custom_legend_labels = ['Intra-Visual Flow', 'Visual-Textual Flow']

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(x, data[0], label=custom_legend_labels[0], alpha=0.5, color=colors[0], width=0.75, edgecolor='black', linewidth=1.5)
    ax.bar(x, data[1], label=custom_legend_labels[1], alpha=0.5, color=colors[1], width=0.75, edgecolor='black', linewidth=1.5)

    font_properties = {'weight': 'bold', 'size': 18}

    plt.xlabel('Transformer Layer', fontdict=font_properties)
    plt.ylabel('Saliency Score', fontdict=font_properties)
    plt.legend(prop={'weight': 'bold', 'size': 12})

    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_attn_alloc(data, path):
    """
    绘制并保存注意力分配 (Attention Allocation) 图。
    """
    normalized_data = data / data.sum(axis=0)
    num_layers = normalized_data.shape[1]
    x = np.arange(num_layers)
    colors = ['#ff5e65', '#90bee0' , '#4B74B2']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # 堆叠柱状图
    ax.bar(x, normalized_data[0], color=colors[0], bottom=normalized_data[2] + normalized_data[1], edgecolor='black', linewidth=1, label='System Prompts')
    ax.bar(x, normalized_data[1], color=colors[1], bottom=normalized_data[2], edgecolor='black', linewidth=1, label='Visual Features')
    ax.bar(x, normalized_data[2], color=colors[2], edgecolor='black', linewidth=1, label='User Instructions')

    ax.set_xlabel('Transfomer Layer', fontsize=18, fontweight='bold')
    ax.set_ylabel('Attention Allocation', fontsize=18, fontweight='bold')
    ax.set_xticks(x[::3])
    ax.set_xlim(-1.5, num_layers + 0.5)

    ax.tick_params(axis='both', labelsize=18)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    for text in legend.get_texts():
        text.set_fontsize(18)
        text.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=str,
        default=os.environ.get("S1MINI_DATA_FILE", "./outputs/analysis/res_coco_random_s1mini.pt"),
    )
    parser.add_argument(
        "--attn-plot",
        type=str,
        default=os.environ.get("S1MINI_ATTN_PLOT", "./outputs/analysis/attn_allocs_s1mini.png"),
    )
    parser.add_argument(
        "--vis-flow-plot",
        type=str,
        default=os.environ.get("S1MINI_VIS_FLOW_PLOT", "./outputs/analysis/vis_flows_s1mini.png"),
    )
    args = parser.parse_args()

    vis_flows, attn_allocs = data_prepare(args.data_file)
    plot_attn_alloc(attn_allocs, args.attn_plot)
    plot_vis_flow(vis_flows, args.vis_flow_plot)