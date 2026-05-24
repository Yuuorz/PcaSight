"""
ClearSight Architectural Tool: Hidden States Subspace PCA Geometric Visualizer.
Plots the high-dimensional spatial distribution changes before and after PCA alignment.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_geometric_subspace_change(text_orig, text_pca, img_tokens, save_path="outputs/analysis/subspace_相变.png"):
    """
    text_orig: [text_len, hidden_dim] - Baseline 原始文本 Token 向量
    text_pca:  [text_len, hidden_dim] - PCA 子空间校准后的文本 Token 向量
    img_tokens: [576, hidden_dim]      - 图片 Token 场矩阵
    """
    # 转为 CPU 上的 numpy 数组
    to_np = lambda x: x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else x
    t_orig = to_np(text_orig)
    t_pca = to_np(text_pca)
    img = to_np(img_tokens)
    
    text_len = t_orig.shape[0]
    img_len = img.shape[0]
    
    # 1. 拼接所有高维向量，统一使用 t-SNE 降维到 3D 空间进行几何拓扑映射
    all_vectors = np.concatenate([t_orig, t_pca, img], axis=0)
    
    print("[ClearSight Visualizer] 正在对高维语义激活空间执行 t-SNE 降维解耦...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, text_len - 1))
    embeddings = tsne.fit_transform(all_vectors)
    
    # 2. 切片剥离各个实体的低维坐标
    embed_orig = embeddings[:text_len]
    embed_pca  = embeddings[text_len:text_len*2]
    embed_img  = embeddings[text_len*2:]
    
    # 3. 开启 3D 画布绘制
    fig = plt.figure(figsize=(16, 7))
    
    # ---- 左图：Baseline 缠结状态 ----
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(embed_img[:, 0], embed_img[:, 1], embed_img[:, 2], c='#1f77b4', alpha=0.3, s=20, label="Visual Tokens (576)")
    ax1.scatter(embed_orig[:, 0], embed_orig[:, 1], embed_orig[:, 2], c='#ff7f0e', alpha=0.9, s=50, edgecolors='k', label="Text Tokens (Baseline)")
    ax1.set_title("1. Baseline: Text & Vision Entangled", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--')
    ax1.legend()
    
    # ---- 右图：Subspace PCA 正交对齐状态 ----
    ax2 = fig.add_subplot(122, projection='3d')
    # 绘制视觉高维超平面底座
    ax2.scatter(embed_img[:, 0], embed_img[:, 1], embed_img[:, 2], c='#1f77b4', alpha=0.2, s=20, label="Visual Subspace Hyperplane")
    # 绘制原始位置作为虚影参考
    ax2.scatter(embed_orig[:, 0], embed_orig[:, 1], embed_orig[:, 2], c='#ff7f0e', alpha=0.15, s=40, linestyle='dashed')
    # 绘制对齐后的新位置
    ax2.scatter(embed_pca[:, 0], embed_pca[:, 1], embed_pca[:, 2], c='#2ca02c', alpha=0.9, s=60, edgecolors='k', label="Text Tokens (Subspace PCA)")
    
    # 绘制投影变相拉力线
    for i in range(text_len):
        ax2.plot([embed_orig[i, 0], embed_pca[i, 0]], 
                 [embed_orig[i, 1], embed_pca[i, 1]], 
                 [embed_orig[i, 2], embed_pca[i, 2]], c='red', linestyle='-', alpha=0.6, linewidth=1.5)
        
    ax2.set_title("2. ClearSight: Multi-Dimensional Orthogonal Pull", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--')
    ax2.legend()
    
    plt.suptitle("Geometric 相变 Visualization: Subspace PCA Projection Effect", fontsize=14, fontweight='bold', y=0.95)
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    print(f"[ClearSight Visualizer] 拓扑相变拓扑可视化图已成功输出至: {save_path}")
    plt.close()
