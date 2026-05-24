# Worker Log

> Worker: Claude Code | 指令源：Designer.md
> 时间线：2026-05-15 ~ 2026-05-17（Task 11/13 于 05-17 当天完成）

## Task 4：自适应分层 AGVP — 乱码问题根因与修复

**状态**：✅ 完成（`infer_pope_s1mini_minimal.py`）

### 根因

**monkey-patch 层数超过 ~13 层导致生成退化（KV cache 污染），与 AGVP 干预代码无关。**

- 测了不同层数：L0-4 (5层) 开始退化，L0-5 (6层) 严重重复，B7-16 + C17-30 (24层) 退化
- B7-16 (10层) 干净、B7-16 + C17-19 (13层) 干净

### 修复

减少 patched 层数到 13 层：B7-16 rotate + C17-19 project，A(L0-6)、D(L20-35) 不 patch。
新建 `infer_pope_s1mini_minimal.py`，使用 soft gating (cos²·imb)。

### 结果

S1-mini COCO random：Acc 93.03%, F1 92.70%（略低于 baseline 93.33%）

---

## Task 8：三数据集 Baseline

**状态**：✅ 完成（旧版图片预处理，直接 resize 有拉伸）

### LLaVA-v1.5-7b

| 数据集 | Acc |
|--------|-----|
| COCO random | 89.20% |
| A-OKVQA random | 89.10% |
| GQA random | 89.10% |
| **三集平均** | **89.13%** |

### Intern-S1-mini

| 数据集 | Acc | F1 |
|--------|-----|-----|
| COCO random | 93.33% | 93.07% |
| A-OKVQA random | 94.20% | 94.34% |
| GQA random | 92.10% | 92.15% |
| **三集平均** | **93.21%** | — |

---

## Task 9：GQA 图片提取

**状态**：✅ 完成

从 `/root/download/GQA_data/` parquet 文件提取 500 张图片到 `data/gqa/images/`。脚本：`extract_gqa_images.py`。

---

## Task 10：LLaVA Baseline 差距排查

**状态**：✅ 已完成排查

### 问题

论文 Table 2: LLaVA-v1.5-7B Regular Random 三集平均 Acc=87.8%、F1=87.5%。

### 排查结果

| 排查项 | 结果 |
|--------|------|
| Eval 方法（substring vs 单词级） | 无差异 |
| Conv mode（vicuna_v1 vs llava_v1） | 无差异 |
| Prompt 后缀（有无 "answer yes or no"） | 无差异 |
| 图片预处理（bug: 直接 resize → 正确 expand2square） | 修复后 Acc 反而更高 |

### 修复预处理后的 LLaVA 三集 baseline

| 数据集 | Acc | F1 |
|--------|-----|-----|
| COCO random | 90.17% | 89.62% |
| A-OKVQA random | 90.93% | 91.09% |
| GQA random | 89.37% | 89.52% |
| **三集平均** | **90.16%** | **90.08%** |

与论文 Acc 87.8% 差 **2.36 点**，方向与图片预处理无关。最可能原因：**LLaVA checkpoint 版本不同**。

---

## Task 6：双模型分区实验（固定预处理）

**状态**：✅ 完成

### LLaVA-v1.5-7b COCO random（固定预处理 process_images）

| Method | Acc | F1 | Δ F1 |
|--------|-----|-----|------|
| baseline | 90.17% | 89.62% | — |
| VAF L0-31 | 90.43% | 90.36% | +0.74 |
| AGVP L0-31 | 89.93% | 89.28% | −0.34 |
| 分区投影 (L2-8 weak×0.5, L9-31×1.0) | 90.00% | 89.38% | −0.24 |

### Intern-S1-mini COCO random

| Method | Acc | F1 | Δ F1 |
|--------|-----|-----|------|
| baseline | 93.33% | 93.07% | — |
| VAF | 93.37% | 93.31% | +0.24 |
| proj_cospos | 93.37% | 93.10% | +0.03 |
| 分区投影 (L7-16 weak×0.5, L17-19×1.0) | 93.23% | 92.95% | −0.12 |
| minimal_adaptive (B rotate + C project) | 93.03% | 92.70% | −0.30 |

### 关键发现

1. **VAF 双模型最优**：LLaVA +0.74 F1，S1-mini +0.24 F1，pre-softmax 方法一致优于 post-softmax 投影。
2. **AGVP/投影在 LLaVA 上掉分**：AGVP −0.34 F1，分区投影 −0.24 F1。post-softmax 方向估计噪声大，级联误差损伤 Llama 架构。
3. **S1-mini 更鲁棒**：投影基本不掉分（分区投影 −0.12 F1），Qwen3 架构对 post-softmax 干预容忍度更高。
4. **分区投影优于全局 AGVP**：分区强度分配修正有帮助（−0.24 vs −0.34），但本质问题仍是 post-softmax 操作而非强度分配。

### 环境

- Conda: `s1minibak` (S1-mini), `clearsight` (LLaVA)
- GPU: 全部空闲

---

## Task 11：无加权方向 AGVP

**状态**：✅ 完成

### 假设

方向估计噪声来自 attention weights。用 value vectors 直接平均替代 attn-weighted 平均，方向更干净，投影更有效。

### 改动

`infer_pope_llava_proj_zonal_noweight.py` — 在分区投影基础上改两行：
```python
# 旧：attention 加权
d_vis = torch.matmul(text_to_img, img_values).mean(dim=2)
d_sys = torch.matmul(text_to_sys, sys_values).mean(dim=2)

# 新：无加权，value space 质心
d_vis = img_values.mean(dim=2)
d_sys = sys_values.mean(dim=2)
```

### 结果

**LLaVA COCO random：**

| Method | Acc | F1 | Δ F1 |
|--------|-----|-----|------|
| baseline | 90.17% | 89.62% | — |
| VAF L0-31 | 90.43% | **90.36%** | +0.74 |
| **无加权方向 AGVP** | **90.00%** | **89.41%** | **−0.21** |
| 分区投影（原始加权） | 90.00% | 89.38% | −0.24 |
| AGVP L0-31（全局） | 89.93% | 89.28% | −0.34 |

无加权方向 F1=89.41% vs 原始加权 89.38% — **几乎无差异**。假设不成立：attention weights 不是方向估计的主要噪声源。

### 结论

1. 无加权方向没有提升，说明方向估计的 attention 噪声不是 post-softmax 掉分的根因
2. 问题更可能出在投影操作本身：单层投影的信噪比低，或者 value space 的方向本身就不够稳定
3. Task 12（全局校准）可能仍值得一试，但预期收益不高
4. 建议 Designer 评估是否该放弃 post-softmax 路线，或设计更根本性的改进

| # | 任务 | 思路 | 优先级 |
|---|------|------|--------|
| 11 | 无加权方向 | value 均值替代 attn 加权 | ⭐⭐⭐ |
| 12 | 全局校准 | 50-100 张图预计算全局 d_vis[l], d_sys[l] | ⭐⭐⭐ |
| 13 | 只拉不推 | 只做 `+ proj_vis`，不做 `- proj_sys` | ⭐⭐ |
| 14 | 跨层平滑 | 相邻层方向滑动平均 | ⭐⭐ |

---

## Task 13：只拉不推 AGVP

**状态**：✅ 完成

### 假设

投影误差主要来自"推离 sys"方向（`- strength * proj_sys`）。如果 sys 方向本身不准确，推离它可能把有用信号也推掉了。只拉不推能避免这个误差。

### 改动

`infer_pope_llava_proj_zonal_pullonly.py` — 分区投影基础上改一行：
```python
# 旧：推 + 拉
text_out = text_out - strength * proj_sys + strength * proj_vis

# 新：只拉
text_out = text_out + strength * proj_vis
```

### 结果

| Method | Acc | F1 | Δ F1 |
|--------|-----|-----|------|
| baseline | 90.17% | 89.62% | — |
| VAF L0-31 | 90.43% | **90.36%** | +0.74 |
| **只拉不推** | **90.27%** | **89.73%** | **+0.11** |
| 分区投影（推+拉） | 90.00% | 89.38% | −0.24 |
| 无加权方向 | 90.00% | 89.41% | −0.21 |
| AGVP L0-31（全局） | 89.93% | 89.28% | −0.34 |

### 分析

1. **只拉不推比推+拉提升 +0.35 F1**（89.73% vs 89.38%），确认"推 sys"方向在损伤
2. 但仍比 baseline 低 0.11 — 光拉的方向向量本身仍有噪声
3. 这是目前为止最好的 post-softmax 结果，但 VAF（pre-softmax）仍领先 0.63
