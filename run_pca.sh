#!/bin/bash

# 创建所需的输出与日志目录
mkdir -p outputs/pca_llava
mkdir -p outputs/pca_s1mini
mkdir -p logs/pca

echo "=== 启动 GPU 2 绝对路径物理隔离 PCA 实验流 ==="

# 指定全局使用 GPU 2
export CUDA_VISIBLE_DEVICES=2

# 提取 clearsight 和 s1mini 环境的绝对 Python 物理路径
LLAVA_PYTHON="/root/miniforge3/envs/clearsight/bin/python"
S1MINI_PYTHON="/root/miniforge3/envs/s1mini/bin/python"

# ================= 任务 1: LLaVA PCA 基础容量测试 (k=3) =================
# echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在运行任务 1/3: LLaVA PCA (k=3)..."
# $LLAVA_PYTHON visaug/inference/infer_pope_llava_pca.py \
#     --question-file data/pope/coco_pope_random.json \
#     --answers-file outputs/pca_llava/res_k3_scale1.5_random.jsonl \
#     --pca-k 3 --scale 1.5 > logs/pca/llava_k3_scale1.5.log 2>&1

# # ================= 任务 2: LLaVA PCA 高保真测试 (k=5) =================
# echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在运行任务 2/3: LLaVA PCA (k=5)..."
# $LLAVA_PYTHON visaug/inference/infer_pope_llava_pca.py \
#     --question-file data/pope/coco_pope_random.json \
#     --answers-file outputs/pca_llava/res_k5_scale2.0_random.jsonl \
#     --pca-k 5 --scale 2.0 > logs/pca/llava_k5_scale2.0.log 2>&1

# ================= 任务 3: S1-mini 跨界深层注入测试 =================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在运行任务 3/3: S1-mini PCA (k=3)..."
$S1MINI_PYTHON visaug/inference/infer_pope_s1mini_pca.py \
    --answers-file outputs/pca_s1mini/res_s1_k3_scale1.0_random.jsonl \
    --pca-k 3 --scale 1.0 > logs/pca/s1mini_k3_scale1.0.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === GPU 2 全套 PCA 串行实验流安全执行完毕！ ==="