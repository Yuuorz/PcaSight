#!/bin/bash

# 激活 conda 基础初始化命令，确保脚本内能正常识别 conda activate
source /root/miniconda3/etc/profile.d/conda.sh || source /root/anaconda3/etc/profile.d/conda.sh

# 创建所需的输出与日志目录
mkdir -p outputs/pca_llava
mkdir -p outputs/pca_s1mini
mkdir -p logs/pca

echo "=== 启动 GPU 2 环境隔离串行 PCA 实验流 ==="

# 指定全局使用 GPU 2
export CUDA_VISIBLE_DEVICES=2

# ================= 任务 1: LLaVA PCA 基础容量测试 (k=3) =================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [环境切换] 正在切入 clearsight 环境..."
conda activate clearsight

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在运行任务 1/3: LLaVA PCA (k=3)..."
python visaug/inference/infer_pope_llava_pca.py \
    --question-file data/pope/coco_pope_random.json \
    --answers-file outputs/pca_llava/res_k3_scale1.5_random.jsonl \
    --pca-k 3 --scale 1.5 > logs/pca/llava_k3_scale1.5.log 2>&1

# ================= 任务 2: LLaVA PCA 高保真测试 (k=5) =================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在运行任务 2/3: LLaVA PCA (k=5)..."
python visaug/inference/infer_pope_llava_pca.py \
    --question-file data/pope/coco_pope_random.json \
    --answers-file outputs/pca_llava/res_k5_scale2.0_random.jsonl \
    --pca-k 5 --scale 2.0 > logs/pca/llava_k5_scale2.0.log 2>&1


# ================= 任务 3: S1-mini 跨界深层注入测试 =================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [环境切换] LLaVA 任务结束。正在切入 s1mini 环境..."
conda activate s1mini

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在运行任务 3/3: S1-mini PCA (k=3)..."
python visaug/inference/infer_pope_s1mini_pca.py \
    --answers-file outputs/pca_s1mini/res_s1_k3_scale1.0_random.jsonl \
    --pca-k 3 --scale 1.0 > logs/pca/s1mini_k3_scale1.0.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === GPU 2 全套多环境 PCA 实验流执行完毕！ ==="