#!/bin/bash
# POPE inference (yes/no QA) — uses legacy pipeline scripts
# See run_opope.sh for open-ended caption experiments.

mkdir -p logs/llava/pope logs/s1mini/pope

# ===================== LLaVA POPE =====================

# CUDA_VISIBLE_DEVICES=0 nohup conda run -n clearsight python visaug/inference/pipeline/run_pope_llava_legacy.py \
#     --question-file data/pope/coco_pope_random.json \
#     --answers-file outputs/llava/pope/baseline.jsonl \
#     --pca-k 3 --scale 1.0 \
#     > logs/llava/pope/baseline.log 2>&1 &

# ===================== S1-mini POPE =====================

# CUDA_VISIBLE_DEVICES=2 nohup conda run -n s1mini python visaug/inference/pipeline/run_pope_s1mini_legacy.py \
#     --question-file data/pope/coco_pope_random.json \
#     --answers-file outputs/s1mini/pope/baseline.jsonl \
#     --pca-k 3 --scale 1.0 \
#     > logs/s1mini/pope/baseline.log 2>&1 &
