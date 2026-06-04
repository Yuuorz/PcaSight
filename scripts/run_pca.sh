#!/bin/bash
# POPE inference (legacy) — uses pipeline/run_pope_*_legacy.py
# See run_opope.sh for active OPOPE experiments.

mkdir -p outputs/llava/pope outputs/s1mini/pope logs/llava/pope logs/s1mini/pope

# LLaVA
# CUDA_VISIBLE_DEVICES=4 nohup conda run -n clearsight python visaug/inference/pipeline/run_pope_llava_legacy.py \
#     --question-file data/pope/coco_pope_random.json \
#     --answers-file outputs/llava/pope/res_k3_scale0.1.jsonl \
#     --pca-k 3 --scale 0.1 > logs/llava/pope/k3_scale0.1.log 2>&1 &

# S1-mini
# CUDA_VISIBLE_DEVICES=0 nohup conda run -n s1mini python visaug/inference/pipeline/run_pope_s1mini_legacy.py \
#     --answers-file outputs/s1mini/pope/res_k2_scale0.2.jsonl \
#     --pca-k 2 --scale 0.2 > logs/s1mini/pope/k2_scale0.2.log 2>&1 &
