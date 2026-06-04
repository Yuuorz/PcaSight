#!/bin/bash

mkdir -p logs/llava/opope logs/s1mini/opope

# ===================== LLaVA OPOPE =====================

CUDA_VISIBLE_DEVICES=0 nohup conda run -n clearsight python visaug/inference/pipeline/run_opope_llava.py \
    --model-path /root/code/llava-v1.5-7b \
    --is-baseline 1 \
    > logs/llava/opope/baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup conda run -n clearsight python visaug/inference/pipeline/run_opope_llava.py \
    --model-path /root/code/llava-v1.5-7b \
    --pca-k 6 --scale 0.2 \
    > logs/llava/opope/k6_scale0.2.log 2>&1 &

# ===================== S1-mini OPOPE =====================

CUDA_VISIBLE_DEVICES=2 nohup conda run -n s1mini python visaug/inference/pipeline/run_opope_s1mini.py \
    --model-path /root/code/Intern-S1-mini \
    --is-baseline 1 \
    > logs/s1mini/opope/baseline.log 2>&1 &

echo "Launched. Check logs/ for progress."
