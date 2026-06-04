#!/bin/bash
# POPE evaluation (yes/no QA) — uses eval_pope.py

python visaug/inference/metrics/eval_pope.py \
    --annotation-file data/pope/coco_pope_random.json \
    --result-file outputs/llava/pope/res_k3_scale0.1_UNCENTERED.jsonl

python visaug/inference/metrics/eval_pope.py \
    --annotation-file data/pope/coco_pope_random.json \
    --result-file outputs/s1mini/pope/res_s1_k2_scale0.2_UNCENTERED.jsonl
