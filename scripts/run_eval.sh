#!/bin/bash

#==============================================>> POPE >>=======================================================

python visaug/inference/metrics/eval_pope.py \
    --annotation-file data/pope/coco_pope_random.json \
    --result-file outputs/s1mini/pope/res_s1_k2_scale0.2_UNCENTERED.jsonl

python visaug/inference/metrics/eval_pope.py \
    --annotation-file data/pope/coco_pope_random.json \
    --result-file outputs/s1mini/pope/res_s1_k1_scale0.2_UNCENTERED.jsonl

python visaug/inference/metrics/eval_pope.py \
    --annotation-file data/pope/coco_pope_random.json \
    --result-file outputs/s1mini/pope/res_s1_k4_scale0.2_UNCENTERED.jsonl

#==============================================<< POPE <<=======================================================
