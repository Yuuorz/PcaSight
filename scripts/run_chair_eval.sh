#!/bin/bash
# CHAIR evaluation on OPOPE caption outputs

INSTANCES=data/coco/annotations/instances_val2014.json

python visaug/inference/metrics/eval_chair.py \
    --cap-file outputs/llava/opope/baseline.jsonl \
    --instances-path $INSTANCES

python visaug/inference/metrics/eval_chair.py \
    --cap-file outputs/llava/opope/k6_scale0.2.jsonl \
    --instances-path $INSTANCES

python visaug/inference/metrics/eval_chair.py \
    --cap-file outputs/s1mini/opope/baseline.jsonl \
    --instances-path $INSTANCES
