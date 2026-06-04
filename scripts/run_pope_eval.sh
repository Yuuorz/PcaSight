#!/bin/bash

# OPOPE evaluation (requires instances_val2014.json)
# Download: wget http://images.cocodataset.org/annotations/instances_val2014.json

INSTANCES=/root/code/ClearSight/data/coco/annotations/instances_val2014.json

python visaug/inference/metrics/eval_opope.py \
    --cap-file outputs/llava/opope/baseline.jsonl \
    --instances-path $INSTANCES

python visaug/inference/metrics/eval_opope.py \
    --cap-file outputs/llava/opope/k6_scale0.2.jsonl \
    --instances-path $INSTANCES

python visaug/inference/metrics/eval_opope.py \
    --cap-file outputs/s1mini/opope/baseline.jsonl \
    --instances-path $INSTANCES
