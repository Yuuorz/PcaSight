#!/bin/bash
# CHAIR evaluation for S1-mini: 5 methods, 500 images
# baseline / VAF / AGVP L0-16 (cos<0) / AGVP L0-35 (cos<0) / AGVP cospos (cos>0, ablation)
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0   # <-- 改这里

MODEL=/root/code/Intern-S1-mini
IMG_FOLDER=/root/code/ClearSight/data/coco/val2014
OUT_DIR=/root/code/ClearSight/outputs/chair
GT_FILE=/root/code/ClearSight/data/coco/coco_ground_truth_segmentation.json
SCRIPT=./visaug/inference/infer_chair_s1mini.py
EVAL_SCRIPT=./visaug/inference/eval_chair.py

NUM_SAMPLES=500
SEED=42

mkdir -p $OUT_DIR

# ============ 1. Baseline ============
OUTFILE=${OUT_DIR}/chair_s1mini_baseline.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== Baseline ==="
    python $SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --num-samples $NUM_SAMPLES --seed $SEED --gt-file $GT_FILE \
        --method baseline \
        --answers-file $OUTFILE
else
    echo "SKIP baseline (exists)"
fi

# ============ 2. VAF ============
OUTFILE=${OUT_DIR}/chair_s1mini_vaf.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== VAF ==="
    python $SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --num-samples $NUM_SAMPLES --seed $SEED --gt-file $GT_FILE \
        --method vaf \
        --start-layer 0 --end-layer 35 \
        --enh-para 1.15 --sup-para 0.95 \
        --answers-file $OUTFILE
else
    echo "SKIP vaf (exists)"
fi

# ============ 3. AGVP L0-16 (cos<0) ============
OUTFILE=${OUT_DIR}/chair_s1mini_agvp_L016.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== AGVP L0-16 (cos<0) ==="
    python $SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --num-samples $NUM_SAMPLES --seed $SEED --gt-file $GT_FILE \
        --method agvp \
        --start-layer 0 --end-layer 16 --cos-gate neg \
        --answers-file $OUTFILE
else
    echo "SKIP agvp L0-16 (exists)"
fi

# ============ 4. AGVP L0-35 (cos<0) ============
OUTFILE=${OUT_DIR}/chair_s1mini_agvp_L035.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== AGVP L0-35 (cos<0) ==="
    python $SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --num-samples $NUM_SAMPLES --seed $SEED --gt-file $GT_FILE \
        --method agvp \
        --start-layer 0 --end-layer 35 --cos-gate neg \
        --answers-file $OUTFILE
else
    echo "SKIP agvp L0-35 (exists)"
fi

# ============ 5. AGVP cospos (cos>0, ablation) ============
OUTFILE=${OUT_DIR}/chair_s1mini_agvp_cospos.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== AGVP cospos (cos>0, ablation) ==="
    python $SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --num-samples $NUM_SAMPLES --seed $SEED --gt-file $GT_FILE \
        --method agvp \
        --start-layer 0 --end-layer 35 --cos-gate pos \
        --answers-file $OUTFILE
else
    echo "SKIP agvp cospos (exists)"
fi

# ============ Evaluation ============
echo ""
echo "============================================"
echo "  CHAIR Evaluation (S1-mini)"
echo "============================================"

for NAME in baseline vaf agvp_L016 agvp_L035 agvp_cospos; do
    CFILE=${OUT_DIR}/chair_s1mini_${NAME}.jsonl
    if [ -f "$CFILE" ]; then
        echo "--- ${NAME} ---"
        python $EVAL_SCRIPT --cap-file $CFILE --instances-path $GT_FILE
        echo ""
    else
        echo "--- ${NAME} --- [SKIP: not found]"
        echo ""
    fi
done
