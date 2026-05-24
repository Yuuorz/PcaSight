#!/bin/bash
# POPE random evaluation for S1-mini: 4 methods
# baseline / vaf / agvp L0-16 (manual) / agvp L0-35 (cos-gated, auto-select)
# AGVP now has cos(d_vis,d_sys)<0 gate: layers with positive cos auto-skip
set -e

export CUDA_VISIBLE_DEVICES=4

MODEL=/root/code/Intern-S1-mini
IMG_FOLDER=/root/code/ClearSight/data/coco/val2014
POPE_FILE=/root/code/ClearSight/data/pope/coco_pope_random.json
OUT_DIR=/root/code/ClearSight/outputs/pope_s1mini
VAF_SCRIPT=./visaug/inference/infer_pope_s1mini.py
AGVP_SCRIPT=./visaug/inference/infer_pope_s1mini_agvp.py
EVAL_SCRIPT=./visaug/inference/eval_pope.py

mkdir -p $OUT_DIR

# ============ 1. Baseline ============
OUTFILE=${OUT_DIR}/res_baseline_random.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== Baseline ==="
    python $VAF_SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --question-file $POPE_FILE \
        --answers-file $OUTFILE
else
    echo "SKIP baseline (exists)"
fi

# ============ 2. VAF ============
OUTFILE=${OUT_DIR}/res_vaf_random.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== VAF (enh=1.15, sup=0.95, L0-35) ==="
    python $VAF_SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --question-file $POPE_FILE \
        --answers-file $OUTFILE \
        --use-visaug --enh-para 1.15 --sup-para 0.95 \
        --vaf-start-layer 0 --vaf-end-layer 35
else
    echo "SKIP vaf (exists)"
fi

# ============ 3. AGVP L0-16 (manual range, cos-gated) ============
OUTFILE=${OUT_DIR}/res_agvp_L016_random.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== AGVP L0-16 (cos-gated) ==="
    python $AGVP_SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --question-file $POPE_FILE \
        --answers-file $OUTFILE \
        --start-layer 0 --end-layer 16
else
    echo "SKIP agvp L0-16 (exists)"
fi

# ============ 4. AGVP L0-35 (full, cos<0 gated) ============
OUTFILE=${OUT_DIR}/res_agvp_L035_random.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== AGVP L0-35 (cos<0 gate) ==="
    python $AGVP_SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --question-file $POPE_FILE \
        --answers-file $OUTFILE \
        --start-layer 0 --end-layer 35 --cos-gate neg
else
    echo "SKIP agvp L0-35 (exists)"
fi

# ============ 5. AGVP L0-35 ablation (cos>0, reverse gate) ============
OUTFILE=${OUT_DIR}/res_agvp_cospos_random.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== AGVP L0-35 ablation (cos>0 gate) ==="
    python $AGVP_SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --question-file $POPE_FILE \
        --answers-file $OUTFILE \
        --start-layer 0 --end-layer 35 --cos-gate pos
else
    echo "SKIP agvp cospos (exists)"
fi

# ============ Evaluation ============
echo ""
echo "========================================"
echo "  POPE Random Results (S1-mini)"
echo "========================================"
printf "%-16s %-8s %-8s %-8s %-8s\n" "Method" "Acc" "Prec" "Recall" "F1"
echo "------------------------------------------------"

for NAME in baseline vaf agvp_L016 agvp_L035 agvp_cospos; do
    RESFILE=${OUT_DIR}/res_${NAME}_random.jsonl
    if [ -f "$RESFILE" ]; then
        RESULT=$(python $EVAL_SCRIPT --annotation-file $POPE_FILE --result-file $RESFILE 2>&1)
        ACC=$(echo "$RESULT" | grep Accurancy | awk '{printf "%.4f", $2}')
        PREC=$(echo "$RESULT" | grep Precision | awk '{printf "%.4f", $2}')
        REC=$(echo "$RESULT" | grep Recall | awk '{printf "%.4f", $2}')
        F1=$(echo "$RESULT" | grep F1_score | awk '{printf "%.4f", $2}')
        printf "%-16s %-8s %-8s %-8s %-8s\n" "$NAME" "$ACC" "$PREC" "$REC" "$F1"
    else
        printf "%-16s %-8s\n" "$NAME" "MISSING"
    fi
done
