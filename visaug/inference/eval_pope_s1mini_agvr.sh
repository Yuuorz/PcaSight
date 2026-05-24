#!/bin/bash
# S1-mini POPE random: AGVR (rotation) vs baseline/VAF/old AGVP
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

# ============ 1. AGVR L0-35 (rotate_vis_imb0, best from LLaVA) ============
OUTFILE=${OUT_DIR}/res_agvr_random.jsonl
if [ ! -f "$OUTFILE" ]; then
    echo "=== AGVR L0-35 (rotate_vis_imb0, cos<0 gate) ==="
    python $AGVP_SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --question-file $POPE_FILE \
        --answers-file $OUTFILE \
        --start-layer 0 --end-layer 35 --cos-gate neg \
        --agvp-mode rotate_vis_imb0
else
    echo "SKIP agvr (exists)"
fi

# ============ Evaluation (include old results for comparison) ============
echo ""
echo "========================================"
echo "  POPE Random Results (S1-mini)"
echo "========================================"
printf "%-16s %-8s %-8s %-8s %-8s\n" "Method" "Acc" "Prec" "Recall" "F1"
echo "------------------------------------------------"

for NAME in baseline vaf agvp_L016 agvp_L035 agvp_cospos agvr; do
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
