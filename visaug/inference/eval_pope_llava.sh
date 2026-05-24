#!/bin/bash
# POPE evaluation for LLaVA: baseline, VAF, AGVP
# 3 methods × 3 splits = 9 experiments

export CUDA_VISIBLE_DEVICES=5

MODEL="/root/code/llava-v1.5-7b"
IMG_FOLDER="/root/code/ClearSight/data/coco/val2014"
POPE_DIR="/root/code/ClearSight/data/pope"
OUT_DIR="/root/code/ClearSight/outputs/pope_llava"
SCRIPT="./visaug/inference/infer_pope_llava.py"
EVAL_SCRIPT="./visaug/inference/eval_pope.py"

METHODS="baseline vaf agvp"
SPLITS="random"

mkdir -p $OUT_DIR

# ============ Inference ============
for METHOD in $METHODS; do
    for SPLIT in $SPLITS; do
        OUTFILE="${OUT_DIR}/res_${METHOD}_${SPLIT}.jsonl"
        if [ -f "$OUTFILE" ]; then
            echo "SKIP $METHOD $SPLIT (already exists)"
            continue
        fi

        echo ""
        echo "=== Inference: $METHOD / $SPLIT ==="
        CMD="python $SCRIPT \
            --model-path $MODEL \
            --image-folder $IMG_FOLDER \
            --question-file ${POPE_DIR}/coco_pope_${SPLIT}.json \
            --answers-file $OUTFILE \
            --method $METHOD"

        if [ "$METHOD" = "vaf" ]; then
            CMD="$CMD --enh-para 1.15 --sup-para 0.95 --start-layer 9 --end-layer 14"
        elif [ "$METHOD" = "agvp" ]; then
            CMD="$CMD --start-layer 0 --end-layer 31 --cos-gate neg"
        fi

        eval $CMD
    done
done

# ============ Evaluation ============
echo ""
echo "========================================"
echo "           POPE Results (LLaVA)"
echo "========================================"
printf "%-12s %-12s %-8s %-8s %-8s %-8s\n" "Method" "Split" "Acc" "Prec" "Recall" "F1"
echo "------------------------------------------------------------------------"

for METHOD in $METHODS; do
    for SPLIT in $SPLITS; do
        RESFILE="${OUT_DIR}/res_${METHOD}_${SPLIT}.jsonl"
        ANNFILE="${POPE_DIR}/coco_pope_${SPLIT}.json"
        if [ -f "$RESFILE" ]; then
            RESULT=$(python $EVAL_SCRIPT \
                --annotation-file $ANNFILE \
                --result-file $RESFILE 2>&1)
            ACC=$(echo "$RESULT" | grep Accurancy | awk '{printf "%.4f", $2}')
            PREC=$(echo "$RESULT" | grep Precision | awk '{printf "%.4f", $2}')
            REC=$(echo "$RESULT" | grep Recall | awk '{printf "%.4f", $2}')
            F1=$(echo "$RESULT" | grep F1_score | awk '{printf "%.4f", $2}')
            printf "%-12s %-12s %-8s %-8s %-8s %-8s\n" "$METHOD" "$SPLIT" "$ACC" "$PREC" "$REC" "$F1"
        else
            printf "%-12s %-12s %-8s\n" "$METHOD" "$SPLIT" "MISSING"
        fi
    done
done
