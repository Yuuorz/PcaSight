#!/bin/bash
# AGVP rotate_vis ablation for LLaVA POPE (random split)
# 4 variants × 1 GPU each, parallel
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="/root/code/llava-v1.5-7b"
IMG_FOLDER="/root/code/ClearSight/data/coco/val2014"
POPE_DIR="/root/code/ClearSight/data/pope"
OUT_DIR="/root/code/ClearSight/outputs/pope_llava"
SCRIPT="./visaug/inference/infer_pope_llava.py"
EVAL_SCRIPT="./visaug/inference/eval_pope.py"
POPE_FILE="${POPE_DIR}/coco_pope_random.json"

mkdir -p $OUT_DIR

run_agvp() {
    local GPU=$1
    local MODE=$2
    local TAG=$3
    export CUDA_VISIBLE_DEVICES=$GPU

    OUTFILE="${OUT_DIR}/res_agvp_${TAG}_random.jsonl"
    if [ -f "$OUTFILE" ]; then
        echo "SKIP ${TAG} (exists)"
        return
    fi

    echo "=== GPU${GPU}: ${TAG} (mode=${MODE}) ==="
    python $SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER \
        --question-file $POPE_FILE \
        --answers-file $OUTFILE \
        --method agvp \
        --start-layer 0 --end-layer 31 --cos-gate neg \
        --agvp-mode $MODE

    echo "[GPU${GPU}] ${TAG} done."
}

# Launch 4 experiments in parallel
run_agvp 2 rotate_vis       rot       &
run_agvp 3 rotate_vis_soft  rot_soft  &
run_agvp 4 rotate_vis_last  rot_last  &
run_agvp 5 rotate_vis_imb   rot_imb   &

echo "4 rotate_vis jobs launched. Waiting..."
wait

# ============ Evaluation ============
echo ""
echo "========================================"
echo "  Rotate-Vis Ablation (LLaVA POPE random)"
echo "========================================"
printf "%-14s %-8s %-8s %-8s %-8s  %s\n" "Mode" "Acc" "Prec" "Recall" "F1" "Description"
echo "----------------------------------------------------------------------"

declare -A DESC
DESC[rot]="alpha=|cos|, all tokens"
DESC[rot_soft]="alpha=|cos|*0.1, all tokens"
DESC[rot_last]="alpha=|cos|, last token only"
DESC[rot_imb]="alpha=|cos|, imb>0.3 layers"

for TAG in rot rot_soft rot_last rot_imb; do
    RESFILE="${OUT_DIR}/res_agvp_${TAG}_random.jsonl"
    if [ -f "$RESFILE" ]; then
        RESULT=$(python $EVAL_SCRIPT \
            --annotation-file $POPE_FILE \
            --result-file $RESFILE 2>&1)
        ACC=$(echo "$RESULT" | grep Accurancy | awk '{printf "%.4f", $2}')
        PREC=$(echo "$RESULT" | grep Precision | awk '{printf "%.4f", $2}')
        REC=$(echo "$RESULT" | grep Recall | awk '{printf "%.4f", $2}')
        F1=$(echo "$RESULT" | grep F1_score | awk '{printf "%.4f", $2}')
        printf "%-14s %-8s %-8s %-8s %-8s  %s\n" "$TAG" "$ACC" "$PREC" "$REC" "$F1" "${DESC[$TAG]}"
    else
        printf "%-14s %-8s\n" "$TAG" "MISSING"
    fi
done
