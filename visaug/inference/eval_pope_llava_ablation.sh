#!/bin/bash
# AGVP mode ablation for LLaVA POPE (random split)
# 4 modes × 1 GPU each, parallel
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

    echo "=== GPU${GPU}: AGVP ${TAG} (mode=${MODE}) ==="
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
run_agvp 2 sys_frac_both  sfrac_both &
run_agvp 3 sys_frac_sup   sfrac_sup  &
run_agvp 4 cos_both       cos_both   &
run_agvp 5 imb_cos_vis    imb_cos_vis &

echo "4 AGVP ablation jobs launched. Waiting..."
wait

# ============ Evaluation ============
echo ""
echo "========================================"
echo "  AGVP Mode Ablation (LLaVA POPE random)"
echo "========================================"
printf "%-16s %-8s %-8s %-8s %-8s\n" "Mode" "Acc" "Prec" "Recall" "F1"
echo "------------------------------------------------------------"

for TAG in sfrac_both sfrac_sup cos_both imb_cos_vis; do
    RESFILE="${OUT_DIR}/res_agvp_${TAG}_random.jsonl"
    if [ -f "$RESFILE" ]; then
        RESULT=$(python $EVAL_SCRIPT \
            --annotation-file $POPE_FILE \
            --result-file $RESFILE 2>&1)
        ACC=$(echo "$RESULT" | grep Accurancy | awk '{printf "%.4f", $2}')
        PREC=$(echo "$RESULT" | grep Precision | awk '{printf "%.4f", $2}')
        REC=$(echo "$RESULT" | grep Recall | awk '{printf "%.4f", $2}')
        F1=$(echo "$RESULT" | grep F1_score | awk '{printf "%.4f", $2}')
        printf "%-16s %-8s %-8s %-8s %-8s\n" "$TAG" "$ACC" "$PREC" "$REC" "$F1"
    else
        printf "%-16s %-8s\n" "$TAG" "MISSING"
    fi
done
