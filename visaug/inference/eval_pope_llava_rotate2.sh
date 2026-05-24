#!/bin/bash
# AGVP rotate_vis round 2: optimize alpha formula
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
run_agvp 2 rotate_vis_adaptive     rot_adp     &
run_agvp 3 rotate_vis_adaptive_sq  rot_adp_sq  &
run_agvp 4 rotate_vis_imb0         rot_imb0    &
run_agvp 5 rotate_vis_sysfrac      rot_sfrac   &

echo "4 jobs launched. Waiting..."
wait

# ============ Evaluation ============
echo ""
echo "============================================="
echo "  Rotate-Vis Round 2 (LLaVA POPE random)"
echo "============================================="
printf "%-14s %-8s %-8s %-8s %-8s  %s\n" "Mode" "Acc" "Prec" "Recall" "F1" "Alpha"
echo "----------------------------------------------------------------------"

declare -A DESC
DESC[rot_adp]="imb*|cos| (zero hyperparam)"
DESC[rot_adp_sq]="imb*cos^2 (gentler)"
DESC[rot_imb0]="imb>0 gate + |cos|"
DESC[rot_sfrac]="sys_frac*|cos| (output-space)"

for TAG in rot_adp rot_adp_sq rot_imb0 rot_sfrac; do
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
