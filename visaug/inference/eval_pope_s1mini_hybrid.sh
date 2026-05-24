#!/bin/bash
# Hybrid ablation: S1-mini only
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="/root/code/Intern-S1-mini"
IMG_FOLDER="/root/code/ClearSight/data/coco/val2014"
POPE_FILE="/root/code/ClearSight/data/pope/coco_pope_random.json"
OUT_DIR="/root/code/ClearSight/outputs/pope_s1mini"
SCRIPT="./visaug/inference/infer_pope_s1mini_agvp.py"
EVAL_SCRIPT="./visaug/inference/eval_pope.py"

mkdir -p $OUT_DIR

run_s1() {
    local GPU=$1; local MODE=$2; local TAG=$3
    OUTFILE="${OUT_DIR}/res_${TAG}_random.jsonl"
    [ -f "$OUTFILE" ] && echo "SKIP ${TAG}" && return
    echo "=== S1-mini GPU${GPU}: ${TAG} ==="
    CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER --gpu-id 0 --no-think \
        --question-file $POPE_FILE --answers-file $OUTFILE \
        --start-layer 0 --end-layer 35 \
        --agvp-mode $MODE
    echo "[S1-mini] ${TAG} done."
}

# 4 modes on 2 GPUs (sequential pairs)
(run_s1 4 hybrid      hybrid      && run_s1 4 proj_cospos proj_cospos) &
(run_s1 5 hybrid_soft hybrid_soft && run_s1 5 rotate_vis_imb0 rot_only) &

echo "S1-mini jobs launched. Waiting..."
wait

echo ""
echo "========================================"
echo "  S1-mini Hybrid Ablation (POPE random)"
echo "========================================"
printf "%-16s %-8s %-8s %-8s %-8s  %s\n" "Mode" "Acc" "Prec" "Recall" "F1" "Desc"
echo "-------------------------------------------------------------------"

declare -A DESC
DESC[baseline]="no intervention"
DESC[vaf]="VAF enh=1.15 sup=0.95"
DESC[agvr]="AGVR (prev run)"
DESC[hybrid]="rot(cos<0) + proj(cos>0)"
DESC[hybrid_soft]="rot+proj soft (cos2*imb)"
DESC[proj_cospos]="proj only (cos>0 layers)"
DESC[rot_only]="rot only (cos<0 layers)"

for TAG in baseline vaf agvr hybrid hybrid_soft proj_cospos rot_only; do
    RESFILE="${OUT_DIR}/res_${TAG}_random.jsonl"
    [ ! -f "$RESFILE" ] && printf "%-16s %-8s\n" "$TAG" "MISSING" && continue
    RESULT=$(python $EVAL_SCRIPT --annotation-file $POPE_FILE --result-file $RESFILE 2>&1)
    ACC=$(echo "$RESULT" | grep Accurancy | awk '{printf "%.4f", $2}')
    PREC=$(echo "$RESULT" | grep Precision | awk '{printf "%.4f", $2}')
    REC=$(echo "$RESULT" | grep Recall | awk '{printf "%.4f", $2}')
    F1=$(echo "$RESULT" | grep F1_score | awk '{printf "%.4f", $2}')
    printf "%-16s %-8s %-8s %-8s %-8s  %s\n" "$TAG" "$ACC" "$PREC" "$REC" "$F1" "${DESC[$TAG]}"
done
