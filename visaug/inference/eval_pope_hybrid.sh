#!/bin/bash
# Hybrid ablation: LLaVA only (needs clearsight env)
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="/root/code/llava-v1.5-7b"
IMG_FOLDER="/root/code/ClearSight/data/coco/val2014"
POPE_FILE="/root/code/ClearSight/data/pope/coco_pope_random.json"
OUT_DIR="/root/code/ClearSight/outputs/pope_llava"
SCRIPT="./visaug/inference/infer_pope_llava.py"
EVAL_SCRIPT="./visaug/inference/eval_pope.py"

mkdir -p $OUT_DIR

run_llava() {
    local GPU=$1; local MODE=$2; local TAG=$3
    OUTFILE="${OUT_DIR}/res_${TAG}_random.jsonl"
    [ -f "$OUTFILE" ] && echo "SKIP ${TAG}" && return
    echo "=== LLaVA GPU${GPU}: ${TAG} ==="
    CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
        --model-path $MODEL --image-folder $IMG_FOLDER \
        --question-file $POPE_FILE --answers-file $OUTFILE \
        --method agvp --start-layer 0 --end-layer 31 \
        --agvp-mode $MODE
    echo "[LLaVA] ${TAG} done."
}

run_llava 2 hybrid       hybrid          &
run_llava 3 hybrid_soft  hybrid_soft     &

echo "LLaVA jobs launched. Waiting..."
wait

echo ""
echo "========================================"
echo "  LLaVA Hybrid Ablation (POPE random)"
echo "========================================"
printf "%-16s %-8s %-8s %-8s %-8s  %s\n" "Mode" "Acc" "Prec" "Recall" "F1" "Desc"
echo "-------------------------------------------------------------------"

declare -A DESC
DESC[baseline]="no intervention"
DESC[agvp_rot_imb]="AGVR best (|cos|, imb>0.3)"
DESC[hybrid]="rot(cos<0) + proj(cos>0)"
DESC[hybrid_soft]="rot+proj soft (cos²×imb)"

for TAG in baseline hybrid hybrid_soft agvp_rot_imb; do
    RESFILE="${OUT_DIR}/res_${TAG}_random.jsonl"
    [ ! -f "$RESFILE" ] && printf "%-16s %-8s\n" "$TAG" "MISSING" && continue
    RESULT=$(python $EVAL_SCRIPT --annotation-file $POPE_FILE --result-file $RESFILE 2>&1)
    ACC=$(echo "$RESULT" | grep Accurancy | awk '{printf "%.4f", $2}')
    PREC=$(echo "$RESULT" | grep Precision | awk '{printf "%.4f", $2}')
    REC=$(echo "$RESULT" | grep Recall | awk '{printf "%.4f", $2}')
    F1=$(echo "$RESULT" | grep F1_score | awk '{printf "%.4f", $2}')
    printf "%-16s %-8s %-8s %-8s %-8s  %s\n" "$TAG" "$ACC" "$PREC" "$REC" "$F1" "${DESC[$TAG]}"
done
