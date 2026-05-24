#!/bin/bash
cd /root/code/ClearSight
EXPS=(
  "0:2.0:res_proj_zonal_pullonly_x2"
  "1:3.0:res_proj_zonal_pullonly_x3"  
  "2:5.0:res_proj_zonal_pullonly_x5"
  "3:fixed0.1:res_proj_zonal_pullonly_f01"
  "4:fixed0.3:res_proj_zonal_pullonly_f03"
  "5:fixed0.5:res_proj_zonal_pullonly_f05"
)

for exp in "${EXPS[@]}"; do
  IFS=':' read -r gpu mult name <<< "$exp"
  extra_args=""
  if [[ "$mult" == fixed* ]]; then
    val="${mult#fixed}"
    extra_args="--fixed-strength $val"
    mult_label="fixed=$val"
  else
    extra_args="--strength-multiplier $mult"
    mult_label="x$mult"
  fi
  
  outfile="outputs/pope_llava/${name}_random.jsonl"
  rm -f "$outfile"
  
  CUDA_VISIBLE_DEVICES=$gpu nohup conda run -n clearsight python visaug/inference/infer_pope_llava_proj_zonal_pullonly.py \
    --model-path /root/code/llava-v1.5-7b \
    --image-folder data/coco/val2014 \
    --question-file data/pope/coco_pope_random.json \
    --answers-file "$outfile" \
    $extra_args > outputs/pope_llava/${name}.log 2>&1 &
  
  echo "GPU $gpu: $mult_label → $name (PID $!)"
done

echo ""
echo "All 6 experiments launched. Check progress:"
echo "  tail -f outputs/pope_llava/*.log"
