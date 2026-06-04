set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0

python ./visaug/analysis/vis_flow_s1mini.py \
    --model-path /root/code/Intern-S1-mini \
    --question-file /root/code/ClearSight/data/pope/coco_pope_random.json \
    --image-folder /root/code/ClearSight/data/coco/val2014 \
    --answers-file /root/code/ClearSight/outputs/analysis/res_coco_random_s1mini.pt \
    --gpu-id 0

sleep 5
python ./visaug/analysis/analysis_plot_s1mini.py --data-file /root/code/ClearSight/outputs/analysis/res_coco_random_s1mini.pt