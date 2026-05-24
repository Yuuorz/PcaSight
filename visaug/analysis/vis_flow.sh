CUDA_VISIBLE_DEVICES=2 python ./visaug/analysis/vis_flow.py \
    --model-path /root/code/2601/llava-v1.5-7b \
    --question-file /root/code/2604/ClearSight/data/pope/coco_pope_random.json \
    --image-folder /root/code/2601/val2014 \
    --answers-file /root/code/2604/ClearSight/outputs/analysis/res_coco_random.pt 

sleep 5
CUDA_VISIBLE_DEVICES=2 python ./visaug/analysis/analysis_plot.py