# ================= 卡槽 0: LLaVA PCA 基础容量测试 (k=3) =================
CUDA_VISIBLE_DEVICES=0 nohup python visaug/inference/infer_pope_llava_pca.py \
    --question-file data/pope/coco_pope_random.json \
    --answers-file outputs/pca_llava/res_k3_scale1.5_random.jsonl \
    --pca-k 3 --scale 1.5 > logs/pca/llava_k3_scale1.5.log 2>&1 &

# ================= 卡槽 1: LLaVA PCA 高保真测试 (k=5) =================
CUDA_VISIBLE_DEVICES=1 nohup python visaug/inference/infer_pope_llava_pca.py \
    --question-file data/pope/coco_pope_random.json \
    --answers-file outputs/pca_llava/res_k5_scale2.0_random.jsonl \
    --pca-k 5 --scale 2.0 > logs/pca/llava_k5_scale2.0.log 2>&1 &

# ================= 卡槽 2: S1-mini 跨界深层注入测试 =================
CUDA_VISIBLE_DEVICES=2 nohup python visaug/inference/infer_pope_s1mini_pca.py \
    --answers-file outputs/pca_s1mini/res_s1_k3_scale1.0_random.jsonl \
    --pca-k 3 --scale 1.0 > logs/pca/s1mini_k3_scale1.0.log 2>&1 &