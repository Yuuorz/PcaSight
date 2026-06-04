"""
OPOPE Data Generation Pipeline for LLaVA-1.5 (Decoupled & Pure).
Supports one-click Baseline switching via --is-baseline.
"""
import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import inject_llava_pca

def main():
    parser = argparse.ArgumentParser(description="Pure LLaVA OPOPE Capture (Supports Baseline Switch)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="data/coco/val2014")
    parser.add_argument("--question-file", type=str, default="data/pope/coco_pope_random.json")
    parser.add_argument("--answers-file", type=str, default=None,
                        help="Output path (auto-generated if not set)")
    parser.add_argument("--is-baseline", type=int, default=0, choices=[0, 1],
                        help="0: Apply PCA projection (Method), 1: Vanilla model (Baseline)")
    parser.add_argument("--model-type", type=str, default="llava", choices=["llava", "s1mini"],
                        help="Model type for PCA injection")
    parser.add_argument("--pca-k", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.1)

    args = parser.parse_args()
    run_as_baseline = (args.is_baseline == 1)

    if not args.answers_file:
        os.makedirs("outputs/llava/opope", exist_ok=True)
        if run_as_baseline:
            args.answers_file = "outputs/llava/opope/baseline.jsonl"
        else:
            args.answers_file = f"outputs/llava/opope/k{args.pca_k}_scale{args.scale}.jsonl"
    print(f"Output -> {args.answers_file}")

    # 1. 离线加载原生 LLaVA 1.5 7B 基座
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, "llava-v1.5-7b")

    # 2. 根据开关决定是否挂载拦截算子
    if run_as_baseline:
        print("====== [Baseline] 纯基线模式开启，不注入任何干预算子 ======")
    else:
        print(f"====== [Method PCA] 注入无中心化算子 (k={args.pca_k}, scale={args.scale}) ======")
        if args.model_type == "llava":
            model = inject_llava_pca(model, args)
        else:
            model = inject_s1mini_pca(model, args)

    model.eval()

    # 3. 聚类 MSCOCO 500 张独特图像资产大盘
    questions = [json.loads(l) for l in open(os.path.expanduser(args.question_file))]
    seen_images = set()
    unique_questions = []
    for q in questions:
        if q["image"] not in seen_images:
            seen_images.add(q["image"])
            unique_questions.append(q)
            if len(unique_questions) >= 500: break

    ans_file = open(os.path.expanduser(args.answers_file), "w")

    for line in tqdm(unique_questions):
        image_file = line["image"]
        qs = "Describe this image in detail."
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_id = int(image_file.split(".")[0].split("_")[-1])

        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token, process_images

        prompt = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()
        
        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=image_tensor, max_new_tokens=512, use_cache=True)
        outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({"image_id": image_id, "caption": outputs, "image_path": image_file}) + "\n")
        ans_file.flush()
        
    ans_file.close()
    print(f"LLaVA OPOPE assets saved successfully at: {args.answers_file}")

if __name__ == "__main__":
    main()