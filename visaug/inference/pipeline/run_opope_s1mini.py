"""
OPOPE Data Generation Pipeline for Intern-S1-mini (Decoupled & Pure).
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
from models import inject_s1mini_pca

def main():
    parser = argparse.ArgumentParser(description="Pure S1-mini OPOPE Capture (Supports Baseline Switch)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="data/coco/val2014")
    parser.add_argument("--question-file", type=str, default="data/pope/coco_pope_random.json")
    parser.add_argument("--answers-file", type=str, default=None,
                        help="Output path (auto-generated if not set)")
    parser.add_argument("--is-baseline", type=int, default=0, choices=[0, 1],
                        help="0: Apply PCA projection (Method), 1: Vanilla model (Baseline)")
    parser.add_argument("--model-type", type=str, default="s1mini", choices=["llava", "s1mini"])
    parser.add_argument("--pca-k", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.1)

    args = parser.parse_args()
    run_as_baseline = (args.is_baseline == 1)

    if not args.answers_file:
        os.makedirs("outputs/s1mini/opope", exist_ok=True)
        if run_as_baseline:
            args.answers_file = "outputs/s1mini/opope/baseline.jsonl"
        else:
            args.answers_file = f"outputs/s1mini/opope/k{args.pca_k}_scale{args.scale}.jsonl"
    print(f"Output -> {args.answers_file}")

    # 1. 加载 S1-mini 强推理核心
    from transformers import AutoProcessor, AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": "cuda:0"})
    model.eval()
    device = model.device
    
    # 2. 根据开关决定是否挂载拦截算子
    if run_as_baseline:
        print("====== 🚨 警告：[Baseline 纯基线模式开启] 绝不注入任何干预算子 ======")
    else:
        print(f"====== 👑 模式：[Method PCA 投影激活] 注入无中心化算子 (k={args.pca_k}, scale={args.scale}) ======")
        model = inject_s1mini_pca(model, args)

    model.eval()

    # 3. 聚类去重 500 张 COCO 真理大盘图片
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

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": qs}]}]
        prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        inputs = processor(text=prompt_str, images=image, return_tensors="pt")

        # 即使是 Baseline 模式，我们也需要让它的 eager 注意力拿到切片边界索引，因此只在非 baseline 下写入控制属性
        if not args.is_baseline:
            img_pos = (inputs["input_ids"][0] == model.config.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) > 0:
                for layer in model.language_model.layers:
                    if hasattr(layer.self_attn, "_sys_len"):
                        layer.self_attn._sys_len = int(img_pos[0].item())
                        layer.self_attn._img_len = int(len(img_pos))

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        with torch.inference_mode():
            generate_ids = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        outputs = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        ans_file.write(json.dumps({"image_id": image_id, "caption": outputs, "image_path": image_file}) + "\n")
        ans_file.flush()
        
    ans_file.close()
    print(f"S1-mini OPOPE assets saved successfully at: {args.answers_file}")

if __name__ == "__main__":
    main()