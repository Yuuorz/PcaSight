import argparse
import torch
import os
import json
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForCausalLM

from PIL import Image
import math
import random
import numpy as np
import time

from AttnAdapter import saliency_compute, attention_compute
import torch.nn.functional as F


def load_intern_s1_mini(model_path, gpu_id):
    print(f"Loading Intern-S1-mini from {model_path}...")
    
    # 1. 加载 Processor（包含 tokenizer + image_processor）
    print("[1/2] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    print("[1/2] Processor loaded.")

    # 限制动态分辨率瓦片数，避免图像 token 过多导致 OOM
    if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'max_patches'):
        processor.image_processor.max_patches = 1
        print(f"[Mem] max_patches set to {processor.image_processor.max_patches}")
    
    # 2. 按官方方式加载 CausalLM（直接放到目标 GPU）
    print(f"[2/2] Loading model weights on CUDA:{gpu_id}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": f"cuda:{gpu_id}"},
        low_cpu_mem_usage=True,
    )
    print(f"[2/2] Model loaded in {time.time() - t0:.1f}s.")

    return processor, model



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/code/Intern-S1-mini")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/root/code/ClearSight/data/coco/val2014")
    parser.add_argument("--question-file", type=str, default="/root/code/ClearSight/data/pope/coco_pope_random.json")
    parser.add_argument("--answers-file", type=str, default="/root/code/ClearSight/outputs/analysis/res_coco_random_s1mini.pt")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    model_path = os.path.expanduser(args.model_path)
    processor, model = load_intern_s1_mini(model_path, args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    if torch.cuda.is_available():
        print(
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}, "
            f"target physical gpu-id={args.gpu_id}, "
            f"logical device: {torch.cuda.current_device()}, "
            f"name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
        )

    print("Model ready. Configuring attention implementation...")
    # 仅切换 language_model 的 attention 实现，避免 vision_tower 也被切到 eager_paged 导致显存暴涨
    if hasattr(model, "language_model") and hasattr(model.language_model, "set_attn_implementation"):
        try:
            model.language_model.set_attn_implementation("eager_paged")
            print("Language-model attention implementation set to eager_paged.")
        except Exception:
            model.language_model.set_attn_implementation("eager")
            print("Language-model attention implementation set to eager (fallback).")
    elif hasattr(model, "set_attn_implementation"):
        # 兜底：老接口只能设置整模型
        try:
            model.set_attn_implementation("eager")
            print("Global attention implementation set to eager (fallback).")
        except Exception as e:
            print(f"Warning: failed to set attention implementation: {e}")
    # checkpointing 可显著降低激活显存占用
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    # 仅需要 attention map 的梯度，不需要模型参数梯度，显著降低显存占用
    for p in model.parameters():
        p.requires_grad_(False)
    # 放开一个很小的语言侧参数，保证计算图可回传到 language attentions，
    # 同时避免对视觉塔输入求梯度带来的超大显存开销。
    grad_anchor = None
    if hasattr(model, "language_model"):
        for name, p in model.language_model.named_parameters():
            if "norm.weight" in name and p.ndim == 1 and p.numel() <= 8192:
                p.requires_grad_(True)
                grad_anchor = name
                break
    print(f"Gradient anchor parameter: {grad_anchor if grad_anchor else 'None'}")
    print("Attention configuration completed.")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    questions = random.sample(questions, 50)
    print(f"Loaded {len(questions)} questions for analysis.")

    results = []

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        qs = qs + " Please just answer yes or no."

        # 将目标标签转换为 id（用 encode 而非 convert_tokens_to_ids，兼容多 subword 标签）
        label = line['label']
        label_ids = processor.tokenizer.encode(label, add_special_tokens=False)
        label = torch.tensor(label_ids[-1], dtype=torch.int64, device=device)

        # 1. 加载本地图片
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')

        # ---------------- 官方标准输入构造法 ----------------
        # 严格按照官方示例的字典结构，不要自己写 <image> 字符串！
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": qs},
                ],
            }
        ]

        # 让 processor 自动去推导它自己到底要几个占位符、长什么样
        # 这行代码等价于自动生成 text_prompt 和 pixel_values
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )

        # 动态计算 token 边界：找到 image_token_id 在 input_ids 中的位置
        input_ids = inputs['input_ids'][0]
        image_token_id = model.config.image_token_id
        img_mask = (input_ids == image_token_id)
        img_positions = img_mask.nonzero(as_tuple=True)[0]
        SYS_LEN = img_positions[0].item()          # 图像 token 之前的都是 system/prompt
        IMG_LEN = img_positions.shape[0]            # 图像 token 总数

        # 先整体搬到模型设备，再仅将 pixel_values 调整为 bf16（避免 input_ids 被转浮点）
        inputs = inputs.to(device)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=torch.bfloat16)
        # --------------------------------------------------

        model.zero_grad()
        
        # 3. 前向传播
        output_ids = model.forward(
            **inputs,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

        # 不替换 self_attn 模块，直接对注意力输出保留梯度（兼容 Qwen3）
        for attn in output_ids['attentions']:
            if attn is not None and attn.requires_grad:
                attn.retain_grad()

        # 取出预测的 logit
        pred_logit = output_ids['logits'][:,-1,:].squeeze(0)
        
        loss = F.cross_entropy(pred_logit, label)
        loss.backward()

        # ---------------- 提取注意力图 ----------------
        img_flow, layers_attn = [], []
        for idx_layer, _ in enumerate(model.language_model.layers):
            attn_tensor = output_ids['attentions'][idx_layer]
            attn_score = attn_tensor.detach().clone().cpu()
            if attn_tensor.grad is None:
                # 极端情况下该层无可用梯度，避免直接崩溃
                attn_grad = torch.zeros_like(attn_tensor).detach().clone().cpu()
            else:
                attn_grad = attn_tensor.grad.detach().clone().cpu()
            saliency = torch.abs(attn_grad * attn_score)

            img_saliency = saliency_compute(saliency, SYS_LEN, IMG_LEN)
            attn_props = attention_compute(attn_score, SYS_LEN, IMG_LEN)

            img_flow.append(img_saliency)
            layers_attn.append(attn_props)

        results.append((img_flow, layers_attn))

        del output_ids, loss, pred_logit
        torch.cuda.empty_cache()

    torch.save(results, args.answers_file)