"""
POPE inference script for LLaVA-v1.5-7b with Multi-Dimensional PCA Subspace Projection.
"""
import argparse
import torch
import os
import sys
import json
import math
from tqdm import tqdm
import shortuuid
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../LLaVA"))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from AGVPAdapter import AGVPAdapter

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    return split_list(lst, n)[k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/code/llava-v1.5-7b")
    parser.add_argument("--image-folder", type=str, default="/root/code/ClearSight/data/coco/val2014")
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    
    # PCA control
    parser.add_argument("--pca-k", type=int, default=3, help="Top-k eigenvalues")
    parser.add_argument("--scale", type=float, default=1.0, help="Subspace pull scalar")
    parser.add_argument("--start-layer", type=int, default=2)
    parser.add_argument("--end-layer", type=int, default=31)
    args = parser.parse_args()

    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, "llava-v1.5-7b")

    print(f"=== Injecting PCA Subspace Adapter into LLaVA ===")
    print(f"Layers: {args.start_layer}-{args.end_layer} | Components: {args.pca_k} | Scale: {args.scale}")

    for i, layer in enumerate(model.model.layers):
        if args.start_layer <= i <= args.end_layer:
            # 继承你之前的强弱分区配置
            zone_weight = 0.5 if i <= 8 else 1.0
            
            adap = AGVPAdapter(layer.self_attn.config)
            adap.load_state_dict(layer.self_attn.state_dict())
            adap._agvp_mode = "subspace_pca"
            adap._pca_k = args.pca_k
            adap._pca_scale = args.scale
            adap._zone_weight = zone_weight
            
            adap = adap.half().cuda()
            layer.self_attn = adap

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    ans_file = open(os.path.expanduser(args.answers_file), "w")

    for line in tqdm(questions):
        idx, image_file, qs = line["question_id"], line["image"], line["text"] + " Please just answer yes or no."
        cur_prompt = qs
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=image_tensor.unsqueeze(0).half().cuda(), max_new_tokens=64, use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str): outputs = outputs[:-len(stop_str)]

        ans_file.write(json.dumps({"question_id": idx, "prompt": cur_prompt, "text": outputs.strip(), "answer_id": shortuuid.uuid()}) + "\n")
        ans_file.flush()
    ans_file.close()
