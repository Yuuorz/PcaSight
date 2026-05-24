"""
POPE inference for LLaVA-v1.5-7b.
Supports 3 methods: baseline, vaf, agvp.
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

from AttnAdapter import AttnAdapter
from AGVPAdapter import AGVPAdapter


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/code/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/root/code/ClearSight/data/coco/val2014")
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    # Method selection
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "vaf", "agvp"])
    # VAF params
    parser.add_argument("--enh-para", type=float, default=1.15)
    parser.add_argument("--sup-para", type=float, default=0.95)
    # Layer range (for both VAF and AGVP)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=31)
    parser.add_argument("--cos-gate", type=str, default="neg", choices=["neg", "pos", "none"],
                        help="Cos gate mode: neg=cos<0 only, pos=cos>0 only (ablation), none=always")
    parser.add_argument("--agvp-mode", type=str, default="sys_frac_both",
                        help="AGVP mode: {sys_frac,cos,imb_cos}_{both,sup,vis}")
    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Apply method
    if args.method == "vaf":
        print(f"Applying VAF (enh={args.enh_para}, sup={args.sup_para}) to layers {args.start_layer}-{args.end_layer}")
        for i, layer in enumerate(model.model.layers):
            if args.start_layer <= i <= args.end_layer:
                adap = AttnAdapter(layer.self_attn.config, args.enh_para, args.sup_para)
                adap.load_state_dict(layer.self_attn.state_dict())
                adap = adap.half().cuda()
                layer.self_attn = adap

    elif args.method == "agvp":
        print(f"Applying AGVP to layers {args.start_layer}-{args.end_layer} (cos_gate={args.cos_gate}, mode={args.agvp_mode})")
        for i, layer in enumerate(model.model.layers):
            if args.start_layer <= i <= args.end_layer:
                adap = AGVPAdapter(layer.self_attn.config)
                adap.load_state_dict(layer.self_attn.state_dict())
                adap._agvp_cos_gate = args.cos_gate
                adap._agvp_mode = args.agvp_mode
                adap = adap.half().cuda()
                layer.self_attn = adap

    else:
        print("Running baseline (no intervention)")

    # Load questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        qs = qs + " Please just answer yes or no."
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=1024,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    print(f"Done. Results saved to {answers_file}")
