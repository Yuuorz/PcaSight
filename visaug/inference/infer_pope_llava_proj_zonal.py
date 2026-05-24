"""
LLaVA 分区投影（proj_zonal）：层区间特定强度的 AGVP 投影。
L0-1: skip, L2-8: 弱投影(×0.5), L9-31: 投影(×1.0)
全投影无旋转，使用修复后的图片预处理（process_images）。
"""
import argparse
import torch
import os
import sys
import json
import math
import torch.nn.functional as F
from tqdm import tqdm
import shortuuid
from PIL import Image
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../LLaVA"))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from AGVPAdapter import AGVPAdapter


class ZoneProjAdapter(AGVPAdapter):
    """AGVP with configurable projection strength scale."""
    def __init__(self, config, strength_scale=1.0):
        super().__init__(config)
        self._agvp_strength_scale = strength_scale

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False):
        """Same as AGVPAdapter forward but strength scaled by _agvp_strength_scale."""
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            query_states = torch.cat([F.linear(hidden_states, qs) for qs in query_slices], dim=-1)
            key_states = torch.cat([F.linear(hidden_states, ks) for ks in key_slices], dim=-1)
            value_states = torch.cat([F.linear(hidden_states, vs) for vs in value_slices], dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states)

        # ---- Zonal projection (always project, strength controlled by _agvp_strength_scale) ----
        SYS_LEN = self.SYS_LEN
        IMG_LEN = self.IMG_LEN
        if kv_seq_len > SYS_LEN + IMG_LEN and q_len > SYS_LEN + IMG_LEN:
            text_start = SYS_LEN + IMG_LEN
            img_values = value_states[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
            sys_values = value_states[:, :, :SYS_LEN, :]
            text_to_img = attn_probs[:, :, text_start:, SYS_LEN:SYS_LEN + IMG_LEN]
            text_to_sys = attn_probs[:, :, text_start:, :SYS_LEN]

            d_vis = torch.matmul(text_to_img, img_values).mean(dim=2)
            d_sys = torch.matmul(text_to_sys, sys_values).mean(dim=2)
            d_vis_hat = F.normalize(d_vis, dim=-1)
            d_sys_hat = F.normalize(d_sys, dim=-1)

            cos_vs = (d_vis_hat * d_sys_hat).sum(dim=-1).mean()
            imb = torch.clamp(
                (text_to_sys.sum(dim=-1).mean() - text_to_img.sum(dim=-1).mean())
                / (text_to_sys.sum(dim=-1).mean() + text_to_img.sum(dim=-1).mean() + 1e-8),
                0, 1
            ).item()

            if imb > 0:
                strength = imb * abs(cos_vs.item()) * self._agvp_strength_scale
                text_out = attn_output[:, :, text_start:, :]
                orig_norm = text_out.norm(dim=-1, keepdim=True)
                proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
                proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)
                text_out = text_out - strength * proj_sys + strength * proj_vis
                text_out = text_out * (orig_norm / (text_out.norm(dim=-1, keepdim=True) + 1e-8))
                attn_output = torch.cat([attn_output[:, :, :text_start, :], text_out], dim=2)
        # ----------------------------------------------------------

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value


def apply_zonal_projection(model, zone_config):
    """
    zone_config: list of (start_layer, end_layer, strength_scale)
    """
    for i, layer in enumerate(model.model.layers):
        scale = None
        for start, end, s in zone_config:
            if start <= i <= end:
                scale = s
                break
        if scale is None:
            continue

        adap = ZoneProjAdapter(layer.self_attn.config, strength_scale=scale)
        adap.load_state_dict(layer.self_attn.state_dict())
        adap._agvp_cos_gate = 'none'  # always project (ignore cos sign)
        adap._agvp_mode = 'imb_cos_vis'
        adap = adap.half().cuda()
        layer.self_attn = adap
        print(f"  Layer {i}: proj (strength_scale={scale})")

    print(f"ZoneProj: {len(zone_config)} zones applied")


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


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
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Zone config: (start, end, strength_scale)
    # L0-1: skip, L2-8: weak, L9-16: normal, L17-31: normal
    zone_config = [(2, 8, 0.5), (9, 16, 1.0), (17, 31, 1.0)]
    apply_zonal_projection(model, zone_config)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"] + " Please just answer yes or no."
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
