"""
CHAIR inference: generate image captions for COCO val2014 images.
Supports baseline / VAF / AGVP / AGVP+VAF modes.

Output format per line (jsonl):
  {"image_id": 123456, "caption": "A cat sitting on a couch."}
"""

import argparse
import torch
import os
import json
import types
import random
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
from PIL import Image
import torch.nn.functional as F


# ====================== VAF ======================

def _vaf_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states_r = repeat_kv(key_states, self.num_key_value_groups)
    value_states_r = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states_r.transpose(2, 3)) * self.scaling

    SYS_LEN = self._vaf_sys_len
    IMG_LEN = self._vaf_img_len
    q_len = query_states.shape[2]
    kv_len = key_states_r.shape[2]

    if SYS_LEN + IMG_LEN > 0 and q_len > SYS_LEN + IMG_LEN:
        # Only apply VAF during prefill, skip during decode to avoid degeneration
        attn_weights[:, :, SYS_LEN + IMG_LEN:, SYS_LEN:SYS_LEN + IMG_LEN] *= self._vaf_enh
        attn_weights[:, :, SYS_LEN + IMG_LEN:, :SYS_LEN] *= self._vaf_sup

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :kv_len]
        attn_weights = attn_weights + causal_mask

    attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_probs_drop = F.dropout(attn_probs, p=0.0, training=self.training)
    attn_output = torch.matmul(attn_probs_drop, value_states_r)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_probs


def apply_vaf(model, start_layer, end_layer, enh, sup):
    layers = model.language_model.layers
    patched = []
    for i in range(start_layer, end_layer + 1):
        attn = layers[i].self_attn
        attn._vaf_sys_len = 0
        attn._vaf_img_len = 0
        attn._vaf_enh = enh
        attn._vaf_sup = sup
        attn.forward = types.MethodType(_vaf_attn_forward, attn)
        patched.append(i)
    return patched


# ====================== AGVP ======================

def _agvp_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states_r = repeat_kv(key_states, self.num_key_value_groups)
    value_states_r = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states_r.transpose(2, 3)) * self.scaling

    SYS_LEN = self._agvp_sys_len
    IMG_LEN = self._agvp_img_len
    q_len = query_states.shape[2]
    kv_len = key_states_r.shape[2]

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :kv_len]
        attn_weights = attn_weights + causal_mask

    attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_probs_drop = F.dropout(attn_probs, p=0.0, training=self.training)
    attn_output = torch.matmul(attn_probs_drop, value_states_r)

    # AGVP projection
    if self._agvp_enabled and SYS_LEN + IMG_LEN > 0 and kv_len > SYS_LEN + IMG_LEN:
        img_values = value_states_r[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
        sys_values = value_states_r[:, :, :SYS_LEN, :]

        if q_len > SYS_LEN + IMG_LEN:
            text_start = SYS_LEN + IMG_LEN
            text_to_img = attn_probs[:, :, text_start:, SYS_LEN:SYS_LEN + IMG_LEN]
            text_to_sys = attn_probs[:, :, text_start:, :SYS_LEN]

            d_vis = torch.matmul(text_to_img, img_values).mean(dim=2)
            d_sys = torch.matmul(text_to_sys, sys_values).mean(dim=2)
            d_vis_hat = F.normalize(d_vis, dim=-1)
            d_sys_hat = F.normalize(d_sys, dim=-1)

            # Cos-gate: decide whether to intervene based on direction separability
            cos_vs = (d_vis_hat * d_sys_hat).sum(dim=-1).mean()
            gate_mode = getattr(self, '_agvp_cos_gate', 'neg')
            do_intervene = (gate_mode == 'none') or \
                           (gate_mode == 'neg' and cos_vs < 0) or \
                           (gate_mode == 'pos' and cos_vs > 0)
            if do_intervene:
                p_vis = text_to_img.sum(dim=-1).mean()
                p_sys = text_to_sys.sum(dim=-1).mean()
                imbalance = torch.clamp((p_sys - p_vis) / (p_sys + p_vis + 1e-8), min=0.0, max=1.0)

                text_out = attn_output[:, :, text_start:, :]
                orig_norm = text_out.norm(dim=-1, keepdim=True)

                proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
                text_out = text_out - imbalance * proj_sys
                proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)
                text_out = text_out + imbalance * proj_vis

                # Norm preservation
                text_out = text_out * (orig_norm / (text_out.norm(dim=-1, keepdim=True) + 1e-8))

                attn_output = torch.cat([attn_output[:, :, :text_start, :], text_out], dim=2)
        else:
            # Decode phase: skip AGVP to avoid error accumulation
            # Prefill correction already propagates through residual stream
            pass

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_probs


def apply_agvp(model, start_layer, end_layer, cos_gate='neg'):
    layers = model.language_model.layers
    patched = []
    for i in range(start_layer, end_layer + 1):
        attn = layers[i].self_attn
        attn._agvp_sys_len = 0
        attn._agvp_img_len = 0
        attn._agvp_enabled = True
        attn._agvp_cos_gate = cos_gate
        attn.forward = types.MethodType(_agvp_attn_forward, attn)
        patched.append(i)
    print(f"AGVP applied to layers {patched} (cos_gate={cos_gate})")
    return patched


def set_boundaries(model, sys_len, img_len, mode="agvp"):
    for layer in model.language_model.layers:
        attn = layer.self_attn
        if mode == "agvp" and hasattr(attn, "_agvp_sys_len"):
            attn._agvp_sys_len = sys_len
            attn._agvp_img_len = img_len
        elif mode == "vaf" and hasattr(attn, "_vaf_sys_len"):
            attn._vaf_sys_len = sys_len
            attn._vaf_img_len = img_len


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of random images to caption")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--gt-file", type=str, default="",
                        help="GT segmentation JSONL to filter images with annotations")

    # Method selection
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "vaf", "agvp", "agvp_vaf"])

    # AGVP params
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=16)
    parser.add_argument("--cos-gate", type=str, default="neg", choices=["neg", "pos", "none"],
                        help="Cos gate mode: neg=cos<0 only, pos=cos>0 only (ablation), none=always")

    # VAF params
    parser.add_argument("--enh-para", type=float, default=1.15)
    parser.add_argument("--sup-para", type=float, default=0.95)

    args = parser.parse_args()

    # 1. Load model
    model_path = os.path.expanduser(args.model_path)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True,
        device_map={"": f"cuda:{args.gpu_id}"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = torch.device(f"cuda:{args.gpu_id}")

    # 2. Apply method
    if args.method == "agvp":
        if hasattr(model, "language_model") and hasattr(model.language_model, "set_attn_implementation"):
            model.language_model.set_attn_implementation("eager")
            print("Attention set to eager for AGVP.")
        apply_agvp(model, args.start_layer, args.end_layer, cos_gate=args.cos_gate)
        boundary_mode = "agvp"
    elif args.method == "vaf":
        if hasattr(model, "language_model") and hasattr(model.language_model, "set_attn_implementation"):
            model.language_model.set_attn_implementation("eager")
            print("Attention set to eager for VAF.")
        apply_vaf(model, args.start_layer, args.end_layer, args.enh_para, args.sup_para)
        boundary_mode = "vaf"
    else:
        boundary_mode = None

    # 3. Collect image files (only those with GT if gt-file provided)
    gt_image_ids = set()
    if args.gt_file:
        with open(args.gt_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                gt_image_ids.add(item["image_id"])
        print(f"Loaded {len(gt_image_ids)} images with GT annotations")

    image_files = sorted([
        f for f in os.listdir(args.image_folder)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])

    if gt_image_ids:
        image_files = [
            f for f in image_files
            if int(f.split('_')[-1].split('.')[0]) in gt_image_ids
        ]
        print(f"Filtered to {len(image_files)} images with GT")

    random.seed(args.seed)
    if args.num_samples < len(image_files):
        image_files = random.sample(image_files, args.num_samples)

    image_token_id = model.config.image_token_id

    # 4. Generate captions
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    prompt = "Describe this image in detail."

    for img_name in tqdm(image_files, desc=f"CHAIR [{args.method}]"):
        # Extract COCO image_id from filename: COCO_val2014_000000123456.jpg -> 123456
        image_id = int(img_name.split("_")[-1].split(".")[0])

        image_path = os.path.join(args.image_folder, img_name)
        image = Image.open(image_path).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        chat_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        if args.no_think:
            chat_kwargs["enable_thinking"] = False

        inputs = processor.apply_chat_template(messages, **chat_kwargs)

        # Set boundaries
        input_ids = inputs["input_ids"][0]
        img_mask = input_ids == image_token_id
        img_positions = img_mask.nonzero(as_tuple=True)[0]

        if len(img_positions) > 0 and boundary_mode:
            sys_len = img_positions[0].item()
            img_len = img_positions.shape[0]
            set_boundaries(model, sys_len, img_len, mode=boundary_mode)

        inputs = inputs.to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        max_tokens = args.max_new_tokens

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        caption = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

        ans_file.write(json.dumps({
            "image_id": image_id,
            "caption": caption,
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f"Done. Captions saved to {answers_file}")


if __name__ == "__main__":
    main()
