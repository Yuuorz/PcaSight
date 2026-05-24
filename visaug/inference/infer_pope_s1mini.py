import argparse
import torch
import os
import json
import types
from tqdm import tqdm
import uuid

from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
from PIL import Image
import math
import torch.nn.functional as F


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# ====================== VAF (Visual Amplification Fusion) ======================

def _vaf_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """
    Drop-in replacement for Qwen3Attention.forward that applies VAF:
      - Enhance text→image attention by enh_para
      - Suppress text→system attention by sup_para
    before the softmax.
    """
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

    # === VAF: enhance visual attention, suppress system attention ===
    SYS_LEN = self._vaf_sys_len
    IMG_LEN = self._vaf_img_len
    q_len = query_states.shape[2]

    if SYS_LEN + IMG_LEN > 0 and q_len > SYS_LEN + IMG_LEN:
        # Only apply VAF during prefill, skip during decode
        attn_weights[:, :, SYS_LEN + IMG_LEN:, SYS_LEN:SYS_LEN + IMG_LEN] *= self._vaf_enh_para
        attn_weights[:, :, SYS_LEN + IMG_LEN:, :SYS_LEN] *= self._vaf_sup_para
    # ================================================================

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states_r.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(
        attn_weights,
        p=0.0 if not self.training else self.attention_dropout,
        training=self.training,
    )
    attn_output = torch.matmul(attn_weights, value_states_r)
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def apply_vaf(model, start_layer, end_layer, enh_para, sup_para):
    """
    Apply VAF to layers [start_layer, end_layer] by monkey-patching
    their self_attn.forward with the VAF-enhanced version.
    Returns the list of layer indices that were patched.
    """
    vaf_layers = []
    for i, layer in enumerate(model.language_model.layers):
        if start_layer <= i <= end_layer:
            attn = layer.self_attn
            attn._vaf_enh_para = enh_para
            attn._vaf_sup_para = sup_para
            attn._vaf_sys_len = 0
            attn._vaf_img_len = 0
            attn.forward = types.MethodType(_vaf_attn_forward, attn)
            vaf_layers.append(i)
    print(f"VAF applied to layers {vaf_layers} (enh={enh_para}, sup={sup_para})")
    return vaf_layers


def set_vaf_boundaries(model, vaf_layers, sys_len, img_len):
    """Set per-sample token boundaries on all VAF layers before generation."""
    for i in vaf_layers:
        attn = model.language_model.layers[i].self_attn
        attn._vaf_sys_len = sys_len
        attn._vaf_img_len = img_len


# ====================== Main ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/code/Intern-S1-mini")
    parser.add_argument("--image-folder", type=str, default="/root/code/ClearSight/data/coco/val2014")
    parser.add_argument("--question-file", type=str, default="/root/code/ClearSight/data/pope/coco_pope_random.json")
    parser.add_argument("--answers-file", type=str, default="/root/code/ClearSight/outputs/inference/res_coco_random_s1mini.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-visaug", action='store_true', default=False)
    parser.add_argument("--enh-para", type=float, default=1.15)
    parser.add_argument("--sup-para", type=float, default=0.95)
    parser.add_argument("--vaf-start-layer", type=int, default=7)
    parser.add_argument("--vaf-end-layer", type=int, default=15)
    parser.add_argument("--no-think", action="store_true", default=False,
                        help="Disable thinking mode for faster inference")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)

    # 1. Load processor & model (official method)
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )

    print(f"Loading model on CUDA:{args.gpu_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": f"cuda:{args.gpu_id}"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = torch.device(f"cuda:{args.gpu_id}")

    # 2. Apply VAF if requested (must use eager attention for manual weight manipulation)
    vaf_layers = []
    if args.use_visaug:
        if hasattr(model, "language_model") and hasattr(model.language_model, "set_attn_implementation"):
            try:
                model.language_model.set_attn_implementation("eager")
                print("Attention implementation set to eager for VAF.")
            except Exception as e:
                print(f"Warning: could not set eager attention: {e}")
        vaf_layers = apply_vaf(
            model, args.vaf_start_layer, args.vaf_end_layer,
            args.enh_para, args.sup_para,
        )

    # 3. Load questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    image_token_id = model.config.image_token_id

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        qs = qs + " Please just answer yes or no."

        # Load image
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")

        # Build input via official chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": qs},
                ],
            }
        ]

        chat_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if args.no_think:
            chat_kwargs["enable_thinking"] = False
        inputs = processor.apply_chat_template(messages, **chat_kwargs)

        # Dynamic token boundaries for VAF
        if args.use_visaug and vaf_layers:
            input_ids = inputs["input_ids"][0]
            img_mask = input_ids == image_token_id
            img_positions = img_mask.nonzero(as_tuple=True)[0]
            if len(img_positions) > 0:
                sys_len = img_positions[0].item()
                img_len = img_positions.shape[0]
                set_vaf_boundaries(model, vaf_layers, sys_len, img_len)

        inputs = inputs.to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        with torch.inference_mode():
            max_tokens = 64 if args.no_think else 1024
            generate_ids = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True)

        input_token_len = inputs["input_ids"].shape[1]
        outputs = processor.decode(
            generate_ids[0, input_token_len:], skip_special_tokens=True
        )
        outputs = outputs.strip()

        ans_id = str(uuid.uuid4())
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": qs,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": os.path.basename(model_path),
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()
    print(f"Results saved to {answers_file}")
