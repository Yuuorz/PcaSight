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
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# ============ Attention-Guided Visual Projection (AGVP) ============
#
# In the middle fusion layers, after computing attention output:
#   1. Extract "visual direction" and "system direction" from
#      attention-weighted Value vectors.
#   2. Measure attention imbalance (system vs visual).
#   3. Project away the system direction and project toward the
#      visual direction on text tokens' hidden states.
#
# Optionally combined with VAF (modifying attention weights before
# softmax) for a dual-level intervention.
# ===================================================================

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

    # ---- Softmax ----
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :kv_len]
        attn_weights = attn_weights + causal_mask

    attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_probs_drop = F.dropout(
        attn_probs, p=0.0 if not self.training else self.attention_dropout,
        training=self.training,
    )
    attn_output = torch.matmul(attn_probs_drop, value_states_r)
    # attn_output: (bsz, heads, q_len, head_dim)

    # ---- AGVP: project away system, project toward visual ----
    if self._agvp_enabled and SYS_LEN + IMG_LEN > 0 and kv_len > SYS_LEN + IMG_LEN:
        img_values = value_states_r[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
        sys_values = value_states_r[:, :, :SYS_LEN, :]

        if q_len > SYS_LEN + IMG_LEN:
            # --- Prefill: modify text tokens ---
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
            mode = getattr(self, '_agvp_mode', 'hybrid')
            cos_abs = abs(cos_vs.item())
            p_vis_val = text_to_img.sum(dim=-1).mean()
            p_sys_val = text_to_sys.sum(dim=-1).mean()
            imb = torch.clamp((p_sys_val - p_vis_val) / (p_sys_val + p_vis_val + 1e-8), min=0.0, max=1.0).item()

            # ---- Hybrid modes: cos sign selects operation ----
            if mode.startswith('hybrid'):
                text_out = attn_output[:, :, text_start:, :]
                orig_norm = text_out.norm(dim=-1, keepdim=True)
                did_modify = False

                if cos_vs < 0 and imb > 0:
                    # Rotation toward d_vis
                    if mode == 'hybrid_soft':
                        alpha = cos_abs * cos_abs * imb
                    else:
                        alpha = cos_abs
                    text_hat = F.normalize(text_out, dim=-1)
                    d_vis_exp = d_vis_hat.unsqueeze(2).expand_as(text_out)
                    new_dir = F.normalize((1 - alpha) * text_hat + alpha * d_vis_exp, dim=-1)
                    text_out = new_dir * orig_norm
                    did_modify = True
                elif cos_vs > 0 and imb > 0:
                    # Projection: -sys +vis
                    if mode == 'hybrid_soft':
                        strength = cos_abs * cos_abs * imb
                    else:
                        strength = imb * cos_abs
                    proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
                    proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)
                    text_out = text_out - strength * proj_sys + strength * proj_vis
                    text_out = text_out * (orig_norm / (text_out.norm(dim=-1, keepdim=True) + 1e-8))
                    did_modify = True

                if did_modify:
                    attn_output = torch.cat(
                        [attn_output[:, :, :text_start, :], text_out], dim=2
                    )

            # ---- proj_cospos: only project when cos > 0 ----
            elif mode == 'proj_cospos':
                if cos_vs > 0 and imb > 0:
                    text_out = attn_output[:, :, text_start:, :]
                    orig_norm = text_out.norm(dim=-1, keepdim=True)
                    strength = imb * cos_abs
                    proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
                    proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)
                    text_out = text_out - strength * proj_sys + strength * proj_vis
                    text_out = text_out * (orig_norm / (text_out.norm(dim=-1, keepdim=True) + 1e-8))
                    attn_output = torch.cat(
                        [attn_output[:, :, :text_start, :], text_out], dim=2
                    )

            # ---- rotate_vis modes (cos < 0 only) ----
            elif mode.startswith('rotate_vis'):
                if cos_vs < 0 and imb > 0:
                    text_out = attn_output[:, :, text_start:, :]
                    orig_norm = text_out.norm(dim=-1, keepdim=True)
                    if mode == 'rotate_vis_adaptive':
                        alpha = imb * cos_abs
                    else:
                        alpha = cos_abs
                    text_hat = F.normalize(text_out, dim=-1)
                    d_vis_exp = d_vis_hat.unsqueeze(2).expand_as(text_out)
                    new_dir = F.normalize((1 - alpha) * text_hat + alpha * d_vis_exp, dim=-1)
                    text_out = new_dir * orig_norm
                    attn_output = torch.cat(
                        [attn_output[:, :, :text_start, :], text_out], dim=2
                    )

            # ---- Legacy projection mode ----
            elif do_intervene:
                text_out = attn_output[:, :, text_start:, :]
                orig_norm = text_out.norm(dim=-1, keepdim=True)
                proj_sys = (text_out * d_sys_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_sys_hat.unsqueeze(2)
                proj_vis = (text_out * d_vis_hat.unsqueeze(2)).sum(dim=-1, keepdim=True) * d_vis_hat.unsqueeze(2)
                strength = imb * cos_abs
                text_out = text_out - strength * proj_sys + strength * proj_vis
                text_out = text_out * (orig_norm / (text_out.norm(dim=-1, keepdim=True) + 1e-8))
                attn_output = torch.cat(
                    [attn_output[:, :, :text_start, :], text_out], dim=2
                )
        else:
            # Decode phase: skip AGVP
            pass

    # ---- Output projection ----
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_probs


def apply_agvp(model, start_layer, end_layer, cos_gate='neg', agvp_mode='rotate_vis_imb0'):
    """
    Apply AGVP/AGVR to layers [start_layer, end_layer].
    cos_gate: 'neg' = only cos<0, 'pos' = only cos>0 (ablation), 'none' = always
    agvp_mode: 'rotate_vis_imb0' (default), 'rotate_vis_adaptive', 'rotate_vis', 'projection'
    Returns list of patched layer indices.
    """
    patched = []
    for i, layer in enumerate(model.language_model.layers):
        if start_layer <= i <= end_layer:
            attn = layer.self_attn
            attn._agvp_enabled = True
            attn._agvp_sys_len = 0
            attn._agvp_img_len = 0
            attn._agvp_cos_gate = cos_gate
            attn._agvp_mode = agvp_mode
            attn.forward = types.MethodType(_agvp_attn_forward, attn)
            patched.append(i)

    print(f"AGVP applied to layers {patched} (cos_gate={cos_gate}, mode={agvp_mode})")
    return patched


def set_boundaries(model, patched_layers, sys_len, img_len):
    """Set per-sample token boundaries before generation."""
    for i in patched_layers:
        attn = model.language_model.layers[i].self_attn
        attn._agvp_sys_len = sys_len
        attn._agvp_img_len = img_len


# ====================== Main ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/code/Intern-S1-mini")
    parser.add_argument("--image-folder", type=str, default="/root/code/ClearSight/data/coco/val2014")
    parser.add_argument("--question-file", type=str, default="/root/code/ClearSight/data/pope/coco_pope_random.json")
    parser.add_argument("--answers-file", type=str, default="/root/code/ClearSight/outputs/inference/res_coco_random_s1mini_agvp.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--start-layer", type=int, default=7)
    parser.add_argument("--end-layer", type=int, default=15)
    parser.add_argument("--cos-gate", type=str, default="neg", choices=["neg", "pos", "none"],
                        help="Cos gate mode: neg=cos<0 only, pos=cos>0 only (ablation), none=always")
    parser.add_argument("--agvp-mode", type=str, default="rotate_vis_imb0",
                        help="AGVP mode: rotate_vis_imb0, rotate_vis_adaptive, rotate_vis, projection")
    parser.add_argument("--no-think", action="store_true", default=False,
                        help="Disable thinking mode for faster inference")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)

    # 1. Load
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

    # 2. Set eager attention (required for manual attention manipulation)
    if hasattr(model, "language_model") and hasattr(model.language_model, "set_attn_implementation"):
        try:
            model.language_model.set_attn_implementation("eager")
            print("Attention implementation set to eager for AGVP.")
        except Exception as e:
            print(f"Warning: could not set eager attention: {e}")

    # 3. Apply AGVP
    patched_layers = apply_agvp(model, args.start_layer, args.end_layer, cos_gate=args.cos_gate, agvp_mode=args.agvp_mode)

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

        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")

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

        # Dynamic token boundaries
        input_ids = inputs["input_ids"][0]
        img_mask = input_ids == image_token_id
        img_positions = img_mask.nonzero(as_tuple=True)[0]
        if len(img_positions) > 0:
            sys_len = img_positions[0].item()
            img_len = img_positions.shape[0]
            set_boundaries(model, patched_layers, sys_len, img_len)

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
