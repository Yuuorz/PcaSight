"""
S1-mini 分区投影（proj_zonal）— 全投影无旋转。
B7-16: 弱投影 (cos²·imb, 无cos门控)
C17-19: 投影 (cos²·imb, 无cos门控)
A0-6 & D20-35: 不 patch
"""
import argparse
import torch
import os
import json
import types
from tqdm import tqdm
import uuid
import math
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
from PIL import Image


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def _zonal_proj_forward(
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

    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

    kr = repeat_kv(k, self.num_key_value_groups)
    vr = repeat_kv(v, self.num_key_value_groups)
    aw = torch.matmul(q, kr.transpose(2, 3)) * self.scaling

    SYS_LEN = self._agvp_sys_len
    IMG_LEN = self._agvp_img_len
    q_len = q.shape[2]
    kv_len = kr.shape[2]

    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, :kv_len]
    ap = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    ao = torch.matmul(ap, vr)

    strength_scale = getattr(self, '_proj_strength', 1.0)
    if SYS_LEN + IMG_LEN > 0 and kv_len > SYS_LEN + IMG_LEN and q_len > SYS_LEN + IMG_LEN:
        ts = SYS_LEN + IMG_LEN
        img_vals = vr[:, :, SYS_LEN:SYS_LEN + IMG_LEN, :]
        sys_vals = vr[:, :, :SYS_LEN, :]
        t2i = ap[:, :, ts:, SYS_LEN:SYS_LEN + IMG_LEN]
        t2s = ap[:, :, ts:, :SYS_LEN]

        d_vis = torch.matmul(t2i, img_vals).mean(dim=2)
        d_sys = torch.matmul(t2s, sys_vals).mean(dim=2)
        dvh = F.normalize(d_vis, dim=-1)
        dsh = F.normalize(d_sys, dim=-1)

        cos_vs = (dvh * dsh).sum(dim=-1).mean()
        imb = torch.clamp(
            (t2s.sum(dim=-1).mean() - t2i.sum(dim=-1).mean())
            / (t2s.sum(dim=-1).mean() + t2i.sum(dim=-1).mean() + 1e-8),
            0, 1,
        ).item()

        # Projection (no cos_vs gate — always project when imb>0)
        if imb > 0:
            to = ao[:, :, ts:, :]
            onorm = to.norm(dim=-1, keepdim=True)
            ca = abs(cos_vs.item())
            strength = ca * ca * imb * strength_scale

            ps = (to * dsh.unsqueeze(2)).sum(dim=-1, keepdim=True) * dsh.unsqueeze(2)
            pv = (to * dvh.unsqueeze(2)).sum(dim=-1, keepdim=True) * dvh.unsqueeze(2)
            to = to - strength * ps + strength * pv
            to = to * (onorm / (to.norm(dim=-1, keepdim=True) + 1e-8))
            ao = torch.cat([ao[:, :, :ts, :], to], dim=2)

    ao = ao.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    return self.o_proj(ao), ap


def apply_zonal_proj(model, weak_layers, strong_layers):
    patched = []
    for i, layer in enumerate(model.language_model.layers):
        if i in weak_layers:
            strength = 0.5
        elif i in strong_layers:
            strength = 1.0
        else:
            continue

        attn = layer.self_attn
        attn._agvp_sys_len = 0
        attn._agvp_img_len = 0
        attn._proj_strength = strength
        attn.forward = types.MethodType(_zonal_proj_forward, attn)
        patched.append((i, strength))

    print(f"ZonalProj: {len(patched)} layers patched "
          f"(weak={len(weak_layers)}, strong={len(strong_layers)})")
    for i, s in patched:
        print(f"  L{i}: strength={s}")
    return [p[0] for p in patched]


def set_boundaries(model, patched_layers, sys_len, img_len):
    for i in patched_layers:
        attn = model.language_model.layers[i].self_attn
        attn._agvp_sys_len = sys_len
        attn._agvp_img_len = img_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/code/Intern-S1-mini")
    parser.add_argument("--image-folder", type=str, default="/root/code/ClearSight/data/coco/val2014")
    parser.add_argument("--question-file", type=str, default="/root/code/ClearSight/data/pope/coco_pope_random.json")
    parser.add_argument("--answers-file", type=str, default="/root/code/ClearSight/outputs/pope_s1mini/res_proj_zonal_random.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=3)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)

    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    gpu = f"cuda:{args.gpu_id}"
    print(f"Loading model on {gpu}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": gpu},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.language_model.set_attn_implementation("eager")

    # B zone L7-16: weak proj, C zone L17-19: strong proj
    weak_layers = list(range(7, 17))
    strong_layers = list(range(17, 20))
    patched_layers = apply_zonal_proj(model, weak_layers, strong_layers)

    questions = [json.loads(l) for l in open(os.path.expanduser(args.question_file))]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    image_token_id = model.config.image_token_id
    device = torch.device(gpu)

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"] + " Please just answer yes or no."

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": qs},
            ]},
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=False,
        )

        input_ids = inputs["input_ids"][0]
        img_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        if len(img_positions) > 0:
            sys_len = img_positions[0].item()
            img_len = len(img_positions)
            set_boundaries(model, patched_layers, sys_len, img_len)

        inputs = inputs.to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs, max_new_tokens=64, use_cache=True,
            )

        input_token_len = inputs["input_ids"].shape[1]
        outputs = processor.decode(
            generate_ids[0, input_token_len:], skip_special_tokens=True
        ).strip()

        ans_file.write(json.dumps({
            "question_id": idx, "prompt": qs, "text": outputs,
            "answer_id": str(uuid.uuid4()),
            "model_id": os.path.basename(model_path),
            "metadata": {},
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f"Results saved to {answers_file}")
