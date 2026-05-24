"""
POPE inference script for Intern-S1-mini with Multi-Dimensional PCA Subspace Projection.
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

def _s1mini_pca_forward(
    self, hidden_states: torch.Tensor, position_embeddings: tuple,
    attention_mask=None, past_key_values=None, cache_position=None, **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})

    kr = repeat_kv(k, self.num_key_value_groups)
    vr = repeat_kv(v, self.num_key_value_groups)
    aw = torch.matmul(q, kr.transpose(2, 3)) * self.scaling

    SYS_LEN, IMG_LEN = self._agvp_sys_len, self._agvp_img_len
    q_len, kv_len = q.shape[2], kr.shape[2]

    if attention_mask is not None: aw = aw + attention_mask[:, :, :, :kv_len]
    ap = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    ao = torch.matmul(ap, vr)

    if SYS_LEN + IMG_LEN > 0 and kv_len > SYS_LEN + IMG_LEN and q_len > SYS_LEN + IMG_LEN:
        ts = SYS_LEN + IMG_LEN
        to = ao[:, :, ts:, :]
        orig_norm = to.norm(dim=-1, keepdim=True)

        pca_k = getattr(self, '_pca_k', 3)
        scale = getattr(self, '_pca_scale', 1.0)
        zone_weight = getattr(self, '_zone_weight', 1.0)

        # 动态提取视觉 Value 场的高维超平面
        img_values = vr[:, :, SYS_LEN:ts, :]
        img_mean = img_values.mean(dim=2, keepdim=True)
        img_centered = img_values - img_mean

        try:
            _, _, V = torch.linalg.svd(img_centered.float(), full_matrices=False)
            W = V[:, :, :pca_k, :].transpose(-1, -2).to(to.dtype)
        except RuntimeError:
            W = F.normalize(img_values.mean(dim=2, keepdim=True), dim=-1).transpose(-1, -2).expand(-1, -1, -1, pca_k)

        # 文本 Token 向视觉子空间投射拉向
        proj_subspace = torch.matmul(torch.matmul(to, W), W.transpose(-1, -2))
        to = to + (scale * zone_weight) * proj_subspace
        to = to * (orig_norm / (to.norm(dim=-1, keepdim=True) + 1e-8))

        ao = torch.cat([ao[:, :, :ts, :], to], dim=2)

    ao = ao.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    return self.o_proj(ao), ap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/code/Intern-S1-mini")
    parser.add_argument("--image-folder", type=str, default="/root/code/ClearSight/data/coco/val2014")
    parser.add_argument("--question-file", type=str, default="/root/code/ClearSight/data/pope/coco_pope_random.json")
    parser.add_argument("--answers-file", type=str)
    
    # PCA control
    parser.add_argument("--pca-k", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--target-layers", type=str, default="7-19", help="Safe zonal expansion")
    args = parser.parse_args()

    layers = list(range(int(args.target_layers.split("-")[0]), int(args.target_layers.split("-")[1]) + 1))
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": "cuda:0"})
    model.eval()
    model.language_model.set_attn_implementation("eager")

    patched = []
    for i, layer in enumerate(model.language_model.layers):
        if i in layers:
            attn = layer.self_attn
            attn._agvp_sys_len, attn._agvp_img_len = 0, 0
            attn._pca_k, attn._pca_scale = args.pca_k, args.scale
            attn._zone_weight = 0.5 if i <= 16 else 1.0  # L17-19 强激活进入缠结区
            attn.forward = types.MethodType(_s1mini_pca_forward, attn)
            patched.append(i)

    print(f"S1-mini PCA Interceptor Active on Layers: {patched}")
    questions = [json.loads(l) for l in open(os.path.expanduser(args.question_file))]
    ans_file = open(os.path.expanduser(args.answers_file), "w")

    for line in tqdm(questions):
        idx, image_file, qs = line["question_id"], line["image"], line["text"] + " Please just answer yes or no."
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": qs}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)

        input_ids = inputs["input_ids"][0]
        img_pos = (input_ids == model.config.image_token_id).nonzero(as_tuple=True)[0]
        if len(img_pos) > 0:
            for l_idx in patched:
                model.language_model.layers[l_idx].self_attn._agvp_sys_len = img_pos[0].item()
                model.language_model.layers[l_idx].self_attn._agvp_img_len = len(img_pos)

        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        with torch.inference_mode():
            generate_ids = model.generate(**inputs, max_new_tokens=64, use_cache=True)

        outputs = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ans_file.write(json.dumps({"question_id": idx, "prompt": qs, "text": outputs, "answer_id": str(uuid.uuid4())}) + "\n")
        ans_file.flush()
    ans_file.close()
