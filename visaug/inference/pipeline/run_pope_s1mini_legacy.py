"""
POPE inference script for Intern-S1-mini with Multi-Dimensional Uncentered PCA Subspace Projection.
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

    SYS_LEN, IMG_LEN = self._sys_len, self._img_len
    q_len, kv_len = q.shape[2], kr.shape[2]

    if attention_mask is not None: aw = aw + attention_mask[:, :, :, :kv_len]
    ap = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    ao = torch.matmul(ap, vr)

    # ==========================================================
    # 核心重构：无中心化（Uncentered）多维子空间正交拉力投影
    # ==========================================================
    if SYS_LEN + IMG_LEN > 0 and kv_len > SYS_LEN + IMG_LEN and q_len > SYS_LEN + IMG_LEN:
        ts = SYS_LEN + IMG_LEN
        text_tokens = ao[:, :, ts:, :]
        orig_norm = text_tokens.norm(dim=-1, keepdim=True)

        pca_k = getattr(self, '_pca_k', 3)
        scale = getattr(self, '_scale', getattr(self, '_pca_scale', 1.0))
        zone_weight = getattr(self, '_zone_weight', 1.0)

        # 动态截取纯视觉特征 Value 场：[B, Heads, 576, Head_Dim]
        img_values = vr[:, :, SYS_LEN:ts, :]

        # 核心数学破局：绝对不减去 mean！直接对绝对全局多模态场做 SVD 
        try:
            _, _, V = torch.linalg.svd(img_values.float(), full_matrices=False)
            U_k = V[:, :, :pca_k, :].to(text_tokens.dtype)
        except RuntimeError:
            U_k = F.normalize(img_values.mean(dim=2, keepdim=True), dim=-1).transpose(-1, -2).expand(-1, -1, pca_k, -1).transpose(-1, -2)

        # 文本 Tokens 往绝对主成分平面投射，重构高保真去噪分量
        proj_coords = torch.matmul(text_tokens, U_k.transpose(-1, -2))
        proj_subspace = torch.matmul(proj_coords, U_k)

        # 施加温和的正交校准引力
        text_modified = text_tokens + (scale * zone_weight) * proj_subspace
        
        # 严格范数球锁定，保护 S1-mini 脆弱的长思维链（CoT）推理链条
        text_tokens = text_modified * (orig_norm / (text_modified.norm(dim=-1, keepdim=True) + 1e-8))

        # 序列重组回填残差流
        ao = torch.cat([ao[:, :, :ts, :], text_tokens], dim=2)
    # ==========================================================

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
    
    # 全局双重 eagerness 锁死，拍死一切 generate 时的自动优化回退
    model.config._attn_implementation = "eager"
    model.language_model.set_attn_implementation("eager")

    patched = []
    for i, layer in enumerate(model.language_model.layers):
        if i in layers:
            attn = layer.self_attn
            attn._sys_len, attn._img_len = 0, 0
            
            # 同时绑定两个配置名，形成完备的接口防御链
            attn._pca_k = args.pca_k
            attn._scale = args.scale
            attn._pca_scale = args.scale
            
            attn._zone_weight = 0.5 if i <= 16 else 1.0  # L17-19 强激活进入缠结区
            attn.forward = types.MethodType(_s1mini_pca_forward, attn)
            patched.append(i)

    print(f"S1-mini Uncentered PCA Interceptor Active on Layers: {patched}")
    questions = [json.loads(l) for l in open(os.path.expanduser(args.question_file))]
    ans_file = open(os.path.expanduser(args.answers_file), "w")

    for line in tqdm(questions):
        idx, image_file, qs = line["question_id"], line["image"], line["text"] + " Please just answer yes or no."
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": qs}]}]
        
        prompt_str = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False, 
            enable_thinking=False
        )

        inputs = processor(
            text=prompt_str, 
            images=image, 
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"][0]
        img_pos = (input_ids == model.config.image_token_id).nonzero(as_tuple=True)[0]
        
        # ==================== 【请在 infer_pope_s1mini_pca.py 里这样替换】 ====================
        # 1. 强化工程防御：int强转，防止多维度张量在KV-Cache调用时发生退化
        if len(img_pos) > 0:
            for l_idx in patched:
                model.language_model.layers[l_idx].self_attn._sys_len = int(img_pos[0].item())
                model.language_model.layers[l_idx].self_attn._img_len = int(len(img_pos))

        # 2. 工业级原生强制设备锁：用原生 dict 展开防御，直接拍死隐藏在嵌套结构里的所有 CPU 漏网之鱼
        cleaned_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                cleaned_inputs[k] = v.to("cuda:0")
            elif isinstance(v, list):
                # 递归处理 Qwen/Intern 架构中可能残留的高阶控制 Tensor 列表
                cleaned_inputs[k] = [x.to("cuda:0") if isinstance(x, torch.Tensor) else x for x in v]
            else:
                cleaned_inputs[k] = v

        # 3. 确保半精度对齐
        if "pixel_values" in cleaned_inputs and isinstance(cleaned_inputs["pixel_values"], torch.Tensor): 
            cleaned_inputs["pixel_values"] = cleaned_inputs["pixel_values"].to(dtype=torch.bfloat16)

        # 4. 送入自回归大门
        with torch.inference_mode():
            generate_ids = model.generate(**cleaned_inputs, max_new_tokens=64, use_cache=True)
            
        # ==================================================================================
        outputs = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ans_file.write(json.dumps({"question_id": idx, "prompt": qs, "text": outputs, "answer_id": str(uuid.uuid4())}) + "\n")
        ans_file.flush()
    ans_file.close()