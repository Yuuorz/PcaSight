# 2026-06-03

## Experiments

| 实验 | 设备 | 配置 | 状态 |
|------|------|------|------|
| LLaVA OPOPE baseline | GPU0 | clearsight, base | running |
| LLaVA OPOPE k6_scale0.2 | GPU1 | clearsight, k=6 s=0.2 | running |
| S1-mini OPOPE baseline | GPU2 | s1mini, base | running |

## Changes

- `AGVPAdapter` → `PCAPullAdapter`（文件名 + 类名），涉及 6 个文件
- 内部属性 `_agvp_sys_len` → `_sys_len`，`_agvp_img_len` → `_img_len`
- 移除死代码 `_agvp_mode = "subspace_pca"`（新类本身就是 PCA 算子）

## Fixes

- `models/__init__.py` 同时导入 `patch_s1mini` 导致 clearsight 环境 import 崩溃 → 改为惰性导入
- `run_opope_llava.py` / `run_opope_s1mini.py` 重复调用 `parse_args()` + 缺 `--model-type` → 合并修复
- `run_opope_s1mini.py` device 硬编码 `"cuda:0"`，`CUDA_VISIBLE_DEVICES` 映射不一致时报错 → 改用 `model.device`
- `max_new_tokens` 128 → 512（之前描述被截断）
- prompt 去掉 `list` 引导，避免模型列点输出

## Decisions

- 输出路径 Python 内自动生成：`outputs/{model}/opope/{baseline|k{K}_scale{S}}.jsonl`
- logs/outputs 按 `{model}/{exp_type}` 分层（llava/s1mini × pope/opope）
- Shell 脚本统一放 `scripts/`

## Cleanup

- 删 `.history/`（2MB 历史快照）、`bak_old/`、`__pycache__/`
- 删旧 AGVP/VAF 结果：`outputs/pope_llava/`、`outputs/pope_s1mini/`、`outputs/chair/`
- 清空旧 logs（POPE/OPOPE 日志全部删除，等重跑生成）
