# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ClearSight is a research codebase for mitigating object hallucination in Multimodal Large Language Models (MLLMs) via visual signal enhancement. It implements two training-free intervention methods — **VAF** (Visual Amplification Fusion) and **AGVP** (Attention-Guided Visual Projection) — that modify attention computation in specific transformer layers to amplify visual signals during modality fusion, reducing hallucinations without sacrificing inference speed.

Two model backends are supported:
- **LLaVA-v1.5-7b** (Llama-based) — original backend
- **Intern-S1-mini** (Qwen3-based) — newer backend with its own inference scripts

## Setup

```bash
conda create -n clearsight python=3.10
conda activate clearsight
cd LLaVA && pip install -e .
```

## Project structure

```
visaug/
  analysis/          # Visual neglect analysis (attention flow, saliency)
    vis_flow.py       # LLaVA: compute attention saliency across layers
    vis_flow_s1mini.py # S1-mini version
    analysis_plot.py   # Plot analysis results
    AttnAdapter.py     # Analysis adapter (captures grad × attn for saliency)
  inference/          # Hallucination mitigation experiments
    AttnAdapter.py     # VAF adapter: pre-softmax attention weight scaling (enh_para / sup_para)
    AGVPAdapter.py     # AGVP adapter: post-softmax geometric projection (suppress sys, enhance vis)
    infer_pope_llava.py  # LLaVA POPE inference (baseline / vaf / agvp)
    infer_pope_s1mini.py # S1-mini POPE inference with VAF (monkey-patching)
    infer_pope_s1mini_agvp.py # S1-mini POPE inference with AGVP
    infer_chair_s1mini.py    # S1-mini CHAIR inference
    eval_pope.py              # POPE evaluation (TP/TN/FP/FN → Acc, Prec, Recall, F1)
    eval_chair.py             # CHAIR evaluation
    *.sh                      # Experiment launchers (infer → eval)
LLaVA/              # LLaVA library (fork installed as editable package)
data/               # POPE/COCO evaluation datasets
outputs/            # Experiment results (jsonl per method/split)
```

## Key concepts

### VAF (Visual Amplification Fusion)
Operates **pre-softmax**: scales text→image attention weights by `enh_para` and text→system attention weights by `sup_para`. Applied only during prefill (not decode) to text tokens. Layer range is configurable.

### AGVP (Attention-Guided Visual Projection)
Operates **post-softmax**: computes visual and system direction vectors from attention-weighted value vectors, then projects text representations away from the system direction and toward the visual direction while preserving norm. Multiple intervention modes exist:
- `imb_cos_vis` / `sys_frac_both` etc.: legacy projection modes with cos-gating
- `hybrid` / `hybrid_soft`: cos sign selects rotation (<0) vs projection (>0)
- `rotate_vis`: rotation toward d_vis when cos<0
- `proj_cospos`: projection only when cos>0 (ablation)

AGVP uses a **cos-gate**: only intervenes when `cos(d_vis, d_sys) < 0` (directions are separable). This auto-skips layers where directions are aligned and intervention would be harmful.

### Token layout convention
Input sequence: `[system_prompt (SYS_LEN=35)] + [image_tokens (IMG_LEN=576)] + [text_tokens]`

## Common workflows

### Run POPE evaluation (LLaVA)
```bash
bash ./visaug/inference/eval_pope_llava.sh
```
This runs baseline, VAF, and AGVP methods and prints the results table.

### Run a single POPE inference (LLaVA)
```bash
python ./visaug/inference/infer_pope_llava.py \
    --model-path /root/code/llava-v1.5-7b \
    --image-folder ./data/coco/val2014 \
    --question-file ./data/pope/coco_pope_random.json \
    --answers-file ./outputs/pope_llava/res_method_random.jsonl \
    --method agvp --start-layer 0 --end-layer 31
```

### Run POPE evaluation (S1-mini)
```bash
bash ./visaug/inference/eval_pope_s1mini_random.sh
```

### Run visual neglect analysis
```bash
bash ./visaug/analysis/vis_flow.sh
```
Outputs per-layer attention metrics (image→image, image→text saliency, attention proportions) saved to `./outputs/analysis/`.

### Visualize AGVP layer metrics (LLaVA)
```bash
python ./visaug/inference/visualize_agvp_llava.py
```
Runs a forward pass, collects per-layer p_vis/p_sys, imbalance, cos(d_vis,d_sys), and projection magnitudes. Saves to `./outputs/visualize/`.

## Adapter injection patterns

**LLaVA (state-dict copy, infer_pope_llava.py:73-89)**: The adapter (VAF or AGVP) is instantiated from the original layer's config, loaded with the original's state_dict, converted to fp16, and replaces `layer.self_attn`. This means the adapter inherits the same Q/K/V/O projection weights.

**S1-mini (monkey-patching, infer_pope_s1mini.py:94-111)**: VAF replaces `layer.self_attn.forward` with a custom function via `types.MethodType`. AGVP uses the same approach in `infer_pope_s1mini_agvp.py`. Eager attention mode must be set before patching:
```python
model.language_model.set_attn_implementation("eager")
```

## Device and GPU assignment

Shell scripts hardcode `CUDA_VISIBLE_DEVICES` (e.g., `export CUDA_VISIBLE_DEVICES=5` for LLaVA, `=4` for S1-mini). Python scripts also accept `--gpu-id` to specify device mapping. Both adapters call `.cuda()` / `.half()` after construction.
