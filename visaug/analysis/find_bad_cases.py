"""
Bad case finder for S1-mini POPE results.
Loads a baseline jsonl and a comparison (AGVP/AGVR) jsonl,
finds cases where baseline is correct but the method flips the answer to wrong.

Usage:
    python visaug/analysis/find_bad_cases.py \
        --baseline outputs/pope_s1mini/res_baseline_random.jsonl \
        --method outputs/pope_s1mini/res_agvr_random.jsonl \
        --annotation data/pope/coco_pope_random.json \
        --output outputs/analysis/bad_cases/bad_cases_index.json
"""
import os
import json
import argparse
import numpy as np


def load_jsonl(path):
    return [json.loads(line) for line in open(path, "r")]


def load_annotation(path):
    return [json.loads(line) for line in open(path, "r")]


def eval_prediction(text, label):
    """Returns True if prediction matches label."""
    text_lower = text.lower().strip()
    if label == "yes":
        return "yes" in text_lower
    else:
        return "no" in text_lower or "not" in text_lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, help="The method to compare (AGVP/AGVR/...)")
    parser.add_argument("--annotation", type=str, required=True)
    parser.add_argument("--output", type=str, default="/root/code/ClearSight/outputs/analysis/bad_cases/bad_cases_index.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--method-name", type=str, default="agvr", help="Display name for the method")
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    method = load_jsonl(args.method)
    annotation = load_annotation(args.annotation)

    print(f"Baseline: {len(baseline)} samples")
    print(f"Method:   {len(method)} samples")
    print(f"Annot:    {len(annotation)} samples")

    # Build annotation lookup
    annot_lookup = {a["question_id"]: {"label": a["label"].lower().strip(), "image": a.get("image", "")} for a in annotation}

    results = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    bad_cases = []

    for b, m in zip(baseline, method):
        assert b["question_id"] == m["question_id"]
        qid = b["question_id"]
        annot_info = annot_lookup.get(qid)
        if annot_info is None:
            continue

        label = annot_info["label"]
        image = annot_info["image"]

        b_correct = eval_prediction(b["text"], label)
        m_correct = eval_prediction(m["text"], label)

        # Track method overall stats (more robust matching)
        m_text = m["text"].lower().strip()
        if label == "yes":
            if "yes" in m_text:
                results["TP"] += 1
            else:
                results["FN"] += 1
        else:
            if "no" in m_text or "not" in m_text:
                results["TN"] += 1
            else:
                results["FP"] += 1

        # Collect bad cases: baseline correct, method wrong
        if b_correct and not m_correct:
            bad_cases.append({
                "question_id": qid,
                "image": image,
                "question": b["prompt"],
                "baseline_answer": b["text"],
                "method_answer": m["text"],
                "label": label,
            })

    # Sort by... we don't have logits, so just use the order they appear
    # But we can try to estimate "confidence" - for now just take top-k
    bad_cases = bad_cases[:args.top_k]

    print(f"\nMethod overall results (from {args.method}):")
    total = sum(results.values())
    acc = (results["TP"] + results["TN"]) / total if total > 0 else 0
    prec = results["TP"] / (results["TP"] + results["FP"]) if (results["TP"] + results["FP"]) > 0 else 0
    rec = results["TP"] / (results["TP"] + results["FN"]) if (results["TP"] + results["FN"]) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    print(f"\nFound {len(bad_cases)} bad cases (baseline correct, {args.method_name} wrong)")
    for i, bc in enumerate(bad_cases[:10]):
        print(f"  {i+1}. qid={bc['question_id']} label={bc['label']} "
              f"baseline='{bc['baseline_answer']}' → method='{bc['method_answer']}'")

    # Save index
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Save as dict for easier reading
    output = {
        "method_name": args.method_name,
        "method_file": args.method,
        "baseline_file": args.baseline,
        "top_k": args.top_k,
        "method_overall": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "TP": results["TP"],
            "TN": results["TN"],
            "FP": results["FP"],
            "FN": results["FN"],
        },
        "bad_cases": bad_cases[:args.top_k],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBad case index saved to {args.output}")
