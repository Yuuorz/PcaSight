"""
OPOPE (Open-ended POPE) Metrics Evaluator.
Computes token-level classification metrics: Accuracy, Precision, Recall, and F-score.
Based on canonical COCO vocabulary and morphological exclusions.
"""
import argparse
import json
import os
import re
from collections import defaultdict

# 引入权威 COCO 80 类标准词库
COCO_OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

SYNONYMS = {
    "person": ["person", "man", "woman", "boy", "girl", "child", "kid", "people", "lady", "guy", "player", "rider", "pedestrian", "baby", "infant", "toddler", "adult", "human"],
    "bicycle": ["bicycle", "bike", "cycling"],
    "car": ["car", "automobile", "sedan", "suv"],
    "motorcycle": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "bus": ["bus"],
    "train": ["train", "locomotive"],
    "truck": ["truck", "pickup"],
    "boat": ["boat", "ship", "vessel", "canoe", "kayak"],
    "traffic light": ["traffic light", "traffic signal", "stoplight"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "bench": ["bench"],
    "bird": ["bird", "parrot", "pigeon", "crow", "eagle", "hawk", "seagull", "duck", "goose", "owl", "sparrow", "penguin"],
    "cat": ["cat", "kitten", "kitty", "feline"],
    "dog": ["dog", "puppy", "canine", "pup", "hound"],
    "horse": ["horse", "pony", "stallion", "mare", "foal"],
    "sheep": ["sheep", "lamb"],
    "cow": ["cow", "cattle", "bull", "calf", "ox"],
    "elephant": ["elephant"],
    "bear": ["bear", "polar bear", "grizzly"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "knapsack", "rucksack"],
    "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "purse"],
    "tie": ["tie", "necktie"],
    "suitcase": ["suitcase", "luggage", "baggage"],
    "frisbee": ["frisbee"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "ball", "baseball", "basketball", "football", "soccer ball", "tennis ball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat"],
    "baseball glove": ["baseball glove"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass", "goblet"],
    "cup": ["cup", "mug"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake"],
    "chair": ["chair", "seat"],
    "couch": ["couch", "sofa", "loveseat"],
    "potted plant": ["potted plant", "plant", "houseplant"],
    "bed": ["bed"],
    "dining table": ["dining table", "table", "desk"],
    "toilet": ["toilet"],
    "tv": ["tv", "television"],
    "laptop": ["laptop", "notebook computer"],
    "mouse": ["mouse"],
    "remote": ["remote", "remote control"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "phone", "cellphone", "smartphone", "mobile phone"],
    "microwave": ["microwave"],
    "oven": ["oven", "stove"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "stuffed animal", "plush"],
    "hair drier": ["hair drier", "hair dryer", "blow dryer"],
    "toothbrush": ["toothbrush"],
}

def build_synonym_to_coco():
    syn2coco = {}
    for coco_name, syns in SYNONYMS.items():
        for s in syns:
            syn2coco[s.lower()] = coco_name
    return syn2coco

def load_coco_gt(instances_path):
    img_objects = defaultdict(set)
    try:
        with open(instances_path, "r") as f:
            first_char = f.read(1)
        if first_char == "{":
            with open(instances_path, "r") as f:
                first_line = f.readline().strip()
                item = json.loads(first_line)
                if "objects" in item:
                    f.seek(0)
                    for line in f:
                        item = json.loads(line.strip())
                        img_objects[item["image_id"]] = set(item["objects"])
                    return img_objects
    except Exception:
        pass

    with open(instances_path, "r") as f:
        data = json.load(f)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    for ann in data["annotations"]:
        img_objects[ann["image_id"]].add(cat_id_to_name[ann["category_id"]])
    return img_objects

def extract_objects_from_caption(caption, syn2coco):
    caption_lower = caption.lower()
    mentioned = set()
    all_syns = sorted(syn2coco.keys(), key=lambda x: -len(x))
    exclude_patterns = {
        "train": r'\btraining\b|\btrained\b|\btrains\b',
        "orange": r'\borange\s+(color|hue|tint|shade)\b',
    }
    for syn in all_syns:
        pattern = r'\b' + re.escape(syn) + r'\b'
        if re.search(pattern, caption_lower):
            if syn in exclude_patterns:
                if re.search(exclude_patterns[syn], caption_lower):
                    clean = re.sub(exclude_patterns[syn], '', caption_lower)
                    if not re.search(pattern, clean): continue
            mentioned.add(syn2coco[syn])
    return mentioned

def evaluate_opope_matrix(cap_file, instances_path):
    syn2coco = build_synonym_to_coco()
    img_objects = load_coco_gt(instances_path)

    captions = []
    with open(cap_file, "r") as f:
        for line in f:
            captions.append(json.loads(line.strip()))

    # 全局词级四大混淆矩阵格位
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_sentences = len(captions)

    for item in captions:
        image_id = item["image_id"]
        caption = item["caption"]

        gt_objects = img_objects.get(image_id, set())
        mentioned_objects = extract_objects_from_caption(caption, syn2coco)

        # 基于集合论解构 Token-Level 事实博弈
        tp_set = mentioned_objects & gt_objects
        fp_set = mentioned_objects - gt_objects
        fn_set = gt_objects - mentioned_objects
        
        # 在 COCO 80 类拓扑空间中，计算未提及且不存在的负样本底噪 (TN)
        tn_count = 80 - len(gt_objects | mentioned_objects)

        total_tp += len(tp_set)
        total_fp += len(fp_set)
        total_fn += len(fn_set)
        total_tn += tn_count

    # 严格根据多模态事实论文标准公式（Table 2 范式）结算
    precision = total_tp / max((total_tp + total_fp), 1)
    recall = total_tp / max((total_tp + total_fn), 1)
    f_score = 2 * precision * recall / max((precision + recall), 1e-8)
    accuracy = (total_tp + total_tn) / max((total_tp + total_tn + total_fp + total_fn), 1)

    print("\n" + "="*60)
    print("      ClearSight Unified OPOPE (Token-Level) Metric Result      ")
    print("="*60)
    print(f"Evaluated Images Count : {total_sentences}")
    print(f"Confusion Matrix Slots : TP={total_tp} | FP={total_fp} | FN={total_fn} | TN={total_tn}")
    print("-" * 60)
    print(f"OPOPE Accuracy  : {accuracy * 100:.2f}%")
    print(f"OPOPE Precision : {precision * 100:.2f}%  <-- (真话概率，彻底暴露物体捏造度)")
    print(f"OPOPE F-score   : {f_score * 100:.2f}%  <-- (论文 Table 2 最终绝杀大招)")
    print("============================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-file", type=str, required=True, help="OPOPE generated jsonl output assets")
    parser.add_argument("--instances-path", type=str, required=True, help="Path to instances_val2014.json")
    args = parser.parse_args()

    evaluate_opope_matrix(args.cap_file, args.instances_path)