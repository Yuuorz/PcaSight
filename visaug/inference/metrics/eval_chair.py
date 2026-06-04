"""
CHAIR (Caption Hallucination Assessment with Image Relevance) evaluation.

Metrics:
  CHAIRs: fraction of sentences with at least one hallucinated object
  CHAIRi: fraction of mentioned objects that are hallucinated
  Recall: fraction of ground-truth objects that are mentioned

Requires COCO annotations:
  - instances_val2014.json (object annotations)
  - captions_val2014.json (ground-truth captions, optional for synonym building)

Usage:
  python eval_chair.py \
      --cap_file outputs/chair/captions_baseline.jsonl \
      --coco_ann_dir data/coco/annotations
"""

import argparse
import json
import os
import re
from collections import defaultdict

# COCO 80 object categories
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

# Common synonyms mapping to canonical COCO names
SYNONYMS = {
    "person": ["person", "man", "woman", "boy", "girl", "child", "kid", "people",
               "lady", "guy", "player", "rider", "pedestrian", "baby", "infant",
               "toddler", "adult", "human"],
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
    "bird": ["bird", "parrot", "pigeon", "crow", "eagle", "hawk", "seagull",
             "duck", "goose", "owl", "sparrow", "penguin"],
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
    "sports ball": ["sports ball", "ball", "baseball", "basketball", "football",
                    "soccer ball", "tennis ball"],
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
    """Build reverse mapping: synonym -> canonical COCO name."""
    syn2coco = {}
    for coco_name, syns in SYNONYMS.items():
        for s in syns:
            syn2coco[s.lower()] = coco_name
    return syn2coco


def load_coco_gt(instances_path):
    """Load COCO ground-truth objects.
    
    Supports two formats:
    1. COCO instances JSON: {"categories": [...], "annotations": [...]}
    2. POPE segmentation JSONL: {"image_id": ..., "objects": [...]} per line
    """
    img_objects = defaultdict(set)

    # Try JSONL format first (POPE segmentation file)
    try:
        with open(instances_path, "r") as f:
            first_char = f.read(1)
        if first_char == "{":
            # Could be JSONL or standard JSON
            with open(instances_path, "r") as f:
                first_line = f.readline().strip()
                item = json.loads(first_line)
                if "objects" in item:
                    # POPE segmentation JSONL format
                    f.seek(0)
                    for line in f:
                        item = json.loads(line.strip())
                        img_id = item["image_id"]
                        for obj in item["objects"]:
                            img_objects[img_id].add(obj)
                    print(f"Loaded GT from POPE segmentation format: {len(img_objects)} images")
                    return img_objects
    except Exception:
        pass

    # Standard COCO instances JSON format
    with open(instances_path, "r") as f:
        data = json.load(f)

    cat_id_to_name = {}
    for cat in data["categories"]:
        cat_id_to_name[cat["id"]] = cat["name"]

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name[ann["category_id"]]
        img_objects[img_id].add(cat_name)

    print(f"Loaded GT from COCO instances format: {len(img_objects)} images")
    return img_objects


def extract_objects_from_caption(caption, syn2coco):
    """Extract COCO objects mentioned in caption using synonym matching."""
    caption_lower = caption.lower()
    mentioned = set()

    # Sort by length (longest first) to match multi-word phrases first
    all_syns = sorted(syn2coco.keys(), key=lambda x: -len(x))

    # Words that need stricter matching to avoid false positives
    exclude_patterns = {
        "train": r'\btraining\b|\btrained\b|\btrains\b',
        "orange": r'\borange\s+(color|hue|tint|shade)\b',
    }

    for syn in all_syns:
        # Word boundary matching
        pattern = r'\b' + re.escape(syn) + r'\b'
        if re.search(pattern, caption_lower):
            # Check for false positive exclusions
            if syn in exclude_patterns:
                if re.search(exclude_patterns[syn], caption_lower):
                    # Only exclude if the match is ONLY the excluded form
                    clean = re.sub(exclude_patterns[syn], '', caption_lower)
                    if not re.search(pattern, clean):
                        continue
            mentioned.add(syn2coco[syn])

    return mentioned


def evaluate_chair(cap_file, instances_path):
    """Compute CHAIR metrics."""
    syn2coco = build_synonym_to_coco()
    img_objects = load_coco_gt(instances_path)

    # Load captions
    captions = []
    with open(cap_file, "r") as f:
        for line in f:
            captions.append(json.loads(line.strip()))

    total_sentences = 0
    hallucinated_sentences = 0
    total_objects_mentioned = 0
    total_objects_hallucinated = 0
    total_gt_objects = 0
    total_gt_recalled = 0

    results = []

    for item in captions:
        image_id = item["image_id"]
        caption = item["caption"]

        gt_objects = img_objects.get(image_id, set())
        mentioned_objects = extract_objects_from_caption(caption, syn2coco)

        hallucinated = mentioned_objects - gt_objects
        recalled = mentioned_objects & gt_objects

        total_sentences += 1
        if len(hallucinated) > 0:
            hallucinated_sentences += 1

        total_objects_mentioned += len(mentioned_objects)
        total_objects_hallucinated += len(hallucinated)
        total_gt_objects += len(gt_objects)
        total_gt_recalled += len(recalled)

        results.append({
            "image_id": image_id,
            "caption": caption,
            "gt_objects": list(gt_objects),
            "mentioned_objects": list(mentioned_objects),
            "hallucinated_objects": list(hallucinated),
            "recalled_objects": list(recalled),
        })

    chairs = hallucinated_sentences / max(total_sentences, 1)
    chairi = total_objects_hallucinated / max(total_objects_mentioned, 1)
    recall = total_gt_recalled / max(total_gt_objects, 1)

    print(f"Total captions: {total_sentences}")
    print(f"CHAIRs: {chairs:.4f}  ({hallucinated_sentences}/{total_sentences} sentences with hallucination)")
    print(f"CHAIRi: {chairi:.4f}  ({total_objects_hallucinated}/{total_objects_mentioned} objects hallucinated)")
    print(f"Recall: {recall:.4f}  ({total_gt_recalled}/{total_gt_objects} GT objects recalled)")

    return {
        "CHAIRs": chairs,
        "CHAIRi": chairi,
        "Recall": recall,
        "total_captions": total_sentences,
        "details": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-file", type=str, required=True,
                        help="JSONL file with {image_id, caption} per line")
    parser.add_argument("--instances-path", type=str, required=True,
                        help="Path to instances_val2014.json")
    parser.add_argument("--save-path", type=str, default="",
                        help="Save detailed results to JSON")
    args = parser.parse_args()

    results = evaluate_chair(args.cap_file, args.instances_path)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        with open(args.save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {args.save_path}")
