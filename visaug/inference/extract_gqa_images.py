"""
Extract GQA images needed for POPE evaluation from parquet files.
"""
import json
import os
from tqdm import tqdm
import pyarrow.parquet as pq
from PIL import Image
import io

GQA_DIR = "/root/download/GQA_data"
POPE_FILE = "/root/code/ClearSight/data/pope/gqa_pope_random.json"
OUT_DIR = "/root/code/ClearSight/data/gqa/images"

# 1. Get list of needed image IDs
needed_ids = set()
for line in open(POPE_FILE):
    needed_ids.add(json.loads(line)["image"])
print(f"Need {len(needed_ids)} unique images")

# 2. Find all image parquet files
image_parquets = []
for root, dirs, files in os.walk(GQA_DIR):
    for f in files:
        if f.endswith(".parquet") and "images" in root:
            image_parquets.append(os.path.join(root, f))
print(f"Found {len(image_parquets)} image parquet files")

# 3. Scan parquet files for needed images
os.makedirs(OUT_DIR, exist_ok=True)
found = 0
remaining = needed_ids.copy()

for pq_file in tqdm(image_parquets, desc="Scanning parquets"):
    if not remaining:
        break
    try:
        table = pq.read_table(pq_file, columns=["id", "image"])
    except Exception as e:
        print(f"  Skip {pq_file}: {e}")
        continue

    for i in range(table.num_rows):
        img_id = table["id"][i].as_py()
        if img_id in remaining:
            img_data = table["image"][i]
            img_bytes = img_data["bytes"].as_py()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            out_path = os.path.join(OUT_DIR, img_id)
            img.save(out_path)
            remaining.remove(img_id)
            found += 1
            if not remaining:
                break

print(f"Extracted {found} images to {OUT_DIR}")
if remaining:
    print(f"WARNING: {len(remaining)} images not found: {list(remaining)[:5]}...")
