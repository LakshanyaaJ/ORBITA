import os
from pathlib import Path
from collections import defaultdict
import numpy as np

classes = ["LOCATION_A", "LOCATION_B", "PEN", "WATCH", "BLUE_BOX", "YELLOW_BOX", "HAND"]

for split in ["train", "val", "test"]:
    label_dir = Path("datasets/orbita/v4/labels") / split
    img_dir = Path("datasets/orbita/v4/images") / split
    if not label_dir.exists():
        continue
    
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    lbl_files = list(label_dir.glob("*.txt"))
    
    class_counts = defaultdict(int)
    class_areas = defaultdict(list)
    class_aspects = defaultdict(list)
    
    for lf in lbl_files:
        lines = lf.read_text().strip().splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cid = int(parts[0])
                w = float(parts[3])
                h = float(parts[4])
                area = w * h
                class_counts[cid] += 1
                class_areas[cid].append(area)
                class_aspects[cid].append(w / (h + 1e-6))
                
    print(f"=== SPLIT: {split} ===")
    print(f"Images: {len(img_files)}, Labels: {len(lbl_files)}")
    for cid in range(len(classes)):
        cname = classes[cid]
        cnt = class_counts[cid]
        if cnt > 0:
            avg_area = np.mean(class_areas[cid])
            min_area = np.min(class_areas[cid])
            max_area = np.max(class_areas[cid])
            print(f"  [{cid}] {cname:<12}: {cnt:4d} instances | Avg Area: {avg_area:.5f} | Min: {min_area:.5f} | Max: {max_area:.5f}")
        else:
            print(f"  [{cid}] {cname:<12}:    0 instances (MISSING!)")
