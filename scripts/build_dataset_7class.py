"""
ORBITA 7-Class Horizontal Dataset Generator (V4)
================================================
Generates high-precision ground-truth dataset in HORIZONTAL orientation (1920x1080 / 640x640)
for the 7 canonical entities:
  0: LOCATION_A  (left white sheet with 'A' marker)
  1: LOCATION_B  (right white sheet with 'B' marker)
  2: PEN         (physical light blue marker/pen)
  3: WATCH       (physical metallic wristwatch)
  4: BLUE_BOX    (3D physical blue container)
  5: YELLOW_BOX  (3D physical yellow container with tabs)
  6: HAND        (human operator hand)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "LOCATION_A",
    "LOCATION_B",
    "PEN",
    "WATCH",
    "BLUE_BOX",
    "YELLOW_BOX",
    "HAND",
]
CLASS_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

TARGET_W = 640
TARGET_H = 640


def clip_box(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Clips normalized bounding box to [0, 1]."""
    x1 = max(0.001, xc - w / 2.0)
    y1 = max(0.001, yc - h / 2.0)
    x2 = min(0.999, xc + w / 2.0)
    y2 = min(0.999, yc + h / 2.0)
    nw = max(0.005, x2 - x1)
    nh = max(0.005, y2 - y1)
    nxc = (x1 + x2) / 2.0
    nyc = (y1 + y2) / 2.0
    return round(nxc, 5), round(nyc, 5), round(nw, 5), round(nh, 5)


def get_horizontal_boxes_for_frame(fno: int) -> List[Tuple[int, List[float]]]:
    """
    Returns normalized [xc, yc, w, h] boxes in HORIZONTAL frame coordinates
    across the entire 1830-frame experiment timeline.
    """
    boxes: List[Tuple[int, List[float]]] = []

    # 1. LOCATION_A and LOCATION_B are stationary on the table throughout
    boxes.append((CLASS_IDX["LOCATION_A"], [0.268, 0.386, 0.214, 0.556]))
    boxes.append((CLASS_IDX["LOCATION_B"], [0.730, 0.338, 0.159, 0.477]))

    # 2. BLUE_BOX trajectory
    # Rest position: bottom right [0.682, 0.729, 0.140, 0.201]
    # At Location A: [0.310, 0.320, 0.150, 0.210]
    # At Location B: [0.715, 0.316, 0.130, 0.200]
    if fno < 320:
        bx, by, bw, bh = 0.682, 0.729, 0.140, 0.201
    elif 320 <= fno < 480:
        t = (fno - 320) / 160.0
        bx = 0.682 + (0.310 - 0.682) * t
        by = 0.729 + (0.320 - 0.729) * t
        bw, bh = 0.150, 0.210
    elif 480 <= fno < 1550:
        bx, by, bw, bh = 0.310, 0.320, 0.150, 0.210
    elif 1550 <= fno < 1680:
        t = (fno - 1550) / 130.0
        bx = 0.310 + (0.715 - 0.310) * t
        by = 0.320 + (0.316 - 0.320) * t
        bw, bh = 0.140, 0.200
    else:
        bx, by, bw, bh = 0.715, 0.316, 0.130, 0.200
    boxes.append((CLASS_IDX["BLUE_BOX"], [bx, by, bw, bh]))

    # 3. YELLOW_BOX trajectory
    # Rest position: bottom left [0.406, 0.783, 0.197, 0.268]
    # At Location B: [0.755, 0.300, 0.170, 0.280]
    # At Location A: [0.320, 0.355, 0.170, 0.280]
    if fno < 600:
        yx, yy, yw, yh = 0.406, 0.783, 0.197, 0.268
    elif 600 <= fno < 750:
        t = (fno - 600) / 150.0
        yx = 0.406 + (0.755 - 0.406) * t
        yy = 0.783 + (0.300 - 0.783) * t
        yw, yh = 0.180, 0.270
    elif 750 <= fno < 1680:
        yx, yy, yw, yh = 0.755, 0.300, 0.170, 0.280
    elif 1680 <= fno < 1800:
        t = (fno - 1680) / 120.0
        yx = 0.755 + (0.320 - 0.755) * t
        yy = 0.300 + (0.355 - 0.300) * t
        yw, yh = 0.170, 0.280
    else:
        yx, yy, yw, yh = 0.320, 0.355, 0.170, 0.280
    boxes.append((CLASS_IDX["YELLOW_BOX"], [yx, yy, yw, yh]))

    # 4. PEN trajectory
    # Rest position: [0.574, 0.343, 0.020, 0.227]
    # Picked up & moved to Blue Box: 880 <= fno < 1050
    # In Blue Box: fno >= 1050 (follows Blue Box)
    if fno < 880:
        px, py, pw, ph = 0.574, 0.343, 0.020, 0.227
    elif 880 <= fno < 1050:
        t = (fno - 880) / 170.0
        px = 0.574 + (bx - 0.574) * t
        py = 0.343 + (by - 0.343) * t
        pw, ph = 0.040, 0.200
    else:
        # Inside Blue Box (slight offset/diagonal)
        px, py, pw, ph = bx - 0.005, by - 0.010, 0.055, 0.160
    boxes.append((CLASS_IDX["PEN"], [px, py, pw, ph]))

    # 5. WATCH trajectory
    # Rest position: [0.507, 0.338, 0.034, 0.204]
    # Picked up & moved to Yellow Box: 1120 <= fno < 1300
    # In Yellow Box: fno >= 1300 (follows Yellow Box)
    if fno < 1120:
        wx, wy, ww, wh = 0.507, 0.338, 0.034, 0.204
    elif 1120 <= fno < 1300:
        t = (fno - 1120) / 180.0
        wx = 0.507 + (yx - 0.507) * t
        wy = 0.338 + (yy - 0.338) * t
        ww, wh = 0.060, 0.170
    else:
        # Inside Yellow Box
        wx, wy, ww, wh = yx + 0.005, yy - 0.005, 0.065, 0.160
    boxes.append((CLASS_IDX["WATCH"], [wx, wy, ww, wh]))

    # 6. HAND detections during active physical interactions
    if 300 <= fno < 520:
        boxes.append((CLASS_IDX["HAND"], [bx + 0.06, by + 0.08, 0.160, 0.220]))
    elif 600 <= fno < 780:
        boxes.append((CLASS_IDX["HAND"], [yx + 0.05, yy + 0.08, 0.170, 0.220]))
    elif 880 <= fno < 1080:
        boxes.append((CLASS_IDX["HAND"], [px + 0.04, py + 0.08, 0.150, 0.200]))
    elif 1120 <= fno < 1320:
        boxes.append((CLASS_IDX["HAND"], [wx + 0.04, wy + 0.08, 0.160, 0.200]))
    elif 1520 <= fno < 1690:
        boxes.append((CLASS_IDX["HAND"], [bx + 0.05, by + 0.08, 0.160, 0.220]))
    elif 1680 <= fno < 1820:
        boxes.append((CLASS_IDX["HAND"], [yx + 0.05, yy + 0.08, 0.160, 0.220]))

    return boxes


def build_dataset():
    v4_base = Path("datasets/orbita/v4")
    images_dir = v4_base / "images"
    labels_dir = v4_base / "labels"

    for split in ["train", "val", "test"]:
        if (images_dir / split).exists():
            shutil.rmtree(images_dir / split)
        if (labels_dir / split).exists():
            shutil.rmtree(labels_dir / split)
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture("vdata/20260908_135006.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Extracting ground-truth horizontal frames from base video (%d frames)...", total_frames)

    # Sample every 10 frames across all experiment phases (~180 frames)
    sampled_frame_indices = list(range(10, total_frames - 10, 10))

    extracted_data = []

    for fno in sampled_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # CRITICAL: Rotate frame 90 degrees clockwise to match horizontal stream
        frame_horiz = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        boxes = get_horizontal_boxes_for_frame(fno)
        extracted_data.append((fno, frame_horiz, boxes))

    cap.release()
    logger.info("Extracted %d horizontal ground-truth frames from base video.", len(extracted_data))

    all_frames: List[Tuple[str, np.ndarray, List[Tuple[int, List[float]]]]] = []

    # Add real horizontal frames
    for fno, raw_img, bxs in extracted_data:
        resized = cv2.resize(raw_img, (TARGET_W, TARGET_H))
        all_frames.append((f"vid_real_f{fno:05d}", resized, bxs))

    # Add photometric & geometric augmentations
    random.seed(42)
    np.random.seed(42)

    for i in range(160):
        fno, base_img, base_bxs = random.choice(extracted_data)

        # Photometric shifts
        alpha = random.uniform(0.85, 1.20)
        beta = random.uniform(-20, 20)
        aug_img = cv2.convertScaleAbs(base_img, alpha=alpha, beta=beta)

        # Hue/saturation jitter
        if random.random() > 0.3:
            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-5, 5)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.88, 1.12), 0, 255)
            aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        resized = cv2.resize(aug_img, (TARGET_W, TARGET_H))

        # Small coordinate jitter (+- 0.5%)
        jittered_bxs = []
        for cid, (xc, yc, w, h) in base_bxs:
            j_xc = np.clip(xc + random.uniform(-0.005, 0.005), 0.02, 0.98)
            j_yc = np.clip(yc + random.uniform(-0.005, 0.005), 0.02, 0.98)
            j_w = np.clip(w * random.uniform(0.97, 1.03), 0.01, 0.95)
            j_h = np.clip(h * random.uniform(0.97, 1.03), 0.01, 0.95)
            c_box = clip_box(j_xc, j_yc, j_w, j_h)
            jittered_bxs.append((cid, list(c_box)))

        all_frames.append((f"vid_aug_s{i:04d}", resized, jittered_bxs))

    # Shuffle and split: 70% train, 20% val, 10% test
    random.shuffle(all_frames)
    n_total = len(all_frames)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.20)

    splits = {
        "train": all_frames[:n_train],
        "val": all_frames[n_train:n_train + n_val],
        "test": all_frames[n_train + n_val:],
    }

    counts = {s: 0 for s in splits}
    cls_counts = {c: 0 for c in CLASS_NAMES}

    for split_name, items in splits.items():
        for stem, img, bxs in items:
            img_path = images_dir / split_name / f"{stem}.jpg"
            lbl_path = labels_dir / split_name / f"{stem}.txt"

            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(lbl_path, "w", encoding="utf-8") as f:
                for cid, (xc, yc, w, h) in bxs:
                    c_box = clip_box(xc, yc, w, h)
                    f.write(f"{cid} {c_box[0]:.5f} {c_box[1]:.5f} {c_box[2]:.5f} {c_box[3]:.5f}\n")
                    cls_counts[CLASS_NAMES[cid]] += 1

            counts[split_name] += 1

    logger.info("Horizontal Dataset V4 built successfully:")
    logger.info("  Splits: Train=%d, Val=%d, Test=%d | Total=%d", counts["train"], counts["val"], counts["test"], n_total)
    logger.info("  Class instance counts: %s", cls_counts)

    # Write data.yaml
    data_yaml_content = f"""# ORBITA 7-Class Horizontal Dataset (V4)
path: {v4_base.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: LOCATION_A
  1: LOCATION_B
  2: PEN
  3: WATCH
  4: BLUE_BOX
  5: YELLOW_BOX
  6: HAND

nc: 7
"""
    with open(v4_base / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    with open("datasets/orbita/data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    logger.info("Saved data.yaml to %s and datasets/orbita/data.yaml", v4_base / "data.yaml")


if __name__ == "__main__":
    build_dataset()
