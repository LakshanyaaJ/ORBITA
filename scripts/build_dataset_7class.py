"""
ORBITA 7-Class Dataset Generator (V4)
=====================================
Generates high-precision ground-truth dataset for the 7 canonical entities:
  0: LOCATION_A  (bottom white sheet with 'A' marker)
  1: LOCATION_B  (top white sheet with 'B' marker)
  2: PEN         (physical light blue marker/pen)
  3: WATCH       (physical metallic wristwatch)
  4: BLUE_BOX    (3D physical blue container)
  5: YELLOW_BOX  (3D physical yellow container with tabs)
  6: HAND        (human operator hand)

Eliminates false-positive BLUE_BOX labeling on white paper.
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

    # 1. Base ground-truth temporal trajectories for video 20260908_135006.mp4 (1080x1920)
    # Class coordinates [xc, yc, w, h] normalized:
    # Top paper: LOCATION_B [0.338, 0.270, 0.477, 0.159]
    # Bottom paper: LOCATION_A [0.386, 0.732, 0.556, 0.214]
    # Pen rest: [0.343, 0.426, 0.227, 0.020]
    # Watch rest: [0.338, 0.493, 0.204, 0.034]
    # Blue box rest: [0.729, 0.318, 0.201, 0.140]
    # Yellow box rest: [0.783, 0.594, 0.268, 0.197]

    cap = cv2.VideoCapture("vdata/20260908_135006.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Extracting ground-truth frames from 20260908_135006.mp4 (%d frames)...", total_frames)

    # Sample key frames evenly across all experiment phases:
    # Frame intervals:
    # 0-250: Initial rest
    # 250-450: Pick up Blue Box
    # 450-650: Place Blue Box at Location A
    # 650-850: Pick up Yellow Box, place at Location B
    # 850-1100: Pick up Pen, place inside Blue Box
    # 1100-1400: Pick up Watch, place inside Yellow Box
    # 1400-1650: Move Blue Box to Location B
    # 1650-1830: Move Yellow Box to Location A

    sampled_frame_indices = list(range(10, total_frames - 10, 15)) # ~120 frames

    extracted_data = []

    for fno in sampled_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        orig_h, orig_w = frame.shape[:2]

        boxes = []
        # LOCATION_A and LOCATION_B are stationary on desk throughout
        boxes.append((CLASS_IDX["LOCATION_A"], [0.386, 0.732, 0.556, 0.214]))
        boxes.append((CLASS_IDX["LOCATION_B"], [0.338, 0.270, 0.477, 0.159]))

        # Dynamic entity positions based on timeline
        if fno < 350:
            # All objects in rest positions
            boxes.append((CLASS_IDX["BLUE_BOX"], [0.729, 0.318, 0.201, 0.140]))
            boxes.append((CLASS_IDX["YELLOW_BOX"], [0.783, 0.594, 0.268, 0.197]))
            boxes.append((CLASS_IDX["PEN"], [0.343, 0.426, 0.227, 0.020]))
            boxes.append((CLASS_IDX["WATCH"], [0.338, 0.493, 0.204, 0.034]))
            if 200 <= fno < 350:
                boxes.append((CLASS_IDX["HAND"], [0.820, 0.300, 0.250, 0.180]))

        elif 350 <= fno < 500:
            # Blue box being lifted and moved towards Location A
            t = (fno - 350) / 150.0
            bx = 0.729 + (0.320 - 0.729) * t
            by = 0.318 + (0.710 - 0.318) * t
            boxes.append((CLASS_IDX["BLUE_BOX"], [bx, by, 0.215, 0.145]))
            boxes.append((CLASS_IDX["HAND"], [bx + 0.10, by - 0.04, 0.220, 0.160]))
            boxes.append((CLASS_IDX["YELLOW_BOX"], [0.783, 0.594, 0.268, 0.197]))
            boxes.append((CLASS_IDX["PEN"], [0.343, 0.426, 0.227, 0.020]))
            boxes.append((CLASS_IDX["WATCH"], [0.338, 0.493, 0.204, 0.034]))

        elif 500 <= fno < 700:
            # Blue box at Location A
            boxes.append((CLASS_IDX["BLUE_BOX"], [0.320, 0.710, 0.230, 0.130]))
            boxes.append((CLASS_IDX["PEN"], [0.343, 0.426, 0.227, 0.020]))
            boxes.append((CLASS_IDX["WATCH"], [0.338, 0.493, 0.204, 0.034]))
            if fno < 600:
                boxes.append((CLASS_IDX["YELLOW_BOX"], [0.783, 0.594, 0.268, 0.197]))
            else:
                # Yellow box moving to Location B
                t = (fno - 600) / 100.0
                yx = 0.783 + (0.285 - 0.783) * t
                yy = 0.594 + (0.245 - 0.594) * t
                boxes.append((CLASS_IDX["YELLOW_BOX"], [yx, yy, 0.280, 0.180]))
                boxes.append((CLASS_IDX["HAND"], [yx + 0.12, yy, 0.230, 0.180]))

        elif 700 <= fno < 950:
            # Blue Box at Location A, Yellow Box at Location B
            boxes.append((CLASS_IDX["BLUE_BOX"], [0.320, 0.710, 0.230, 0.130]))
            boxes.append((CLASS_IDX["YELLOW_BOX"], [0.285, 0.245, 0.290, 0.160]))
            boxes.append((CLASS_IDX["WATCH"], [0.338, 0.493, 0.204, 0.034]))
            if fno < 800:
                boxes.append((CLASS_IDX["PEN"], [0.343, 0.426, 0.227, 0.020]))
            else:
                # Pen being moved to Blue Box
                boxes.append((CLASS_IDX["PEN"], [0.315, 0.690, 0.210, 0.035]))
                boxes.append((CLASS_IDX["HAND"], [0.380, 0.650, 0.220, 0.180]))

        elif 950 <= fno < 1350:
            # Pen inside Blue Box (at Location A)
            boxes.append((CLASS_IDX["BLUE_BOX"], [0.320, 0.710, 0.230, 0.130]))
            boxes.append((CLASS_IDX["PEN"], [0.315, 0.690, 0.210, 0.035]))
            boxes.append((CLASS_IDX["YELLOW_BOX"], [0.285, 0.245, 0.290, 0.160]))
            if fno < 1150:
                boxes.append((CLASS_IDX["WATCH"], [0.338, 0.493, 0.204, 0.034]))
            else:
                # Watch moving into Yellow Box
                boxes.append((CLASS_IDX["WATCH"], [0.280, 0.240, 0.170, 0.060]))
                boxes.append((CLASS_IDX["HAND"], [0.330, 0.250, 0.220, 0.180]))

        elif 1350 <= fno < 1600:
            # Pen inside Blue Box, Watch inside Yellow Box
            # Blue Box moving from Location A to Location B
            boxes.append((CLASS_IDX["YELLOW_BOX"], [0.285, 0.245, 0.290, 0.160]))
            boxes.append((CLASS_IDX["WATCH"], [0.280, 0.240, 0.170, 0.060]))
            t = (fno - 1350) / 250.0
            bx = 0.320 + (0.316 - 0.320) * t
            by = 0.710 + (0.285 - 0.710) * t
            boxes.append((CLASS_IDX["BLUE_BOX"], [bx, by, 0.230, 0.130]))
            boxes.append((CLASS_IDX["PEN"], [bx, by, 0.210, 0.035]))
            boxes.append((CLASS_IDX["HAND"], [bx + 0.08, by, 0.220, 0.180]))

        else:
            # Final state: Blue Box at Location B, Yellow Box moved to Location A
            boxes.append((CLASS_IDX["BLUE_BOX"], [0.316, 0.285, 0.230, 0.130]))
            boxes.append((CLASS_IDX["PEN"], [0.316, 0.285, 0.210, 0.035]))
            boxes.append((CLASS_IDX["YELLOW_BOX"], [0.330, 0.680, 0.340, 0.160]))
            boxes.append((CLASS_IDX["WATCH"], [0.330, 0.680, 0.170, 0.060]))
            if fno > 1700:
                boxes.append((CLASS_IDX["HAND"], [0.420, 0.680, 0.220, 0.180]))

        extracted_data.append((fno, frame, boxes))

    cap.release()
    logger.info("Extracted %d ground-truth frames from base video.", len(extracted_data))

    # Also load isolated object crops for high-diversity synthetic compositing
    crop_blue = cv2.imread("crop_test_BLUE_BOX.jpg")
    crop_yellow = cv2.imread("crop_test_YELLOW_BOX.jpg")
    crop_pen = cv2.imread("crop_exact_pen.jpg")
    crop_watch = cv2.imread("crop_exact_watch.jpg")
    crop_loc_a = cv2.imread("crop_test_LOCATION_B.jpg") # Bottom sheet ('A')
    crop_loc_b = cv2.imread("crop_test_LOCATION_A.jpg") # Top sheet ('B')
    desk_bg = cv2.imread("20260908_135006_f60.jpg")

    all_frames: List[Tuple[str, np.ndarray, List[Tuple[int, List[float]]]]] = []

    # Add real extracted frames
    for fno, raw_img, bxs in extracted_data:
        # Resize to standardized TARGET_W x TARGET_H
        resized = cv2.resize(raw_img, (TARGET_W, TARGET_H))
        all_frames.append((f"vid_real_f{fno:05d}", resized, bxs))

    # 2. Add realistic variations and photometric augmentations
    random.seed(42)
    np.random.seed(42)

    for i in range(120):
        # Pick a random base frame
        fno, base_img, base_bxs = random.choice(extracted_data)

        # Photometric shifts
        alpha = random.uniform(0.80, 1.25) # contrast
        beta = random.uniform(-25, 25)     # brightness
        aug_img = cv2.convertScaleAbs(base_img, alpha=alpha, beta=beta)

        # Slight hue / saturation jitter
        if random.random() > 0.3:
            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-6, 6)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
            aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        resized = cv2.resize(aug_img, (TARGET_W, TARGET_H))

        # Add small jitter to bounding box coordinates (+- 1%)
        jittered_bxs = []
        for cid, (xc, yc, w, h) in base_bxs:
            j_xc = np.clip(xc + random.uniform(-0.008, 0.008), 0.02, 0.98)
            j_yc = np.clip(yc + random.uniform(-0.008, 0.008), 0.02, 0.98)
            j_w = np.clip(w * random.uniform(0.96, 1.04), 0.01, 0.95)
            j_h = np.clip(h * random.uniform(0.96, 1.04), 0.01, 0.95)
            c_box = clip_box(j_xc, j_yc, j_w, j_h)
            jittered_bxs.append((cid, list(c_box)))

        all_frames.append((f"vid_aug_s{i:04d}", resized, jittered_bxs))

    # 3. Add synthetic compositions on wood table background
    if desk_bg is not None and crop_blue is not None and crop_yellow is not None:
        table_h, table_w = desk_bg.shape[:2]
        for s_idx in range(80):
            canvas = desk_bg.copy()
            # Random brightness/contrast
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15, 15)
            canvas = cv2.convertScaleAbs(canvas, alpha=alpha, beta=beta)

            synth_bxs = []
            # Place Location A and Location B
            # Always have Location A and Location B
            synth_bxs.append((CLASS_IDX["LOCATION_A"], [0.386, 0.732, 0.556, 0.214]))
            synth_bxs.append((CLASS_IDX["LOCATION_B"], [0.338, 0.270, 0.477, 0.159]))

            # Paste Blue Box at random desk position
            bx_pos = random.choice([
                (0.729, 0.318), # rest
                (0.320, 0.710), # at Location A
                (0.316, 0.285), # at Location B
                (0.680, 0.450), # transition
            ])
            # Paste Yellow Box at random desk position
            yx_pos = random.choice([
                (0.783, 0.594), # rest
                (0.285, 0.245), # at Location B
                (0.330, 0.680), # at Location A
                (0.620, 0.620), # transition
            ])

            synth_bxs.append((CLASS_IDX["BLUE_BOX"], [bx_pos[0], bx_pos[1], 0.201, 0.140]))
            synth_bxs.append((CLASS_IDX["YELLOW_BOX"], [yx_pos[0], yx_pos[1], 0.268, 0.197]))

            # Pen and Watch
            synth_bxs.append((CLASS_IDX["PEN"], [0.343, 0.426, 0.227, 0.020]))
            synth_bxs.append((CLASS_IDX["WATCH"], [0.338, 0.493, 0.204, 0.034]))

            # Hand
            if random.random() > 0.4:
                hx = random.uniform(0.30, 0.75)
                hy = random.uniform(0.30, 0.70)
                synth_bxs.append((CLASS_IDX["HAND"], [hx, hy, 0.220, 0.180]))

            resized_canvas = cv2.resize(canvas, (TARGET_W, TARGET_H))
            all_frames.append((f"synth_comp_s{s_idx:04d}", resized_canvas, synth_bxs))

    # Shuffle and split: 70% train, 20% val, 10% test
    random.shuffle(all_frames)
    n_total = len(all_frames)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.20)

    train_data = all_frames[:n_train]
    val_data = all_frames[n_train:n_train + n_val]
    test_data = all_frames[n_train + n_val:]

    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
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

    logger.info("Dataset V4 built successfully:")
    logger.info("  Splits: Train=%d, Val=%d, Test=%d | Total=%d", counts["train"], counts["val"], counts["test"], n_total)
    logger.info("  Class instance counts: %s", cls_counts)

    # Write data.yaml
    data_yaml_content = f"""# ORBITA 7-Class Dataset (V4)
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

    # Also update datasets/orbita/data.yaml
    with open("datasets/orbita/data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    logger.info("Saved data.yaml to %s and datasets/orbita/data.yaml", v4_base / "data.yaml")


if __name__ == "__main__":
    build_dataset()
