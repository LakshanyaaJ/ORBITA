"""
ORBITA YOLO Multi-Threshold Diagnostic Scan Script
=================================================
Runs diagnostic threshold scan at 0.05, 0.10, 0.20, 0.30, 0.40, 0.50
to determine whether detector failures are Case A (no candidate boxes),
Case B (low confidence), Case C (coord transformation), or Case D (class mismatch).
"""

import os
import sys
import json
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from core_ai.app.config import load_config


def run_threshold_scan(video_path: str = "vdata/20260905_145858.mp4", max_frames: int = 150):
    cfg = load_config()
    model_path = cfg.detection.yolo_model_path
    yolo = YOLO(model_path)
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    print(f"=== YOLO MULTI-THRESHOLD DIAGNOSTIC SCAN ===")
    print(f"Model Path: {model_path}")
    print(f"Video Path: {video_path}")
    print(f"Thresholds Tested: {thresholds}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file {video_path}")
        return

    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    scan_results = {}

    for thresh in thresholds:
        print(f"\nScanning threshold conf={thresh:.2f} on {len(frames)} frames...")
        per_class_counts = {c: 0 for c in yolo.names.values()}
        total_boxes = 0

        for frame in frames:
            res = yolo.predict(frame, imgsz=cfg.detection.yolo_imgsz, conf=thresh, verbose=False)
            for r in res:
                if r.boxes is not None:
                    total_boxes += len(r.boxes)
                    for b in r.boxes:
                        cls_idx = int(b.cls[0])
                        cls_name = yolo.names[cls_idx]
                        per_class_counts[cls_name] = per_class_counts.get(cls_name, 0) + 1

        scan_results[str(thresh)] = {
            "total_boxes_detected": total_boxes,
            "avg_boxes_per_frame": round(total_boxes / len(frames), 2),
            "per_class_counts": per_class_counts,
        }
        print(f"  Conf {thresh:.2f} -> Total Boxes: {total_boxes:4d} | Per-Class: {per_class_counts}")

    os.makedirs("reports", exist_ok=True)
    report_file = "reports/yolo_threshold_scan_results.json"
    with open(report_file, "w") as f:
        json.dump(scan_results, f, indent=2)

    print(f"\nThreshold scan complete! JSON written to: {report_file}")


if __name__ == "__main__":
    run_threshold_scan()
