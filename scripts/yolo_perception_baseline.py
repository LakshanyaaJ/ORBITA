"""
ORBITA YOLO Perception Baseline Script
======================================
Freezes the baseline performance of orbita_yolo_detector_v4.pt on target MP4 video vdata/20260905_145858.mp4.
Collects 300 representative frames, records detailed per-class detection statistics, saving:
  - reports/yolo_baseline_real_video.json
  - reports/yolo_samples/*.jpg
"""

import os
import sys
import json
import time
import cv2
import numpy as np

# Ensure project root in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from core_ai.perception.object_detector import ObjectDetector, audit_model_classes
from core_ai.app.config import load_config


def run_baseline(video_path: str = "vdata/20260905_145858.mp4", max_frames: int = 300):
    cfg = load_config()
    model_path = cfg.detection.yolo_model_path
    print(f"=== YOLO PERCEPTION BASELINE RUN ===")
    print(f"Model Path: {model_path}")
    print(f"Video Path: {video_path}")
    print(f"Max Frames: {max_frames}")

    # Ensure reports directory
    os.makedirs("reports/yolo_samples", exist_ok=True)

    # Inspect Ultralytics Model
    yolo = YOLO(model_path)
    model_names = getattr(yolo, "names", {})
    print("\nModel Checkpoint Metadata:")
    print(f"  Task: {getattr(yolo, 'task', 'N/A')}")
    print(f"  Class Count: {len(model_names)}")
    print(f"  Class Names: {model_names}")

    missing = audit_model_classes(model_names)
    print(f"  Missing Application Classes: {missing}")

    detector = ObjectDetector(cfg.detection)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file {video_path}")
        return

    frame_count = 0
    per_class_stats = {
        cls_name: {
            "frames_with_detection": 0,
            "total_detections": 0,
            "confidences": [],
            "widths": [],
            "heights": [],
            "areas": [],
            "norm_areas": [],
            "centroids": [],
        }
        for cls_name in cfg.detection.classes
    }

    raw_yolo_boxes_count = 0
    nms_boxes_count = 0
    saved_samples_count = 0

    print(f"\nProcessing {max_frames} frames from video...")
    t0 = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]
        t_stamp = time.time()

        # Execute raw Ultralytics prediction at diagnostic low conf threshold (0.05) to capture raw candidate boxes
        raw_results = yolo.predict(frame, imgsz=cfg.detection.yolo_imgsz, conf=0.05, verbose=False)
        for r in raw_results:
            if r.boxes is not None:
                raw_yolo_boxes_count += len(r.boxes)

        # Execute detector pipeline
        objs = detector.detect(frame, t_stamp, frame_id=frame_count)
        nms_boxes_count += len(objs)

        detected_in_frame = set()
        for obj in objs:
            cls_name = str(obj.semantic_identity or obj.class_name).upper()
            if cls_name in per_class_stats:
                st = per_class_stats[cls_name]
                st["total_detections"] += 1
                st["confidences"].append(float(obj.confidence))
                bx, by, bw, bh = obj.bbox
                st["widths"].append(bw)
                st["heights"].append(bh)
                st["areas"].append(bw * bh)
                st["norm_areas"].append((bw * bh) / (w * h))
                st["centroids"].append([int(obj.centroid[0]), int(obj.centroid[1])])
                detected_in_frame.add(cls_name)

        for cls_name in detected_in_frame:
            per_class_stats[cls_name]["frames_with_detection"] += 1

        # Save sample annotated frame if detections exist or every 50 frames
        if (objs or frame_count % 50 == 0) and saved_samples_count < 10:
            vis = detector.draw(frame, objs)
            sample_file = f"reports/yolo_samples/frame_{frame_count:04d}.jpg"
            cv2.imwrite(sample_file, vis)
            saved_samples_count += 1

    cap.release()
    elapsed = time.time() - t0

    # Summary calculations
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(model_path),
        "video_path": str(video_path),
        "frames_processed": frame_count,
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(frame_count / max(0.001, elapsed), 2),
        "raw_yolo_candidate_boxes_total": raw_yolo_boxes_count,
        "nms_postprocessed_boxes_total": nms_boxes_count,
        "per_class_summary": {},
    }

    for cls_name, st in per_class_stats.items():
        confs = st["confidences"]
        areas = st["areas"]
        norm_areas = st["norm_areas"]
        cov_pct = round((st["frames_with_detection"] / max(1, frame_count)) * 100.0, 2)
        summary_report["per_class_summary"][cls_name] = {
            "unannotated_detection_coverage_pct": cov_pct,
            "frames_with_detection": st["frames_with_detection"],
            "total_detections": st["total_detections"],
            "max_confidence": round(float(np.max(confs)), 4) if confs else 0.0,
            "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            "median_confidence": round(float(np.median(confs)), 4) if confs else 0.0,
            "p95_confidence": round(float(np.percentile(confs, 95)), 4) if confs else 0.0,
            "mean_bbox_width_px": round(float(np.mean(st["widths"])), 1) if st["widths"] else 0.0,
            "mean_bbox_height_px": round(float(np.mean(st["heights"])), 1) if st["heights"] else 0.0,
            "mean_bbox_area_px": round(float(np.mean(areas)), 1) if areas else 0.0,
            "mean_normalized_area_pct": round(float(np.mean(norm_areas)) * 100.0, 4) if norm_areas else 0.0,
        }

    os.makedirs("reports", exist_ok=True)
    report_file = "reports/yolo_baseline_real_video.json"
    with open(report_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\nBaseline run complete!")
    print(f"JSON Report written to: {report_file}")
    print("\nPer-Class Summary:")
    for cls_name, s in summary_report["per_class_summary"].items():
        print(f"  {cls_name:12s} | Cov: {s['unannotated_detection_coverage_pct']:6.2f}% | Detections: {s['total_detections']:4d} | MaxConf: {s['max_confidence']:.4f} | MeanConf: {s['mean_confidence']:.4f}")


if __name__ == "__main__":
    run_baseline()
