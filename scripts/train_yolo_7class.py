"""
ORBITA YOLOv8 7-Class Model Training Pipeline
=============================================
Trains YOLOv8 on the 7 canonical classes:
  0: LOCATION_A
  1: LOCATION_B
  2: PEN
  3: WATCH
  4: BLUE_BOX
  5: YELLOW_BOX
  6: HAND

Saves production checkpoint to models/orbita_yolo_detector_v4.pt.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from ultralytics import YOLO

CLASS_NAMES = [
    "LOCATION_A",
    "LOCATION_B",
    "PEN",
    "WATCH",
    "BLUE_BOX",
    "YELLOW_BOX",
    "HAND",
]


def train_7class_model():
    data_yaml = Path("datasets/orbita/v4/data.yaml")
    if not data_yaml.exists():
        data_yaml = Path("datasets/orbita/data.yaml")

    base_model = "models/yolov8n.pt"
    if not Path(base_model).exists():
        logger.info("Downloading base yolov8n.pt...")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(base_model)

    logger.info("Starting YOLOv8 7-class training...")
    t0 = time.time()

    train_args = dict(
        data=str(data_yaml),
        epochs=12,
        batch=8,
        imgsz=640,
        device="cpu",
        degrees=4.0,
        translate=0.06,
        scale=0.08,
        fliplr=0.0,    # Strictly keep 0 so LOCATION_A (left) and LOCATION_B (right) stay correct
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.25,
        hsv_v=0.25,
        plots=False,
        save=True,
        val=True,
        verbose=True,
    )

    results = model.train(**train_args)
    duration = time.time() - t0
    logger.info("Training completed in %.1f seconds", duration)

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = save_dir / "weights" / "last.pt"

    dest_weights = Path("models/orbita_yolo_detector_v4.pt")
    shutil.copy2(best_weights, dest_weights)
    logger.info("Exported best weights to %s", dest_weights)

    # Also back up to versions directory
    v4_dir = Path("models/versions/orbita_yolo_detector_v4")
    v4_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, v4_dir / "orbita_yolo_detector_v4.pt")
    shutil.copy2(best_weights, v4_dir / "orbita_yolo_detector_v3.pt")
    logger.info("Backed up weights to %s", v4_dir)

    # Evaluate on validation split
    eval_model = YOLO(str(dest_weights))
    logger.info("Model class names: %s", eval_model.names)

    print("\nMODEL CLASSES:")
    for idx, name in sorted(eval_model.names.items(), key=lambda x: int(x[0])):
        print(f"{idx} -> {name}")

    val_res = eval_model.val(data=str(data_yaml), split="val", imgsz=640)
    mAP50 = float(val_res.results_dict.get("metrics/mAP50(B)", 0.0))
    prec = float(val_res.results_dict.get("metrics/precision(B)", 0.0))
    rec = float(val_res.results_dict.get("metrics/recall(B)", 0.0))

    logger.info("Validation Results: Precision=%.3f, Recall=%.3f, mAP50=%.3f", prec, rec, mAP50)

    # Update model_registry.json
    reg_path = Path("models/model_registry.json")
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                registry = json.load(f)

            # Ensure orbita_yolo_detector_v4 is production and has correct weights and classes
            registry["orbita_yolo_detector_v4"] = {
                "version": "orbita_yolo_detector_v4",
                "model_type": "yolo_detector",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                "weights_path": str(dest_weights.resolve()),
                "dataset_version": "v4",
                "training_samples_count": 341,
                "classes": [eval_model.names[i] for i in range(len(eval_model.names))],
                "is_production": True,
                "metrics": {
                    "mAP50": mAP50,
                    "precision": prec,
                    "recall": rec
                },
                "promotion_notes": f"Promoted: Horizontal 7-class model with mAP50={mAP50:.4f}"
            }

            registry["active_yolo_model"] = "models/orbita_yolo_detector_v4.pt"
            registry["yolo_candidates"] = registry.get("yolo_candidates", {})
            registry["yolo_candidates"]["v4_7class"] = {
                "path": "models/orbita_yolo_detector_v4.pt",
                "classes": list(eval_model.names.values()),
                "mAP50": mAP50,
                "precision": prec,
                "recall": rec,
                "timestamp": time.time(),
            }
            with open(reg_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
            logger.info("Updated model registry with v4_7class production model.")
        except Exception as e:
            logger.warning("Could not update model_registry.json: %s", e)

    return str(dest_weights)


if __name__ == "__main__":
    train_7class_model()
