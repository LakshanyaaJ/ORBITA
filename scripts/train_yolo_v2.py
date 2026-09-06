"""
ORBITA YOLO V2 Training & Evaluation Pipeline
=============================================
Trains YOLOv8 on the expanded Dataset V2 using verified physical experiment data.
Performs:
  1. Supervised training on Train split with realistic augmentation
  2. Validation on Val split
  3. Final evaluation on untouched held-out TEST split
  4. Per-class metrics breakdown (Precision, Recall, mAP50, mAP50-95)
  5. Registration and candidate promotion in ModelRegistry
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from core_ai.training.model_registry import ModelRegistry
from core_ai.training.train_yolo import check_dataset_readiness

logger = logging.getLogger(__name__)

def train_and_evaluate_v2():
    data_yaml = Path("datasets/orbita/data.yaml")
    base_model = "models/yolov8n.pt"

    print("=== Step 1: Checking Dataset V2 Readiness ===")
    ready, verified_count, msg = check_dataset_readiness(data_yaml, require_verified=True)
    print(f"Status: ready={ready}, count={verified_count}, msg={msg}")
    if not ready:
        raise RuntimeError(f"Dataset not ready: {msg}")

    print("\n=== Step 2: Training YOLO V2 Candidate on Dataset V2 ===")
    t0 = time.time()
    model = YOLO(base_model)

    train_args = dict(
        data=str(data_yaml),
        epochs=5,
        batch=8,
        imgsz=416,
        device="cpu",
        degrees=10.0,
        translate=0.1,
        scale=0.15,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        plots=False,
        save=True,
        val=True,
        verbose=False,
    )
    print("Training configuration:", train_args)
    results = model.train(**train_args)
    train_duration = round(time.time() - t0, 1)
    print(f"Training completed in {train_duration}s")

    # Step 3: Validation on Val split
    print("\n=== Step 3: Evaluating on Validation Split ===")
    val_results = model.val(data=str(data_yaml), split="val", verbose=False)
    val_metrics = {
        "precision": float(val_results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(val_results.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
    }
    print("Validation Metrics:", val_metrics)

    # Step 4: Untouched Held-Out Test Set Evaluation
    print("\n=== Step 4: Evaluating on Held-Out Test Set ===")
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path(results.save_dir) / "weights" / "last.pt"

    eval_model = YOLO(str(best_weights))
    test_results = eval_model.val(data=str(data_yaml), split="test", verbose=False)
    test_metrics = {
        "precision": float(test_results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(test_results.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(test_results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(test_results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
    }
    print("Test Set Metrics:", test_metrics)

    # Per-class metrics from test set
    class_names = ["PERSON", "MAIN_BOX", "RED_BOX", "YELLOW_BOX", "SAMPLE", "TOOL"]
    per_class = {}
    try:
        if hasattr(test_results, "box"):
            box_res = test_results.box
            p_list = box_res.p.tolist() if hasattr(box_res.p, "tolist") else []
            r_list = box_res.r.tolist() if hasattr(box_res.r, "tolist") else []
            map50_list = box_res.ap50.tolist() if hasattr(box_res.ap50, "tolist") else []
            map_list = box_res.ap.tolist() if hasattr(box_res.ap, "tolist") else []

            for idx, cname in enumerate(class_names):
                per_class[cname] = {
                    "precision": round(p_list[idx], 4) if idx < len(p_list) else 0.0,
                    "recall": round(r_list[idx], 4) if idx < len(r_list) else 0.0,
                    "mAP50": round(map50_list[idx], 4) if idx < len(map50_list) else 0.0,
                    "mAP50-95": round(map_list[idx], 4) if idx < len(map_list) else 0.0,
                }
    except Exception as e:
        print("Note: per-class extraction exception:", e)

    print("Per-Class Test Performance:", json.dumps(per_class, indent=2))

    # Step 5: Register in ModelRegistry & evaluate for promotion
    print("\n=== Step 5: Model Registry Registration & Promotion Evaluation ===")
    registry = ModelRegistry()
    record = registry.register_candidate_version(
        weights_file=best_weights,
        model_type="yolo_detector",
        dataset_version="v2",
        training_samples=verified_count,
        classes=class_names,
        metrics=test_metrics,
    )
    print(f"Candidate registered as: {record.version} at {record.weights_path}")

    promoted, promo_msg = registry.evaluate_and_promote(record.version, min_improvement=0.001)
    print(f"Promotion Result: promoted={promoted}, msg={promo_msg}")

    summary = {
        "version": record.version,
        "is_production": record.is_production,
        "dataset_version": "v2",
        "training_samples": verified_count,
        "train_duration_s": train_duration,
        "weights_path": str(record.weights_path),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "per_class": per_class,
        "promoted": promoted,
        "promotion_message": promo_msg,
    }

    out_file = Path("models/yolo_v2_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\nEvaluation summary saved to {out_file}")
    return summary

if __name__ == "__main__":
    train_and_evaluate_v2()
