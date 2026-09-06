"""
ORBITA YOLOv8 Training & Validation Pipeline
============================================
Supervised fine-tuning of YOLO object detection models on the ORBITA dataset.

Safety Rules:
  1. Checks for annotated labels before attempting training.
  2. Rejects training if annotations are missing, explicitly reporting:
     "Frames have been extracted, but annotations are required before supervised training."
  3. Computes standard evaluation metrics: Precision, Recall, mAP50, mAP50-95.
  4. Registers candidate version in ModelRegistry for promotion comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def check_dataset_readiness(
    dataset_yaml_path: str | Path,
    require_verified: bool = True,
) -> tuple[bool, int, str]:
    """
    Verifies that the dataset has valid non-empty annotation files.
    When require_verified is True, also verifies that annotations have been confirmed by a human reviewer.
    Returns: (is_ready, label_count, status_message)
    """
    yaml_path = Path(dataset_yaml_path)
    if not yaml_path.exists():
        return False, 0, f"data.yaml not found at {yaml_path}"

    base_dir = yaml_path.parent
    labels_train = base_dir / "labels" / "train"

    msg_not_ready = "Frames have been extracted, but annotations are required before supervised object-detection training."
    if not labels_train.exists():
        return False, 0, msg_not_ready

    label_files = list(labels_train.glob("*.txt"))
    non_empty_labels = [f for f in label_files if f.stat().st_size > 0]

    if not non_empty_labels:
        return False, 0, msg_not_ready

    if require_verified:
        import json
        verified_count = 0
        for lf in non_empty_labels:
            meta_f = lf.with_suffix(".meta.json")
            if meta_f.exists():
                try:
                    with open(meta_f, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    if meta_data.get("status") == "verified":
                        verified_count += 1
                except Exception:
                    pass
        if verified_count == 0:
            return False, 0, (
                "Frames have been extracted and pre-annotated, but human verification is required "
                "before supervised object-detection training. The system will not silently train on unverified pseudo-labels."
            )
        return True, verified_count, f"Dataset ready with {verified_count} verified training frames."

    return True, len(non_empty_labels), f"Dataset ready with {len(non_empty_labels)} annotated training frames."


def train_yolo_model(
    data_yaml: str | Path = "datasets/orbita/data.yaml",
    base_model: str = "models/yolov8n.pt",
    epochs: int = 10,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Execute YOLO fine-tuning if dataset is ready, otherwise return status requirements.
    """
    data_yaml_p = Path(data_yaml)
    is_ready, label_count, message = check_dataset_readiness(data_yaml_p)

    if not is_ready:
        logger.warning("YOLO Training blocked: %s", message)
        return {
            "status": "ANNOTATIONS_REQUIRED",
            "message": message,
            "trainable": False,
            "annotated_samples": label_count,
        }

    try:
        from ultralytics import YOLO
        from core_ai.training.model_registry import ModelRegistry

        logger.info("Starting YOLO training on %s (epochs=%d, batch=%d)", data_yaml, epochs, batch_size)
        model = YOLO(base_model)

        # Train model
        train_results = model.train(
            data=str(data_yaml_p),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            plots=False,
            save=True,
            val=True,
        )

        # Run validation on val set
        val_results = model.val(data=str(data_yaml_p))

        metrics = {
            "precision": float(val_results.results_dict.get("metrics/precision(B)", 0.0)),
            "recall": float(val_results.results_dict.get("metrics/recall(B)", 0.0)),
            "mAP50": float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50-95": float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        }

        # Best checkpoint path
        best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = Path(train_results.save_dir) / "weights" / "last.pt"

        registry = ModelRegistry()
        record = registry.register_candidate_version(
            weights_file=best_pt,
            model_type="yolo_detector",
            dataset_version="v1",
            training_samples=label_count,
            classes=["PERSON", "MAIN_BOX", "RED_BOX", "YELLOW_BOX", "SAMPLE", "TOOL"],
            metrics=metrics,
        )

        promoted, promo_msg = registry.evaluate_and_promote(record.version)

        return {
            "status": "TRAINING_COMPLETE",
            "message": f"YOLO model trained successfully. {promo_msg}",
            "trainable": True,
            "version": record.version,
            "is_production": record.is_production,
            "metrics": metrics,
            "checkpoint_path": record.weights_path,
        }

    except Exception as exc:
        logger.error("YOLO training failed: %s", exc)
        return {
            "status": "TRAINING_FAILED",
            "message": str(exc),
            "trainable": True,
        }
