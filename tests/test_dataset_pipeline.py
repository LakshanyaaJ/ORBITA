"""
Unit Tests for ORBITA Closed-Loop Dataset & Training System
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core_ai.dataset.dataset_manager import DatasetManager, ORBITA_CLASSES
from core_ai.dataset.video_ingest import VideoIngestPipeline, VideoMetadata, IngestionReport
from core_ai.dataset.annotator import AssistedAnnotator, AnnotationStatus, BoundingBoxLabel
from core_ai.recording.quality_filter import DatasetQualityFilter
from core_ai.recording.run_collector import RunCollector
from core_ai.dataset.review_queue import ReviewQueue
from core_ai.training.model_registry import ModelRegistry
from core_ai.training.train_yolo import check_dataset_readiness


@pytest.fixture
def temp_workspace():
    tmp_dir = tempfile.mkdtemp(prefix="orbita_test_")
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_video_split_assignment(temp_workspace):
    """Verify that split assignment strictly splits by video/trial without leakage."""
    dm = DatasetManager(base_dir=temp_workspace / "datasets" / "orbita")
    dm.initialize_directories()

    videos = ["trial_01", "trial_02", "trial_03"]
    splits = dm.assign_video_splits(videos)

    assert len(splits) == 3
    assert splits["trial_01"] == "train"
    assert splits["trial_02"] == "val"
    assert splits["trial_03"] == "test"

    # Verify data.yaml was created properly
    assert dm.data_yaml_path.exists()
    content = dm.data_yaml_path.read_text(encoding="utf-8")
    assert "names:" in content
    assert "PERSON" in content
    assert "RED_BOX" in content


def test_quality_filter_evaluates_correctly(temp_workspace):
    """Verify that DatasetQualityFilter rejects low quality / incomplete runs."""
    q_filter = DatasetQualityFilter(min_overall_score=70.0, min_procedure_pct=100.0)

    # Incomplete run
    incomplete_meta = {
        "duration_seconds": 30.0,
        "procedure_completeness_pct": 50.0, # only 50%
        "mean_detection_confidence": 0.85,
        "mean_action_confidence": 0.80,
    }
    metrics = q_filter.evaluate_run([], incomplete_meta)
    assert not metrics.is_acceptable
    assert any("Incomplete procedure" in r for r in metrics.rejection_reasons)


def test_review_queue_lifecycle(temp_workspace):
    """Verify candidate staging, human approval, and rejection."""
    candidates_dir = temp_workspace / "candidates"
    dataset_dir = temp_workspace / "dataset"
    queue = ReviewQueue(candidates_dir=candidates_dir, dataset_base_dir=dataset_dir)

    # Create dummy candidate run
    run_id = "EXP001_20260905_120000"
    run_folder = candidates_dir / f"pending_{run_id}"
    images_dir = run_folder / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Put a dummy image
    (images_dir / f"{run_id}_f0001.jpg").write_bytes(b"dummy_image_data")

    meta = {
        "run_id": run_id,
        "experiment_id": "EXP001",
        "recorded_at": "2026-09-05T12:00:00Z",
        "duration_seconds": 25.0,
        "saved_samples_count": 1,
        "quality": {"overall_score": 88.0, "is_acceptable": True, "rejection_reasons": []},
    }
    with open(run_folder / "run_metadata.json", "w") as f:
        json.dump(meta, f)

    # 1. List candidates
    pending_list = queue.list_candidates(status_filter="pending")
    assert len(pending_list) == 1
    assert pending_list[0].run_id == run_id

    # 2. Approve candidate
    res = queue.approve_candidate(run_id, reviewer_name="astronaut_lead", target_split="train")
    assert res["status"] == "approved"
    assert (dataset_dir / "images" / "train" / f"{run_id}_f0001.jpg").exists()

    # Verify folder renamed to approved_
    assert (candidates_dir / f"approved_{run_id}").exists()


def test_model_registry_promotion(temp_workspace):
    """Verify candidate version registration and performance promotion comparison."""
    models_dir = temp_workspace / "models"
    registry_file = models_dir / "model_registry.json"
    registry = ModelRegistry(models_dir=models_dir, registry_file=registry_file)

    # Create dummy weights
    weights1 = models_dir / "baseline.pt"
    weights1.parent.mkdir(parents=True, exist_ok=True)
    weights1.write_bytes(b"fake_yolo_weights_1")

    # 1. Register baseline model v1 (first model becomes production automatically)
    rec1 = registry.register_candidate_version(
        weights_file=weights1,
        model_type="yolo_detector",
        dataset_version="v1",
        training_samples=100,
        classes=list(ORBITA_CLASSES),
        metrics={"mAP50": 0.750, "recall": 0.70},
    )
    assert rec1.is_production
    assert registry.get_production_model("yolo_detector").version == rec1.version

    # 2. Register candidate v2 with lower performance -> should NOT promote
    weights2 = models_dir / "candidate_lower.pt"
    weights2.write_bytes(b"fake_yolo_weights_2")

    rec2 = registry.register_candidate_version(
        weights_file=weights2,
        model_type="yolo_detector",
        dataset_version="v2",
        training_samples=150,
        classes=list(ORBITA_CLASSES),
        metrics={"mAP50": 0.720, "recall": 0.68},
    )
    assert not rec2.is_production
    promoted, msg = registry.evaluate_and_promote(rec2.version, primary_metric_key="mAP50")
    assert not promoted
    assert registry.get_production_model("yolo_detector").version == rec1.version

    # 3. Register candidate v3 with higher performance -> SHOULD promote!
    weights3 = models_dir / "candidate_higher.pt"
    weights3.write_bytes(b"fake_yolo_weights_3")

    rec3 = registry.register_candidate_version(
        weights_file=weights3,
        model_type="yolo_detector",
        dataset_version="v2",
        training_samples=200,
        classes=list(ORBITA_CLASSES),
        metrics={"mAP50": 0.810, "recall": 0.79},
    )
    promoted, msg = registry.evaluate_and_promote(rec3.version, primary_metric_key="mAP50")
    assert promoted
    assert registry.get_production_model("yolo_detector").version == rec3.version


def test_dataset_readiness_checker(temp_workspace):
    """Verify that training is safely blocked when labels are missing."""
    yaml_file = temp_workspace / "data.yaml"
    yaml_file.write_text("dummy yaml")

    # No labels directory
    ready, count, msg = check_dataset_readiness(yaml_file)
    assert not ready
    assert "required before supervised" in msg
