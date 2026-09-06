"""
Unit Tests for ORBITA Dataset and Governance API Endpoints
"""

import pytest
from fastapi.testclient import TestClient

from core_ai.backend.api import create_app, _state

client = TestClient(create_app())


def test_api_dataset_status():
    """Verify that /api/dataset/status returns valid manifest and model info."""
    response = client.get("/api/dataset/status")
    assert response.status_code == 200
    data = response.json()
    assert "stage" in data
    assert "manifest" in data
    assert "production_models" in data
    assert "review_queue_counts" in data


def test_api_models_versions():
    """Verify listing of registered model versions."""
    response = client.get("/api/models/versions")
    assert response.status_code == 200
    versions = response.json()
    assert isinstance(versions, list)
    # Check that har_gru_v1 is present
    if versions:
        assert any("har_gru" in v["model_type"] for v in versions)


def test_api_dataset_candidates():
    """Verify /api/dataset/candidates endpoint returns list."""
    response = client.get("/api/dataset/candidates")
    assert response.status_code == 200
    candidates = response.json()
    assert isinstance(candidates, list)


def test_api_training_retrain_har():
    """Verify controlled HAR retraining endpoint executes and returns valid metrics."""
    response = client.post("/api/training/retrain", json={"model_type": "har", "epochs": 2})
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "TRAINING_COMPLETE"
    assert "metrics" in res
    assert "accuracy" in res["metrics"]


def test_api_training_retrain_yolo_safety_gate(monkeypatch):
    """Verify that unverified YOLO training is properly blocked."""
    import core_ai.training.train_yolo as ty
    monkeypatch.setattr(
        ty, 
        "check_dataset_readiness", 
        lambda *args, **kwargs: (False, 0, "Human verification is required before supervised object-detection training.")
    )
    response = client.post("/api/training/retrain", json={"model_type": "yolo", "require_verified": True})
    assert response.status_code == 200
    res = response.json()
    # It must report that annotations/verification are required before training
    assert res["status"] in ("ANNOTATIONS_REQUIRED", "TRAINING_FAILED")
    assert not res["trainable"]
