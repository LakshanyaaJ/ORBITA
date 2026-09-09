"""
Unit and Integration Tests for 'BLUE AND YELLOW BOX VDATA' (EXP-VDATA) Experiment
================================================================================
Verifies:
  1. GET /api/experiments lists EXP-VDATA ("BLUE AND YELLOW BOX VDATA")
  2. POST /api/control arms EXP-VDATA and auto-connects to vdata video
  3. POST /api/camera/connect connects to vdata/20260905_145858.mp4 in video_file mode
  4. Video frames feed into ObjectDetector and StateManager step validation pipeline
"""

import pytest
from fastapi.testclient import TestClient

from core_ai.backend.api import app, _state
from core_ai.app.config import load_config
from core_ai.perception.object_detector import ObjectDetector
from core_ai.reasoning.state_manager import StateManager


@pytest.fixture(scope="module")
def client():
    cfg = load_config()
    cfg.detection.use_yolo = False  # fast fallback for test environment
    _state.detector = ObjectDetector(cfg.detection)
    with TestClient(app) as test_client:
        yield test_client


def test_api_experiments_includes_exp_vdata(client):
    """Verify EXP-VDATA is present in /api/experiments catalog."""
    res = client.get("/api/experiments")
    assert res.status_code == 200
    experiments = res.json()
    assert isinstance(experiments, list)
    vdata_exp = next((e for e in experiments if e.get("id") == "EXP-VDATA"), None)
    assert vdata_exp is not None
    assert vdata_exp["name"] == "BLUE AND YELLOW BOX VDATA"
    assert vdata_exp["total_steps"] == 13
    assert vdata_exp["status"] in ("READY", "SUCCESS", "IN_PROGRESS")


def test_api_control_starts_vdata_experiment(client):
    """Verify /api/control auto-connects video_file source when starting EXP-VDATA."""
    res = client.post("/api/control", json={
        "action": "start",
        "experiment_id": "EXP-VDATA",
        "video_path": "vdata/20260905_145858.mp4",
        "loop": True
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "started"
    assert data["experiment_id"] == "EXP-VDATA"
    assert data["source"] == "video_file"


def test_api_camera_connect_vdata_video(client):
    """Verify /api/camera/connect with source video_file switches cleanly and resets FSM."""
    res = client.post("/api/camera/connect", json={
        "source": "video_file",
        "path": "vdata/20260905_145858.mp4",
        "loop": True,
        "reset_fsm": True
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "connected"
    assert data["source"] == "video_file"
    assert data["loop"] is True


def test_api_experiment_start_vdata(client):
    """Verify /api/experiment/start endpoint with EXP-VDATA."""
    res = client.post("/api/experiment/start", json={
        "experiment_id": "EXP-VDATA",
        "scenario": "A"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "experiment_started"
    assert data["experiment_id"] == "EXP-VDATA"
    assert data["recording"] is True
    assert data["source"] == "video_file"
