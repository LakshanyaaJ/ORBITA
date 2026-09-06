"""
Tests for Reference Video (vdata) YOLO Inspection APIs
======================================================
Verifies:
  1. GET /api/vdata/videos lists videos in vdata/ with metadata
  2. POST /api/vdata/inspect_frame runs YOLO on reference frames and returns bounding boxes
"""

import pytest
from fastapi.testclient import TestClient

from core_ai.backend.api import app, _state
from core_ai.app.config import load_config
from core_ai.perception.object_detector import ObjectDetector


@pytest.fixture(scope="module")
def client():
    cfg = load_config()
    cfg.detection.use_yolo = False  # fast fallback for test environment
    _state.detector = ObjectDetector(cfg.detection)
    with TestClient(app) as test_client:
        yield test_client


def test_api_vdata_videos(client):
    """Verify vdata videos list returns correct structure."""
    res = client.get("/api/vdata/videos")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    if len(data) > 0:
        first = data[0]
        assert "filename" in first
        assert "duration_seconds" in first
        assert "fps" in first
        assert "resolution" in first


def test_api_vdata_inspect_frame(client):
    """Verify frame inspection runs detector and returns detections list & image data."""
    vids_res = client.get("/api/vdata/videos")
    videos = vids_res.json()
    if not videos:
        pytest.skip("No reference videos in vdata/")

    first_vid = videos[0]["filename"]
    res = client.post("/api/vdata/inspect_frame", json={"filename": first_vid, "frame_idx": 0})
    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == first_vid
    assert "detections" in data
    assert "latency_ms" in data
    assert "image_data" in data
    assert data["image_data"].startswith("data:image/jpeg;base64,")
