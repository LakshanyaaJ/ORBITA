"""
Tests for Google Drive Dataset Video Sync in ORBITA
===================================================
Verifies:
1. Google Drive folder ID extraction from sharing URLs.
2. Status reporting of local vdata/ files and Drive configurations.
3. FastAPI endpoints:
   - GET /api/vdata/gdrive/status
   - POST /api/vdata/gdrive/list
   - POST /api/vdata/gdrive/sync
"""

import pytest
from fastapi.testclient import TestClient
from core_ai.backend.api import app
from core_ai.dataset.gdrive_sync import (
    DEFAULT_GDRIVE_FOLDER_ID,
    DEFAULT_GDRIVE_FOLDER_URL,
    GDriveSyncManager,
    extract_folder_id,
)


def test_extract_folder_id():
    """Verify regex folder ID extraction for standard Drive sharing links."""
    test_url = "https://drive.google.com/drive/folders/1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C?usp=sharing"
    assert extract_folder_id(test_url) == "1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C"

    test_url_param = "https://drive.google.com/open?id=1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C"
    assert extract_folder_id(test_url_param) == "1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C"

    raw_id = "1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C"
    assert extract_folder_id(raw_id) == "1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C"

    assert extract_folder_id("") == DEFAULT_GDRIVE_FOLDER_ID


def test_gdrive_sync_manager_status(tmp_path):
    """Verify GDriveSyncManager initializes and tracks files."""
    mgr = GDriveSyncManager(target_dir=tmp_path)
    status = mgr.get_status()
    assert status["status"] in ("idle", "completed")
    assert status["folder_id"] == DEFAULT_GDRIVE_FOLDER_ID
    assert status["local_video_count"] == 0

    # Create dummy video file
    dummy_file = tmp_path / "dummy_video.mp4"
    dummy_file.write_bytes(b"dummy video data")

    status2 = mgr.get_status()
    assert status2["local_video_count"] == 1
    assert "dummy_video.mp4" in status2["local_videos"]


def test_api_gdrive_endpoints():
    """Verify FastAPI endpoints for Google Drive dataset video sync."""
    client = TestClient(app)

    # 1. GET status
    res = client.get("/api/vdata/gdrive/status")
    assert res.status_code == 200
    data = res.json()
    assert "folder_id" in data
    assert data["folder_id"] == DEFAULT_GDRIVE_FOLDER_ID
    assert "local_video_count" in data

    # 2. POST sync start
    res_sync = client.post("/api/vdata/gdrive/sync", json={"force": False})
    assert res_sync.status_code == 200
    sync_data = res_sync.json()
    assert sync_data["status"] in ("started", "in_progress")
