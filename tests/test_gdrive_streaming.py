"""
Tests for Google Drive Online Video Streaming & Authorization in ORBITA
=======================================================================
Verifies:
1. Google Drive catalog metadata and playback URLs.
2. Strict security authorization: prevents streaming arbitrary Google Drive file IDs.
3. HTTP Range streaming endpoint:
   - 206 Partial Content
   - Content-Range header
   - Content-Length header
   - Accept-Ranges: bytes
   - Seeking forward in the video byte stream
4. Rejection of unauthorized file IDs (HTTP 403).
5. GET /api/vdata/gdrive/videos endpoint response format.
"""

import pytest
from fastapi.testclient import TestClient
from core_ai.backend.api import app
from core_ai.dataset.gdrive_sync import (
    DEFAULT_GDRIVE_FOLDER_ID,
    DEFAULT_GDRIVE_FOLDER_URL,
    KNOWN_FOLDER_VIDEOS,
    extract_folder_id,
    gdrive_sync_manager,
    validate_file_id,
)

VALID_TEST_FILE_ID = "1k202zi-kg68lc5Nr5RR-FhW_neqEP8OW"
UNAUTHORIZED_FILE_ID = "1ABCxyz9876543210_fake_unauthorized_id"


def test_folder_and_file_id_validation():
    """Verify folder ID extraction and file ID format validation."""
    assert extract_folder_id(DEFAULT_GDRIVE_FOLDER_URL) == DEFAULT_GDRIVE_FOLDER_ID
    assert validate_file_id(VALID_TEST_FILE_ID) is True
    assert validate_file_id("invalid-id") is False
    assert validate_file_id("../path/traversal") is False
    assert validate_file_id("") is False


def test_file_authorization_security():
    """Security check: ensure only authorized dataset videos can be accessed."""
    assert gdrive_sync_manager.is_file_authorized(VALID_TEST_FILE_ID) is True
    assert gdrive_sync_manager.is_file_authorized(UNAUTHORIZED_FILE_ID) is False

    with pytest.raises(PermissionError):
        gdrive_sync_manager.get_stream_response(UNAUTHORIZED_FILE_ID)


def test_gdrive_catalog_metadata():
    """Verify catalog entries contain valid metadata and playback URLs."""
    videos = gdrive_sync_manager.list_remote_videos()
    assert len(videos) >= 4

    first = videos[0]
    assert "id" in first
    assert "name" in first
    assert "mimeType" in first
    assert "size_mb" in first
    assert "driveUrl" in first
    assert "playbackUrl" in first
    assert first["playbackUrl"].startswith("/api/vdata/gdrive/video/")
    assert first["status"] == "online"


def test_api_gdrive_videos_endpoint():
    """Verify GET /api/vdata/gdrive/videos returns structured online catalog."""
    client = TestClient(app)
    res = client.get("/api/vdata/gdrive/videos")
    assert res.status_code == 200
    data = res.json()
    assert "videos" in data
    assert "count" in data
    assert "online_count" in data
    assert data["count"] >= 4
    assert data["online_count"] >= 4


def test_api_gdrive_streaming_initial_range():
    """
    Verify GET /api/vdata/gdrive/video/{file_id} supports HTTP Range requests (206 Partial Content).
    """
    client = TestClient(app)
    res = client.get(
        f"/api/vdata/gdrive/video/{VALID_TEST_FILE_ID}",
        headers={"Range": "bytes=0-1000"},
    )
    assert res.status_code == 206
    assert res.headers.get("accept-ranges") == "bytes"
    assert "bytes 0-1000/" in res.headers.get("content-range", "")
    assert res.headers.get("content-length") == "1001"
    assert "video/mp4" in res.headers.get("content-type", "")
    assert len(res.content) == 1001


def test_api_gdrive_streaming_seeking():
    """
    Verify seeking forward into the video byte stream without reading earlier bytes.
    """
    client = TestClient(app)
    # Seek to byte 50,000,000 (around middle of video)
    res = client.get(
        f"/api/vdata/gdrive/video/{VALID_TEST_FILE_ID}",
        headers={"Range": "bytes=50000000-50002000"},
    )
    assert res.status_code == 206
    assert "bytes 50000000-50002000/" in res.headers.get("content-range", "")
    assert len(res.content) == 2001


def test_api_gdrive_streaming_unauthorized_blocking():
    """
    Verify unauthorized file IDs are rejected with 403 Forbidden.
    """
    client = TestClient(app)
    res = client.get(f"/api/vdata/gdrive/video/{UNAUTHORIZED_FILE_ID}")
    assert res.status_code == 403
    assert "not an authorized video" in res.json().get("detail", "")
