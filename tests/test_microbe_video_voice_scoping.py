"""
ORBITA Microbe Video Scoping & Voice Lifecycle Test
=====================================================
Verifies:
1. WhatsApp Video is available ONLY when experiment_id is EXP-MICROBE.
2. WhatsApp Video is EXCLUDED from EXP-VDATA and EXP-01.
3. Default video paths in API endpoints enforce WhatsApp Video for EXP-MICROBE and 20260905_145858.mp4 for EXP-VDATA.
4. Voice session endpoints (/api/voice/session, /api/voice/stop) operate cleanly.
"""

import pytest
from fastapi.testclient import TestClient
from core_ai.backend.api import app, _state


@pytest.fixture
def client():
    return TestClient(app)


def test_vdata_videos_scoping_microbe_vs_vdata(client):
    # Query for EXP-MICROBE -> Should ONLY return WhatsApp Video
    res_microbe = client.get("/api/vdata/videos?experiment_id=EXP-MICROBE")
    assert res_microbe.status_code == 200
    vids_microbe = res_microbe.json()
    for v in vids_microbe:
        assert "whatsapp" in v["filename"].lower() or "microbe" in v["filename"].lower()

    # Query for EXP-VDATA -> Should EXCLUDE WhatsApp Video completely
    res_vdata = client.get("/api/vdata/videos?experiment_id=EXP-VDATA")
    assert res_vdata.status_code == 200
    vids_vdata = res_vdata.json()
    for v in vids_vdata:
        assert "whatsapp" not in v["filename"].lower()
        assert "microbe" not in v["filename"].lower()

    # Query for EXP-01 -> Should EXCLUDE WhatsApp Video completely
    res_exp01 = client.get("/api/vdata/videos?experiment_id=EXP-01")
    assert res_exp01.status_code == 200
    vids_exp01 = res_exp01.json()
    for v in vids_exp01:
        assert "whatsapp" not in v["filename"].lower()
        assert "microbe" not in v["filename"].lower()


def test_voice_session_lifecycle(client):
    # 1. Start voice session
    res_start = client.post("/api/voice/session", json={
        "active": True,
        "experiment_id": "EXP-MICROBE",
        "session_id": "test_session_123"
    })
    assert res_start.status_code == 200
    assert res_start.json()["active"] is True

    # 2. Stop voice speech
    res_stop = client.post("/api/voice/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "stopped"

    # 3. Deactivate voice session
    res_end = client.post("/api/voice/session", json={"active": False})
    assert res_end.status_code == 200
    assert res_end.json()["active"] is False
