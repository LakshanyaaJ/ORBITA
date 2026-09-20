"""
ORBITA Strict Voice Session Gating & Lifecycle Authorization Test Suite
=======================================================================
Verifies that:
1. No voice requests are executed on backend without an active voice session (Default: Dropped).
2. /api/voice/session activates voice ONLY for the specific experiment_id and session_id.
3. Speech calls with wrong experiment_id or stale session_id are dropped.
4. /api/voice/stop deactivates voice session immediately and clears the TTS queue.
"""

import pytest
from fastapi.testclient import TestClient
from core_ai.backend.api import app, _state


def test_api_voice_strict_gating_lifecycle():
    with TestClient(app) as client:
        # 1. Verify default backend state (No active session)
        res_default = client.post("/api/voice/speak", json={"text": "Step 1 guidance", "experiment_id": "EXP-MICROBE"})
        assert res_default.status_code == 200
        assert res_default.json()["status"] == "dropped"
        assert "No active experiment voice session" in res_default.json()["reason"]

        # 2. Activate session for EXP-MICROBE
        sess_id = "EXP-MICROBE_session_12345"
        res_sess = client.post("/api/voice/session", json={
            "active": True,
            "experiment_id": "EXP-MICROBE",
            "session_id": sess_id
        })
        assert res_sess.status_code == 200
        assert res_sess.json()["voice_session_active"] is True
        assert res_sess.json()["active_experiment_id"] == "EXP-MICROBE"

        # 3. Speak with valid session -> OK (or TTS status)
        res_ok = client.post("/api/voice/speak", json={
            "text": "Step 1 prepare microbial sample.",
            "experiment_id": "EXP-MICROBE",
            "session_id": sess_id,
            "clear_previous": True
        })
        assert res_ok.status_code == 200
        assert res_ok.json()["status"] in ("ok", "error")  # ok if SAPI/pyttsx3 ready, error if audio device missing in headless runner

        # 4. Speak with wrong experiment ID (e.g. EXP-01 event arriving while EXP-MICROBE active) -> Dropped
        res_wrong_exp = client.post("/api/voice/speak", json={
            "text": "Step 1 identify blue box.",
            "experiment_id": "EXP-01",
            "session_id": sess_id
        })
        assert res_wrong_exp.status_code == 200
        assert res_wrong_exp.json()["status"] == "dropped"
        assert "Experiment ID mismatch" in res_wrong_exp.json()["reason"]

        # 5. Speak with stale session ID -> Dropped
        res_stale = client.post("/api/voice/speak", json={
            "text": "Step 2 prepare microbial sample.",
            "experiment_id": "EXP-MICROBE",
            "session_id": "old_stale_session_99999"
        })
        assert res_stale.status_code == 200
        assert res_stale.json()["status"] == "dropped"
        assert "Stale voice session ID" in res_stale.json()["reason"]

        # 6. Stop session (Exit experiment) -> Deactivates session & clears queue
        res_stop = client.post("/api/voice/stop")
        assert res_stop.status_code == 200
        assert res_stop.json()["status"] == "ok"

        # 7. Post-exit speak -> Dropped
        res_post_exit = client.post("/api/voice/speak", json={
            "text": "Delayed message after exit",
            "experiment_id": "EXP-MICROBE",
            "session_id": sess_id
        })
        assert res_post_exit.status_code == 200
        assert res_post_exit.json()["status"] == "dropped"
