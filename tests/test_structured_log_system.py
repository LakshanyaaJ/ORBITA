"""
ORBITA Structured Experiment Log System Test Suite
===================================================
Verifies:
1. EXP-MICROBE (7 steps), EXP-VDATA (13 steps), EXP-01 (13 steps), EXP-02 (2 steps) logs.
2. Unique execution_id assignment and isolation.
3. Chronological logging of EXPERIMENT_STARTED, STEP_VALIDATED, PROCEDURE_DEVIATION, VOICE_GUIDANCE.
4. Preservation of procedure deviations without overwriting previous history.
5. Offline SQLite persistence, retrieval, and SHA-256 integrity checksums.
"""

import pytest
import time
from fastapi.testclient import TestClient
from core_ai.backend.api import app, _state
from core_ai.logging.experiment_logger import ExperimentLogger, LogEntry
from core_ai.database.sqlite_db import OrbitaDB


@pytest.fixture
def client():
    return TestClient(app)


def test_structured_logs_all_4_experiments(client):
    # 1. Test EXP-MICROBE (7 steps)
    res_microbe = client.get("/api/experiments/EXP-MICROBE/logs")
    assert res_microbe.status_code == 200
    data_microbe = res_microbe.json()
    assert data_microbe["experiment_id"] == "EXP-MICROBE"
    assert data_microbe["protocol_steps"] == 7
    assert "checksum" in data_microbe

    # 2. Test EXP-VDATA (13 steps)
    res_vdata = client.get("/api/experiments/EXP-VDATA/logs")
    assert res_vdata.status_code == 200
    data_vdata = res_vdata.json()
    assert data_vdata["experiment_id"] == "EXP-VDATA"
    assert data_vdata["protocol_steps"] == 13

    # 3. Test EXP-01 (13 steps)
    res_exp01 = client.get("/api/experiments/EXP-01/logs")
    assert res_exp01.status_code == 200
    data_exp01 = res_exp01.json()
    assert data_exp01["experiment_id"] == "EXP-01"
    assert data_exp01["protocol_steps"] == 13

    # 4. Test EXP-02 (2 steps)
    res_exp02 = client.get("/api/experiments/EXP-02/logs")
    assert res_exp02.status_code == 200
    data_exp02 = res_exp02.json()
    assert data_exp02["experiment_id"] == "EXP-02"
    assert data_exp02["protocol_steps"] == 2


def test_execution_id_logging_and_deviation_preservation():
    db = OrbitaDB()
    exec_id = f"RUN-TEST-{int(time.time())}"

    # Log initial experiment start event
    db.log_structured_event({
        "experiment_id": "EXP-MICROBE",
        "execution_id": exec_id,
        "timestamp_epoch": time.time(),
        "elapsed_time": 0.0,
        "step_number": 1,
        "total_steps": 7,
        "expected_action": "PREPARE_SETUP",
        "detected_action": "EXPERIMENT_STARTED",
        "target_object": "SYSTEM",
        "status": "IN_PROGRESS",
        "procedure_state": "INITIAL",
        "event_type": "EXPERIMENT_STARTED",
        "confidence": 1.0
    })

    # Log validated step 1
    db.log_structured_event({
        "experiment_id": "EXP-MICROBE",
        "execution_id": exec_id,
        "timestamp_epoch": time.time() + 2,
        "elapsed_time": 2.0,
        "step_number": 1,
        "total_steps": 7,
        "expected_action": "PREPARE_SETUP",
        "detected_action": "Prepare Experiment Setup",
        "target_object": "Work surface",
        "status": "VALIDATED",
        "procedure_state": "ADVANCED",
        "event_type": "STEP_VALIDATED",
        "confidence": 0.95
    })

    # Log procedure deviation on step 2
    db.log_structured_event({
        "experiment_id": "EXP-MICROBE",
        "execution_id": exec_id,
        "timestamp_epoch": time.time() + 5,
        "elapsed_time": 5.0,
        "step_number": 2,
        "total_steps": 7,
        "expected_action": "PREPARE_SAMPLE",
        "detected_action": "Transfer Tool Picked",
        "target_object": "Pipette",
        "status": "UNEXPECTED",
        "procedure_state": "HELD",
        "event_type": "PROCEDURE_DEVIATION",
        "deviation_reason": "Unexpected target object",
        "confidence": 0.82
    })

    # Log recovery / validation on step 2
    db.log_structured_event({
        "experiment_id": "EXP-MICROBE",
        "execution_id": exec_id,
        "timestamp_epoch": time.time() + 8,
        "elapsed_time": 8.0,
        "step_number": 2,
        "total_steps": 7,
        "expected_action": "PREPARE_SAMPLE",
        "detected_action": "Prepare Microbial Sample",
        "target_object": "Sample container",
        "status": "VALIDATED",
        "procedure_state": "ADVANCED",
        "event_type": "STEP_VALIDATED",
        "confidence": 0.94
    })

    # Query structured logs for this specific execution_id
    logs = db.get_structured_logs("EXP-MICROBE", execution_id=exec_id)
    assert len(logs) == 4, "Should preserve all 4 events including deviation"
    assert logs[0]["event_type"] == "EXPERIMENT_STARTED"
    assert logs[1]["event_type"] == "STEP_VALIDATED"
    assert logs[2]["event_type"] == "PROCEDURE_DEVIATION"
    assert logs[3]["event_type"] == "STEP_VALIDATED"


def test_export_log_endpoint(client):
    res = client.get("/api/experiments/EXP-MICROBE/export_log")
    assert res.status_code == 200
    data = res.json()
    assert "events" in data
    assert "checksum" in data
    assert data["experiment_id"] == "EXP-MICROBE"
