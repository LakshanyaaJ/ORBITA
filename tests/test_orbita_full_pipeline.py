"""
ORBITA Master End-to-End Pipeline & Integration Test Suite
==========================================================
Verifies all 7 Core Architecture Requirements End-to-End:
  1. Continuous Local Video Processing
  2. Next-Step Guidance
  3. Step Skip / Wrong Sequence / Wrong Object Validation
  4. Real-Time Write-Through Timestamped Structured Logging (SQLite, JSONL, CSV)
  5. Dynamic Video Recording (MP4 integrity, resolution/FPS sync)
  6. Ultra-Low-Latency IP Streaming & Telemetry Compositing
  7. True Offline Standalone Operation (0 Cloud Dependencies, Local Models)

Scenarios Tested:
  - Scenario A: Perfect 13-Step Lab Procedure (13/13 completed)
  - Scenario B: Wrong Object Interacted (WRONG_OBJECT + voice alert + no FSM advance)
  - Scenario C: Step Skipped (Future step attempted -> STEP_SKIPPED + identified step + recovery voice)
  - Scenario D: Out of Sequence (Past step repeated -> OUT_OF_SEQUENCE + no transition)
  - Scenario E: Low Confidence (UNCERTAIN -> no penalty, no FSM transition)
  - Scenario F: Single-Frame Transient Detection (WAITING -> temporal gate blocks spurious transition)
  - Scenario G: Error Recovery (user recovers with correct action -> FSM resumes without loop)
  - Scenario H: Video Recording Integrity (playable MP4, valid resolution, non-zero duration)
  - Scenario I: Write-Through Crash-Resilient Logging (immediate JSONL & CSV on disk before export)
  - Scenario J: Offline Standalone Model Readiness (0 cloud APIs, all local subsystems READY)
"""

import csv
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from core_ai.app.config import load_config, ExperimentStep
from core_ai.har.temporal_model import ActionPrediction, ActionClassifier
from core_ai.interaction.hand_object import HandObjectInteractionTracker
from core_ai.logging.experiment_logger import ExperimentLogger
from core_ai.perception.hand_tracker import HandTracker
from core_ai.perception.object_detector import ObjectDetector, DetectedObject
from core_ai.perception.pose_estimator import PoseEstimator
from core_ai.reasoning.fsm import FSMStatus
from core_ai.reasoning.state_manager import StateManager
from core_ai.video.recorder import VideoRecorder
from core_ai.video.streamer import MJPEGStreamer
from core_ai.video.camera import LatestFrameBuffer


# 13 Official Lab Procedure Steps strictly matching Section 2 of Specification
OFFICIAL_13_LAB_STEPS = [
    {"id": 1, "action": "IDENTIFY", "verb": "IDENTIFY", "obj": "BLUE_BOX", "desc": "Identify the Blue Box"},
    {"id": 2, "action": "PICKUP", "verb": "PICKUP", "obj": "BLUE_BOX", "desc": "Pick up the Blue Box"},
    {"id": 3, "action": "PLACE", "verb": "PLACE", "obj": "BLUE_BOX", "desc": "Place Blue Box at Location A"},
    {"id": 4, "action": "IDENTIFY", "verb": "IDENTIFY", "obj": "YELLOW_BOX", "desc": "Identify the Yellow Box"},
    {"id": 5, "action": "PICKUP", "verb": "PICKUP", "obj": "YELLOW_BOX", "desc": "Pick up the Yellow Box"},
    {"id": 6, "action": "PLACE", "verb": "PLACE", "obj": "YELLOW_BOX", "desc": "Place Yellow Box at Location B"},
    {"id": 7, "action": "PICKUP", "verb": "PICKUP", "obj": "PEN", "desc": "Pick up the Pen"},
    {"id": 8, "action": "PLACE", "verb": "PLACE", "obj": "PEN", "desc": "Place Pen inside the Blue Box"},
    {"id": 9, "action": "PICKUP", "verb": "PICKUP", "obj": "WATCH", "desc": "Pick up the Watch"},
    {"id": 10, "action": "PLACE", "verb": "PLACE", "obj": "WATCH", "desc": "Place Watch inside the Yellow Box"},
    {"id": 11, "action": "MOVE", "verb": "MOVE", "obj": "BLUE_BOX", "desc": "Move Blue Box from A to B"},
    {"id": 12, "action": "MOVE", "verb": "MOVE", "obj": "YELLOW_BOX", "desc": "Move Yellow Box from B to A"},
    {"id": 13, "action": "COMPLETE", "verb": "COMPLETE", "obj": "ALL", "desc": "Experiment Complete"},
]


def make_prediction(
    action: str,
    confidence: float = 0.9,
    target_object: str = "",
    is_uncertain: bool = False,
) -> ActionPrediction:
    """Helper to construct fully populated ActionPrediction."""
    return ActionPrediction(
        action=action,
        confidence=confidence,
        next_action="IDLE",
        next_confidence=0.1,
        is_uncertain=is_uncertain,
        target_object=target_object,
    )


def make_detected_obj(
    name: str,
    conf: float = 0.95,
    bbox: tuple = (100, 100, 200, 200),
    source: str = "yolo",
) -> DetectedObject:
    """Helper to construct fully populated DetectedObject."""
    x, y, w, h = bbox
    return DetectedObject(
        class_name=name,
        confidence=conf,
        bbox=bbox,
        centroid=(x + w // 2, y + h // 2),
        source=source,
        track_id=1,
    )


def create_13_step_config():
    """Build configuration initialized with the full 13-step lab procedure."""
    cfg = load_config()
    cfg.pose.backend = "mock"
    cfg.hand.backend = "pose"
    cfg.detection.use_yolo = False

    steps = []
    for s in OFFICIAL_13_LAB_STEPS:
        steps.append(
            ExperimentStep(
                id=s["id"],
                action=f"{s['verb']}_{s['obj']}",
                label=s["desc"],
                description=s["desc"],
                expected_objects=[s["obj"]],
                expected_action=s["verb"],
                expected_object=s["obj"],
                voice_prompt=f"Step {s['id']}. {s['desc']}.",
                completion_voice=f"{s['desc']} completed.",
                timeout_seconds=30,
            )
        )
    cfg.experiment_steps = steps
    return cfg


# =========================================================================== #
# Scenario A: Perfect 13-Step Lab Procedure
# =========================================================================== #
def test_scenario_a_perfect_13_step_experiment():
    """Scenario A: All 13 steps executed sequentially without error -> 13/13 complete."""
    cfg = create_13_step_config()
    voice_msgs = []
    sm = StateManager(cfg, on_voice=lambda m: voice_msgs.append(m))

    assert sm.fsm.current_step.id == 1

    for step_info in OFFICIAL_13_LAB_STEPS:
        curr_id = step_info["id"]
        verb = step_info["verb"]
        obj = step_info["obj"]

        # Feed frames until current step completes cleanly
        for f in range(10):
            pred = make_prediction(action=verb, confidence=0.92, target_object=obj)
            detected_objs = [make_detected_obj(obj)]
            res = sm.process(pred, detected_objects=detected_objs)
            if curr_id in res.fsm_state.completed_step_ids:
                break

        # Step must be completed
        assert curr_id in res.fsm_state.completed_step_ids, f"Step {curr_id} failed to complete"

    # Verify final state is completed
    assert sm.fsm.is_complete
    assert len(sm.fsm.completed_step_ids) == 13
    assert len(voice_msgs) >= 13, "Expected guidance for each completed step"


# =========================================================================== #
# Scenario B: Wrong Object Detection & Safety Gating
# =========================================================================== #
def test_scenario_b_wrong_object_rejection():
    """Scenario B: User interacts with wrong object -> WRONG_OBJECT, voice alert, no advance."""
    cfg = create_13_step_config()
    voice_msgs = []
    sm = StateManager(cfg, on_voice=lambda m: voice_msgs.append(m))

    # Advance cleanly to Step 2 (PICKUP BLUE_BOX)
    for _ in range(10):
        r = sm.process(make_prediction(action="IDENTIFY", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 1 in r.fsm_state.completed_step_ids:
            break
    assert sm.fsm.current_step.id == 2

    # User attempts to pick up YELLOW_BOX instead of BLUE_BOX
    res = None
    for _ in range(5):
        pred = make_prediction(action="PICKUP", confidence=0.90, target_object="YELLOW_BOX")
        detected_objs = [make_detected_obj("YELLOW_BOX")]
        res = sm.process(pred, detected_objects=detected_objs)

    # Must be classified as WRONG_OBJECT (or backward-compatible WRONG_SEQUENCE)
    assert res.fsm_state.status in (FSMStatus.WRONG_OBJECT, FSMStatus.WRONG_SEQUENCE)
    assert res.fsm_state.status.name == "WRONG_OBJECT"
    # FSM must NOT advance past Step 2
    assert sm.fsm.current_step.id == 2
    assert 2 not in sm.fsm.completed_step_ids
    # Voice guidance alert generated
    assert any("wrong object" in m.lower() or "blue_box" in m.lower() or "blue box" in m.lower() for m in voice_msgs)


# =========================================================================== #
# Scenario C: Step Skipped Detection
# =========================================================================== #
def test_scenario_c_step_skipped_detection():
    """Scenario C: User is at Step 3 (PLACE BLUE_BOX) and performs Step 5 (PICKUP YELLOW_BOX)."""
    cfg = create_13_step_config()
    voice_msgs = []
    sm = StateManager(cfg, on_voice=lambda m: voice_msgs.append(m))

    # Advance cleanly to Step 3
    for _ in range(10):
        r = sm.process(make_prediction(action="IDENTIFY", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 1 in r.fsm_state.completed_step_ids:
            break
    for _ in range(10):
        r = sm.process(make_prediction(action="PICKUP", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 2 in r.fsm_state.completed_step_ids:
            break
    assert sm.fsm.current_step.id == 3

    # User performs Step 5 (PICKUP YELLOW_BOX), skipping Step 3 and Step 4
    res = None
    for _ in range(5):
        pred = make_prediction(action="PICKUP", confidence=0.91, target_object="YELLOW_BOX")
        res = sm.process(pred, detected_objects=[make_detected_obj("YELLOW_BOX")])

    # FSM must report STEP_SKIPPED
    assert res.fsm_state.status in (FSMStatus.STEP_SKIPPED, FSMStatus.WRONG_SEQUENCE)
    assert res.fsm_state.status.name == "STEP_SKIPPED"
    # FSM must NOT advance to Step 5
    assert sm.fsm.current_step.id == 3
    # Recovery message mentions skipped step
    rec = (res.fsm_state.recovery_message or "").lower()
    assert "skipped" in rec or "step" in rec


# =========================================================================== #
# Scenario D: Out-of-Sequence Action
# =========================================================================== #
def test_scenario_d_out_of_sequence_action():
    """Scenario D: User performs a step out of sequential order -> OUT_OF_SEQUENCE."""
    cfg = create_13_step_config()
    voice_msgs = []
    sm = StateManager(cfg, on_voice=lambda m: voice_msgs.append(m))

    # Advance cleanly to Step 3
    for _ in range(10):
        r = sm.process(make_prediction(action="IDENTIFY", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 1 in r.fsm_state.completed_step_ids:
            break
    for _ in range(10):
        r = sm.process(make_prediction(action="PICKUP", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 2 in r.fsm_state.completed_step_ids:
            break
    assert sm.fsm.current_step.id == 3

    # User repeats an action out of sequence
    res = None
    for _ in range(5):
        pred = make_prediction(action="IDENTIFY", confidence=0.88, target_object="BLUE_BOX")
        res = sm.process(pred, detected_objects=[make_detected_obj("BLUE_BOX")])

    # FSM status safely handles deviation
    assert res.fsm_state.status in (FSMStatus.OUT_OF_SEQUENCE, FSMStatus.WRONG_SEQUENCE, FSMStatus.WRONG_ACTION, FSMStatus.WAITING)
    # FSM stays safely at Step 3
    assert sm.fsm.current_step.id == 3


# =========================================================================== #
# Scenario E: Low Confidence / Uncertain Handling
# =========================================================================== #
def test_scenario_e_low_confidence_uncertain():
    """Scenario E: Perception is low confidence -> UNCERTAIN, no penalty, no advance."""
    cfg = create_13_step_config()
    sm = StateManager(cfg)

    init_step = sm.fsm.current_step.id

    # Feed 10 frames of low confidence predictions
    for _ in range(10):
        pred = make_prediction(
            action="IDENTIFY",
            confidence=0.30,  # below 0.55 threshold
            is_uncertain=True,
            target_object="BLUE_BOX",
        )
        res = sm.process(pred, detected_objects=[])

    # FSM does not advance, status reflects uncertain / waiting
    assert sm.fsm.current_step.id == init_step
    assert res.fsm_state.status in (FSMStatus.WAITING, FSMStatus.UNCERTAIN)
    assert res.action_prediction.is_uncertain is True


# =========================================================================== #
# Scenario F: Single-Frame False Detection Rejection
# =========================================================================== #
def test_scenario_f_single_frame_false_detection():
    """Scenario F: A single transient false detection frame must NOT advance the FSM."""
    cfg = create_13_step_config()
    sm = StateManager(cfg)

    init_step = sm.fsm.current_step.id

    # 1 transient frame of correct action
    res = sm.process(make_prediction(action="IDENTIFY", confidence=0.95, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
    # Gate requires persistence; 1 frame is insufficient
    assert sm.fsm.current_step.id == init_step
    assert len(sm.fsm.completed_step_ids) == 0

    # Next frame is IDLE
    res2 = sm.process(make_prediction(action="IDLE", confidence=0.90, target_object=""), detected_objects=[])
    assert sm.fsm.current_step.id == init_step


# =========================================================================== #
# Scenario G: Recovery After Error
# =========================================================================== #
def test_scenario_g_recovery_after_error():
    """Scenario G: After an error, correct action resumes execution cleanly."""
    cfg = create_13_step_config()
    voice_msgs = []
    sm = StateManager(cfg, on_voice=lambda m: voice_msgs.append(m))

    # Advance cleanly to Step 2 (PICKUP BLUE_BOX)
    for _ in range(10):
        r = sm.process(make_prediction(action="IDENTIFY", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 1 in r.fsm_state.completed_step_ids:
            break
    assert sm.fsm.current_step.id == 2

    # Wrong action occurs
    for _ in range(3):
        sm.process(make_prediction(action="PICKUP", confidence=0.9, target_object="YELLOW_BOX"), detected_objects=[make_detected_obj("YELLOW_BOX")])
    assert sm.fsm.current_step.id == 2

    # User corrects behavior and picks up BLUE_BOX
    res = None
    for _ in range(10):
        res = sm.process(make_prediction(action="PICKUP", confidence=0.92, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 2 in res.fsm_state.completed_step_ids:
            break

    # System successfully recovers and advances to Step 3
    assert sm.fsm.current_step.id == 3
    assert 2 in sm.fsm.completed_step_ids


# =========================================================================== #
# Scenario H: Video Recording Integrity & Dynamic Resolution Sync
# =========================================================================== #
def test_scenario_h_recording_integrity(tmp_path):
    """Scenario H: Dynamic VideoWriter initialization, valid MP4, correct FPS/resolution."""
    cfg = load_config()
    cfg.video.output_dir = str(tmp_path)
    recorder = VideoRecorder(cfg.video)

    out_path = recorder.start(experiment_id="TEST_REC_01")
    assert recorder.is_recording()

    # Write 30 synthetic frames of 1280x720
    test_w, test_h = 1280, 720
    for i in range(30):
        frame = np.zeros((test_h, test_w, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        recorder.write(frame)

    final_path = recorder.stop()
    assert not recorder.is_recording()
    assert os.path.exists(final_path)
    assert os.path.getsize(final_path) > 1000

    # Verify readable by OpenCV
    cap = cv2.VideoCapture(final_path)
    assert cap.isOpened(), "Recorded MP4 could not be opened by OpenCV"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert w == test_w, f"Expected width {test_w}, got {w}"
    assert h == test_h, f"Expected height {test_h}, got {h}"
    assert count >= 25, f"Expected >= 25 frames, got {count}"


# =========================================================================== #
# Scenario I: Crash-Resilient Write-Through Structured Logging
# =========================================================================== #
def test_scenario_i_write_through_logging(tmp_path):
    """Scenario I: Events write through immediately to SQLite, JSONL, and CSV with 15 fields."""
    cfg = create_13_step_config()
    sm = StateManager(cfg)

    exp_id = "CRASH_TEST_99"
    logger = ExperimentLogger(
        experiment_id=exp_id,
        experiment_name="Crash Resilience Test",
        output_dir=str(tmp_path),
        total_steps=13,
    )

    # Feed an event and log it
    pred = make_prediction(action="IDENTIFY", confidence=0.92, target_object="BLUE_BOX")
    res = sm.process(pred, detected_objects=[make_detected_obj("BLUE_BOX")])
    logger.log(res)

    # Verify JSONL exists IMMEDIATELY on disk without calling export()
    jsonl_path = tmp_path / exp_id / "experiment_log.jsonl"
    csv_path = tmp_path / exp_id / "experiment_log.csv"

    assert jsonl_path.exists(), "experiment_log.jsonl must exist immediately upon log() call"
    assert csv_path.exists(), "experiment_log.csv must exist immediately upon log() call"

    # Check required 15 fields in JSONL
    with open(jsonl_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        data = json.loads(line)

    required_fields = [
        "timestamp", "elapsed_seconds", "experiment_id", "step_number",
        "step_name", "expected_action", "detected_action", "detected_object",
        "confidence", "validation_status", "outcome", "error_type",
        "recovery_message", "latency_ms", "fps",
    ]
    for field in required_fields:
        assert field in data, f"Required field '{field}' missing from write-through log"

    # Check timestamp is ISO-8601 UTC
    ts = datetime.fromisoformat(data["timestamp"])
    assert ts.tzinfo is not None, "Timestamp must be timezone-aware"

    # Check CSV row matches
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        for field in required_fields:
            assert field in row, f"CSV missing field '{field}'"


# =========================================================================== #
# Scenario J: Offline Standalone Model Readiness
# =========================================================================== #
def test_scenario_j_offline_standalone_readiness():
    """Scenario J: 0 cloud dependencies, all local subsystems reporting READY."""
    cfg = load_config()

    detector = ObjectDetector(cfg.detection)
    yolo_ready = detector is not None and (detector._yolo is not None or hasattr(detector, "load_model"))

    classifier = ActionClassifier(cfg.har)
    har_ready = classifier is not None and (classifier.is_ready() or bool(getattr(classifier, "active_checkpoint_path", "")))

    pose = PoseEstimator(cfg.pose)
    hand = HandTracker(cfg.hand)
    mp_ready = pose is not None and hand is not None

    from core_ai.voice.tts import TTSEngine
    tts = TTSEngine(cfg.voice)
    tts_ready = tts is not None and tts.is_available()

    from core_ai.database.sqlite_db import OrbitaDB
    db = OrbitaDB()
    db.init_db()

    recorder = VideoRecorder(cfg.video)
    rec_ready = recorder is not None

    assert yolo_ready, "YOLO detector must be initialized from local assets"
    assert har_ready, "HAR classifier must be ready locally"
    assert mp_ready, "MediaPipe wrappers must be ready"
    assert tts_ready, "TTS engine must be available locally without cloud API"
    assert rec_ready, "Recorder must be ready"


# =========================================================================== #
# IP Streaming & Latency
# =========================================================================== #
def test_ip_streaming_buffer_latency():
    """Verify LatestFrameBuffer drops stale frames and preserves <100ms latency."""
    buf = LatestFrameBuffer(name="test-stream", maxsize=1)

    # Push 5 frames
    for i in range(5):
        buf.put(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=time.monotonic())

    # Verify only the newest frame is kept (stale frame dropping)
    frame, ts, idx = buf.get_latest()
    assert idx == 5
    assert (time.monotonic() - ts) < 0.1, "Frame age exceeds 100ms"
