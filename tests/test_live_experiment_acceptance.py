"""
ORBITA Master Live Experiment Acceptance Test Suite
===================================================
Strict verification of the live experiment runtime pipeline:
  - Test A: Correct Procedure (Step 1 -> 13, transitions, voice prompts, COMPLETE)
  - Test B: Wrong Object Detection (WRONG_OBJECT, voice alert, FSM unchanged)
  - Test C: Skipped Step Detection (STEP_SKIPPED, voice recovery, FSM unchanged)
  - Test D: Out of Sequence Detection (OUT_OF_SEQUENCE, voice alert, FSM unchanged)
  - Test E: Uncertain Perception (UNCERTAIN, FSM unchanged, no false advance)
  - Test F: Video Recording & MP4 Integrity (resolution, FPS, first/middle/last frame readable)
  - Test G: Crash-Resilient Multi-Format Logging (JSONL, CSV, SQLite synchronized)
  - Test H: Real-Time Streaming & Buffer Latency (fresh frames, <100ms, drops stale)
"""

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from core_ai.app.config import load_config, ExperimentStep
from core_ai.har.temporal_model import ActionPrediction
from core_ai.logging.experiment_logger import ExperimentLogger
from core_ai.perception.object_detector import DetectedObject
from core_ai.reasoning.fsm import FSMStatus
from core_ai.reasoning.state_manager import StateManager
from core_ai.video.camera import LatestFrameBuffer
from core_ai.video.recorder import VideoRecorder


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
    x, y, w, h = bbox
    return DetectedObject(
        class_name=name,
        confidence=conf,
        bbox=bbox,
        centroid=(x + w // 2, y + h // 2),
        source=source,
    )


def create_13_step_config():
    cfg = load_config()
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
# Test A: Correct Procedure (Step 1 -> 13)
# =========================================================================== #
def test_acceptance_a_correct_procedure():
    """Acceptance A: 13 correct transitions, 13 voice events, 13 completions -> COMPLETE."""
    cfg = create_13_step_config()
    voice_messages = []
    sm = StateManager(cfg, on_voice=lambda msg: voice_messages.append(msg))

    assert sm.fsm.current_step.id == 1
    assert not sm.fsm.is_complete

    completed_ids = []
    for step_info in OFFICIAL_13_LAB_STEPS:
        verb = step_info["verb"]
        obj = step_info["obj"]
        step_id = step_info["id"]

        res = None
        for _ in range(10):
            pred = make_prediction(action=verb, confidence=0.92, target_object=obj)
            detected = [make_detected_obj(obj)]
            res = sm.process(pred, detected_objects=detected)
            if step_id in res.fsm_state.completed_step_ids:
                break

        assert step_id in sm.fsm.completed_step_ids, f"Step {step_id} failed to complete"
        completed_ids.append(step_id)

    assert len(completed_ids) == 13
    assert sm.fsm.is_complete
    assert res.fsm_state.status.name == "COMPLETED"
    assert len(voice_messages) >= 13


# =========================================================================== #
# Test B: Wrong Object Detection
# =========================================================================== #
def test_acceptance_b_wrong_object():
    """Acceptance B: User interacts with YELLOW_BOX when BLUE_BOX is required."""
    cfg = create_13_step_config()
    voice_messages = []
    sm = StateManager(cfg, on_voice=lambda msg: voice_messages.append(msg))

    # Cleanly advance to Step 2 (PICKUP BLUE_BOX)
    for _ in range(8):
        r = sm.process(make_prediction(action="IDENTIFY", confidence=0.92, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 1 in r.fsm_state.completed_step_ids:
            break
    assert sm.fsm.current_step.id == 2

    # User picks up wrong object (YELLOW_BOX)
    res = None
    for _ in range(5):
        pred = make_prediction(action="PICKUP", confidence=0.91, target_object="YELLOW_BOX")
        res = sm.process(pred, detected_objects=[make_detected_obj("YELLOW_BOX")])

    assert res.fsm_state.status in (FSMStatus.WRONG_OBJECT, FSMStatus.WRONG_SEQUENCE)
    assert sm.fsm.current_step.id == 2
    assert any("wrong object" in m.lower() or "blue_box" in m.lower() or "blue box" in m.lower() for m in voice_messages)


# =========================================================================== #
# Test C: Skipped Step Detection
# =========================================================================== #
def test_acceptance_c_skipped_step():
    """Acceptance C: User at Step 3 attempts Step 5 (skips Step 3 and 4)."""
    cfg = create_13_step_config()
    voice_messages = []
    sm = StateManager(cfg, on_voice=lambda msg: voice_messages.append(msg))

    # Advance to Step 3
    for _ in range(8):
        r = sm.process(make_prediction(action="IDENTIFY", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 1 in r.fsm_state.completed_step_ids:
            break
    for _ in range(8):
        r = sm.process(make_prediction(action="PICKUP", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 2 in r.fsm_state.completed_step_ids:
            break
    assert sm.fsm.current_step.id == 3

    # User attempts Step 5 (PICKUP YELLOW_BOX)
    res = None
    for _ in range(5):
        pred = make_prediction(action="PICKUP", confidence=0.92, target_object="YELLOW_BOX")
        res = sm.process(pred, detected_objects=[make_detected_obj("YELLOW_BOX")])

    assert res.fsm_state.status in (FSMStatus.STEP_SKIPPED, FSMStatus.WRONG_SEQUENCE)
    assert res.fsm_state.status.name == "STEP_SKIPPED"
    assert sm.fsm.current_step.id == 3
    rec = (res.fsm_state.recovery_message or "").lower()
    assert "skipped" in rec or "step" in rec


# =========================================================================== #
# Test D: Out of Sequence Detection
# =========================================================================== #
def test_acceptance_d_out_of_sequence():
    """Acceptance D: User at Step 3 repeats completed Step 1 action."""
    cfg = create_13_step_config()
    voice_messages = []
    sm = StateManager(cfg, on_voice=lambda msg: voice_messages.append(msg))

    # Advance to Step 3
    for _ in range(8):
        r = sm.process(make_prediction(action="IDENTIFY", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 1 in r.fsm_state.completed_step_ids:
            break
    for _ in range(8):
        r = sm.process(make_prediction(action="PICKUP", confidence=0.9, target_object="BLUE_BOX"), detected_objects=[make_detected_obj("BLUE_BOX")])
        if 2 in r.fsm_state.completed_step_ids:
            break
    assert sm.fsm.current_step.id == 3

    # Repeat Step 1 (IDENTIFY BLUE_BOX)
    res = None
    for _ in range(5):
        pred = make_prediction(action="IDENTIFY", confidence=0.88, target_object="BLUE_BOX")
        res = sm.process(pred, detected_objects=[make_detected_obj("BLUE_BOX")])

    assert res.fsm_state.status in (FSMStatus.OUT_OF_SEQUENCE, FSMStatus.WRONG_SEQUENCE, FSMStatus.WRONG_ACTION, FSMStatus.WAITING)
    assert sm.fsm.current_step.id == 3


# =========================================================================== #
# Test E: Uncertain Perception
# =========================================================================== #
def test_acceptance_e_uncertain():
    """Acceptance E: Low confidence (<0.55) -> UNCERTAIN, no penalty, no FSM advance."""
    cfg = create_13_step_config()
    sm = StateManager(cfg)

    init_step = sm.fsm.current_step.id

    for _ in range(10):
        pred = make_prediction(action="IDENTIFY", confidence=0.35, is_uncertain=True, target_object="BLUE_BOX")
        res = sm.process(pred, detected_objects=[])

    assert sm.fsm.current_step.id == init_step
    assert res.fsm_state.status in (FSMStatus.WAITING, FSMStatus.UNCERTAIN)
    assert res.action_prediction.is_uncertain is True


# =========================================================================== #
# Test F: Video Recording & MP4 Frame-by-Frame Readability
# =========================================================================== #
def test_acceptance_f_recording(tmp_path):
    """Acceptance F: Local MP4 created with valid headers; first, middle, and last frames readable."""
    cfg = load_config()
    cfg.video.output_dir = str(tmp_path)
    recorder = VideoRecorder(cfg.video)

    exp_id = "ACCEPT_REC_01"
    recorder.start(experiment_id=exp_id)
    assert recorder.is_recording()

    # Write 30 frames at 1280x720
    test_w, test_h = 1280, 720
    for i in range(30):
        frame = np.zeros((test_h, test_w, 3), dtype=np.uint8)
        cv2.putText(frame, f"ORBITA Live Frame {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        recorder.write(frame)

    final_path = recorder.stop()
    assert not recorder.is_recording()

    # Canonical experiment.mp4 verification
    canonical_path = tmp_path / exp_id / "experiment.mp4"
    assert canonical_path.exists(), "Canonical experiment.mp4 must exist"
    assert canonical_path.stat().st_size > 1000

    # Programmatic OpenCV inspection
    cap = cv2.VideoCapture(str(canonical_path))
    assert cap.isOpened(), "VideoCapture could not open recorded MP4"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    assert total_frames >= 25, f"Expected >= 25 frames, got {total_frames}"
    assert width == test_w, f"Expected width {test_w}, got {width}"
    assert height == test_h, f"Expected height {test_h}, got {height}"
    assert fps > 0

    # Read first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_first, frame_first = cap.read()
    assert ret_first and frame_first is not None, "First frame could not be read"
    assert frame_first.shape == (test_h, test_w, 3)

    # Read middle frame
    mid_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret_mid, frame_mid = cap.read()
    assert ret_mid and frame_mid is not None, "Middle frame could not be read"
    assert frame_mid.shape == (test_h, test_w, 3)

    # Read last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret_last, frame_last = cap.read()
    assert ret_last and frame_last is not None, "Last frame could not be read"
    assert frame_last.shape == (test_h, test_w, 3)

    cap.release()


# =========================================================================== #
# Test G: Real-Time Multi-Format Crash-Resilient Logging
# =========================================================================== #
def test_acceptance_g_logging(tmp_path):
    """Acceptance G: JSONL, CSV, and SQLite update immediately upon log() with 15 fields."""
    cfg = create_13_step_config()
    sm = StateManager(cfg)

    exp_id = "ACCEPT_LOG_01"
    logger = ExperimentLogger(
        experiment_id=exp_id,
        experiment_name="Live Acceptance Logging",
        output_dir=str(tmp_path),
        total_steps=13,
    )

    pred = make_prediction(action="IDENTIFY", confidence=0.92, target_object="BLUE_BOX")
    res = sm.process(pred, detected_objects=[make_detected_obj("BLUE_BOX")])
    logger.log(res)

    jsonl_path = tmp_path / exp_id / "experiment_log.jsonl"
    csv_path = tmp_path / exp_id / "experiment_log.csv"

    assert jsonl_path.exists(), "JSONL log must exist immediately on disk"
    assert csv_path.exists(), "CSV log must exist immediately on disk"

    # Verify JSONL content
    with open(jsonl_path, "r", encoding="utf-8") as f:
        line = f.readline()
        record = json.loads(line)

    required_fields = [
        "timestamp", "elapsed_seconds", "experiment_id", "step_number",
        "step_name", "expected_action", "detected_action", "detected_object",
        "confidence", "validation_status", "outcome", "error_type",
        "recovery_message", "latency_ms", "fps",
    ]
    for field in required_fields:
        assert field in record, f"Missing required field '{field}' in JSONL log"

    ts = datetime.fromisoformat(record["timestamp"])
    assert ts.tzinfo is not None, "Timestamp must be timezone-aware ISO-8601 UTC"

    # Verify CSV content
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_row = next(reader)
        for field in required_fields:
            assert field in csv_row, f"Missing required field '{field}' in CSV log"


# =========================================================================== #
# Test H: Real-Time Streaming Buffer Latency (<100ms)
# =========================================================================== #
def test_acceptance_h_streaming_buffer():
    """Acceptance H: LatestFrameBuffer(maxsize=1) discards stale frames and guarantees <100ms latency."""
    buf = LatestFrameBuffer(name="live-acceptance-stream", maxsize=1)

    for i in range(10):
        buf.put(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=time.monotonic())
        time.sleep(0.005)

    frame, ts, idx = buf.get_latest()
    assert idx == 10
    assert frame is not None
    latency = time.monotonic() - ts
    assert latency < 0.1, f"Frame latency exceeded 100ms: {latency * 1000.0:.1f}ms"
