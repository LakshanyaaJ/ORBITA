"""
Tests for ORBITA Multi-Object YOLO Vision Pipeline, Hand Tracking Integration,
Action Validation Gate, and Real-Time Debug Overlay.
"""

from __future__ import annotations

import io
import sys
import numpy as np
import pytest

from core_ai.app.config import DetectionConfig, OrbitaConfig, load_config
from core_ai.perception.hand_tracker import HandState
from core_ai.perception.object_detector import (
    DEFAULT_CLASS_THRESHOLDS,
    REQUIRED_YOLO_CLASSES,
    DetectedObject,
    NormalizedClassName,
    ObjectDetector,
    apply_class_aware_nms,
    audit_model_classes,
    draw_debug_overlay,
)
from core_ai.reasoning.action_gate import ActionConfirmationGate, ActionGateStatus
from core_ai.reasoning.fsm import ExperimentFSM, FSMStatus
from core_ai.reasoning.state_manager import StateManager
from core_ai.har.temporal_model import ActionPrediction


class TestModelAuditAndNormalization:
    """Test YOLO model class inspection and reporting."""

    def test_audit_model_classes_reports_missing(self, capsys):
        # Simulate model.names from orbita_yolo_detector_v3.pt
        v3_names = {0: "PERSON", 1: "MAIN_BOX", 2: "RED_BOX", 3: "YELLOW_BOX", 4: "SAMPLE", 5: "TOOL"}
        missing = audit_model_classes(v3_names)
        captured = capsys.readouterr().out

        assert "MODEL CLASSES:" in captured
        assert "0 -> PERSON" in captured
        assert "3 -> YELLOW_BOX" in captured
        assert "MODEL DOES NOT CONTAIN REQUIRED CLASS: HAND" in captured

        # When a model is missing classes, audit_model_classes reports each one
        sparse_names = {0: "PERSON", 1: "OTHER"}
        sparse_missing = audit_model_classes(sparse_names)
        sparse_captured = capsys.readouterr().out
        for req in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH", "HAND"]:
            assert f"MODEL DOES NOT CONTAIN REQUIRED CLASS: {req}" in sparse_captured
            assert req in sparse_missing

    def test_normalized_class_name_bidirectional_equality(self):
        c1 = NormalizedClassName("yellow_box")
        assert c1 == "YELLOW_BOX"
        assert c1 == "yellow_box"

        c2 = NormalizedClassName("MAIN_BOX")
        assert c2 == "blue_box"
        assert c2 == "BLUE_BOX"

        c3 = NormalizedClassName("TOOL")
        assert c3 == "pen"
        assert c3 == "PEN"

        c4 = NormalizedClassName("SAMPLE")
        assert c4 == "watch"
        assert c4 == "WATCH"

        c5 = NormalizedClassName("HAND")
        assert c5 == "hand"


class TestMultiObjectDetectionAndGrouping:
    """Test multi-class extraction and schema output."""

    def test_detected_object_dictionary_schema(self):
        det = DetectedObject(
            class_name="yellow_box",
            confidence=0.92,
            bbox=(100, 120, 80, 60),
            centroid=(140, 150),
            timestamp=10.5,
            source="yolo",
        )
        d = det.to_detection_dict()

        assert d["class_name"] == "yellow_box"
        assert d["confidence"] == 0.92
        assert d["bounding_box"] == [100, 120, 180, 180]  # [x1, y1, x2, y2]
        assert d["center_x"] == 140
        assert d["center_y"] == 150
        assert d["frame_timestamp"] == 10.5

    def test_simultaneous_five_class_extraction_no_single_winner(self):
        cfg = DetectionConfig(use_yolo=False)
        detector = ObjectDetector(cfg)

        # Create live scene with all 5 required classes simultaneously
        yolo_dets = [
            DetectedObject(class_name="yellow_box", confidence=0.95, bbox=(500, 100, 100, 100), centroid=(550, 150), source="yolo"),
            DetectedObject(class_name="blue_box", confidence=0.88, bbox=(100, 100, 120, 120), centroid=(160, 160), source="yolo"),
            DetectedObject(class_name="pen", confidence=0.78, bbox=(300, 200, 20, 80), centroid=(310, 240), source="yolo"),
            DetectedObject(class_name="watch", confidence=0.82, bbox=(420, 200, 40, 40), centroid=(440, 220), source="yolo"),
        ]
        detector._detect_yolo = lambda frame, ts: yolo_dets

        # Hand detection from hand tracker
        hand_left = HandState(
            hand_id=1,
            side="left",
            confidence=0.89,
            bbox=(280, 180, 60, 60),
            position=np.array([310, 210]),
            velocity=np.array([0.0, 0.0]),
            speed=0.0,
            is_visible=True,
            grasp_confidence=0.75,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame, timestamp=1.0, hands=(hand_left, None))

        # Check grouped output
        grouped = detector.get_grouped_detections()
        for req_cls in REQUIRED_YOLO_CLASSES:
            assert req_cls in grouped, f"Missing key {req_cls} in grouped detections"
            assert len(grouped[req_cls]) >= 1, f"Class {req_cls} should have survived extraction"

        assert grouped["yellow_box"][0]["confidence"] == 0.95
        assert grouped["blue_box"][0]["confidence"] == 0.88
        assert grouped["pen"][0]["confidence"] == 0.78
        assert grouped["watch"][0]["confidence"] == 0.82
        assert grouped["hand"][0]["confidence"] == 0.89

        # Ensure no single-winner bias (yellow box does NOT eliminate pen or watch or hand)
        result_classes = {str(o.class_name).lower() for o in results}
        assert "yellow_box" in result_classes
        assert "blue_box" in result_classes
        assert "pen" in result_classes
        assert "watch" in result_classes
        assert "hand" in result_classes

    def test_class_aware_nms_preserves_different_overlapping_classes(self):
        # Hand holding pen: Bounding boxes overlap heavily
        dets = [
            DetectedObject(class_name="hand", confidence=0.85, bbox=(200, 200, 80, 80), centroid=(240, 240), source="hand_tracker"),
            DetectedObject(class_name="pen", confidence=0.80, bbox=(210, 210, 30, 60), centroid=(225, 240), source="yolo"),
            # Duplicate duplicate hand that should be suppressed
            DetectedObject(class_name="hand", confidence=0.50, bbox=(202, 198, 78, 82), centroid=(241, 239), source="hand_tracker"),
        ]

        kept = apply_class_aware_nms(dets, iou_threshold=0.45)
        assert len(kept) == 2, f"Expected 2 detections (1 hand, 1 pen), got {len(kept)}"
        classes = [str(o.class_name).lower() for o in kept]
        assert "hand" in classes
        assert "pen" in classes
        # Higher confidence hand survived
        hand_det = [o for o in kept if str(o.class_name).lower() == "hand"][0]
        assert hand_det.confidence == 0.85


class TestShortTermTemporalSmoothing:
    """Test 1–3 frames persistence without flickering."""

    def test_object_missed_for_one_to_three_frames_persists(self):
        cfg = DetectionConfig(use_yolo=False)
        detector = ObjectDetector(cfg)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1: Pen detected
        pen = [DetectedObject(class_name="pen", confidence=0.85, bbox=(200, 200, 30, 80), centroid=(215, 240), source="yolo")]
        detector._detect_yolo = lambda f, ts: pen
        res1 = detector.detect(frame, timestamp=1.0)
        assert any(str(o.class_name).lower() == "pen" for o in res1)

        # Frame 2: Missed (frame 1 missed) -> MUST persist
        detector._detect_yolo = lambda f, ts: []
        res2 = detector.detect(frame, timestamp=1.033)
        assert any(str(o.class_name).lower() == "pen" for o in res2), "Pen should persist on 1st missed frame"

        # Frame 3: Missed (frame 2 missed) -> MUST persist
        res3 = detector.detect(frame, timestamp=1.066)
        assert any(str(o.class_name).lower() == "pen" for o in res3), "Pen should persist on 2nd missed frame"

        # Frame 4: Missed (frame 3 missed) -> MUST persist
        res4 = detector.detect(frame, timestamp=1.100)
        assert any(str(o.class_name).lower() == "pen" for o in res4), "Pen should persist on 3rd missed frame"

        # Frame 5: Missed (frame 4 missed) -> Removed
        res5 = detector.detect(frame, timestamp=1.133)
        assert not any(str(o.class_name).lower() == "pen" for o in res5), "Pen should be dropped after > 3 missed frames"


class TestActionValidationGateAndProtocolAdvancement:
    """Test that protocol timeline never advances from YOLO detection alone."""

    def test_yolo_detection_alone_does_not_advance_pickup_pen(self):
        gate = ActionConfirmationGate()
        step_pickup_pen = {
            "id": 7,
            "action": "PICKUP_PEN",
            "expected_action": "PICKUP",
            "expected_object": "PEN",
        }
        gate.reset_for_step(step_pickup_pen)

        pen_det = DetectedObject(class_name="pen", confidence=0.90, bbox=(300, 200, 20, 80), centroid=(310, 240), source="yolo")

        # Merely detecting PEN without hand or motion must remain WAITING
        for _ in range(10):
            res = gate.evaluate(
                current_step=step_pickup_pen,
                detected_objects=[pen_det],
                tracks=[pen_det],
                interactions=[],
                har_action="IDLE",
                har_confidence=0.3,
            )
            assert res.status == ActionGateStatus.WAITING
            assert res.validation_state in ("OBJECT_DETECTED", "WAITING")

    def test_hand_near_pen_without_lift_does_not_advance(self):
        gate = ActionConfirmationGate()
        step_pickup_pen = {
            "id": 7,
            "action": "PICKUP_PEN",
            "expected_action": "PICKUP",
            "expected_object": "PEN",
        }
        gate.reset_for_step(step_pickup_pen)

        pen_det = DetectedObject(class_name="pen", confidence=0.90, bbox=(300, 200, 20, 80), centroid=(310, 240), source="yolo")

        # Hand near pen (distance 50px) but no displacement/lift
        class MockInteraction:
            object_class = "PEN"
            distance_px = 45.0
            hand_object_overlap = 0.1
            relative_motion = 0.0
            is_holding = False
            is_pointing = False
            hand_speed = 5.0
            object_speed = 0.0
            contact_duration = 1
            confidence = 0.75
            hand_confidence = 0.85
            state = 1  # NEAR

        for _ in range(5):
            res = gate.evaluate(
                current_step=step_pickup_pen,
                detected_objects=[pen_det],
                tracks=[pen_det],
                interactions=[MockInteraction()],
                har_action="IDLE",
            )
            assert res.status == ActionGateStatus.WAITING
            assert res.validation_state in ("HAND_NEAR_OBJECT", "HAND_INTERACTING")

    def test_valid_hand_motion_coupling_advances_pickup_pen(self):
        gate = ActionConfirmationGate()
        step_pickup_pen = {
            "id": 7,
            "action": "PICKUP_PEN",
            "expected_action": "PICKUP",
            "expected_object": "PEN",
        }
        gate.reset_for_step(step_pickup_pen)

        # Pen initial position
        pen_init = DetectedObject(class_name="pen", confidence=0.90, bbox=(300, 200, 20, 80), centroid=(310, 240), source="yolo")
        gate.evaluate(step_pickup_pen, [pen_init], [pen_init], [])

        # Pen lifted 60px with hand holding it
        pen_lifted = DetectedObject(class_name="pen", confidence=0.92, bbox=(300, 140, 20, 80), centroid=(310, 180), source="yolo")

        class MockHoldInteraction:
            object_class = "PEN"
            distance_px = 10.0
            hand_object_overlap = 0.8
            relative_motion = 0.6
            is_holding = True
            is_pointing = False
            hand_speed = 80.0
            object_speed = 75.0
            contact_duration = 5
            confidence = 0.90
            hand_confidence = 0.92
            state = 2  # HOLDING

        # Evaluate across consecutive confirmation frames
        confirmed = False
        for _ in range(5):
            res = gate.evaluate(
                current_step=step_pickup_pen,
                detected_objects=[pen_lifted],
                tracks=[pen_lifted],
                interactions=[MockHoldInteraction()],
                har_action="TAKE",
                har_confidence=0.85,
            )
            if res.status == ActionGateStatus.CONFIRMED_CORRECT:
                confirmed = True
                break

        assert confirmed, "Pickup pen should be confirmed when hand holds and lifts pen"

    def test_place_pen_inside_yellow_box_requires_box_and_release(self):
        gate = ActionConfirmationGate()
        step_place_pen = {
            "id": 8,
            "action": "PLACE_PEN_YELLOW_BOX",
            "expected_action": "PLACE",
            "expected_object": "PEN",
            "expected_target": "YELLOW_BOX",
        }
        gate.reset_for_step(step_place_pen)

        # Yellow Box at (400, 300, 120, 120)
        yellow_box = DetectedObject(class_name="yellow_box", confidence=0.92, bbox=(400, 300, 120, 120), centroid=(460, 360), source="yolo")
        # Pen placed inside yellow box centroid (450, 350)
        pen_in_box = DetectedObject(class_name="pen", confidence=0.88, bbox=(440, 320, 20, 60), centroid=(450, 350), source="yolo")

        # 1. When hand is still holding the pen, placement is NOT complete
        class MockHoldingInBox:
            object_class = "PEN"
            distance_px = 10.0
            is_holding = True
            is_pointing = False
            object_speed = 0.0
            hand_speed = 0.0
            confidence = 0.85

        for _ in range(4):
            res = gate.evaluate(
                current_step=step_place_pen,
                detected_objects=[yellow_box, pen_in_box],
                tracks=[yellow_box, pen_in_box],
                interactions=[MockHoldingInBox()],
                har_action="PLACE",
            )
            assert res.status == ActionGateStatus.WAITING

        # 2. Hand releases pen inside yellow box and moves away
        class MockReleasedInBox:
            object_class = "PEN"
            distance_px = 75.0  # Hand moved away
            is_holding = False
            is_pointing = False
            object_speed = 0.0
            hand_speed = 40.0
            confidence = 0.85

        confirmed = False
        for _ in range(5):
            res = gate.evaluate(
                current_step=step_place_pen,
                detected_objects=[yellow_box, pen_in_box],
                tracks=[yellow_box, pen_in_box],
                interactions=[MockReleasedInBox()],
                har_action="PLACE",
                har_confidence=0.80,
            )
            if res.status == ActionGateStatus.CONFIRMED_CORRECT:
                confirmed = True
                break

        assert confirmed, "Placing pen inside yellow box must be confirmed after hand releases"


class TestDebugOverlayAndTelemetry:
    """Test debug overlay rendering and telemetry payload."""

    def test_draw_debug_overlay_renders_properly(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        grouped = {
            "yellow_box": [{"class_name": "yellow_box", "confidence": 0.94, "bounding_box": [10, 10, 50, 50], "center_x": 30, "center_y": 30}],
            "blue_box": [{"class_name": "blue_box", "confidence": 0.89, "bounding_box": [60, 60, 100, 100], "center_x": 80, "center_y": 80}],
            "pen": [],
            "watch": [],
            "hand": [{"class_name": "hand", "confidence": 0.91, "bounding_box": [150, 150, 200, 200], "center_x": 175, "center_y": 175}],
        }
        action_state = {
            "current_step": "Step 1: Identify the Blue Box",
            "required_object": "BLUE_BOX",
            "hand_detected": "YES",
            "object_detected": "YES",
            "hand_object_distance": "42px",
            "spatial_relationship": "NEAR",
            "motion_detected": "YES",
            "validation_progress": "80%",
            "step_complete": "NO",
        }

        annotated = draw_debug_overlay(frame, grouped, action_state)
        assert annotated is not None
        assert annotated.shape == (480, 640, 3)

    def test_state_manager_action_state_telemetry(self):
        cfg = load_config()
        sm = StateManager(cfg)

        pred = ActionPrediction(action="IDLE", confidence=0.5, next_action="IDLE", next_confidence=0.3, is_uncertain=False, target_object="")
        blue_box = DetectedObject(class_name="blue_box", confidence=0.92, bbox=(100, 100, 80, 80), centroid=(140, 140), source="yolo")

        res = sm.process(prediction=pred, detected_objects=[blue_box], fps=15.0, latency_ms=12.0)
        d = res.to_dict()

        assert "yolo_detections_summary" in d
        assert "yellow_box" in d["yolo_detections_summary"]
        assert "blue_box" in d["yolo_detections_summary"]
        assert d["yolo_detections_summary"]["blue_box"]["detected"] is True

        assert "action_state" in d
        ast = d["action_state"]
        assert "current_step" in ast
        assert "required_object" in ast
        assert "hand_detected" in ast
        assert "object_detected" in ast
        assert "hand_object_distance" in ast
        assert "spatial_relationship" in ast
        assert "motion_detected" in ast
        assert "validation_progress" in ast
        assert "step_complete" in ast
