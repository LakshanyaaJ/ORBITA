"""
Unit tests for ORBITA Geometric Zones, Multi-Object Tracking Robustness,
Hand-Object Hysteresis, and Debug Overlay (Sections 2, 7, 8, 9, 13).
"""

import time
import numpy as np
import pytest

from core_ai.app.config import ZoneDefinition, load_config
from core_ai.perception.object_tracker import MultiObjectTracker
from core_ai.perception.object_detector import DetectedObject, NormalizedClassName
from core_ai.interaction.hand_object import HandObjectInteractionManager, InteractionState, MANIPULABLE_CLASSES
from core_ai.perception.hand_tracker import HandState
from core_ai.perception.debug_visualizer import DebugVisualizer
from core_ai.reasoning.action_gate import ActionConfirmationGate, ActionGateStatus


def test_zone_definition_contains_point():
    """Verify ZoneDefinition correctly identifies inside/outside points."""
    # Normalized rectangle (left half of screen)
    zone = ZoneDefinition(
        name="LOCATION_A",
        zone_type="polygon",
        coordinates=[[0.10, 0.20], [0.40, 0.20], [0.40, 0.80], [0.10, 0.80]],
    )
    frame_w, frame_h = 1000, 1000

    # Inside point (0.25 * 1000, 0.50 * 1000)
    assert zone.contains_point((250, 500), frame_w, frame_h) is True

    # Outside points
    assert zone.contains_point((50, 500), frame_w, frame_h) is False   # Too far left
    assert zone.contains_point((500, 500), frame_w, frame_h) is False  # Too far right
    assert zone.contains_point((250, 100), frame_w, frame_h) is False  # Too high


def test_zone_definition_bbox_overlap():
    """Verify bbox overlap calculation."""
    zone = ZoneDefinition(
        name="LOCATION_B",
        zone_type="rectangle",
        coordinates=[0.60, 0.20, 0.90, 0.80],
    )
    frame_w, frame_h = 1000, 1000

    # Bounding box fully inside
    inside_bbox = (650, 300, 100, 100)
    assert zone.bbox_overlap(inside_bbox, frame_w, frame_h) > 0.5

    # Bounding box far away
    outside_bbox = (100, 100, 50, 50)
    assert zone.bbox_overlap(outside_bbox, frame_w, frame_h) == 0.0


def test_tracker_temporal_persistence():
    """Verify that an active track persists across brief occlusions (Section 7 & 8)."""
    tracker = MultiObjectTracker(max_age=15, min_hits=1, iou_threshold=0.25, object_lost_frames=5)

    # Frame 1: Object detected
    det1 = DetectedObject(
        class_name=NormalizedClassName("pen"),
        confidence=0.80,
        bbox=(200, 200, 50, 20),
        centroid=(225, 210),
        timestamp=1.0,
    )
    tracked_1 = tracker.update([det1], timestamp=1.0)
    assert len(tracked_1) == 1
    pen_id = tracked_1[0].track_id
    assert pen_id > 0

    # Frame 2: YOLO dropped the object (empty detection list)
    tracked_2 = tracker.update([], timestamp=1.033)
    assert len(tracked_2) == 1
    assert tracked_2[0].track_id == pen_id
    assert tracked_2[0].confidence < 0.80  # Decayed confidence
    assert tracked_2[0].confidence > 0.50

    # Frame 3: Object reappears
    det3 = DetectedObject(
        class_name=NormalizedClassName("pen"),
        confidence=0.85,
        bbox=(202, 201, 50, 20),
        centroid=(227, 211),
        timestamp=1.066,
    )
    tracked_3 = tracker.update([det3], timestamp=1.066)
    assert len(tracked_3) == 1
    assert tracked_3[0].track_id == pen_id  # Stable ID preserved!


def test_hand_object_manipulable_classes():
    """Verify that PEN and WATCH are recognized as manipulable lab items."""
    assert "PEN" in MANIPULABLE_CLASSES
    assert "WATCH" in MANIPULABLE_CLASSES
    assert "BLUE_BOX" in MANIPULABLE_CLASSES
    assert "YELLOW_BOX" in MANIPULABLE_CLASSES


def test_hand_object_holding_hysteresis():
    """Verify HOLDING state does not immediately oscillate on single-frame separation."""
    mgr = HandObjectInteractionManager()

    # Hand and pen at close proximity
    landmarks = np.zeros((21, 2), dtype=np.float32)
    landmarks[4] = [298.0, 302.0]
    landmarks[8] = [305.0, 302.0]
    hand = HandState(
        hand_id=1,
        is_visible=True,
        side="right",
        position=np.array([300.0, 300.0], dtype=np.float32),
        velocity=np.array([0.0, 0.0], dtype=np.float32),
        bbox=(280, 280, 60, 60),
        confidence=0.90,
        speed=0.0,
        grasp_confidence=0.85,
        finger_landmarks=landmarks,
    )
    pen = DetectedObject(
        class_name="PEN",
        confidence=0.80,
        bbox=(300, 300, 40, 15),
        centroid=(320, 307),
    )

    # Transition into CONTACT -> HOLDING over 5 frames
    for t in range(6):
        res = mgr.evaluate(hand, None, [pen], timestamp=float(t))

    # Should be in HOLDING
    primary = mgr.get_primary_interaction(res)
    assert primary is not None
    assert primary.state == InteractionState.HOLDING

    # 1 single frame of momentary jitter/separation (e.g. distance spikes)
    hand_jitter = HandState(
        hand_id=1,
        is_visible=True,
        side="right",
        position=np.array([380.0, 380.0], dtype=np.float32),
        velocity=np.array([10.0, 0.0], dtype=np.float32),
        bbox=(350, 350, 60, 60),
        confidence=0.85,
        speed=10.0,
        grasp_confidence=0.50,
    )
    res_jitter = mgr.evaluate(hand_jitter, None, [pen], timestamp=7.0)
    primary_jitter = mgr.get_primary_interaction(res_jitter)

    # State must NOT immediately drop to NOT_INTERACTING due to hysteresis!
    assert primary_jitter is not None
    assert primary_jitter.state in (InteractionState.HOLDING, InteractionState.RELEASING)


def test_action_gate_uses_geometric_zones():
    """Verify ActionConfirmationGate evaluates destination using geometric zones."""
    gate = ActionConfirmationGate()

    # Step 3: Place Blue Box at Location A
    step_spec = {
        "id": 3,
        "action": "PLACE_BLUE_BOX_LOCATION_A",
        "label": "Place Blue Box at Location A",
        "description": "Place the Blue Box down at designated Location A.",
        "expected_objects": ["BLUE_BOX"],
        "expected_action": "PLACE",
        "expected_object": "BLUE_BOX",
        "expected_target": "LOCATION_A",
    }
    gate.reset_for_step(step_spec)

    # Blue Box sitting stably in Location A (cx=250, cy=500 in 1000x1000 frame)
    blue_box = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.85,
        bbox=(200, 450, 100, 100),
        centroid=(250, 500),
        semantic_identity="BLUE_BOX",
    )

    # Even with NO visual Location A detection from YOLO:
    conf_action = gate.evaluate(
        current_step=step_spec,
        detected_objects=[blue_box],
        tracks=[blue_box],
        interactions=[],
        har_action="PLACE",
        har_confidence=0.85,
        features={"object_velocity": 0.0, "is_holding": False, "hand_object_distance": 150.0},
        frame_shape=(1000, 1000),
    )

    # Should recognize that Blue Box is in Location A geometrically
    assert conf_action is not None
    assert gate._loc_a_detected is True
    assert len(gate._loc_a_bbox) == 4


def test_debug_visualizer_rendering():
    """Verify DebugVisualizer renders overlay without error."""
    vis = DebugVisualizer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    det = DetectedObject(
        class_name="PEN",
        confidence=0.88,
        bbox=(400, 300, 120, 20),
        centroid=(460, 310),
        semantic_identity="PEN",
        track_id=4,
    )
    cfg = load_config()

    out = vis.render_debug_overlay(
        frame=frame,
        detections=[det],
        hands=None,
        interactions=[],
        zones=cfg.detection.zones,
        fsm_state={"status": "IN_PROGRESS", "current_step": {"action": "PICKUP_PEN", "label": "Pick up pen"}},
        latencies_ms={"decode": 2.5, "yolo": 25.0, "total": 30.0},
        fps=32.0,
    )
    assert out.shape == (720, 1280, 3)
    assert out.dtype == np.uint8
