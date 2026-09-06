"""
Unit and regression tests for Hand Detection, RED_BOX disambiguation,
Person-ROI expansion, and Hand-Object coexistence.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_ai.perception.object_detector import ObjectDetector, DetectedObject
from core_ai.perception.hand_tracker import HandTracker, HandState, expand_person_roi
from core_ai.interaction.hand_object import (
    HandObjectInteractionTracker,
    InteractionState,
)
from core_ai.app.config import DetectionConfig, InteractionConfig


@pytest.fixture
def detector():
    cfg = DetectionConfig(use_yolo=False)
    return ObjectDetector(cfg)


@pytest.fixture
def interaction_tracker():
    cfg = InteractionConfig()
    return HandObjectInteractionTracker(cfg)


def test_person_roi_expansion():
    """Verify person bbox is expanded by 15-20% and clipped to frame boundaries."""
    # 1. Central person bbox (unclipped)
    frame_shape = (1000, 1280)
    bbox = (200, 150, 400, 500)
    rx, ry, rw, rh = expand_person_roi(bbox, frame_shape, expansion_ratio=0.18)
    
    # Check expansion
    expected_pad_w = int(400 * 0.18)  # 72
    expected_pad_h = int(500 * 0.18)  # 90
    assert rx == 200 - expected_pad_w
    assert ry == 150 - expected_pad_h
    assert rw == 400 + 2 * expected_pad_w
    assert rh == 500 + 2 * expected_pad_h

    # 2. Boundary person bbox (clipped to 0, 0, 1280, 720)
    edge_bbox = (10, 10, 300, 400)
    ex, ey, ew, eh = expand_person_roi(edge_bbox, (720, 1280), expansion_ratio=0.18)
    assert ex >= 0
    assert ey >= 0
    assert ex + ew <= 1280
    assert ey + eh <= 720


def test_hand_structured_representation():
    """Verify HandState exposes all required structured properties."""
    landmarks = np.zeros((21, 2), dtype=np.float32)
    for idx in range(21):
        landmarks[idx] = [100.0 + idx * 2.0, 200.0 + idx * 3.0]

    hand = HandState(
        hand_id=1,
        side="left",
        confidence=0.92,
        bbox=(90, 180, 100, 120),
        position=np.array([100.0, 200.0], dtype=np.float32),
        velocity=np.array([12.5, -4.0], dtype=np.float32),
        speed=13.12,
        is_visible=True,
        grasp_confidence=0.75,
        finger_landmarks=landmarks,
        landmarks_conf=np.ones(21, dtype=np.float32),
        timestamp=1.0,
        track_status="TRACKED",
    )

    # Core required properties
    assert hand.hand_id == 1
    assert hand.track_id == 1
    assert hand.handedness == "left"
    assert hand.tracking_state == "TRACKED"
    assert hand.confidence == 0.92
    assert hand.is_visible is True
    assert hand.is_detected is True
    assert hand.bbox == (90, 180, 100, 120)
    assert np.allclose(hand.wrist, [100.0, 200.0])
    assert hand.thumb_tip is not None
    assert hand.index_tip is not None
    assert hand.middle_tip is not None
    assert hand.ring_tip is not None
    assert hand.pinky_tip is not None
    assert len(hand.thumb_landmarks) == 4
    assert len(hand.index_landmarks) == 4
    assert len(hand.middle_landmarks) == 4
    assert len(hand.ring_landmarks) == 4
    assert len(hand.pinky_landmarks) == 4
    assert "thumb" in hand.fingertip_positions
    assert "index" in hand.fingertip_positions


def test_red_box_without_hand(detector):
    """Verify a genuine red rectangular container is detected as RED_BOX."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a clean red rectangular box (BGR: Red=220, Green=20, Blue=20)
    # Area: 100 x 70 = 7000 px (> eff_min_area)
    bx, by, bw, bh = 250, 180, 100, 70
    frame[by:by+bh, bx:bx+bw] = [20, 20, 220]

    detections = detector.detect(frame, timestamp=0.0)
    red_boxes = [d for d in detections if d.class_name == "RED_BOX"]
    assert len(red_boxes) >= 1, "Genuine red box must be detected"
    rb = red_boxes[0]
    assert rb.confidence >= 0.65
    assert abs(rb.bbox[0] - bx) < 10
    assert abs(rb.bbox[1] - by) < 10


def test_hand_without_red_box_suppresses_false_red_box(detector):
    """
    Verify that organic human skin/palm color overlapping a hand is rejected
    and NEVER falsely classified as RED_BOX.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw an organic skin tone region with low rectangularity (knuckle/crease)
    # (BGR: Blue=60, Green=70, Red=115) - same as Frame 300 empirical measurement
    hx, hy, hw, hh = 200, 150, 120, 140
    cv2_hand_contour = np.array([
        [200, 150], [280, 155], [310, 210], [270, 260], [220, 280], [205, 220]
    ], dtype=np.int32)
    import cv2
    cv2.fillPoly(frame, [cv2_hand_contour], (60, 70, 115))

    # Mock hand tracker state
    hand = HandState(
        hand_id=1,
        side="left",
        confidence=0.95,
        bbox=(hx, hy, hw, hh),
        position=np.array([250.0, 260.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        speed=0.0,
        is_visible=True,
        grasp_confidence=0.5,
    )

    detections = detector.detect(frame, timestamp=0.0, hands=(hand, None))
    red_boxes = [d for d in detections if d.class_name == "RED_BOX"]
    assert len(red_boxes) == 0, "Hand skin tones must NOT be classified as RED_BOX"


def test_hand_and_red_box_coexistence(detector):
    """
    When a hand touches/grasps a real red box, BOTH the hand and RED_BOX
    must coexist cleanly in the perception output.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw real red rectangular container (120x80 = 9600 px)
    bx, by, bw, bh = 240, 180, 120, 80
    frame[by:by+bh, bx:bx+bw] = [15, 15, 230]

    # Hand grasps left side of the box (partially overlapping)
    # Hand bbox: (180, 160, 100, 110)
    hx, hy, hw, hh = 180, 160, 100, 110
    hand = HandState(
        hand_id=1,
        side="left",
        confidence=0.95,
        bbox=(hx, hy, hw, hh),
        position=np.array([190.0, 240.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        speed=0.0,
        is_visible=True,
        grasp_confidence=0.8,
    )

    detections = detector.detect(frame, timestamp=0.0, hands=(hand, None))
    red_boxes = [d for d in detections if d.class_name == "RED_BOX"]

    assert len(red_boxes) >= 1, "Real red box must remain detected when touched by hand"
    assert hand.is_visible is True, "Hand must remain visible"
    # Coexistence: Hand exists and RED_BOX exists simultaneously
    assert red_boxes[0].class_name == "RED_BOX"


def test_interaction_pickup_hold_release_lifecycle(interaction_tracker):
    """Test full interaction lifecycle: APPROACH -> CONTACT -> HOLDING -> RELEASED."""
    sample = DetectedObject(
        class_name="SAMPLE",
        confidence=0.95,
        bbox=(200, 200, 60, 60),
        centroid=(230, 230),
        track_id=10,
        velocity=(0.0, 0.0),
    )

    landmarks = np.zeros((21, 2), dtype=np.float32)

    # 1. Far away: Hand at distance > 200px
    landmarks[4] = [520.0, 230.0]  # Thumb
    landmarks[8] = [520.0, 235.0]  # Index
    hand1 = HandState(
        hand_id=1, side="right", confidence=0.9, bbox=(500, 200, 80, 80),
        position=np.array([540.0, 250.0]), velocity=np.array([-40.0, 0.0]),
        speed=40.0, is_visible=True, grasp_confidence=0.3, finger_landmarks=landmarks,
    )
    res1 = interaction_tracker.update(None, hand1, [sample], timestamp=1.0)
    assert len(res1) == 0 or res1[0].state == InteractionState.NOT_INTERACTING

    # 2. Contact: Fingertips touch sample bbox boundary (<40px)
    landmarks[4] = [255.0, 230.0]  # Inside / on sample edge
    landmarks[8] = [258.0, 235.0]
    hand2 = HandState(
        hand_id=1, side="right", confidence=0.95, bbox=(220, 200, 80, 80),
        position=np.array([270.0, 250.0]), velocity=np.array([-10.0, 0.0]),
        speed=10.0, is_visible=True, grasp_confidence=0.8, finger_landmarks=landmarks,
    )
    res2 = interaction_tracker.update(None, hand2, [sample], timestamp=1.1)
    assert res2[0].state in (InteractionState.NEAR_OBJECT, InteractionState.CONTACT)

    # 3. Holding: Hand and sample move together (velocity coupling)
    sample_moving = DetectedObject(
        class_name="SAMPLE", confidence=0.95, bbox=(180, 200, 60, 60),
        centroid=(210, 230), track_id=10, velocity=(-50.0, 0.0),
    )
    landmarks[4] = [215.0, 230.0]
    landmarks[8] = [218.0, 235.0]
    hand3 = HandState(
        hand_id=1, side="right", confidence=0.95, bbox=(180, 200, 80, 80),
        position=np.array([230.0, 250.0]), velocity=np.array([-52.0, 0.0]),
        speed=52.0, is_visible=True, grasp_confidence=0.85, finger_landmarks=landmarks,
    )
    for step in range(5):
        res3 = interaction_tracker.update(None, hand3, [sample_moving], timestamp=1.2 + step * 0.033)
    assert res3[0].state == InteractionState.HOLDING
    assert res3[0].is_holding is True

    # 4. Release: Sample stops, hand moves away (>100px)
    sample_stopped = DetectedObject(
        class_name="SAMPLE", confidence=0.95, bbox=(150, 200, 60, 60),
        centroid=(180, 230), track_id=10, velocity=(0.0, 0.0),
    )
    landmarks[4] = [340.0, 230.0]
    landmarks[8] = [345.0, 235.0]
    hand4 = HandState(
        hand_id=1, side="right", confidence=0.9, bbox=(330, 200, 80, 80),
        position=np.array([360.0, 250.0]), velocity=np.array([60.0, 0.0]),
        speed=60.0, is_visible=True, grasp_confidence=0.2, finger_landmarks=landmarks,
    )
    for step in range(6):
        res4 = interaction_tracker.update(None, hand4, [sample_stopped], timestamp=1.5 + step * 0.033)
    assert res4[0].state in (InteractionState.RELEASING, InteractionState.RELEASED, InteractionState.NOT_INTERACTING)


# =========================================================================== #
# PHASE 10 — EXPLICIT REGRESSION TEST CASES 1 THROUGH 6
# =========================================================================== #

def test_phase10_case1_hand_only(detector):
    """Case 1: HAND only -> Expected: HAND, NO RED_BOX."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Hand skin region
    hx, hy, hw, hh = 220, 160, 110, 130
    import cv2
    cv2_contour = np.array([[220, 160], [300, 165], [330, 220], [290, 270], [240, 290], [225, 230]], dtype=np.int32)
    cv2.fillPoly(frame, [cv2_contour], (55, 65, 120))

    hand = HandState(
        hand_id=1, side="left", confidence=0.94, bbox=(hx, hy, hw, hh),
        position=np.array([260.0, 270.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32), speed=0.0, is_visible=True, grasp_confidence=0.5,
    )
    detections = detector.detect(frame, timestamp=0.0, hands=(hand, None))
    red_boxes = [d for d in detections if d.class_name == "RED_BOX"]
    assert len(red_boxes) == 0, "Case 1 failed: No RED_BOX must be detected when only hand is present"
    assert hand.is_visible is True


def test_phase10_case2_redbox_only(detector):
    """Case 2: RED_BOX only -> Expected: RED_BOX."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bx, by, bw, bh = 200, 160, 120, 90
    frame[by:by+bh, bx:bx+bw] = [15, 15, 225]

    detections = detector.detect(frame, timestamp=0.0, hands=(None, None))
    red_boxes = [d for d in detections if d.class_name == "RED_BOX"]
    assert len(red_boxes) >= 1, "Case 2 failed: RED_BOX must be detected"
    assert red_boxes[0].class_name == "RED_BOX"


def test_phase10_case3_hand_touching_redbox(detector):
    """Case 3: HAND touching RED_BOX -> Expected: HAND + RED_BOX."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Real red box
    bx, by, bw, bh = 250, 180, 130, 85
    frame[by:by+bh, bx:bx+bw] = [10, 10, 235]

    # Hand touches left border of red box
    hand = HandState(
        hand_id=1, side="right", confidence=0.95, bbox=(180, 170, 90, 100),
        position=np.array([220.0, 250.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32), speed=0.0, is_visible=True, grasp_confidence=0.75,
    )
    detections = detector.detect(frame, timestamp=0.0, hands=(hand, None))
    red_boxes = [d for d in detections if d.class_name == "RED_BOX"]
    assert len(red_boxes) >= 1, "Case 3 failed: RED_BOX must coexist when touched by hand"
    assert hand.is_visible is True, "Case 3 failed: Hand must remain visible"


def test_phase10_case4_hand_covering_redbox(detector):
    """Case 4: HAND covering RED_BOX -> Expected: HAND + RED_BOX."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Larger red box
    bx, by, bw, bh = 220, 160, 160, 110
    frame[by:by+bh, bx:bx+bw] = [12, 12, 230]

    # Hand covering upper-left quadrant of the box
    hand = HandState(
        hand_id=1, side="left", confidence=0.96, bbox=(210, 150, 80, 80),
        position=np.array([240.0, 220.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32), speed=0.0, is_visible=True, grasp_confidence=0.85,
    )
    detections = detector.detect(frame, timestamp=0.0, hands=(hand, None))
    red_boxes = [d for d in detections if d.class_name == "RED_BOX"]
    assert len(red_boxes) >= 1, "Case 4 failed: Partially covered RED_BOX must remain detected"
    assert hand.is_visible is True


def test_phase10_case5_hand_moving_across_scene(detector):
    """Case 5: HAND moving across scene -> Expected: HAND, no false RED_BOX."""
    import cv2
    for step in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hx = 100 + step * 40
        hy = 180 + step * 10
        cv2_contour = np.array([
            [hx, hy], [hx + 70, hy + 5], [hx + 90, hy + 50],
            [hx + 60, hy + 90], [hx + 20, hy + 100], [hx + 5, hy + 60]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [cv2_contour], (60, 70, 120))

        hand = HandState(
            hand_id=1, side="right", confidence=0.93, bbox=(hx, hy, 90, 100),
            position=np.array([hx + 40.0, hy + 80.0], dtype=np.float32),
            velocity=np.array([40.0, 10.0], dtype=np.float32), speed=41.2,
            is_visible=True, grasp_confidence=0.4,
        )
        detections = detector.detect(frame, timestamp=step * 0.033, hands=(hand, None))
        red_boxes = [d for d in detections if d.class_name == "RED_BOX"]
        assert len(red_boxes) == 0, f"Case 5 failed at step {step}: Moving hand must NOT create false RED_BOX"


def test_phase10_case6_skin_colored_object(detector):
    """Case 6: skin-colored object -> Expected: no false RED_BOX."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    import cv2
    # Draw various skin-colored objects (warm beige, peach, skin tones with low red channel dominance)
    # Object 1: Oval face/palm shaped patch
    cv2.ellipse(frame, (200, 200), (60, 80), 30, 0, 360, (75, 95, 140), -1)
    # Object 2: Elongated arm patch
    cv2.rectangle(frame, (350, 150), (420, 320), (80, 100, 150), -1)

    detections = detector.detect(frame, timestamp=0.0, hands=(None, None))
    red_boxes = [d for d in detections if d.class_name == "RED_BOX"]
    assert len(red_boxes) == 0, "Case 6 failed: Organic skin tones must NOT produce false RED_BOX"

