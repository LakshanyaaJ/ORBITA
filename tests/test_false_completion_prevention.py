"""
Comprehensive Test Suite: Prevention of False Step Completion in ORBITA
======================================================================
Verifies all 10 core requirements from the Master Prompt:
  1. Target object detection alone (BLUE_BOX) does NOT complete the step.
  2. Hand detection alone does NOT complete the step.
  3. Hand + BLUE_BOX overlap / proximity alone does NOT complete the step.
  4. The actual required action must occur (pointing for identification, lift for pickup, etc.).
  5. Action must persist for the temporal confirmation window (debouncing/hysteresis).
  6. Expected postcondition must be satisfied (displacement threshold, containment, release).
  7. Only then does the central FSM gate advance the procedure.
  8. Incorrect actions trigger voice correction.
  9. Next step cannot be reached through false positives or flickering detections.
 10. The exact reason and checklist for every completed step is logged.
"""

import pytest
import logging
from core_ai.app.config import load_config
from core_ai.reasoning.state_manager import StateManager
from core_ai.reasoning.action_gate import ActionGateStatus
from core_ai.reasoning.fsm import FSMStatus
from core_ai.perception.object_detector import DetectedObject
from core_ai.interaction.hand_object import HandObjectInteraction, InteractionState
from core_ai.har.temporal_model import ActionPrediction


@pytest.fixture
def config():
    return load_config()


def test_1_object_detection_alone_does_not_complete_step(config):
    """Requirement 1: BLUE_BOX detection alone produces OBJECT_DETECTED and does NOT advance step."""
    manager = StateManager(config, experiment_id="EXP_TEST_01")
    manager.reset()

    assert manager.fsm.current_step_idx == 0
    blue_det = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.95,
        bbox=(100, 100, 80, 80),
        centroid=(140, 140),
        source="yolo",
    )
    pred_idle = ActionPrediction(action="IDLE", confidence=0.8, next_action="IDLE", next_confidence=0.2, is_uncertain=False)

    # Stream 15 frames of blue box sitting on the table
    for _ in range(15):
        res = manager.process(pred_idle, detected_objects=[blue_det])
        assert res.fsm_state.current_step_idx == 0, "Object detection alone must never advance step"
        assert res.confirmed_action.status == ActionGateStatus.WAITING
        assert res.confirmed_action.validation_state == "OBJECT_DETECTED"


def test_2_hand_detection_alone_does_not_complete_step(config):
    """Requirement 2: Hand detection alone produces HAND_DETECTED/WAITING and does NOT advance step."""
    manager = StateManager(config, experiment_id="EXP_TEST_02")
    manager.reset()

    pred_idle = ActionPrediction(action="IDLE", confidence=0.8, next_action="IDLE", next_confidence=0.2, is_uncertain=False)
    # Hand interaction far from target
    hand_inter = HandObjectInteraction(
        hand_side="right",
        object_class="OTHER",
        state=InteractionState.NOT_INTERACTING,
        distance_px=350.0,
        approach_velocity=0.0,
        duration_frames=10,
        object_displacement=0.0,
        confidence=0.85,
        hand_confidence=0.90,
    )

    for _ in range(10):
        res = manager.process(pred_idle, detected_objects=[], interactions=[hand_inter])
        assert res.fsm_state.current_step_idx == 0
        assert res.confirmed_action.status == ActionGateStatus.WAITING


def test_3_hand_and_object_overlap_alone_does_not_complete_step(config):
    """Requirement 3: Hand near or overlapping target object alone does NOT complete the step."""
    manager = StateManager(config, experiment_id="EXP_TEST_03")
    manager.reset()

    blue_det = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.92,
        bbox=(100, 100, 80, 80),
        centroid=(140, 140),
        source="yolo",
    )
    pred_idle = ActionPrediction(action="IDLE", confidence=0.7, next_action="IDLE", next_confidence=0.2, is_uncertain=False)

    # Hand overlapping blue box (IoU 0.45, distance 15px), but NOT pointing and NOT lifting/moving
    overlap_inter = HandObjectInteraction(
        hand_side="right",
        object_class="BLUE_BOX",
        state=InteractionState.CONTACT,
        distance_px=15.0,
        approach_velocity=0.0,
        duration_frames=15,
        object_displacement=0.0,
        confidence=0.88,
        hand_confidence=0.90,
        hand_object_overlap=0.45,
        is_pointing=False,
        is_holding=False,
    )

    for _ in range(15):
        res = manager.process(pred_idle, detected_objects=[blue_det], interactions=[overlap_inter])
        assert res.fsm_state.current_step_idx == 0, "Hand overlap alone must never complete the step"
        assert res.confirmed_action.status == ActionGateStatus.WAITING
        # Validation state must indicate hand is interacting/near object, not confirmed!
        assert res.confirmed_action.validation_state in ("HAND_NEAR_OBJECT", "OBJECT_DETECTED", "HAND_INTERACTING")


def test_4_actual_required_action_must_occur(config):
    """Requirement 4: The actual required action (pointing gesture for Step 1) must occur."""
    manager = StateManager(config, experiment_id="EXP_TEST_04")
    manager.reset()

    blue_det = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.95,
        bbox=(100, 100, 80, 80),
        centroid=(140, 140),
        source="yolo",
    )
    pred_idle = ActionPrediction(action="IDLE", confidence=0.7, next_action="IDLE", next_confidence=0.2, is_uncertain=False)

    # Verified pointing gesture directed at the Blue Box
    pointing_inter = HandObjectInteraction(
        hand_side="right",
        object_class="BLUE_BOX",
        state=InteractionState.CONTACT,
        distance_px=60.0,
        approach_velocity=5.0,
        duration_frames=1,
        object_displacement=0.0,
        confidence=0.92,
        hand_confidence=0.92,
        is_pointing=True,
    )

    # Single frame of pointing: action is IN PROGRESS, not yet completed!
    res = manager.process(pred_idle, detected_objects=[blue_det], interactions=[pointing_inter])
    assert res.fsm_state.current_step_idx == 0
    assert res.confirmed_action.validation_state == "ACTION_IN_PROGRESS"


def test_5_temporal_confirmation_window_required(config):
    """Requirement 5: Action must persist across temporal confirmation window before advancing."""
    manager = StateManager(config, experiment_id="EXP_TEST_05")
    manager.reset()

    blue_det = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.95,
        bbox=(100, 100, 80, 80),
        centroid=(140, 140),
        source="yolo",
    )
    pred_idle = ActionPrediction(action="IDLE", confidence=0.7, next_action="IDLE", next_confidence=0.2, is_uncertain=False)

    pointing_inter = HandObjectInteraction(
        hand_side="right",
        object_class="BLUE_BOX",
        state=InteractionState.CONTACT,
        distance_px=60.0,
        approach_velocity=0.0,
        duration_frames=1,
        object_displacement=0.0,
        confidence=0.92,
        hand_confidence=0.92,
        is_pointing=True,
    )

    # 2 frames: within window (req is 3 frames), remains Step 1
    for frame in range(2):
        res = manager.process(pred_idle, detected_objects=[blue_det], interactions=[pointing_inter])
        assert res.fsm_state.current_step_idx == 0
        assert res.confirmed_action.validation_state == "ACTION_IN_PROGRESS"

    # Frame 3: confirmation window fulfilled -> ACTION_CONFIRMED -> transitions to Step 2
    res_final = manager.process(pred_idle, detected_objects=[blue_det], interactions=[pointing_inter])
    assert res_final.confirmed_action.validation_state == "ACTION_CONFIRMED"
    assert res_final.fsm_state.current_step_idx == 1  # Advanced to Step 2


def test_6_and_7_step_2_pickup_postcondition_and_central_fsm_gate(config):
    """Requirements 6 & 7: Pickup requires grasp + displacement + coupling + postcondition satisfaction."""
    manager = StateManager(config, experiment_id="EXP_TEST_06")
    manager.reset()

    blue_det_init = DetectedObject(class_name="BLUE_BOX", confidence=0.92, bbox=(100, 100, 80, 80), centroid=(140, 140))
    pointing_inter = HandObjectInteraction(
        hand_side="right", object_class="BLUE_BOX", state=InteractionState.CONTACT,
        distance_px=60.0, approach_velocity=0.0, duration_frames=1, object_displacement=0.0,
        confidence=0.92, hand_confidence=0.92, is_pointing=True,
    )
    pred_idle = ActionPrediction(action="IDLE", confidence=0.8, next_action="IDLE", next_confidence=0.2, is_uncertain=False)

    # Complete Step 1 (3 frames)
    for _ in range(3):
        manager.process(pred_idle, detected_objects=[blue_det_init], interactions=[pointing_inter])
    assert manager.fsm.current_step_idx == 1, "Must be at Step 2 (PICKUP_BLUE_BOX)"

    # Step 2: Test that hand touching blue box without lift/displacement does NOT advance Step 2
    touch_inter = HandObjectInteraction(
        hand_side="right", object_class="BLUE_BOX", state=InteractionState.CONTACT,
        distance_px=10.0, approach_velocity=0.0, duration_frames=10, object_displacement=0.0,
        confidence=0.90, hand_confidence=0.90, is_holding=True, contact_duration=10,
    )
    # Target centroid has NOT moved from initial (140, 140)
    for _ in range(10):
        res = manager.process(pred_idle, detected_objects=[blue_det_init], interactions=[touch_inter])
        assert res.fsm_state.current_step_idx == 1, "Grasp without displacement must NOT complete pickup"
        assert res.confirmed_action.status == ActionGateStatus.WAITING

    # Now simulate actual lift and physical displacement (> 25px)
    blue_det_lifted = DetectedObject(class_name="BLUE_BOX", confidence=0.92, bbox=(100, 50, 80, 80), centroid=(140, 90))  # 50px upward displacement!
    lift_inter = HandObjectInteraction(
        hand_side="right", object_class="BLUE_BOX", state=InteractionState.HOLDING,
        distance_px=10.0, approach_velocity=0.0, duration_frames=10, object_displacement=50.0,
        confidence=0.95, hand_confidence=0.92, is_holding=True, contact_duration=10,
        relative_hand_object_motion=0.85,
    )

    # Frame 1: accumulating in temporal confirmation window (req is 2 frames)
    res = manager.process(pred_idle, detected_objects=[blue_det_lifted], interactions=[lift_inter])
    assert res.fsm_state.current_step_idx == 1
    assert res.confirmed_action.validation_state == "ACTION_IN_PROGRESS"

    # Frame 2: completes Step 2 and advances to Step 3!
    res = manager.process(pred_idle, detected_objects=[blue_det_lifted], interactions=[lift_inter])
    assert res.confirmed_action.status == ActionGateStatus.CONFIRMED_CORRECT
    assert res.confirmed_action.validation_state == "ACTION_CONFIRMED"
    assert res.fsm_state.current_step_idx == 2, "Central FSM gate must advance to Step 3"


def test_8_incorrect_action_triggers_voice_correction(config):
    """Requirement 8: Performing wrong action triggers voice correction."""
    spoken = []
    manager = StateManager(config, on_voice=lambda msg: spoken.append(msg), experiment_id="EXP_TEST_08")
    manager.reset()
    spoken.clear()

    # On Step 1 (expecting Blue Box identification), operator points to/holds Yellow Box
    yellow_det = DetectedObject(class_name="YELLOW_BOX", confidence=0.92, bbox=(400, 100, 80, 80), centroid=(440, 140))
    wrong_inter = HandObjectInteraction(
        hand_side="right", object_class="YELLOW_BOX", state=InteractionState.CONTACT,
        distance_px=20.0, approach_velocity=0.0, duration_frames=5, object_displacement=0.0,
        confidence=0.90, hand_confidence=0.90, is_pointing=True,
    )
    pred_idle = ActionPrediction(action="IDLE", confidence=0.8, next_action="IDLE", next_confidence=0.2, is_uncertain=False)

    res = manager.process(pred_idle, detected_objects=[yellow_det], interactions=[wrong_inter])
    assert res.confirmed_action.status == ActionGateStatus.CONFIRMED_WRONG
    assert res.fsm_state.status == FSMStatus.WRONG_SEQUENCE
    assert len(spoken) >= 1
    assert "blue box" in spoken[-1].lower()


def test_9_next_step_cannot_be_reached_through_false_positives(config):
    """Requirement 9: Flickering or momentary touches cannot trigger advancement."""
    manager = StateManager(config, experiment_id="EXP_TEST_09")
    manager.reset()

    blue_det = DetectedObject(class_name="BLUE_BOX", confidence=0.95, bbox=(100, 100, 80, 80), centroid=(140, 140))
    pred_idle = ActionPrediction(action="IDLE", confidence=0.8, next_action="IDLE", next_confidence=0.2, is_uncertain=False)

    # Frame 1: Transient pointing
    touch_inter = HandObjectInteraction(hand_side="right", object_class="BLUE_BOX", state=InteractionState.CONTACT, distance_px=40.0, approach_velocity=0.0, duration_frames=1, object_displacement=0.0, confidence=0.9, hand_confidence=0.9, is_pointing=True)
    manager.process(pred_idle, detected_objects=[blue_det], interactions=[touch_inter])
    assert manager.fsm.current_step_idx == 0

    # Frame 2: Hand moves away (lost)
    manager.process(pred_idle, detected_objects=[blue_det], interactions=[])
    assert manager.fsm.current_step_idx == 0

    # Frame 3: Another transient touch
    manager.process(pred_idle, detected_objects=[blue_det], interactions=[touch_inter])
    assert manager.fsm.current_step_idx == 0, "Transient flickering touches must not advance step"


def test_10_exact_reason_and_checklist_logged(config, caplog):
    """Requirement 10: Step completion includes structured explanation checklist and reasons."""
    caplog.set_level(logging.INFO)
    manager = StateManager(config, experiment_id="EXP_TEST_10")
    manager.reset()

    blue_det = DetectedObject(class_name="BLUE_BOX", confidence=0.95, bbox=(100, 100, 80, 80), centroid=(140, 140))
    pointing_inter = HandObjectInteraction(hand_side="right", object_class="BLUE_BOX", state=InteractionState.CONTACT, distance_px=60.0, approach_velocity=0.0, duration_frames=1, object_displacement=0.0, confidence=0.92, hand_confidence=0.92, is_pointing=True)
    pred_idle = ActionPrediction(action="IDLE", confidence=0.8, next_action="IDLE", next_confidence=0.2, is_uncertain=False)

    res = None
    for _ in range(3):
        res = manager.process(pred_idle, detected_objects=[blue_det], interactions=[pointing_inter])

    assert res.confirmed_action.status == ActionGateStatus.CONFIRMED_CORRECT
    checklist = res.confirmed_action.why_completed_checklist
    assert len(checklist) >= 4, "Must have comprehensive criteria checklist"
    assert all(item["satisfied"] for item in checklist), "All checklist items must be satisfied"

    log_text = caplog.text
    assert "FSM_WHY_COMPLETED" in log_text
    assert "BLUE_BOX detected" in log_text


def test_11_untouched_object_not_flagged_wrong_when_touching_target(config):
    """Untouched stationary object (YELLOW_BOX) must NEVER be flagged CONFIRMED_WRONG when touching BLUE_BOX."""
    manager = StateManager(config, experiment_id="EXP_TEST_11")
    manager.reset()

    blue_det = DetectedObject(class_name="BLUE_BOX", confidence=0.92, bbox=(100, 50, 120, 100), centroid=(160, 100))
    yellow_det = DetectedObject(class_name="YELLOW_BOX", confidence=0.88, bbox=(100, 250, 120, 100), centroid=(160, 300))

    # Hand is touching blue box, completely away from yellow box
    blue_touch = HandObjectInteraction(
        hand_side="right",
        object_class="BLUE_BOX",
        state=InteractionState.CONTACT,
        distance_px=10.0,
        approach_velocity=0.0,
        duration_frames=10,
        object_displacement=0.0,
        confidence=0.90,
        hand_confidence=0.95,
        hand_object_overlap=0.35,
        is_pointing=False,
        is_holding=False,
        is_fingertip_contact=True,
    )

    # Even if HAR produces a noisy PLACE prediction:
    pred_noisy_place = ActionPrediction(
        action="PLACE",
        confidence=0.65,
        next_action="IDLE",
        next_confidence=0.2,
        is_uncertain=False,
        target_object="YELLOW_BOX",  # e.g. raw classification artifact
    )

    for _ in range(5):
        res = manager.process(
            pred_noisy_place,
            detected_objects=[blue_det, yellow_det],
            interactions=[blue_touch],
        )
        # Must NEVER be CONFIRMED_WRONG on YELLOW_BOX because yellow box was never touched!
        assert res.confirmed_action.status != ActionGateStatus.CONFIRMED_WRONG, (
            "Untouched YELLOW_BOX must never be flagged as CONFIRMED_WRONG"
        )
        # Should resolve to BLUE_BOX because the hand is touching BLUE_BOX
        assert res.confirmed_action.object_name == "BLUE_BOX"
        assert res.confirmed_action.status == ActionGateStatus.WAITING


def test_12_hand_tracker_multi_fingertip_contact_and_overlap(config):
    """Verify HandObjectInteractionTracker detects CONTACT with all fingertips and hand bbox overlap."""
    import numpy as np
    from core_ai.interaction.hand_object import HandObjectInteractionTracker
    from core_ai.perception.hand_tracker import HandState

    tracker = HandObjectInteractionTracker(config)
    blue_obj = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.90,
        bbox=(200, 100, 150, 120),
        centroid=(275, 160),
    )

    # Hand with middle and ring fingers touching blue box bbox
    landmarks = np.zeros((21, 2), dtype=np.float32)
    landmarks[0] = [280, 20]     # wrist (outside)
    landmarks[4] = [230, 60]     # thumb (outside)
    landmarks[8] = [260, 95]     # index tip (close)
    landmarks[12] = [275, 110]   # middle tip (INSIDE blue box bbox: y=110 in [100, 220])
    landmarks[16] = [290, 115]   # ring tip (INSIDE blue box bbox)
    landmarks[20] = [305, 120]   # pinky tip (INSIDE blue box bbox)

    hand = HandState(
        hand_id=2,
        side="right",
        confidence=0.95,
        bbox=(220, 30, 100, 100),
        position=landmarks[0],
        velocity=np.array([0.0, 0.0]),
        speed=15.0,
        is_visible=True,
        grasp_confidence=0.4,
        finger_landmarks=landmarks,
        track_status="TRACKED",
    )

    interactions = tracker.update(left=None, right=hand, objects=[blue_obj], timestamp=1.0)
    assert len(interactions) >= 1
    int_blue = [i for i in interactions if i.object_class == "BLUE_BOX"][0]
    assert int_blue.state == InteractionState.CONTACT
    assert int_blue.is_fingertip_contact is True

