"""
ORBITA Step Validation and FSM Transition Test Suite
===================================================
Validates:
  1. Action-type-specific confirmation gates for all 13 steps.
  2. Step 1 Identification gate:
     - BLUE_BOX + IDENTIFY -> CONFIRMED_CORRECT -> Advance to Step 2
     - YELLOW_BOX + IDENTIFY -> CONFIRMED_WRONG -> Remain on Step 1 with wrong-step voice
     - Single noisy frame / low confidence / missing detection -> WAITING (does not advance, does not error)
     - Stale frame (is_stale=True) -> blocked from validation
  3. Steps 2 through 12 physical action confirmation (PICKUP, PLACE, MOVE).
  4. Terminal Step 13 completion.
  5. Strict 13-step progression (no step skips, 1 -> 2 -> ... -> 13).
"""

import pytest
import logging
from core_ai.app.config import load_config
from core_ai.reasoning.state_manager import StateManager
from core_ai.reasoning.action_gate import ActionConfirmationGate, ActionGateStatus, ConfirmedAction
from core_ai.reasoning.fsm import FSMStatus
from core_ai.perception.object_detector import DetectedObject
from core_ai.perception.object_tracker import TrackedState, ObjectPhysicalState
from core_ai.har.temporal_model import ActionPrediction


@pytest.fixture
def config():
    return load_config()


def test_step_1_identification_success(config, caplog):
    """Step 1: Stable BLUE_BOX identification transitions FSM from Step 1 to Step 2."""
    caplog.set_level(logging.INFO)
    spoken = []
    manager = StateManager(config, on_voice=lambda msg: spoken.append(msg), experiment_id="EXP_TEST_01")
    manager.reset()

    assert manager.fsm.current_step_idx == 0
    assert len(spoken) == 1
    assert "Step 1. Identify the Blue Box." in spoken[0]

    blue_det = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.91,
        bbox=(100, 100, 80, 80),
        centroid=(140, 140),
        source="yolo",
        semantic_identity="BLUE_BOX",
    )
    pred_identify = ActionPrediction(
        action="IDENTIFY",
        confidence=0.94,
        next_action="IDLE",
        next_confidence=0.2,
        is_uncertain=False,
        target_object="BLUE_BOX",
    )

    # Frames 1 and 2: Accumulating evidence (WAITING / CONFIRMING)
    res1 = manager.process(pred_identify, detected_objects=[blue_det])
    assert res1.fsm_state.current_step_idx == 0
    assert res1.confirmed_action.status == ActionGateStatus.WAITING

    res2 = manager.process(pred_identify, detected_objects=[blue_det])
    assert res2.fsm_state.current_step_idx == 0
    assert res2.confirmed_action.status == ActionGateStatus.WAITING

    # Frame 3: Stability window satisfied (min_identify_frames = 3) -> CONFIRMED_CORRECT
    res3 = manager.process(pred_identify, detected_objects=[blue_det])
    assert res3.confirmed_action.status == ActionGateStatus.CONFIRMED_CORRECT
    assert res3.fsm_state.current_step_idx == 1  # Successfully advanced to Step 2!
    assert len(spoken) == 2
    assert "Step 2. Pick up the Blue Box." in spoken[1]

    # Verify structured logs appeared
    log_text = caplog.text
    assert "STEP_VALIDATION" in log_text
    assert "FSM_TRANSITION" in log_text
    assert "from_step=1" in log_text
    assert "to_step=2" in log_text
    assert "reason=IDENTIFICATION_CONFIRMED" in log_text


def test_step_1_wrong_identification(config, caplog):
    """Step 1: Identifying YELLOW_BOX instead of BLUE_BOX triggers CONFIRMED_WRONG and stays on Step 1."""
    caplog.set_level(logging.INFO)
    spoken = []
    manager = StateManager(config, on_voice=lambda msg: spoken.append(msg), experiment_id="EXP_TEST_WRONG")
    manager.reset()
    spoken.clear()

    yellow_det = DetectedObject(
        class_name="YELLOW_BOX",
        confidence=0.93,
        bbox=(500, 100, 80, 80),
        centroid=(540, 140),
        source="yolo",
        semantic_identity="YELLOW_BOX",
    )
    pred_identify_yellow = ActionPrediction(
        action="IDENTIFY",
        confidence=0.90,
        next_action="IDLE",
        next_confidence=0.2,
        is_uncertain=False,
        target_object="YELLOW_BOX",
    )

    for _ in range(3):
        res = manager.process(pred_identify_yellow, detected_objects=[yellow_det])

    assert res.confirmed_action.status == ActionGateStatus.CONFIRMED_WRONG
    assert res.fsm_state.current_step_idx == 0  # Remains Step 1!
    assert res.fsm_state.status == FSMStatus.WRONG_SEQUENCE
    assert len(spoken) >= 1
    assert "Wrong sequence" in spoken[-1]
    assert "blue box" in spoken[-1].lower()


def test_stale_frame_protection(config):
    """Stale frames (latency > 500ms / is_stale=True) are blocked from advancing FSM."""
    manager = StateManager(config, experiment_id="EXP_TEST_STALE")
    manager.reset()

    blue_det = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.95,
        bbox=(100, 100, 80, 80),
        centroid=(140, 140),
        source="yolo",
        semantic_identity="BLUE_BOX",
    )
    pred_identify = ActionPrediction(action="IDENTIFY", confidence=0.95, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")

    # Pass 5 frames with is_stale=True
    for _ in range(5):
        res = manager.process(pred_identify, detected_objects=[blue_det], frame_age_ms=800.0, is_stale=True)
        assert res.fsm_state.current_step_idx == 0, "Stale frame must not advance FSM"
        assert res.confirmed_action.status == ActionGateStatus.WAITING
        assert res.is_stale is True


def test_unstable_detection_does_not_advance(config):
    """Flickering detection that drops out does not advance FSM."""
    manager = StateManager(config, experiment_id="EXP_TEST_UNSTABLE")
    manager.reset()

    blue_det = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.90,
        bbox=(100, 100, 80, 80),
        centroid=(140, 140),
        source="yolo",
        semantic_identity="BLUE_BOX",
    )
    pred_identify = ActionPrediction(action="IDENTIFY", confidence=0.85, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")

    # Frame 1: seen
    res1 = manager.process(pred_identify, detected_objects=[blue_det])
    assert res1.fsm_state.current_step_idx == 0

    # Frame 2: missing
    res2 = manager.process(pred_identify, detected_objects=[])
    assert res2.fsm_state.current_step_idx == 0

    # Frame 3: missing
    res3 = manager.process(pred_identify, detected_objects=[])
    assert res3.fsm_state.current_step_idx == 0

    # Frame 4: missing
    res4 = manager.process(pred_identify, detected_objects=[])
    assert res4.fsm_state.current_step_idx == 0


def test_complete_deterministic_13_step_advancement(config, caplog):
    """
    Validates end-to-end deterministic progression through all 13 steps:
      1. Identify Blue Box
      2. Pick up Blue Box
      3. Place Blue Box at Location A
      4. Identify Yellow Box
      5. Pick up Yellow Box
      6. Place Yellow Box at Location B
      7. Pick up Pen
      8. Place Pen inside Blue Box
      9. Pick up Watch
      10. Place Watch inside Yellow Box
      11. Move Blue Box A->B
      12. Move Yellow Box B->A
      13. Experiment Complete
    """
    caplog.set_level(logging.INFO)
    spoken = []
    manager = StateManager(config, on_voice=lambda msg: spoken.append(msg), experiment_id="EXP_ALL_13")
    manager.reset()

    # Step 1 -> 2: Identify Blue Box
    blue_det = DetectedObject(class_name="BLUE_BOX", confidence=0.92, bbox=(100, 100, 80, 80), centroid=(140, 140), source="yolo", semantic_identity="BLUE_BOX")
    pred_identify = ActionPrediction(action="IDENTIFY", confidence=0.90, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")
    for _ in range(3):
        res = manager.process(pred_identify, detected_objects=[blue_det])
    assert res.fsm_state.current_step_idx == 1, "Must advance to Step 2"

    # Step 2 -> 3: Pick up Blue Box
    blue_track_held = TrackedState(track_id=1, class_name="BLUE_BOX", bbox=(100, 80, 80, 80), centroid=(140, 120), confidence=0.90, identity="BLUE_BOX", state=ObjectPhysicalState.BEING_HELD, hand_contact=True, held_duration=0.5)
    pred_pickup = ActionPrediction(action="TAKE", confidence=0.88, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")
    for _ in range(3):
        res = manager.process(pred_pickup, detected_objects=[blue_det], tracks=[blue_track_held])
    assert res.fsm_state.current_step_idx == 2, "Must advance to Step 3"

    # Step 3 -> 4: Place Blue Box at Location A
    blue_det_a = DetectedObject(class_name="BLUE_BOX", confidence=0.92, bbox=(150, 400, 80, 80), centroid=(190, 440), source="yolo", semantic_identity="BLUE_BOX")
    blue_track_a = TrackedState(track_id=1, class_name="BLUE_BOX", bbox=(150, 400, 80, 80), centroid=(190, 440), confidence=0.90, identity="BLUE_BOX", state=ObjectPhysicalState.PLACED, hand_contact=False)
    pred_place = ActionPrediction(action="PLACE", confidence=0.88, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[blue_det_a], tracks=[blue_track_a])
    assert res.fsm_state.current_step_idx == 3, "Must advance to Step 4"

    # Step 4 -> 5: Identify Yellow Box
    yellow_det = DetectedObject(class_name="YELLOW_BOX", confidence=0.94, bbox=(500, 100, 80, 80), centroid=(540, 140), source="yolo", semantic_identity="YELLOW_BOX")
    pred_identify_y = ActionPrediction(action="IDENTIFY", confidence=0.90, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="YELLOW_BOX")
    for _ in range(3):
        res = manager.process(pred_identify_y, detected_objects=[yellow_det])
    assert res.fsm_state.current_step_idx == 4, "Must advance to Step 5"

    # Step 5 -> 6: Pick up Yellow Box
    yellow_track_held = TrackedState(track_id=2, class_name="YELLOW_BOX", bbox=(500, 80, 80, 80), centroid=(540, 120), confidence=0.91, identity="YELLOW_BOX", state=ObjectPhysicalState.BEING_HELD, hand_contact=True, held_duration=0.5)
    pred_pickup_y = ActionPrediction(action="TAKE", confidence=0.88, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="YELLOW_BOX")
    for _ in range(3):
        res = manager.process(pred_pickup_y, detected_objects=[yellow_det], tracks=[yellow_track_held])
    assert res.fsm_state.current_step_idx == 5, "Must advance to Step 6"

    # Step 6 -> 7: Place Yellow Box at Location B
    yellow_det_b = DetectedObject(class_name="YELLOW_BOX", confidence=0.93, bbox=(800, 400, 80, 80), centroid=(840, 440), source="yolo", semantic_identity="YELLOW_BOX")
    yellow_track_b = TrackedState(track_id=2, class_name="YELLOW_BOX", bbox=(800, 400, 80, 80), centroid=(840, 440), confidence=0.91, identity="YELLOW_BOX", state=ObjectPhysicalState.PLACED, hand_contact=False)
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[yellow_det_b], tracks=[yellow_track_b])
    assert res.fsm_state.current_step_idx == 6, "Must advance to Step 7"

    # Step 7 -> 8: Pick up Pen
    pen_det = DetectedObject(class_name="TOOL", confidence=0.88, bbox=(300, 300, 40, 40), centroid=(320, 320), source="yolo", semantic_identity="PEN")
    pen_track = TrackedState(track_id=3, class_name="TOOL", bbox=(300, 280, 40, 40), centroid=(320, 300), confidence=0.85, identity="PEN", state=ObjectPhysicalState.BEING_HELD, hand_contact=True, held_duration=0.4)
    pred_pickup_pen = ActionPrediction(action="TAKE", confidence=0.85, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="PEN")
    for _ in range(3):
        res = manager.process(pred_pickup_pen, detected_objects=[pen_det], tracks=[pen_track])
    assert res.fsm_state.current_step_idx == 7, "Must advance to Step 8"

    # Step 8 -> 9: Place Pen inside Blue Box
    pen_det_box = DetectedObject(class_name="TOOL", confidence=0.88, bbox=(160, 410, 30, 30), centroid=(175, 425), source="yolo", semantic_identity="PEN")
    pen_track_box = TrackedState(track_id=3, class_name="TOOL", bbox=(160, 410, 30, 30), centroid=(175, 425), confidence=0.85, identity="PEN", state=ObjectPhysicalState.PLACED, hand_contact=False)
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[pen_det_box, blue_det_a], tracks=[pen_track_box, blue_track_a])
    assert res.fsm_state.current_step_idx == 8, "Must advance to Step 9"

    # Step 9 -> 10: Pick up Watch
    watch_det = DetectedObject(class_name="SAMPLE", confidence=0.86, bbox=(450, 300, 40, 40), centroid=(470, 320), source="yolo", semantic_identity="WATCH")
    watch_track = TrackedState(track_id=4, class_name="SAMPLE", bbox=(450, 280, 40, 40), centroid=(470, 300), confidence=0.84, identity="WATCH", state=ObjectPhysicalState.BEING_HELD, hand_contact=True, held_duration=0.4)
    pred_pickup_watch = ActionPrediction(action="TAKE", confidence=0.85, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="WATCH")
    for _ in range(3):
        res = manager.process(pred_pickup_watch, detected_objects=[watch_det], tracks=[watch_track])
    assert res.fsm_state.current_step_idx == 9, "Must advance to Step 10"

    # Step 10 -> 11: Place Watch inside Yellow Box
    watch_det_box = DetectedObject(class_name="SAMPLE", confidence=0.86, bbox=(810, 410, 30, 30), centroid=(825, 425), source="yolo", semantic_identity="WATCH")
    watch_track_box = TrackedState(track_id=4, class_name="SAMPLE", bbox=(810, 410, 30, 30), centroid=(825, 425), confidence=0.84, identity="WATCH", state=ObjectPhysicalState.PLACED, hand_contact=False)
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[watch_det_box, yellow_det_b], tracks=[watch_track_box, yellow_track_b])
    assert res.fsm_state.current_step_idx == 10, "Must advance to Step 11"

    # Step 11 -> 12: Move Blue Box from A to B
    blue_det_at_b = DetectedObject(class_name="BLUE_BOX", confidence=0.92, bbox=(800, 400, 80, 80), centroid=(840, 440), source="yolo", semantic_identity="BLUE_BOX")
    blue_track_at_b = TrackedState(track_id=1, class_name="BLUE_BOX", bbox=(800, 400, 80, 80), centroid=(840, 440), confidence=0.90, identity="BLUE_BOX", state=ObjectPhysicalState.PLACED, hand_contact=False)
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[blue_det_at_b], tracks=[blue_track_at_b])
    assert res.fsm_state.current_step_idx == 11, "Must advance to Step 12"

    # Step 12 -> 13: Move Yellow Box from B to A
    yellow_det_at_a = DetectedObject(class_name="YELLOW_BOX", confidence=0.92, bbox=(150, 400, 80, 80), centroid=(190, 440), source="yolo", semantic_identity="YELLOW_BOX")
    yellow_track_at_a = TrackedState(track_id=2, class_name="YELLOW_BOX", bbox=(150, 400, 80, 80), centroid=(190, 440), confidence=0.91, identity="YELLOW_BOX", state=ObjectPhysicalState.PLACED, hand_contact=False)
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[yellow_det_at_a], tracks=[yellow_track_at_a])

    # Terminal Step 13: Completed
    assert res.fsm_state.is_complete
    assert res.fsm_state.status == FSMStatus.COMPLETED
    assert len(spoken) == 13
    assert "Experiment complete" in spoken[-1]
