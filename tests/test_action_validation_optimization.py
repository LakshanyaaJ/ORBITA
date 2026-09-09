"""
ORBITA Action Validation & Step Transition Optimization Test Suite
==================================================================
Targeted tests for:
  1. IDENTIFY_YELLOW_BOX completes without requiring release or hand interaction.
  2. Stationary objects (e.g. BLUE_BOX) do not block or interfere with IDENTIFY_YELLOW_BOX.
  3. Transient detection noise (1-2 frames) does not trigger WRONG_SEQUENCE.
  4. PICK_UP validation requires actual holding/lifting evidence, not mere contact/proximity.
  5. Validator state cleanly resets between step transitions (Step 3 -> 4, 4 -> 5).
  6. FSM transitions exactly once with duplicate advances guarded.
  7. Voice guidance triggers once per state transition and debounces wrong sequences.
"""

import logging
import pytest
from core_ai.app.config import load_config
from core_ai.reasoning.state_manager import StateManager
from core_ai.reasoning.action_gate import ActionConfirmationGate, ActionGateStatus, ConfirmedAction
from core_ai.reasoning.fsm import FSMStatus
from core_ai.perception.object_detector import DetectedObject
from core_ai.perception.object_tracker import TrackedState, ObjectPhysicalState
from core_ai.interaction.hand_object import HandObjectInteraction, InteractionState
from core_ai.har.temporal_model import ActionPrediction


@pytest.fixture
def config():
    return load_config()


def test_identify_yellow_box_decoupled_from_hands_and_releasing(config, caplog):
    """
    CRITICAL ACCEPTANCE TEST:
    Verifies that on Step 4 (IDENTIFY_YELLOW_BOX), valid detection of YELLOW_BOX
    advances to Step 5 in 3 frames even when:
      - Hands are actively moving (high velocity telemetry)
      - Interaction state shows RELEASING / RELEASE
      - BLUE_BOX is present in the workspace
      - Hand contact exists with another object
    """
    caplog.set_level(logging.INFO)
    spoken = []
    manager = StateManager(config, on_voice=lambda msg: spoken.append(msg), experiment_id="EXP_TEST_YELLOW")
    manager.reset()

    # Manually fast-forward to Step 4 (current_step_idx = 3)
    manager.fsm._current_idx = 3
    manager._last_spoken_step_idx = 3
    manager._last_step_id = 4
    manager.action_gate.reset_for_step(manager.fsm.current_step)
    spoken.clear()

    assert manager.fsm.current_step.id == 4
    assert manager.fsm.current_step.action == "IDENTIFY_YELLOW_BOX"

    # Both objects are in view, hand is releasing near yellow box
    yellow_det = DetectedObject(
        class_name="YELLOW_BOX",
        confidence=0.82,
        bbox=(200, 200, 100, 100),
        centroid=(250, 250),
        source="hybrid",
        semantic_identity="YELLOW_BOX",
    )
    blue_det = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.88,
        bbox=(500, 200, 100, 100),
        centroid=(550, 250),
        source="yolo",
        semantic_identity="BLUE_BOX",
    )
    releasing_interaction = HandObjectInteraction(
        hand_side="left",
        object_class="YELLOW_BOX",
        state=InteractionState.RELEASING,
        distance_px=35.0,
        approach_velocity=-280.0,
        duration_frames=10,
        object_displacement=5.0,
        confidence=0.85,
        is_holding=False,
    )
    pred_identify = ActionPrediction(
        action="IDENTIFY",
        confidence=0.80,
        next_action="IDLE",
        next_confidence=0.2,
        is_uncertain=False,
        target_object="YELLOW_BOX",
    )

    # Frame 1: Confirming 1/3
    res1 = manager.process(
        pred_identify,
        detected_objects=[yellow_det, blue_det],
        interactions=[releasing_interaction],
    )
    assert res1.fsm_state.current_step_idx == 3
    assert res1.confirmed_action.status == ActionGateStatus.WAITING
    assert res1.confirmed_action.validation_state in ("CONFIRMING", "ACTION_IN_PROGRESS")
    assert "YELLOW_BOX stable for 1/3" in res1.confirmed_action.validation_reason

    # Frame 2: Confirming 2/3
    res2 = manager.process(
        pred_identify,
        detected_objects=[yellow_det, blue_det],
        interactions=[releasing_interaction],
    )
    assert res2.fsm_state.current_step_idx == 3
    assert res2.confirmed_action.status == ActionGateStatus.WAITING
    assert "YELLOW_BOX stable for 2/3" in res2.confirmed_action.validation_reason

    # Frame 3: Confirmed Correct! Advances to Step 5 (Pick up Yellow Box)
    res3 = manager.process(
        pred_identify,
        detected_objects=[yellow_det, blue_det],
        interactions=[releasing_interaction],
    )
    assert res3.confirmed_action.status == ActionGateStatus.CONFIRMED_CORRECT
    assert res3.fsm_state.current_step_idx == 4  # Advanced to Step 5!
    assert res3.fsm_state.current_step.id == 5
    assert "Pick up the Yellow Box" in res3.fsm_state.current_step.label

    # Voice prompt announced Step 5 once and NO wrong sequence error occurred
    assert len(spoken) == 1
    assert "Step 5. Pick up the Yellow Box." in spoken[0]
    assert not any("Wrong sequence" in s for s in spoken)


def test_identify_ignores_other_stationary_objects(config):
    """Presence of BLUE_BOX sitting on table does not trigger wrong sequence during Step 4."""
    manager = StateManager(config, experiment_id="EXP_TEST_STAT")
    manager.reset()
    manager.fsm._current_idx = 3
    manager.action_gate.reset_for_step(manager.fsm.current_step)

    yellow_det = DetectedObject(class_name="YELLOW_BOX", confidence=0.85, bbox=(100, 100, 80, 80), centroid=(140, 140), source="yolo", semantic_identity="YELLOW_BOX")
    blue_det = DetectedObject(class_name="BLUE_BOX", confidence=0.90, bbox=(400, 100, 80, 80), centroid=(440, 140), source="yolo", semantic_identity="BLUE_BOX")
    pred = ActionPrediction(action="IDENTIFY", confidence=0.85, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="YELLOW_BOX")

    for _ in range(3):
        res = manager.process(pred, detected_objects=[yellow_det, blue_det])

    assert res.fsm_state.current_step_idx == 4  # Advanced cleanly to Step 5
    assert res.fsm_state.status != FSMStatus.WRONG_SEQUENCE


def test_transient_noise_does_not_trigger_wrong_sequence(config):
    """1-2 frames of an irrelevant object do not trigger wrong-sequence voice or error."""
    spoken = []
    manager = StateManager(config, on_voice=lambda m: spoken.append(m), experiment_id="EXP_NOISE")
    manager.reset()
    spoken.clear()

    # Step 1: Identify Blue Box
    # Introduce 2 transient frames of WATCH
    watch_det = DetectedObject(class_name="SAMPLE", confidence=0.85, bbox=(200, 200, 30, 30), centroid=(215, 215), source="yolo", semantic_identity="WATCH")
    pred = ActionPrediction(action="IDLE", confidence=0.5, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="")

    res1 = manager.process(pred, detected_objects=[watch_det])
    assert res1.fsm_state.status == FSMStatus.WAITING
    assert res1.confirmed_action.status == ActionGateStatus.WAITING

    res2 = manager.process(pred, detected_objects=[watch_det])
    assert res2.fsm_state.status == FSMStatus.WAITING
    assert res2.confirmed_action.status == ActionGateStatus.WAITING
    assert len(spoken) == 0  # No wrong sequence voice spoken!


def test_pickup_validation_requires_actual_hold(config):
    """Pickup validation requires actual holding/carrying, not mere proximity or touch."""
    manager = StateManager(config, experiment_id="EXP_PICKUP_HOLD")
    manager.reset()

    # Fast forward to Step 2: Pick up Blue Box
    manager.fsm._current_idx = 1
    manager.action_gate.reset_for_step(manager.fsm.current_step)

    blue_det = DetectedObject(class_name="BLUE_BOX", confidence=0.90, bbox=(100, 100, 80, 80), centroid=(140, 140), source="yolo", semantic_identity="BLUE_BOX")
    pred = ActionPrediction(action="IDLE", confidence=0.5, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="")

    # 1. Proximity / Near object touch only -> does NOT confirm pickup
    touch_track = TrackedState(track_id=1, class_name="BLUE_BOX", bbox=(100, 100, 80, 80), centroid=(140, 140), confidence=0.90, identity="BLUE_BOX", state=ObjectPhysicalState.HAND_CONTACT, hand_contact=True)
    res_touch = manager.process(pred, detected_objects=[blue_det], tracks=[touch_track])
    assert res_touch.fsm_state.current_step_idx == 1  # Remains on Step 2
    assert res_touch.confirmed_action.status == ActionGateStatus.WAITING

    # 2. Actual hold state -> frame 1 accumulating, frame 2 confirmed correct
    held_track = TrackedState(track_id=1, class_name="BLUE_BOX", bbox=(100, 80, 80, 80), centroid=(140, 120), confidence=0.90, identity="BLUE_BOX", state=ObjectPhysicalState.BEING_HELD, hand_contact=True, held_duration=0.5)
    res_h1 = manager.process(pred, detected_objects=[blue_det], tracks=[held_track])
    assert res_h1.fsm_state.current_step_idx == 1  # Still on Step 2
    assert res_h1.confirmed_action.status == ActionGateStatus.WAITING

    res_h2 = manager.process(pred, detected_objects=[blue_det], tracks=[held_track])
    assert res_h2.confirmed_action.status == ActionGateStatus.CONFIRMED_CORRECT
    assert res_h2.fsm_state.current_step_idx == 2  # Advanced to Step 3!


def test_validator_clean_reset_on_step_transition(config):
    """Step transition automatically resets temporal accumulators for clean context."""
    manager = StateManager(config, experiment_id="EXP_RESET_TRANS")
    manager.reset()

    # Start at Step 3: Place Blue Box
    manager.fsm._current_idx = 2
    manager.action_gate.reset_for_step(manager.fsm.current_step)

    # Accumulate held state for Blue Box
    manager.action_gate._consecutive_held["BLUE_BOX"] = 10
    manager.action_gate._was_held["BLUE_BOX"] = True

    # Complete Step 3
    blue_det_a = DetectedObject(class_name="BLUE_BOX", confidence=0.92, bbox=(150, 400, 80, 80), centroid=(190, 440), source="yolo", semantic_identity="BLUE_BOX")
    blue_track_a = TrackedState(track_id=1, class_name="BLUE_BOX", bbox=(150, 400, 80, 80), centroid=(190, 440), confidence=0.90, identity="BLUE_BOX", state=ObjectPhysicalState.PLACED, hand_contact=False)
    pred_place = ActionPrediction(action="PLACE", confidence=0.88, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")

    for _ in range(2):
        res = manager.process(pred_place, detected_objects=[blue_det_a], tracks=[blue_track_a])
    assert res.fsm_state.current_step_idx == 3  # Transitioned to Step 4 (Identify Yellow Box)

    # Verify that action gate accumulators were cleanly reset for Step 4
    assert manager.action_gate._consecutive_held.get("BLUE_BOX", 0) == 0
    assert manager.action_gate._consecutive_seen.get("YELLOW_BOX", 0) == 0
    assert manager.action_gate._was_held.get("BLUE_BOX", False) is False
    assert manager.action_gate._current_gate_status == ActionGateStatus.WAITING


def test_duplicate_advances_prevented(config):
    """FSM cannot advance multiple times for the same completed step."""
    manager = StateManager(config, experiment_id="EXP_DUP_PREVENT")
    manager.reset()

    blue_det = DetectedObject(class_name="BLUE_BOX", confidence=0.92, bbox=(100, 100, 80, 80), centroid=(140, 140), source="yolo", semantic_identity="BLUE_BOX")
    pred = ActionPrediction(action="IDENTIFY", confidence=0.90, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")

    # Satisfy Step 1
    for _ in range(3):
        res = manager.process(pred, detected_objects=[blue_det])
    assert res.fsm_state.current_step_idx == 1  # At Step 2

    # Step 1 ID is completed
    assert 1 in manager.fsm.completed_step_ids

    # Try calling _advance directly with already completed step
    manager.fsm._advance()
    assert manager.fsm.current_step_idx == 2
    # Ensure completed_ids has no duplicates
    assert len(manager.fsm.completed_step_ids) == len(set(manager.fsm.completed_step_ids))
