"""
Automated Integration Test for ORBITA Step-by-Step Voice Guidance
and Action Confirmation Synchronization (Section 23).
"""

import pytest
import time
from core_ai.app.config import load_config, OrbitaConfig, ExperimentStep
from core_ai.reasoning.state_manager import StateManager
from core_ai.reasoning.action_gate import ActionConfirmationGate, ActionGateStatus, ConfirmedAction
from core_ai.reasoning.fsm import FSMStatus
from core_ai.perception.object_detector import DetectedObject
from core_ai.perception.object_tracker import TrackedState, ObjectPhysicalState
from core_ai.har.temporal_model import ActionPrediction


@pytest.fixture
def official_config():
    cfg = load_config()
    return cfg


def test_full_13_step_sequence_with_voice_synchronization(official_config):
    """
    Validates:
    - Step 1 spoken at experiment start
    - Actions 1..12 confirmed advance FSM Step 1 -> 13
    - Voice speaks exactly next step on transition
    - Completion voice speaks exactly once at Step 13
    - Voice latch prevents duplicate voice prompts during repeated frames
    """
    spoken_messages = []

    def voice_sink(text: str):
        spoken_messages.append(text)

    manager = StateManager(official_config, on_voice=voice_sink, experiment_id="EXP_VOICE_01")
    # Reset triggers Step 1 voice prompt
    manager.reset()

    assert len(spoken_messages) == 1
    assert "Step 1. Identify the Blue Box." in spoken_messages[0]

    # Repeated WAITING frames must NOT re-speak Step 1 (Voice Latch)
    pred_idle = ActionPrediction(action="IDLE", confidence=0.7, next_action="IDLE", next_confidence=0.3, is_uncertain=False, target_object="")
    for _ in range(5):
        manager.process(pred_idle, detected_objects=[])
    assert len(spoken_messages) == 1, "Voice latch must prevent repeating Step 1"

    # Step 1: Identify Blue Box
    # Provide persistent blue box detection
    blue_det = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.92,
        bbox=(100, 100, 80, 80),
        centroid=(140, 140),
        source="yolo",
        semantic_identity="BLUE_BOX",
    )
    pred_identify = ActionPrediction(action="IDENTIFY", confidence=0.92, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")
    for _ in range(3):
        res = manager.process(pred_identify, detected_objects=[blue_det])
    
    # Step 1 confirmed -> Advanced to Step 2 -> Voice Step 2
    assert res.fsm_state.current_step_idx == 1
    assert len(spoken_messages) == 2
    assert "Step 2. Pick up the Blue Box." in spoken_messages[1]

    # Step 2: Pick up Blue Box
    blue_track = TrackedState(
        track_id=1,
        class_name="BLUE_BOX",
        bbox=(100, 80, 80, 80),
        centroid=(140, 120),
        confidence=0.90,
        identity="BLUE_BOX",
        state=ObjectPhysicalState.BEING_HELD,
        hand_contact=True,
        held_duration=0.5,
    )
    pred_pickup = ActionPrediction(action="TAKE", confidence=0.85, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")
    for _ in range(3):
        res = manager.process(pred_pickup, detected_objects=[blue_det], tracks=[blue_track])

    assert res.fsm_state.current_step_idx == 2
    assert len(spoken_messages) == 3
    assert "Step 3. Place the Blue Box at Location A." in spoken_messages[2]

    # Step 3: Place Blue Box at Location A
    blue_det_a = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.92,
        bbox=(150, 400, 80, 80),
        centroid=(190, 440),
        source="yolo",
        semantic_identity="BLUE_BOX",
    )
    blue_track_a = TrackedState(
        track_id=1,
        class_name="BLUE_BOX",
        bbox=(150, 400, 80, 80),
        centroid=(190, 440),
        confidence=0.90,
        identity="BLUE_BOX",
        state=ObjectPhysicalState.PLACED,
        hand_contact=False,
    )
    pred_place = ActionPrediction(action="PLACE", confidence=0.88, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="BLUE_BOX")
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[blue_det_a], tracks=[blue_track_a])

    assert res.fsm_state.current_step_idx == 3
    assert len(spoken_messages) == 4
    assert "Step 4. Identify the Yellow Box." in spoken_messages[3]

    # Step 4: Identify Yellow Box
    yellow_det = DetectedObject(
        class_name="YELLOW_BOX",
        confidence=0.94,
        bbox=(500, 100, 80, 80),
        centroid=(540, 140),
        source="yolo",
        semantic_identity="YELLOW_BOX",
    )
    pred_identify_y = ActionPrediction(action="IDENTIFY", confidence=0.92, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="YELLOW_BOX")
    for _ in range(3):
        res = manager.process(pred_identify_y, detected_objects=[yellow_det])

    assert res.fsm_state.current_step_idx == 4
    assert len(spoken_messages) == 5
    assert "Step 5. Pick up the Yellow Box." in spoken_messages[4]

    # Step 5: Pick up Yellow Box
    yellow_track = TrackedState(
        track_id=2,
        class_name="YELLOW_BOX",
        bbox=(500, 80, 80, 80),
        centroid=(540, 120),
        confidence=0.91,
        identity="YELLOW_BOX",
        state=ObjectPhysicalState.BEING_HELD,
        hand_contact=True,
        held_duration=0.5,
    )
    for _ in range(3):
        res = manager.process(pred_pickup, detected_objects=[yellow_det], tracks=[yellow_track])

    assert res.fsm_state.current_step_idx == 5
    assert len(spoken_messages) == 6
    assert "Step 6. Place the Yellow Box at Location B." in spoken_messages[5]

    # Step 6: Place Yellow Box at Location B
    yellow_det_b = DetectedObject(
        class_name="YELLOW_BOX",
        confidence=0.93,
        bbox=(800, 400, 80, 80),
        centroid=(840, 440),
        source="yolo",
        semantic_identity="YELLOW_BOX",
    )
    yellow_track_b = TrackedState(
        track_id=2,
        class_name="YELLOW_BOX",
        bbox=(800, 400, 80, 80),
        centroid=(840, 440),
        confidence=0.91,
        identity="YELLOW_BOX",
        state=ObjectPhysicalState.PLACED,
        hand_contact=False,
    )
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[yellow_det_b], tracks=[yellow_track_b])

    assert res.fsm_state.current_step_idx == 6
    assert len(spoken_messages) == 7
    assert "Step 7. Pick up the Pen." in spoken_messages[6]

    # Step 7: Pick up Pen
    pen_det = DetectedObject(
        class_name="TOOL",
        confidence=0.88,
        bbox=(300, 300, 40, 40),
        centroid=(320, 320),
        source="yolo",
        semantic_identity="PEN",
    )
    pen_track = TrackedState(
        track_id=3,
        class_name="TOOL",
        bbox=(300, 280, 40, 40),
        centroid=(320, 300),
        confidence=0.85,
        identity="PEN",
        state=ObjectPhysicalState.BEING_HELD,
        hand_contact=True,
        held_duration=0.4,
    )
    for _ in range(3):
        res = manager.process(pred_pickup, detected_objects=[pen_det], tracks=[pen_track])

    assert res.fsm_state.current_step_idx == 7
    assert len(spoken_messages) == 8
    assert "Step 8. Place the Pen inside the Blue Box." in spoken_messages[7]

    # Step 8: Place Pen inside Blue Box
    pen_det_in_box = DetectedObject(
        class_name="TOOL",
        confidence=0.88,
        bbox=(160, 410, 30, 30),
        centroid=(175, 425),
        source="yolo",
        semantic_identity="PEN",
    )
    pen_track_in_box = TrackedState(
        track_id=3,
        class_name="TOOL",
        bbox=(160, 410, 30, 30),
        centroid=(175, 425),
        confidence=0.85,
        identity="PEN",
        state=ObjectPhysicalState.PLACED,
        hand_contact=False,
    )
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[pen_det_in_box, blue_det_a], tracks=[pen_track_in_box, blue_track_a])

    assert res.fsm_state.current_step_idx == 8
    assert len(spoken_messages) == 9
    assert "Step 9. Pick up the Watch." in spoken_messages[8]

    # Step 9: Pick up Watch
    watch_det = DetectedObject(
        class_name="SAMPLE",
        confidence=0.86,
        bbox=(450, 300, 40, 40),
        centroid=(470, 320),
        source="yolo",
        semantic_identity="WATCH",
    )
    watch_track = TrackedState(
        track_id=4,
        class_name="SAMPLE",
        bbox=(450, 280, 40, 40),
        centroid=(470, 300),
        confidence=0.84,
        identity="WATCH",
        state=ObjectPhysicalState.BEING_HELD,
        hand_contact=True,
        held_duration=0.4,
    )
    for _ in range(3):
        res = manager.process(pred_pickup, detected_objects=[watch_det], tracks=[watch_track])

    assert res.fsm_state.current_step_idx == 9
    assert len(spoken_messages) == 10
    assert "Step 10. Place the Watch inside the Yellow Box." in spoken_messages[9]

    # Step 10: Place Watch inside Yellow Box
    watch_det_in_box = DetectedObject(
        class_name="SAMPLE",
        confidence=0.86,
        bbox=(810, 410, 30, 30),
        centroid=(825, 425),
        source="yolo",
        semantic_identity="WATCH",
    )
    watch_track_in_box = TrackedState(
        track_id=4,
        class_name="SAMPLE",
        bbox=(810, 410, 30, 30),
        centroid=(825, 425),
        confidence=0.84,
        identity="WATCH",
        state=ObjectPhysicalState.PLACED,
        hand_contact=False,
    )
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[watch_det_in_box, yellow_det_b], tracks=[watch_track_in_box, yellow_track_b])

    assert res.fsm_state.current_step_idx == 10
    assert len(spoken_messages) == 11
    assert "Step 11. Move the Blue Box from Location A to Location B." in spoken_messages[10]

    # Step 11: Move Blue Box from A -> B
    blue_det_at_b = DetectedObject(
        class_name="BLUE_BOX",
        confidence=0.92,
        bbox=(800, 400, 80, 80),
        centroid=(840, 440),
        source="yolo",
        semantic_identity="BLUE_BOX",
    )
    blue_track_at_b = TrackedState(
        track_id=1,
        class_name="BLUE_BOX",
        bbox=(800, 400, 80, 80),
        centroid=(840, 440),
        confidence=0.90,
        identity="BLUE_BOX",
        state=ObjectPhysicalState.PLACED,
        hand_contact=False,
    )
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[blue_det_at_b], tracks=[blue_track_at_b])

    assert res.fsm_state.current_step_idx == 11
    assert len(spoken_messages) == 12
    assert "Step 12. Move the Yellow Box from Location B to Location A." in spoken_messages[11]

    # Step 12: Move Yellow Box from B -> A
    yellow_det_at_a = DetectedObject(
        class_name="YELLOW_BOX",
        confidence=0.92,
        bbox=(150, 400, 80, 80),
        centroid=(190, 440),
        source="yolo",
        semantic_identity="YELLOW_BOX",
    )
    yellow_track_at_a = TrackedState(
        track_id=2,
        class_name="YELLOW_BOX",
        bbox=(150, 400, 80, 80),
        centroid=(190, 440),
        confidence=0.91,
        identity="YELLOW_BOX",
        state=ObjectPhysicalState.PLACED,
        hand_contact=False,
    )
    for _ in range(3):
        res = manager.process(pred_place, detected_objects=[yellow_det_at_a], tracks=[yellow_track_at_a])

    # Transition to Step 13: Completed
    assert res.fsm_state.is_complete
    assert len(spoken_messages) == 13
    assert "Experiment complete. All steps were successfully completed." in spoken_messages[12]

    # Repeated calls at terminal state must NOT repeat completion message
    for _ in range(5):
        manager.process(pred_idle, detected_objects=[])
    assert len(spoken_messages) == 13, "Completion message must not be repeated"


def test_wrong_sequence_handling(official_config):
    """
    Validates:
    - Wrong confirmed action triggers WRONG_SEQUENCE
    - FSM does not advance
    - Appropriate voice warning is spoken
    """
    spoken_messages = []
    manager = StateManager(official_config, on_voice=lambda msg: spoken_messages.append(msg), experiment_id="EXP_WRONG_01")
    manager.reset()

    # Move to Step 5 (current_step = Step 5: Pick up Yellow Box)
    manager.fsm._current_idx = 4
    manager.speak_step(4)
    spoken_messages.clear()

    # Operator picks up Pen instead of Yellow Box
    pen_track = TrackedState(
        track_id=3,
        class_name="TOOL",
        bbox=(300, 280, 40, 40),
        centroid=(320, 300),
        confidence=0.88,
        identity="PEN",
        state=ObjectPhysicalState.BEING_HELD,
        hand_contact=True,
        held_duration=0.5,
    )
    pred_pickup_pen = ActionPrediction(action="TAKE", confidence=0.85, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="PEN")

    # Confirmed wrong action (requires multi-frame temporal confirmation)
    for _ in range(3):
        res = manager.process(pred_pickup_pen, detected_objects=[], tracks=[pen_track])

    assert res.fsm_state.current_step_idx == 4, "FSM must not advance on wrong action"
    assert res.fsm_state.status == FSMStatus.WRONG_SEQUENCE
    assert len(spoken_messages) >= 1
    assert "Wrong sequence" in spoken_messages[-1]
    assert "yellow box" in spoken_messages[-1].lower()


def test_noisy_perception_stays_waiting(official_config):
    """
    Validates:
    - Raw noisy detection for 1 frame stays in WAITING
    - No wrong sequence error
    - No premature voice output
    """
    spoken_messages = []
    manager = StateManager(official_config, on_voice=lambda msg: spoken_messages.append(msg), experiment_id="EXP_NOISE_01")
    manager.reset()
    assert len(spoken_messages) == 1  # Step 1 prompt

    # Transient 1-frame detection of an unrelated object
    weak_obj = DetectedObject(
        class_name="SAMPLE",
        confidence=0.45,
        bbox=(10, 10, 20, 20),
        centroid=(20, 20),
        source="yolo",
        semantic_identity="WATCH",
    )
    pred_idle = ActionPrediction(action="IDLE", confidence=0.5, next_action="IDLE", next_confidence=0.2, is_uncertain=True, target_object="")

    res = manager.process(pred_idle, detected_objects=[weak_obj])

    assert res.fsm_state.current_step_idx == 0
    assert res.fsm_state.status == FSMStatus.WAITING
    assert len(spoken_messages) == 1, "Noisy single-frame perception must not trigger voice prompt"


def test_tracked_state_section_7_and_37_fields():
    """Validates that TrackedState contains all required Section 7 & 37 fields and properties."""
    track = TrackedState(
        track_id=42,
        class_name="MAIN_BOX",
        bbox=(100, 200, 80, 80),
        centroid=(140, 240),
        confidence=0.95,
        raw_label="MAIN_BOX",
        source="yolo",
        identity="BLUE_BOX",
        state=ObjectPhysicalState.ON_TABLE,
    )

    # Section 7 fields
    assert track.track_id == 42
    assert track.raw_class == "MAIN_BOX"
    assert track.semantic_identity == "BLUE_BOX"
    assert track.confidence == 0.95
    assert track.bbox == (100, 200, 80, 80)
    assert track.center == (140, 240)
    assert track.velocity == (0.0, 0.0)
    assert track.physical_state == "ON_TABLE"
    assert track.hand_contact is False
    assert track.held_duration == 0.0
    assert len(track.position_history) >= 1

    # Section 37 fields
    assert track.initial_position == (140, 240)
    assert track.current_position == (140, 240)
    assert len(track.velocity_history) >= 1
    assert len(track.hand_contact_history) >= 1
    assert len(track.physical_state_history) >= 1
    assert track.pickup_time is None
    assert track.release_time is None
    assert track.placement_location == ""


def test_wrong_action_correction_latch_no_loop(official_config):
    """
    Validates Section 25:
    - Wrong action voice guidance is spoken once when confirmed wrong
    - Repeated frames of the SAME wrong action do NOT repeat voice guidance
    - Only after cooldown expires does it speak again
    """
    spoken_messages = []
    manager = StateManager(official_config, on_voice=lambda msg: spoken_messages.append(msg), experiment_id="EXP_LATCH_01")
    manager.reset()
    spoken_messages.clear()

    # Step 2: Pick up Blue Box (index 1)
    manager.fsm._current_idx = 1
    manager.speak_step(1)
    spoken_messages.clear()

    # Operator picks up Pen
    pen_track = TrackedState(
        track_id=3,
        class_name="TOOL",
        bbox=(300, 280, 40, 40),
        centroid=(320, 300),
        confidence=0.88,
        identity="PEN",
        state=ObjectPhysicalState.BEING_HELD,
        hand_contact=True,
        held_duration=0.5,
    )
    pred_pickup_pen = ActionPrediction(action="TAKE", confidence=0.85, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="PEN")

    # Confirm wrong action across 3 frames
    for _ in range(3):
        res = manager.process(pred_pickup_pen, detected_objects=[], tracks=[pen_track])

    assert res.fsm_state.status == FSMStatus.WRONG_SEQUENCE
    assert len(spoken_messages) == 1
    assert "Wrong sequence" in spoken_messages[0]
    assert "Blue Box" in spoken_messages[0] or "blue box" in spoken_messages[0].lower()

    # Simulate 10 more consecutive frames of the same wrong action (within cooldown)
    for _ in range(10):
        manager.process(pred_pickup_pen, detected_objects=[], tracks=[pen_track])

    # Correction latch MUST prevent voice looping
    assert len(spoken_messages) == 1, "Correction latch must prevent voice looping on repeated frames"


def test_session_reset_allows_step_1_replay(official_config):
    """Validates Section 42: Resetting the session resets tracking, FSM, and allows Step 1 prompt again."""
    spoken_messages = []
    manager = StateManager(official_config, on_voice=lambda msg: spoken_messages.append(msg), experiment_id="EXP_RESET_01")
    manager.reset()
    assert len(spoken_messages) == 1
    assert "Step 1. Identify the Blue Box." in spoken_messages[0]

    # Finish or advance
    manager.fsm._current_idx = 12
    manager.speak_complete()
    assert len(spoken_messages) == 2

    # Reset session
    spoken_messages.clear()
    manager.reset()
    assert len(spoken_messages) == 1
    assert "Step 1. Identify the Blue Box." in spoken_messages[0]
    assert manager.fsm.current_step_idx == 0


def test_wrong_action_event_lifecycle_release_and_repick(official_config):
    """
    Validates Section 10 (Wrong-Action Event Lifecycle):
    Operator picks up Pen -> Pen pickup confirmed -> WRONG_ACTION_EVENT created -> VOICE spoken
    -> same Pen remains held -> NO NEW EVENT (voice count = 1)
    -> operator releases Pen -> wrong interaction ended
    -> operator picks up Pen again -> NEW WRONG ACTION -> NEW EVENT -> NEW CORRECTION (voice count = 2)
    """
    spoken_messages = []
    manager = StateManager(official_config, on_voice=lambda msg: spoken_messages.append(msg), experiment_id="EXP_EVENT_01")
    manager.reset()
    spoken_messages.clear()

    # Step 2: Pick up Blue Box
    manager.fsm._current_idx = 1
    manager.speak_step(1)
    spoken_messages.clear()

    pen_track_held = TrackedState(
        track_id=3,
        class_name="TOOL",
        bbox=(300, 280, 40, 40),
        centroid=(320, 300),
        confidence=0.88,
        identity="PEN",
        state=ObjectPhysicalState.BEING_HELD,
        hand_contact=True,
        held_duration=0.5,
    )
    pred_pickup_pen = ActionPrediction(action="TAKE", confidence=0.85, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="PEN")

    # Confirm wrong action (frames 1..3)
    for _ in range(3):
        manager.process(pred_pickup_pen, detected_objects=[], tracks=[pen_track_held])
    assert len(spoken_messages) == 1
    assert "Wrong sequence" in spoken_messages[0]

    # Hold pen for 20 frames -> voice count must remain 1
    for _ in range(20):
        manager.process(pred_pickup_pen, detected_objects=[], tracks=[pen_track_held])
    assert len(spoken_messages) == 1

    # Operator releases Pen -> ON_TABLE
    pen_track_table = TrackedState(
        track_id=3,
        class_name="TOOL",
        bbox=(300, 350, 40, 40),
        centroid=(320, 370),
        confidence=0.88,
        identity="PEN",
        state=ObjectPhysicalState.ON_TABLE,
        hand_contact=False,
    )
    pred_idle = ActionPrediction(action="IDLE", confidence=0.7, next_action="IDLE", next_confidence=0.2, is_uncertain=False, target_object="")
    for _ in range(5):
        manager.process(pred_idle, detected_objects=[], tracks=[pen_track_table])
    assert len(spoken_messages) == 1

    # Cooldown timer elapsed
    manager._last_wrong_voice_time = time.time() - 5.0

    # Operator picks up Pen AGAIN
    for _ in range(3):
        manager.process(pred_pickup_pen, detected_objects=[], tracks=[pen_track_held])
    assert len(spoken_messages) == 2, "Re-picking up wrong object after release must trigger new wrong event"

