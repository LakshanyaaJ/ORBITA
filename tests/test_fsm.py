"""Tests for ORBITA Finite State Machine and Rule Engine."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orbita.app.config import ExperimentStep, load_config
from orbita.reasoning.fsm import ExperimentFSM, FSMStatus
from orbita.reasoning.rules import evaluate_rules


def make_steps():
    return [
        ExperimentStep(1, "OPEN_MAIN_BOX", "Open Main Box", "Open the main box", ["MAIN_BOX"],
                       "Open the main box.", "Main box opened.", 30),
        ExperimentStep(2, "TAKE_RED_BOX", "Take Red Box", "Take the red box", ["RED_BOX"],
                       "Take the red box.", "Red box retrieved.", 30),
        ExperimentStep(3, "TAKE_YELLOW_BOX", "Take Yellow Box", "Take the yellow box", ["YELLOW_BOX"],
                       "Take the yellow box.", "Yellow box retrieved.", 30),
        ExperimentStep(4, "OPEN_RED_BOX", "Open Red Box", "Open the red box", ["RED_BOX"],
                       "Open the red box.", "Red box opened.", 30),
    ]


class TestFSMInitialState:
    def test_initial_step_is_1(self):
        fsm = ExperimentFSM(make_steps())
        state = fsm.get_current_state()
        assert state.current_step_idx == 0
        assert state.current_step.id == 1

    def test_initial_status_is_waiting(self):
        fsm = ExperimentFSM(make_steps())
        state = fsm.process("IDLE", "", 0.9)
        assert state.status == FSMStatus.WAITING

    def test_uncertain_action_does_not_advance(self):
        fsm = ExperimentFSM(make_steps(), confidence_threshold=0.55)
        state = fsm.process("OPEN", "MAIN_BOX", confidence=0.30)
        assert state.status == FSMStatus.UNCERTAIN
        assert state.current_step_idx == 0  # Did not advance


class TestFSMStepAdvancement:
    def test_correct_action_advances_fsm(self):
        fsm = ExperimentFSM(make_steps())
        state = fsm.process("OPEN", "MAIN_BOX", confidence=0.85)
        assert state.status == FSMStatus.CORRECT
        # After CORRECT, FSM is now on step 2
        state2 = fsm.get_current_state()
        assert state2.current_step_idx == 1
        assert state2.current_step.id == 2

    def test_all_steps_complete(self):
        fsm = ExperimentFSM(make_steps())
        pairs = [
            ("OPEN", "MAIN_BOX"),
            ("TAKE", "RED_BOX"),
            ("TAKE", "YELLOW_BOX"),
            ("OPEN", "RED_BOX"),
        ]
        for verb, obj in pairs:
            state = fsm.process(verb, obj, confidence=0.85)
        assert state.status == FSMStatus.COMPLETED

    def test_completed_steps_tracked(self):
        fsm = ExperimentFSM(make_steps())
        fsm.process("OPEN", "MAIN_BOX", 0.85)
        fsm.process("TAKE", "RED_BOX", 0.85)
        state = fsm.get_current_state()
        assert 1 in state.completed_step_ids
        assert 2 in state.completed_step_ids
        assert 3 not in state.completed_step_ids


class TestFSMErrorPreservesState:
    def test_wrong_object_does_not_advance(self):
        fsm = ExperimentFSM(make_steps())
        # Step 1: OPEN_MAIN_BOX
        state = fsm.process("TAKE", "YELLOW_BOX", confidence=0.85)
        assert state.status in (FSMStatus.WRONG_OBJECT, FSMStatus.WRONG_ACTION, FSMStatus.STEP_SKIPPED)
        # FSM must still be at step 0
        assert state.current_step_idx == 0

    def test_step_skipped_preserves_state(self):
        fsm = ExperimentFSM(make_steps())
        # Currently at step 1 (OPEN_MAIN_BOX), try step 2 directly
        state = fsm.process("TAKE", "RED_BOX", confidence=0.85)
        assert state.status == FSMStatus.STEP_SKIPPED
        assert state.current_step_idx == 0  # Still at step 1


class TestFSMValidator:
    def test_correct_action(self):
        fsm = ExperimentFSM(make_steps())
        state = fsm.process("OPEN", "MAIN_BOX", 0.85)
        assert state.status == FSMStatus.CORRECT

    def test_wrong_object_detected(self):
        fsm = ExperimentFSM(make_steps())
        # Step 1 expects OPEN_MAIN_BOX. Try OPEN_RED_BOX → should be WRONG_OBJECT
        # (verb OPEN matches, but object RED_BOX ≠ MAIN_BOX)
        state = fsm.process("OPEN", "RED_BOX", 0.85)
        assert state.status in (FSMStatus.WRONG_OBJECT, FSMStatus.WRONG_ACTION, FSMStatus.STEP_SKIPPED)
        # Most important: state must NOT advance
        assert state.current_step_idx == 0

    def test_step_skipped(self):
        fsm = ExperimentFSM(make_steps())
        # At step 1 (OPEN), jump to step 2 (TAKE_RED_BOX)
        state = fsm.process("TAKE", "RED_BOX", 0.85)
        assert state.status == FSMStatus.STEP_SKIPPED

    def test_uncertain_fallback(self):
        fsm = ExperimentFSM(make_steps(), confidence_threshold=0.55)
        state = fsm.process("OPEN", "MAIN_BOX", confidence=0.40)
        assert state.status == FSMStatus.UNCERTAIN


class TestFSMReset:
    def test_reset_returns_to_step_1(self):
        fsm = ExperimentFSM(make_steps())
        fsm.process("OPEN", "MAIN_BOX", 0.85)
        fsm.process("TAKE", "RED_BOX", 0.85)
        fsm.reset()
        state = fsm.get_current_state()
        assert state.current_step_idx == 0
        assert len(state.completed_step_ids) == 0


class TestRuleEngine:
    def test_rule_R01_wrong_object(self):
        steps = make_steps()
        matches = evaluate_rules(
            detected_action="TAKE",
            detected_object="YELLOW_BOX",
            confidence=0.85,
            current_step=steps[1],  # TAKE_RED_BOX
            all_steps=steps,
            completed_step_ids=[1],
            step_start_time=0.0,
        )
        error_types = [m.error_type for m in matches]
        assert "WRONG_OBJECT" in error_types

    def test_rule_R02_step_skipped(self):
        steps = make_steps()
        matches = evaluate_rules(
            detected_action="TAKE",
            detected_object="RED_BOX",
            confidence=0.85,
            current_step=steps[0],  # OPEN_MAIN_BOX
            all_steps=steps,
            completed_step_ids=[],
            step_start_time=0.0,
        )
        error_types = [m.error_type for m in matches]
        assert "STEP_SKIPPED" in error_types

    def test_rule_R05_uncertain(self):
        steps = make_steps()
        matches = evaluate_rules(
            detected_action="OPEN",
            detected_object="MAIN_BOX",
            confidence=0.30,
            current_step=steps[0],
            all_steps=steps,
            completed_step_ids=[],
            step_start_time=0.0,
        )
        assert any(m.error_type == "UNCERTAIN" for m in matches)
