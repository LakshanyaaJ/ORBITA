"""
ORBITA Finite State Machine (FSM)
===================================
Deterministic procedural reasoning engine.

The FSM is the SAFETY-CRITICAL decision maker.
It does NOT use an LLM or any probabilistic model for transitions.

Each state corresponds to one experiment step.
Transitions are allowed only when the detected action matches
the expected action for the current step.

FSM properties:
  - State only advances on CORRECT action
  - State is NEVER advanced on ERROR or UNCERTAIN
  - Full history is maintained (completed, failed, skipped steps)
  - Recovery mode re-prompts without state change
  - End state is terminal (experiment complete)

Error types:
  CORRECT        — expected action detected
  WRONG_OBJECT   — action verb correct, wrong object
  WRONG_ACTION   — completely different action
  STEP_SKIPPED   — action matches a future step (skip detected)
  OUT_OF_SEQUENCE — action matches a past completed step
  REPEATED_ACTION — current step already completed
  UNCERTAIN      — confidence below threshold
  TIMEOUT        — step exceeded time limit (future: timer integration)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from orbita.app.config import ExperimentStep

logger = logging.getLogger(__name__)


class FSMStatus(Enum):
    WAITING = auto()       # No action detected yet for current step
    CORRECT = auto()       # Current action matches expected step
    WRONG_OBJECT = auto()  # Right action type, wrong object
    WRONG_ACTION = auto()  # Completely wrong action
    STEP_SKIPPED = auto()  # Action matches a future step (gap)
    OUT_OF_SEQUENCE = auto()  # Action matches an already-completed step
    REPEATED_ACTION = auto()  # Current step already done
    UNCERTAIN = auto()     # AI confidence too low
    COMPLETED = auto()     # Experiment fully completed


@dataclass
class FSMState:
    """Full experiment state snapshot at one point in time."""
    current_step_idx: int               # 0-indexed
    total_steps: int
    status: FSMStatus
    current_step: Optional[ExperimentStep]
    next_step: Optional[ExperimentStep]
    completed_step_ids: list[int]
    failed_step_ids: list[int]
    skipped_step_ids: list[int]
    detected_action: str
    detected_object: str
    error_type: Optional[str]
    recovery_message: Optional[str]
    experiment_id: str
    start_time: float
    elapsed_seconds: float
    step_start_time: float

    @property
    def is_complete(self) -> bool:
        return self.status == FSMStatus.COMPLETED

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return 100.0 * len(self.completed_step_ids) / self.total_steps

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "current_step_idx": self.current_step_idx,
            "total_steps": self.total_steps,
            "status": self.status.name,
            "current_step": {
                "id": self.current_step.id,
                "action": self.current_step.action,
                "label": self.current_step.label,
                "description": self.current_step.description,
            } if self.current_step else None,
            "next_step": {
                "id": self.next_step.id,
                "action": self.next_step.action,
                "label": self.next_step.label,
            } if self.next_step else None,
            "completed_steps": self.completed_step_ids,
            "failed_steps": self.failed_step_ids,
            "skipped_steps": self.skipped_step_ids,
            "detected_action": self.detected_action,
            "detected_object": self.detected_object,
            "error_type": self.error_type,
            "recovery_message": self.recovery_message,
            "progress_pct": round(self.progress_pct, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


class ExperimentFSM:
    """
    Finite State Machine for experiment procedure validation.

    Usage:
        fsm = ExperimentFSM(steps, experiment_id="EXP001")
        state = fsm.process("TAKE", "RED_BOX", confidence=0.75)
        if state.status == FSMStatus.CORRECT:
            ...
    """

    def __init__(
        self,
        steps: list[ExperimentStep],
        experiment_id: str = "EXP001",
        confidence_threshold: float = 0.55,
    ):
        if not steps:
            raise ValueError("FSM requires at least one experiment step.")

        self.steps = steps
        self.experiment_id = experiment_id
        self.confidence_threshold = confidence_threshold

        # Mutable FSM state
        self._current_idx: int = 0
        self._completed_ids: list[int] = []
        self._failed_ids: list[int] = []
        self._skipped_ids: list[int] = []
        self._start_time: float = time.time()
        self._step_start: float = time.time()
        self._last_status: FSMStatus = FSMStatus.WAITING
        self._last_detected_action: str = "IDLE"
        self._last_detected_object: str = ""

    # ----------------------------------------------------------------------- #
    # Core processing
    # ----------------------------------------------------------------------- #
    def process(
        self,
        detected_action: str,
        detected_object: str = "",
        confidence: float = 1.0,
    ) -> FSMState:
        """
        Evaluate detected action against the current expected step.

        This is the FSM's primary entry point. Called once per frame
        (or per stable action classification).

        Args:
            detected_action: e.g. "TAKE", "OPEN", "IDLE"
            detected_object: e.g. "RED_BOX", "YELLOW_BOX"
            confidence: Action classification confidence [0, 1]

        Returns:
            FSMState snapshot
        """
        self._last_detected_action = detected_action
        self._last_detected_object = detected_object
        now = time.time()

        # --- Terminal state ---
        if self._current_idx >= len(self.steps):
            return self._make_state(FSMStatus.COMPLETED, None, None, now)

        current = self.steps[self._current_idx]
        next_step = self.steps[self._current_idx + 1] if self._current_idx + 1 < len(self.steps) else None

        # --- UNCERTAIN ---
        if confidence < self.confidence_threshold:
            return self._make_state(FSMStatus.UNCERTAIN, current, next_step, now,
                                    error="UNCERTAIN")

        # --- IDLE — nothing happening ---
        if detected_action in ("IDLE", "", "UNKNOWN"):
            return self._make_state(FSMStatus.WAITING, current, next_step, now)

        # --- Build full detected action label ---
        detected_full = f"{detected_action}_{detected_object}" if detected_object else detected_action

        # --- Check against current expected action ---
        expected_action = current.action  # e.g. "TAKE_RED_BOX"
        expected_verb, expected_obj = self._split_action(expected_action)

        detected_verb, detected_obj = detected_action.upper(), detected_object.upper()

        # CORRECT match
        if self._matches(detected_verb, detected_obj, expected_action):
            self._advance()
            new_idx = self._current_idx
            new_current = self.steps[new_idx] if new_idx < len(self.steps) else None
            new_next = self.steps[new_idx + 1] if new_idx + 1 < len(self.steps) else None
            if new_current is None:
                return self._make_state(FSMStatus.COMPLETED, None, None, now)
            return self._make_state(FSMStatus.CORRECT, current, next_step, now)

        # REPEATED — already completed
        if current.id in self._completed_ids:
            return self._make_state(FSMStatus.REPEATED_ACTION, current, next_step, now,
                                    error="REPEATED_ACTION",
                                    recovery=f"Step {current.id} already completed.")

        # WRONG_OBJECT — correct verb, wrong object (check BEFORE step-skip)
        # This catches cases like TAKE_YELLOW_BOX when TAKE_RED_BOX expected.
        if detected_verb == expected_verb and expected_obj and detected_obj and detected_obj != expected_obj:
            self._failed_ids.append(current.id)
            return self._make_state(FSMStatus.WRONG_OBJECT, current, next_step, now,
                                    error="WRONG_OBJECT",
                                    recovery=f"Wrong object. Expected: {expected_obj.replace('_', ' ').lower()}.")

        # STEP_SKIPPED — detected action matches a future step (exact verb+obj match required)
        for future_step in self.steps[self._current_idx + 1:]:
            if self._matches_strict(detected_verb, detected_obj, future_step.action):
                self._failed_ids.append(current.id)
                self._skipped_ids.append(current.id)
                return self._make_state(FSMStatus.STEP_SKIPPED, current, next_step, now,
                                        error="STEP_SKIPPED",
                                        recovery=f"Please complete step {current.id}: {current.label} first.")

        # OUT_OF_SEQUENCE — matches a past step
        for past_step in self.steps[:self._current_idx]:
            if self._matches_strict(detected_verb, detected_obj, past_step.action):
                return self._make_state(FSMStatus.OUT_OF_SEQUENCE, current, next_step, now,
                                        error="OUT_OF_SEQUENCE",
                                        recovery=f"Step {past_step.id} is already done. Please proceed with step {current.id}.")

        # WRONG_ACTION — completely different
        self._failed_ids.append(current.id)
        return self._make_state(FSMStatus.WRONG_ACTION, current, next_step, now,
                                error="WRONG_ACTION",
                                recovery=f"Incorrect action. Please: {current.label}.")

    def reset(self) -> None:
        """Reset experiment to beginning."""
        self._current_idx = 0
        self._completed_ids.clear()
        self._failed_ids.clear()
        self._skipped_ids.clear()
        self._start_time = time.time()
        self._step_start = time.time()
        self._last_status = FSMStatus.WAITING
        logger.info("FSM reset.")

    def get_current_state(self) -> FSMState:
        """Get current state without processing a new action."""
        now = time.time()
        if self._current_idx >= len(self.steps):
            return self._make_state(FSMStatus.COMPLETED, None, None, now)
        current = self.steps[self._current_idx]
        next_step = self.steps[self._current_idx + 1] if self._current_idx + 1 < len(self.steps) else None
        return self._make_state(self._last_status, current, next_step, now)

    # ----------------------------------------------------------------------- #
    # Internal helpers
    # ----------------------------------------------------------------------- #
    def _advance(self) -> None:
        current = self.steps[self._current_idx]
        self._completed_ids.append(current.id)
        self._current_idx += 1
        self._step_start = time.time()
        logger.info("FSM: Step %d (%s) COMPLETED.", current.id, current.action)

    @staticmethod
    def _split_action(action: str) -> tuple[str, str]:
        """Split 'TAKE_RED_BOX' into ('TAKE', 'RED_BOX')."""
        parts = action.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ""

    @staticmethod
    def _matches(verb: str, obj: str, expected_action: str) -> bool:
        """
        Check if detected (verb, obj) matches an expected action string.
        Strict: only exact VERB+OBJ or VERB-only (when expected has no object) match.
        """
        expected_action = expected_action.upper()
        full_detected = f"{verb}_{obj}" if obj else verb
        full_detected = full_detected.upper()

        # Exact match (TAKE_RED_BOX == TAKE_RED_BOX)
        if full_detected == expected_action:
            return True

        # Verb-only detected, expected has no object component
        exp_verb, exp_obj = ExperimentFSM._split_action(expected_action)
        if verb == exp_verb and not obj and not exp_obj:
            return True

        # Object is a valid substring component of a compound expected (MAIN_BOX in OPEN_MAIN_BOX)
        if verb == exp_verb and obj and exp_obj and obj == exp_obj:
            return True

        return False

    @staticmethod
    def _matches_strict(verb: str, obj: str, expected_action: str) -> bool:
        """
        Strict match: requires exact VERB+OBJ match (no partial).
        Used for step-skip and out-of-sequence detection to avoid false positives.
        """
        expected_action = expected_action.upper()
        full_detected = f"{verb}_{obj}" if obj else verb
        full_detected = full_detected.upper()
        exp_verb, exp_obj = ExperimentFSM._split_action(expected_action)

        # Exact full match
        if full_detected == expected_action:
            return True
        # Verb+obj exact component match
        if verb == exp_verb and obj == exp_obj:
            return True
        return False

    def _make_state(
        self,
        status: FSMStatus,
        current: Optional[ExperimentStep],
        next_step: Optional[ExperimentStep],
        now: float,
        error: Optional[str] = None,
        recovery: Optional[str] = None,
    ) -> FSMState:
        self._last_status = status
        return FSMState(
            current_step_idx=self._current_idx,
            total_steps=len(self.steps),
            status=status,
            current_step=current,
            next_step=next_step,
            completed_step_ids=list(self._completed_ids),
            failed_step_ids=list(self._failed_ids),
            skipped_step_ids=list(self._skipped_ids),
            detected_action=self._last_detected_action,
            detected_object=self._last_detected_object,
            error_type=error,
            recovery_message=recovery,
            experiment_id=self.experiment_id,
            start_time=self._start_time,
            elapsed_seconds=now - self._start_time,
            step_start_time=self._step_start,
        )
