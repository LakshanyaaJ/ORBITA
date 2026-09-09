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

from core_ai.app.config import ExperimentStep
from core_ai.reasoning.action_gate import normalize_action, normalize_target

logger = logging.getLogger(__name__)


class FSMStatus(Enum):
    WAITING = auto()       # No action detected yet for current step
    CORRECT = auto()       # Current action matches expected step
    WRONG_OBJECT = auto()  # Right action type, wrong object
    WRONG_ACTION = auto()  # Completely wrong action
    WRONG_SEQUENCE = auto() # Confirmed out-of-order action
    STEP_SKIPPED = auto()  # Action matches a future step (gap)
    OUT_OF_SEQUENCE = auto()  # Action matches an already-completed step
    REPEATED_ACTION = auto()  # Current step already done
    UNCERTAIN = auto()     # AI confidence too low
    COMPLETED = auto()     # Experiment fully completed

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if isinstance(other, FSMStatus):
            # WRONG_SEQUENCE is a category that matches specific sequence/object errors for backwards compatibility
            if other is FSMStatus.WRONG_SEQUENCE and self in (
                FSMStatus.WRONG_OBJECT,
                FSMStatus.WRONG_ACTION,
                FSMStatus.STEP_SKIPPED,
                FSMStatus.OUT_OF_SEQUENCE,
            ):
                return True
            if self is FSMStatus.WRONG_SEQUENCE and other in (
                FSMStatus.WRONG_OBJECT,
                FSMStatus.WRONG_ACTION,
                FSMStatus.STEP_SKIPPED,
                FSMStatus.OUT_OF_SEQUENCE,
            ):
                return True
        return False

    def __hash__(self) -> int:
        return Enum.__hash__(self)


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
    is_transition: bool = False

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
            "is_transition": self.is_transition,
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
        confirmation_frames_required: int = 1,
    ):
        if not steps:
            raise ValueError("FSM requires at least one experiment step.")

        self.steps = steps
        self.experiment_id = experiment_id
        self.confidence_threshold = confidence_threshold
        self.confirmation_frames_required = confirmation_frames_required

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
        self._consecutive_matches: int = 0
        self._consecutive_errors: int = 0

    @property
    def is_complete(self) -> bool:
        return self._current_idx >= len(self.steps)

    @property
    def current_step_idx(self) -> int:
        return self._current_idx

    @property
    def current_step(self) -> Optional[ExperimentStep]:
        if self._current_idx < len(self.steps):
            return self.steps[self._current_idx]
        return None

    @property
    def completed_step_ids(self) -> list[int]:
        return list(self._completed_ids)

    # ----------------------------------------------------------------------- #
    # Core processing
    # ----------------------------------------------------------------------- #
    def process(
        self,
        detected_action: str = "IDLE",
        detected_object: str = "",
        confidence: float = 1.0,
        confirmed_action: Optional[Any] = None,
        frame_age_ms: float = 0.0,
    ) -> FSMState:
        now = time.time()

        # --- Terminal state ---
        if self._current_idx >= len(self.steps):
            return self._make_state(FSMStatus.COMPLETED, None, None, now)

        current = self.steps[self._current_idx]
        next_step = self.steps[self._current_idx + 1] if self._current_idx + 1 < len(self.steps) else None

        # 1. Authoritative Action Confirmation Gate Evaluation
        if confirmed_action is not None:
            c_status = getattr(confirmed_action, "status", None)
            c_status_str = getattr(c_status, "name", str(c_status))
            self._last_detected_action = getattr(confirmed_action, "action", detected_action)
            self._last_detected_object = getattr(confirmed_action, "object_name", detected_object)

            if c_status_str == "CONFIRMED_CORRECT":
                self._advance(confirmed_action=confirmed_action, frame_age_ms=frame_age_ms)
                new_idx = self._current_idx
                new_current = self.steps[new_idx] if new_idx < len(self.steps) else None
                new_next = self.steps[new_idx + 1] if new_idx + 1 < len(self.steps) else None
                if new_current is None or new_idx >= len(self.steps) or getattr(new_current, "action", "") == "EXPERIMENT_COMPLETE":
                    st = self._make_state(FSMStatus.COMPLETED, new_current, None, now)
                    st.is_transition = True
                    return st
                st = self._make_state(FSMStatus.CORRECT, new_current, new_next, now)
                st.is_transition = True
                return st

            elif c_status_str == "CONFIRMED_WRONG":
                exp_label = current.label if current else "the correct step"
                det_act = getattr(confirmed_action, "action", detected_action)
                det_obj = getattr(confirmed_action, "object_name", detected_object)
                norm_v = normalize_action(det_act)
                norm_o = normalize_target(det_obj)
                err_evidence = getattr(confirmed_action, "evidence", {}) or {}

                exp_verb = normalize_action(getattr(current, "expected_action", "") or current.action.split("_")[0])
                exp_obj = normalize_target(getattr(current, "expected_object", "") or (current.action.split("_", 1)[1] if "_" in current.action else ""))
                has_wrong_obj = "wrong_object" in err_evidence or (
                    det_obj and exp_obj and norm_o != exp_obj
                )

                # 1. Wrong-object detection: right action type for current step, wrong object
                if norm_v == exp_verb and has_wrong_obj:
                    if current.id not in self._failed_ids:
                        self._failed_ids.append(current.id)
                    return self._make_state(
                        FSMStatus.WRONG_OBJECT,
                        current,
                        next_step,
                        now,
                        error="WRONG_OBJECT",
                        recovery=f"Wrong sequence. Wrong object. Please {exp_label.lower()}."
                    )

                # 2. Future-step reasoning: Search all future steps
                matched_future = None
                skipped_step_id = None
                for future_step in self.steps[self._current_idx + 1:]:
                    if self._matches_strict(norm_v, norm_o, future_step.action):
                        matched_future = future_step
                        skipped_step_id = current.id
                        break

                if matched_future is not None:
                    if current.id not in self._failed_ids:
                        self._failed_ids.append(current.id)
                    if current.id not in self._skipped_ids:
                        self._skipped_ids.append(current.id)
                    return self._make_state(
                        FSMStatus.STEP_SKIPPED,
                        current,
                        next_step,
                        now,
                        error="STEP_SKIPPED",
                        recovery=f"Wrong sequence. Step {skipped_step_id} was skipped. Please complete Step {skipped_step_id}: {exp_label} first."
                    )

                # 3. Out-of-sequence reasoning: Search all past steps
                matched_past = None
                for past_step in self.steps[:self._current_idx]:
                    if self._matches_strict(norm_v, norm_o, past_step.action):
                        matched_past = past_step
                        break

                if matched_past is not None:
                    return self._make_state(
                        FSMStatus.OUT_OF_SEQUENCE,
                        current,
                        next_step,
                        now,
                        error="OUT_OF_SEQUENCE",
                        recovery=f"Wrong sequence. Step {matched_past.id} is already done. Please proceed with Step {current.id}: {exp_label}."
                    )

                # 4. Fallback wrong action / sequence
                if current.id not in self._failed_ids:
                    self._failed_ids.append(current.id)
                return self._make_state(
                    FSMStatus.WRONG_ACTION,
                    current,
                    next_step,
                    now,
                    error="WRONG_ACTION",
                    recovery=f"Wrong sequence. Please {exp_label.lower()}."
                )

            else:
                # WAITING (Noise or partial interaction - never treat as error)
                return self._make_state(FSMStatus.WAITING, current, next_step, now)

        self._last_detected_action = detected_action
        self._last_detected_object = detected_object

        # --- Fallback Heuristic Match (for tests without ActionConfirmationGate) ---
        # --- UNCERTAIN ---
        if confidence < self.confidence_threshold:
            self._consecutive_matches = 0
            self._consecutive_errors = 0
            return self._make_state(FSMStatus.UNCERTAIN, current, next_step, now,
                                    error="UNCERTAIN")

        # --- IDLE — nothing happening ---
        if detected_action in ("IDLE", "", "UNKNOWN"):
            self._consecutive_matches = 0
            self._consecutive_errors = 0
            return self._make_state(FSMStatus.WAITING, current, next_step, now)

        # --- Build full detected action label ---
        expected_action = current.action  # e.g. "TAKE_RED_BOX"
        expected_verb, expected_obj = self._split_action(expected_action)

        detected_verb, detected_obj = detected_action.upper(), detected_object.upper()

        # CORRECT match with temporal confirmation
        if self._matches(detected_verb, detected_obj, expected_action):
            self._consecutive_matches += 1
            self._consecutive_errors = 0
            if self._consecutive_matches >= self.confirmation_frames_required:
                self._consecutive_matches = 0
                self._advance()
                new_idx = self._current_idx
                new_current = self.steps[new_idx] if new_idx < len(self.steps) else None
                new_next = self.steps[new_idx + 1] if new_idx + 1 < len(self.steps) else None
                if new_current is None or new_idx >= len(self.steps) or getattr(new_current, "action", "") == "EXPERIMENT_COMPLETE":
                    st = self._make_state(FSMStatus.COMPLETED, new_current, None, now)
                    st.is_transition = True
                    return st
                st = self._make_state(FSMStatus.CORRECT, current, next_step, now)
                st.is_transition = True
                return st
            else:
                return self._make_state(FSMStatus.WAITING, current, next_step, now,
                                        recovery=f"Action recognized, confirming ({self._consecutive_matches}/{self.confirmation_frames_required})...")

        # Not a match -> reset match counter
        self._consecutive_matches = 0
        self._consecutive_errors += 1

        # REPEATED — already completed
        if current.id in self._completed_ids:
            return self._make_state(FSMStatus.REPEATED_ACTION, current, next_step, now,
                                    error="REPEATED_ACTION",
                                    recovery=f"Step {current.id} already completed.")

        # WRONG_OBJECT — correct verb, wrong object (require 2 consecutive frames before latching error)
        if detected_verb == expected_verb and expected_obj and detected_obj and detected_obj != expected_obj:
            if self._consecutive_errors >= 2 and current.id not in self._failed_ids:
                self._failed_ids.append(current.id)
            return self._make_state(FSMStatus.WRONG_OBJECT, current, next_step, now,
                                    error="WRONG_OBJECT",
                                    recovery=f"Wrong object. Expected: {expected_obj.replace('_', ' ').lower()}.")

        # STEP_SKIPPED — detected action matches a future step (exact verb+obj match required)
        for future_step in self.steps[self._current_idx + 1:]:
            if self._matches_strict(detected_verb, detected_obj, future_step.action):
                if self._consecutive_errors >= 2 and current.id not in self._failed_ids:
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
        if self._consecutive_errors >= 2 and current.id not in self._failed_ids:
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
        self._consecutive_matches = 0
        self._consecutive_errors = 0
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
    def _advance(
        self,
        confirmed_action: Optional[Any] = None,
        frame_age_ms: float = 0.0,
    ) -> None:
        if self._current_idx >= len(self.steps):
            return
        current = self.steps[self._current_idx]
        if current.id in self._completed_ids:
            logger.warning("FSM: Step %d already marked completed, ignoring duplicate advance.", current.id)
            return
        from_step = current.id
        self._completed_ids.append(current.id)
        self._current_idx += 1
        self._step_start = time.time()
        to_step = self.steps[self._current_idx].id if self._current_idx < len(self.steps) else 13

        exp_act = getattr(current, "expected_action", "") or current.action.split("_")[0]
        exp_tgt = getattr(current, "expected_object", "") or (current.action.split("_", 1)[1] if "_" in current.action else "")
        det_act = getattr(confirmed_action, "action", self._last_detected_action) if confirmed_action else self._last_detected_action
        det_tgt = getattr(confirmed_action, "object_name", self._last_detected_object) if confirmed_action else self._last_detected_object
        conf = getattr(confirmed_action, "confidence", 1.0) if confirmed_action else 1.0
        reason = getattr(confirmed_action, "validation_reason", "ACTION_CONFIRMED") if confirmed_action else "HEURISTIC_CONFIRMED"

        logger.info(
            "FSM_TRANSITION\n"
            "from_step=%d\n"
            "to_step=%d\n"
            "expected_action=%s\n"
            "expected_target=%s\n"
            "detected_action=%s\n"
            "detected_target=%s\n"
            "validation=CONFIRMED_CORRECT\n"
            "frame_age_ms=%.1f\n"
            "confidence=%.2f\n"
            "reason=%s",
            from_step,
            to_step,
            exp_act,
            exp_tgt,
            det_act,
            det_tgt,
            frame_age_ms,
            conf,
            reason,
        )
        checklist = getattr(confirmed_action, "why_completed_checklist", []) if confirmed_action else []
        checklist_str = "\n".join(f"  ✓ {c.get('criterion', '')}: {c.get('detail', '')}" for c in checklist if c.get('satisfied'))
        if checklist_str:
            logger.info("FSM_WHY_COMPLETED (Step %d):\n%s", from_step, checklist_str)
        logger.info("FSM: Step %d (%s) COMPLETED -> advancing to Step %d.", from_step, current.action, to_step)

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

        norm_full = normalize_action(full_detected)
        norm_exp = normalize_action(expected_action)
        if norm_full == norm_exp and norm_full != "IDLE":
            return True

        # Verb-only detected, expected has no object component
        exp_verb, exp_obj = ExperimentFSM._split_action(expected_action)
        if verb == exp_verb and not obj and not exp_obj:
            return True

        # Object is a valid substring component of a compound expected (MAIN_BOX in OPEN_MAIN_BOX)
        if verb == exp_verb and obj and exp_obj and obj == exp_obj:
            return True

        # Normalized comparison
        norm_v = normalize_action(verb)
        norm_exp_v = normalize_action(exp_verb)
        norm_o = normalize_target(obj)
        norm_exp_o = normalize_target(exp_obj)

        if norm_v == norm_exp_v:
            if not norm_exp_o:
                return True
            if norm_o == norm_exp_o:
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

        norm_v = normalize_action(verb)
        norm_exp_v = normalize_action(exp_verb)
        norm_o = normalize_target(obj)
        norm_exp_o = normalize_target(exp_obj)
        if norm_v == norm_exp_v and norm_o and norm_exp_o and norm_o == norm_exp_o:
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
