"""
ORBITA State Manager
=====================
Orchestrates the full reasoning pipeline:
  1. Receives action prediction from HAR
  2. Maps HAR action to experiment-specific action (verb + object)
  3. Runs FSM + rule engine
  4. Returns ValidationResult with voice message
  5. Triggers voice alert via callback

This is the single integration point between AI and FSM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from orbita.app.config import ExperimentStep, OrbitaConfig
from orbita.har.temporal_model import ActionPrediction
from orbita.reasoning.fsm import ExperimentFSM, FSMState, FSMStatus
from orbita.reasoning.rules import RuleMatch, RuleResult, evaluate_rules
from orbita.perception.object_detector import DetectedObject

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Complete result of processing one action frame."""
    fsm_state: FSMState
    rule_matches: list[RuleMatch]
    voice_message: str
    hud_message: str
    alert_level: str           # "info" | "warning" | "error" | "success"
    action_prediction: Optional[ActionPrediction] = None
    detected_objects: list[DetectedObject] = field(default_factory=list)
    fps: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = self.fsm_state.to_dict()
        d["voice_message"] = self.voice_message
        d["hud_message"] = self.hud_message
        d["alert_level"] = self.alert_level
        d["fps"] = round(self.fps, 1)
        d["latency_ms"] = round(self.latency_ms, 2)
        d["rules"] = [
            {"id": r.rule_id, "result": r.result.name, "message": r.message}
            for r in self.rule_matches
        ]
        if self.action_prediction:
            d["action_confidence"] = round(self.action_prediction.confidence, 3)
            d["next_action"] = self.action_prediction.next_action
            d["next_confidence"] = round(self.action_prediction.next_confidence, 3)
            d["is_uncertain"] = self.action_prediction.is_uncertain
        return d


# Mapping from HAR action verbs to experiment action verbs
# The HAR model outputs generic verbs; this maps them to experiment vocabulary
HAR_TO_EXPERIMENT_VERB: dict[str, str] = {
    "OPEN": "OPEN",
    "TAKE": "TAKE",
    "PLACE": "PLACE",
    "TRANSFER": "TRANSFER",
    "CLOSE": "CLOSE",
    "ACTIVATE": "ACTIVATE",
    "PERFORM": "PERFORM",
    "STORE": "STORE",
    "IDLE": "IDLE",
}


class StateManager:
    """
    Central orchestrator for ORBITA's reasoning pipeline.

    Usage:
        mgr = StateManager(config)
        result = mgr.process(action_prediction, detected_objects)
    """

    def __init__(
        self,
        config: OrbitaConfig,
        on_voice: Optional[Callable[[str], None]] = None,
        experiment_id: Optional[str] = None,
    ):
        self.config = config
        self.on_voice = on_voice or (lambda msg: None)

        # Generate experiment ID
        ts = int(time.time()) % 100000
        self.experiment_id = experiment_id or f"{config.experiment_id_prefix}{ts:05d}"

        # Initialize FSM
        self.fsm = ExperimentFSM(
            steps=config.experiment_steps,
            experiment_id=self.experiment_id,
            confidence_threshold=config.har.action_confidence_min,
        )

        self._last_voice_message: str = ""
        self._last_status: FSMStatus = FSMStatus.WAITING
        self._voice_debounce_seconds: float = 3.0
        self._last_voice_time: float = 0.0

        logger.info("StateManager initialized. Experiment: %s", self.experiment_id)

    # ----------------------------------------------------------------------- #
    # Main process method
    # ----------------------------------------------------------------------- #
    def process(
        self,
        prediction: ActionPrediction,
        detected_objects: list[DetectedObject],
        fps: float = 0.0,
        latency_ms: float = 0.0,
    ) -> ValidationResult:
        """
        Process one action prediction and return complete validation result.

        Args:
            prediction: HAR action prediction
            detected_objects: Current frame's detected objects
            fps: Current processing frame rate
            latency_ms: Current end-to-end latency

        Returns:
            ValidationResult
        """
        # Map HAR action to experiment action + object
        action_verb = HAR_TO_EXPERIMENT_VERB.get(prediction.action.upper(), "IDLE")
        detected_object = (
            prediction.target_object
            if prediction.target_object is not None
            else self._resolve_object(prediction, detected_objects)
        )

        # Run FSM
        fsm_state = self.fsm.process(
            detected_action=action_verb,
            detected_object=detected_object,
            confidence=prediction.confidence,
        )

        # Run rule engine (supplementary checks)
        rule_matches: list[RuleMatch] = []
        if fsm_state.current_step is not None and not prediction.is_uncertain:
            rule_matches = evaluate_rules(
                detected_action=action_verb,
                detected_object=detected_object,
                confidence=prediction.confidence,
                current_step=fsm_state.current_step,
                all_steps=self.config.experiment_steps,
                completed_step_ids=fsm_state.completed_step_ids,
                step_start_time=fsm_state.step_start_time,
                confidence_threshold=self.config.har.action_confidence_min,
            )

        # Generate messages
        voice_msg, hud_msg, alert_level = self._generate_messages(
            fsm_state, rule_matches, prediction
        )

        # Trigger voice (with debounce)
        self._maybe_speak(voice_msg, fsm_state.status)

        return ValidationResult(
            fsm_state=fsm_state,
            rule_matches=rule_matches,
            voice_message=voice_msg,
            hud_message=hud_msg,
            alert_level=alert_level,
            action_prediction=prediction,
            detected_objects=detected_objects,
            fps=fps,
            latency_ms=latency_ms,
        )

    def reset(self) -> None:
        """Reset experiment to step 1."""
        self.fsm.reset()
        self._last_voice_message = ""
        self._last_status = FSMStatus.WAITING
        ts = int(time.time()) % 100000
        self.experiment_id = f"{self.config.experiment_id_prefix}{ts:05d}"
        logger.info("StateManager reset. New experiment: %s", self.experiment_id)

    def get_current_state(self) -> FSMState:
        return self.fsm.get_current_state()

    # ----------------------------------------------------------------------- #
    # Object resolution
    # ----------------------------------------------------------------------- #
    def _resolve_object(
        self,
        prediction: ActionPrediction,
        detected_objects: list[DetectedObject],
    ) -> str:
        """
        Determine which object the action is performed on.

        Strategy: find the object closest to the primary interaction
        (by highest confidence detection of action-relevant class).
        """
        if not detected_objects:
            return ""

        # Filter out PERSON
        candidates = [o for o in detected_objects if o.class_name != "PERSON"]
        if not candidates:
            return ""

        # If current expected step mentions specific objects, prioritize those
        current = self.fsm.get_current_state().current_step
        if current:
            expected_objs = {o.upper() for o in current.expected_objects}
            matching = [o for o in candidates if o.class_name in expected_objs]
            if matching:
                return max(matching, key=lambda x: x.confidence).class_name

        # Otherwise return highest-confidence non-person object
        return max(candidates, key=lambda x: x.confidence).class_name

    # ----------------------------------------------------------------------- #
    # Message generation
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _generate_messages(
        fsm_state: FSMState,
        rule_matches: list[RuleMatch],
        prediction: ActionPrediction,
    ) -> tuple[str, str, str]:
        status = fsm_state.status

        if status == FSMStatus.COMPLETED:
            return (
                "Experiment complete. All steps successfully executed. Well done.",
                "EXPERIMENT COMPLETE ✓",
                "success",
            )

        if status == FSMStatus.UNCERTAIN or prediction.is_uncertain:
            return (
                "Unable to confidently identify the action. Please hold your position.",
                "UNCERTAIN — HOLD POSITION",
                "warning",
            )

        if status == FSMStatus.WAITING:
            if fsm_state.current_step:
                return (
                    fsm_state.current_step.voice_prompt,
                    f"AWAITING: {fsm_state.current_step.label.upper()}",
                    "info",
                )

        if status == FSMStatus.CORRECT:
            # Step just completed
            prev_step = None
            for s in [fsm_state.current_step]:
                if s:
                    prev_step = s
            msg = prev_step.completion_voice if prev_step else "Step completed."
            next_label = fsm_state.next_step.label if fsm_state.next_step else "none"
            return (
                msg,
                f"✓ COMPLETED → NEXT: {next_label.upper()}",
                "success",
            )

        if status == FSMStatus.WRONG_OBJECT:
            recovery = fsm_state.recovery_message or "Please use the correct object."
            return (
                recovery,
                f"⚠ WRONG OBJECT — {recovery}",
                "error",
            )

        if status == FSMStatus.STEP_SKIPPED:
            recovery = fsm_state.recovery_message or "Please complete the skipped step."
            return (
                recovery,
                f"⚠ STEP SKIPPED — {recovery}",
                "error",
            )

        if status == FSMStatus.OUT_OF_SEQUENCE:
            recovery = fsm_state.recovery_message or "Action out of sequence."
            return (
                recovery,
                f"⚠ OUT OF SEQUENCE — {recovery}",
                "warning",
            )

        if status == FSMStatus.WRONG_ACTION:
            recovery = fsm_state.recovery_message or "Please perform the correct action."
            return (
                recovery,
                f"⚠ WRONG ACTION — {recovery}",
                "error",
            )

        if status == FSMStatus.REPEATED_ACTION:
            return (
                "This step is already done. Please proceed to the next step.",
                "STEP ALREADY COMPLETED",
                "warning",
            )

        # Rule engine override messages (if FSM passed but rules flagged)
        for rm in rule_matches:
            if rm.result == RuleResult.FAIL and rm.recovery:
                return rm.recovery, f"⚠ {rm.error_type} — {rm.message}", "error"

        # Default
        step = fsm_state.current_step
        if step:
            return step.voice_prompt, f"STEP {step.id}: {step.label.upper()}", "info"
        return "Ready.", "READY", "info"

    # ----------------------------------------------------------------------- #
    # Voice debounce
    # ----------------------------------------------------------------------- #
    def _maybe_speak(self, message: str, status: FSMStatus) -> None:
        now = time.time()
        # Always speak on state changes; debounce repeated identical messages
        state_changed = status != self._last_status
        message_changed = message != self._last_voice_message
        debounce_elapsed = (now - self._last_voice_time) > self._voice_debounce_seconds

        if (state_changed or message_changed) and debounce_elapsed:
            if message:
                self.on_voice(message)
                self._last_voice_message = message
                self._last_voice_time = now
        self._last_status = status
