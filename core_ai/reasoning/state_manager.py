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

from core_ai.app.config import ExperimentStep, OrbitaConfig
from core_ai.har.temporal_model import ActionPrediction
from core_ai.reasoning.fsm import ExperimentFSM, FSMState, FSMStatus
from core_ai.reasoning.rules import RuleMatch, RuleResult, evaluate_rules
from core_ai.perception.object_detector import DetectedObject

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
    left_hand: Any = None
    right_hand: Any = None
    interactions: list[Any] = field(default_factory=list)

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

        # Rich Hand Telemetry
        primary_int = None
        if self.interactions:
            # Sort by HOLDING > CONTACT > NEAR
            candidates = sorted(self.interactions, key=lambda x: (getattr(x, "state", 0), getattr(x, "confidence", 0)), reverse=True)
            if candidates:
                primary_int = candidates[0]

        d["hand_telemetry"] = {
            "left": {
                "hand_id": getattr(self.left_hand, "hand_id", -1),
                "tracked": getattr(self.left_hand, "is_visible", False),
                "confidence": round(getattr(self.left_hand, "confidence", 0.0), 3),
                "status": getattr(self.left_hand, "track_status", "LOST"),
                "speed_px_s": round(getattr(self.left_hand, "speed", 0.0), 1),
                "landmarks_count": 21 if (getattr(self.left_hand, "finger_landmarks", None) is not None) else 0,
            },
            "right": {
                "hand_id": getattr(self.right_hand, "hand_id", -1),
                "tracked": getattr(self.right_hand, "is_visible", False),
                "confidence": round(getattr(self.right_hand, "confidence", 0.0), 3),
                "status": getattr(self.right_hand, "track_status", "LOST"),
                "speed_px_s": round(getattr(self.right_hand, "speed", 0.0), 1),
                "landmarks_count": 21 if (getattr(self.right_hand, "finger_landmarks", None) is not None) else 0,
            },
            "primary_interaction": {
                "hand_side": getattr(primary_int, "hand_side", ""),
                "object_class": getattr(primary_int, "object_class", ""),
                "state": getattr(getattr(primary_int, "state", None), "name", "NOT_INTERACTING"),
                "confidence": round(getattr(primary_int, "confidence", 0.0), 3),
                "is_holding": getattr(primary_int, "is_holding", False),
                "transfer_event": getattr(primary_int, "transfer_event", None),
            } if primary_int else None,
        }

        # Critical Perception Debug Mode breakdown
        raw_yolo = [
            {"class": o.class_name, "conf": round(float(o.confidence), 3), "bbox": list(o.bbox)}
            for o in self.detected_objects if getattr(o, "source", "") == "yolo"
        ]
        hybrid = [
            {"class": o.class_name, "conf": round(float(o.confidence), 3), "bbox": list(o.bbox)}
            for o in self.detected_objects if getattr(o, "source", "") == "chroma"
        ]
        d["perception_debug"] = {
            "raw_yolo": raw_yolo,
            "hybrid": hybrid,
            "hand": {
                "left": f"LEFT_HAND {round(float(getattr(self.left_hand, 'confidence', 0.0)), 2)}" if getattr(self.left_hand, 'is_visible', False) else "NOT_DETECTED",
                "right": f"RIGHT_HAND {round(float(getattr(self.right_hand, 'confidence', 0.0)), 2)}" if getattr(self.right_hand, 'is_visible', False) else "NOT_DETECTED",
            },
            "tracks": [
                f"{o.class_name} #{o.track_id}" for o in self.detected_objects if getattr(o, "track_id", -1) > 0
            ] + ([f"LEFT_HAND #{self.left_hand.hand_id}"] if getattr(self.left_hand, "is_visible", False) else [])
              + ([f"RIGHT_HAND #{self.right_hand.hand_id}"] if getattr(self.right_hand, "is_visible", False) else []),
            "interactions": [
                f"HAND #{getattr(i, 'hand_side', '')} <-> {getattr(i, 'object_class', '')} = {getattr(getattr(i, 'state', None), 'name', str(getattr(i, 'state', '')))}"
                for i in self.interactions if getattr(i, "state", 0) != 0
            ],
        }

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
            confirmation_frames_required=3,
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
        left_hand: Any = None,
        right_hand: Any = None,
        interactions: Optional[list[Any]] = None,
    ) -> ValidationResult:
        """
        Process one action prediction and return complete validation result.

        Args:
            prediction: HAR action prediction
            detected_objects: Current frame's detected objects
            fps: Current processing frame rate
            latency_ms: Current end-to-end latency
            left_hand: Tracked left hand state
            right_hand: Tracked right hand state
            interactions: List of active hand-object interactions

        Returns:
            ValidationResult
        """
        # Map HAR action to experiment action + object
        action_verb = HAR_TO_EXPERIMENT_VERB.get(prediction.action.upper(), "IDLE")
        detected_object = (
            prediction.target_object
            if prediction.target_object is not None
            else self._resolve_object(prediction, detected_objects, interactions)
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
            left_hand=left_hand,
            right_hand=right_hand,
            interactions=interactions or [],
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
        interactions: Optional[list[Any]] = None,
    ) -> str:
        """
        Determine which object the action is performed on.
        Prioritizes direct physical interactions (HOLDING, CONTACT) over distance heuristics.
        """
        # 1. First check physical interactions from hand tracking
        if interactions:
            # Check for actively held objects
            holding = [i for i in interactions if getattr(i, "is_holding", False)]
            if holding:
                best_hold = max(holding, key=lambda x: getattr(x, "confidence", 0.0))
                return best_hold.object_class

            # Check for contact objects
            contact = [i for i in interactions if getattr(getattr(i, "state", None), "name", "") in ("CONTACT", "RELEASING")]
            if contact:
                best_contact = max(contact, key=lambda x: getattr(x, "confidence", 0.0))
                return best_contact.object_class

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
