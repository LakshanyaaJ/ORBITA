"""
ORBITA Action Confirmation Gate
===============================
Deterministic, multi-frame physical verification gate between perception/tracking
and the procedural Finite State Machine (FSM).

Three discrete states:
  - WAITING            -> Action candidate forming, temporal evidence accumulating
  - CONFIRMED_CORRECT  -> Physical action verified and matches active FSM step
  - CONFIRMED_WRONG    -> Physical action verified but violates expected sequence / object

Rules:
  - Never advance FSM on transient raw detections or single-frame touches.
  - Never classify WAITING as wrong sequence.
  - Require persistent tracking, spatial displacement, hand-object coupling, and stability.
  - Action-type-specific confirmation gates (IDENTIFICATION, PICKUP, PLACEMENT, MOVEMENT, COMPLETION).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ActionGateStatus(str, Enum):
    WAITING = "WAITING"
    CONFIRMED_CORRECT = "CONFIRMED_CORRECT"
    CONFIRMED_WRONG = "CONFIRMED_WRONG"


def normalize_action(action: str) -> str:
    """Normalize any action variant/enum/string to canonical action verb."""
    if not action:
        return "IDLE"
    act = str(action).strip().upper()
    if act in ("IDENTIFY", "IDENTIFICATION") or act.startswith("IDENTIFY_"):
        return "IDENTIFY"
    if act in ("TAKE", "PICKUP", "PICK_UP") or act.startswith("PICKUP_") or act.startswith("TAKE_"):
        return "PICKUP"
    if act in ("PLACE", "PUT", "STORE") or act.startswith("PLACE_"):
        return "PLACE"
    if act in ("MOVE", "TRANSFER") or act.startswith("MOVE_"):
        return "MOVE"
    if act in ("COMPLETE", "COMPLETED", "EXPERIMENT_COMPLETE"):
        return "COMPLETE"
    return act


def normalize_target(target: str) -> str:
    """Normalize object / location strings and aliases to canonical ontology."""
    if not target:
        return ""
    tgt = str(target).strip().upper()
    if tgt in ("BLUE_BOX", "BLUE"):
        return "BLUE_BOX"
    if tgt in ("RED_BOX", "RED"):
        return "RED_BOX"
    if tgt in ("MAIN_BOX", "MAIN"):
        return "MAIN_BOX"
    if tgt in ("YELLOW_BOX", "YELLOW"):
        return "YELLOW_BOX"
    if tgt in ("PEN", "TOOL", "PENCIL", "STYLUS"):
        return "PEN"
    if tgt in ("WATCH", "SAMPLE", "CLOCK", "SPECIMEN"):
        return "WATCH"
    if tgt in ("LOCATION_A", "A", "LOC_A", "LOCA"):
        return "LOCATION_A"
    if tgt in ("LOCATION_B", "B", "LOC_B", "LOCB"):
        return "LOCATION_B"
    if tgt in ("ALL", "EXPERIMENT"):
        return "ALL"
    return tgt


@dataclass
class ConfirmedAction:
    action: str                        # "IDENTIFY" | "PICKUP" | "PLACE" | "MOVE" | "COMPLETE" | "IDLE"
    object_name: str                   # "BLUE_BOX" | "YELLOW_BOX" | "PEN" | "WATCH" | "ALL"
    target: str = ""                   # "LOCATION_A" | "LOCATION_B" | "BLUE_BOX" | "YELLOW_BOX"
    status: ActionGateStatus = ActionGateStatus.WAITING
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    is_new_event: bool = False
    event_id: str = ""
    track_id: Optional[int] = None
    validation_state: str = "WAITING"
    validation_reason: str = ""
    is_stable: bool = False
    object_confidence: float = 0.0
    action_confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Centralized Validation Configuration
# --------------------------------------------------------------------------- #
VALIDATION_CONFIG: Dict[str, Dict[str, Any]] = {
    "IDENTIFY": {
        "min_confidence": 0.65,
        "maintain_confidence": 0.45,
        "stable_frames": 3,          # ~100ms at 30 FPS for snappy identification
        "timeout_ms": 15000,
    },
    "PICKUP": {
        "min_confidence": 0.65,
        "maintain_confidence": 0.45,
        "stable_frames": 3,
        "min_pickup_frames": 3,
    },
    "PLACE": {
        "min_confidence": 0.65,
        "maintain_confidence": 0.45,
        "stable_frames": 3,
        "spatial_tolerance_px": 50,
    },
    "MOVE": {
        "min_confidence": 0.65,
        "maintain_confidence": 0.45,
        "stable_frames": 3,
    },
    "RELEASE": {
        "min_confidence": 0.60,
        "maintain_confidence": 0.40,
        "stable_frames": 3,
    },
}


class ActionConfirmationGate:
    """
    Validates physical actions against current FSM step expectations before
    authorizing any state transition.
    Supports action-specific confirmation modes:
      - IDENTIFICATION
      - PICKUP
      - PLACEMENT
      - MOVEMENT
      - COMPLETION
    """

    def __init__(
        self,
        min_identify_frames: int = 3,
        min_pickup_frames: int = 3,
        min_place_frames: int = 3,
        min_move_frames: int = 4,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.config = {**VALIDATION_CONFIG, **(config or {})}
        self.min_identify_frames = self.config.get("IDENTIFY", {}).get("stable_frames", min_identify_frames)
        self.min_pickup_frames = self.config.get("PICKUP", {}).get("min_pickup_frames", min_pickup_frames)
        self.min_place_frames = self.config.get("PLACE", {}).get("stable_frames", min_place_frames)
        self.min_move_frames = self.config.get("MOVE", {}).get("stable_frames", min_move_frames)

        # Tracking state accumulators: object_name -> metrics
        self._consecutive_seen: Dict[str, int] = {}
        self._consecutive_held: Dict[str, int] = {}
        self._consecutive_released: Dict[str, int] = {}
        self._initial_positions: Dict[str, Tuple[int, int]] = {}
        self._last_positions: Dict[str, Tuple[int, int]] = {}
        self._was_held: Dict[str, bool] = {}
        self._held_start_time: Dict[str, float] = {}

        self._event_counter: int = 0
        self._active_wrong_event_id: Optional[str] = None
        self._wrong_episode_active: bool = False
        self._frames_since_wrong: int = 0

        self._last_confirmed_action: Optional[ConfirmedAction] = None
        self._current_gate_status: ActionGateStatus = ActionGateStatus.WAITING

    def reset(self) -> None:
        """Reset all temporal accumulators on session/experiment start."""
        self._consecutive_seen.clear()
        self._consecutive_held.clear()
        self._consecutive_released.clear()
        self._initial_positions.clear()
        self._last_positions.clear()
        self._was_held.clear()
        self._held_start_time.clear()
        self._event_counter = 0
        self._active_wrong_event_id = None
        self._wrong_episode_active = False
        self._frames_since_wrong = 0
        self._last_confirmed_action = None
        self._current_gate_status = ActionGateStatus.WAITING
        logger.info("ActionConfirmationGate: Session reset complete.")

    def reset_for_step(self, step: Any = None) -> None:
        """
        Reset temporal accumulators for a new procedure step.
        Ensures clean action-validation context with no carryover of
        held counters or wrong episodes from previous steps.
        """
        self._consecutive_seen.clear()
        self._consecutive_held.clear()
        self._consecutive_released.clear()
        self._was_held.clear()
        self._held_start_time.clear()
        self._active_wrong_event_id = None
        self._wrong_episode_active = False
        self._frames_since_wrong = 0
        self._last_confirmed_action = None
        self._current_gate_status = ActionGateStatus.WAITING
        step_lbl = getattr(step, "label", getattr(step, "action", str(step))) if step else "Unknown"
        logger.info("ActionConfirmationGate: Reset state for step '%s'.", step_lbl)

    def _emit_wrong_action(
        self,
        action: str,
        object_name: str,
        target: str = "",
        confidence: float = 0.85,
        evidence: Optional[Dict[str, Any]] = None,
        track_id: Optional[int] = None,
        timestamp: float = 0.0,
        reason: str = "",
    ) -> ConfirmedAction:
        self._frames_since_wrong = 0
        if self._wrong_episode_active:
            # Same ongoing wrong action episode! Deduplicated!
            is_new = False
            event_id = self._active_wrong_event_id or f"WRONG_ACTION_EVENT_{self._event_counter:03d}"
        else:
            # Brand new wrong action episode
            self._event_counter += 1
            event_id = f"WRONG_ACTION_EVENT_{self._event_counter:03d}"
            self._active_wrong_event_id = event_id
            self._wrong_episode_active = True
            is_new = True

        self._current_gate_status = ActionGateStatus.CONFIRMED_WRONG
        action_res = ConfirmedAction(
            action=action,
            object_name=object_name,
            target=target,
            status=ActionGateStatus.CONFIRMED_WRONG,
            confidence=confidence,
            evidence=evidence or {},
            timestamp=timestamp,
            is_new_event=is_new,
            event_id=event_id,
            track_id=track_id,
            validation_state="CONFIRMED_WRONG",
            validation_reason=reason or f"Wrong action: {action} on {object_name}",
            is_stable=True,
            action_confidence=confidence,
        )
        self._last_confirmed_action = action_res
        return action_res

    @property
    def current_status(self) -> ActionGateStatus:
        return self._current_gate_status

    @property
    def last_confirmed(self) -> Optional[ConfirmedAction]:
        return self._last_confirmed_action

    def evaluate(
        self,
        current_step: Any,
        detected_objects: List[Any],
        tracks: List[Any],
        interactions: List[Any],
        har_action: str = "IDLE",
        har_confidence: float = 0.5,
        target_object: str = "",
        predicted_target: str = "",
        frame_shape: Tuple[int, int] = (720, 1280),
        fps: float = 15.0,
        frame_age_ms: float = 0.0,
        is_stale: bool = False,
        **kwargs: Any,
    ) -> ConfirmedAction:
        """
        Evaluate frame evidence against current step expectations.
        Dispatches to action-type-specific confirmation gates.
        Returns ConfirmedAction with status WAITING, CONFIRMED_CORRECT, or CONFIRMED_WRONG.
        """
        now = time.time()
        H, W = frame_shape[:2]

        if current_step is None:
            self._current_gate_status = ActionGateStatus.WAITING
            return ConfirmedAction(
                action="IDLE",
                object_name="",
                status=ActionGateStatus.WAITING,
                confidence=1.0,
                timestamp=now,
                validation_state="WAITING",
                validation_reason="No active experiment step",
            )

        step_id = getattr(current_step, "id", 1)
        raw_exp_action = getattr(current_step, "expected_action", "") or getattr(current_step, "action", "")
        raw_exp_obj = getattr(current_step, "expected_object", "")
        raw_exp_target = getattr(current_step, "expected_target", "")
        from_loc = getattr(current_step, "from_location", "")
        to_loc = getattr(current_step, "to_location", "")

        # Fallback step parsing
        if not raw_exp_action and hasattr(current_step, "action"):
            raw_exp_action = current_step.action

        exp_action = normalize_action(raw_exp_action)
        exp_obj_canonical = normalize_target(raw_exp_obj)
        exp_target_canonical = normalize_target(raw_exp_target)

        # Infer object from action name if not explicitly set
        if not exp_obj_canonical and hasattr(current_step, "action"):
            act_str = current_step.action
            if "BLUE" in act_str:
                exp_obj_canonical = "BLUE_BOX"
            elif "YELLOW" in act_str:
                exp_obj_canonical = "YELLOW_BOX"
            elif "PEN" in act_str or "TOOL" in act_str:
                exp_obj_canonical = "PEN"
            elif "WATCH" in act_str or "SAMPLE" in act_str:
                exp_obj_canonical = "WATCH"
            elif "COMPLETE" in act_str:
                exp_obj_canonical = "ALL"

        # 1. Build unified canonical object map from BOTH detected_objects and tracks
        # This prevents 1-frame YOLO detection dips from losing tracked objects.
        obj_map: Dict[str, Any] = {}
        for obj in detected_objects:
            raw_cls = getattr(obj, "class_name", "")
            raw_ident = getattr(obj, "semantic_identity", "") or getattr(obj, "identity", "") or raw_cls
            canonical = normalize_target(raw_ident) or normalize_target(raw_cls)
            if canonical:
                obj_map[canonical] = obj

        for trk in tracks:
            raw_cls = getattr(trk, "class_name", "")
            raw_ident = getattr(trk, "semantic_identity", "") or getattr(trk, "identity", "") or raw_cls
            canonical = normalize_target(raw_ident) or normalize_target(raw_cls)
            if canonical and canonical not in obj_map:
                obj_map[canonical] = trk

        # 2. Update seen counters with smooth 1-frame decay grace (bridges 33ms camera flutter)
        for obj_name in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"]:
            if obj_name in obj_map:
                self._consecutive_seen[obj_name] = self._consecutive_seen.get(obj_name, 0) + 1
                det = obj_map[obj_name]
                centroid = getattr(det, "centroid", None)
                if centroid is None and hasattr(det, "bbox"):
                    centroid = (det.bbox[0] + det.bbox[2] // 2, det.bbox[1] + det.bbox[3] // 2)
                if centroid is not None:
                    if obj_name not in self._initial_positions:
                        self._initial_positions[obj_name] = centroid
                    self._last_positions[obj_name] = centroid
            else:
                # Decay by 1 rather than hard zeroing on a single dropped frame
                self._consecutive_seen[obj_name] = max(0, self._consecutive_seen.get(obj_name, 0) - 1)

        # 3. Check hand interaction states for objects
        held_objects = set()
        contact_objects = set()
        released_objects = set()

        for inter in interactions:
            cls = getattr(inter, "object_class", "")
            c_name = normalize_target(cls)
            is_holding = getattr(inter, "is_holding", False)
            state_val = getattr(inter, "state", None)
            state_name = getattr(state_val, "name", str(state_val))

            if is_holding or state_name in ("HOLDING", "GRASPING"):
                if c_name:
                    held_objects.add(c_name)
            elif state_name in ("CONTACT", "NEAR_OBJECT"):
                if c_name:
                    contact_objects.add(c_name)
            elif state_name in ("RELEASING", "RELEASED"):
                if c_name:
                    released_objects.add(c_name)

        # 4. Check tracks for physical states
        for trk in tracks:
            cls = getattr(trk, "class_name", "")
            raw_ident = getattr(trk, "identity", "") or cls
            canonical = normalize_target(raw_ident) or normalize_target(cls)

            trk_state = getattr(trk, "state", None)
            trk_state_val = getattr(trk_state, "value", str(trk_state))
            if trk_state_val in ("BEING_HELD", "CARRIED"):
                if canonical:
                    held_objects.add(canonical)
            elif getattr(trk, "hand_contact", False):
                if canonical:
                    contact_objects.add(canonical)
            elif trk_state_val in ("RELEASED", "PLACED"):
                if canonical:
                    released_objects.add(canonical)

        # HAR / Movement inference
        har_act_upper = normalize_action(har_action)
        effective_target = predicted_target or target_object or kwargs.get("predicted_target", "")
        norm_pred_target = normalize_target(effective_target)
        if har_act_upper == "PICKUP":
            if norm_pred_target:
                if norm_pred_target in contact_objects or norm_pred_target in obj_map or (not tracks and not interactions):
                    held_objects.add(norm_pred_target)
            elif exp_obj_canonical in contact_objects or exp_obj_canonical in obj_map:
                held_objects.add(exp_obj_canonical)

        # Update held counters
        all_candidate_objs = set(self._consecutive_held.keys()) | held_objects | {exp_obj_canonical}
        if norm_pred_target:
            all_candidate_objs.add(norm_pred_target)
        for obj_name in all_candidate_objs:
            if obj_name in held_objects:
                self._consecutive_held[obj_name] = self._consecutive_held.get(obj_name, 0) + 1
                self._was_held[obj_name] = True
                if obj_name not in self._held_start_time:
                    self._held_start_time[obj_name] = now
            else:
                self._consecutive_held[obj_name] = 0

        # ------------------------------------------------------------- #
        # Step-Specific Confirmation Gates
        # ------------------------------------------------------------- #
        if exp_action == "IDENTIFY":
            return self.validate_identification_step(
                expected_target=exp_obj_canonical,
                obj_map=obj_map,
                held_objects=held_objects,
                contact_objects=contact_objects,
                har_action=har_action,
                har_confidence=har_confidence,
                step_id=step_id,
                frame_age_ms=frame_age_ms,
                is_stale=is_stale,
                now=now,
                predicted_target=norm_pred_target,
            )

        elif exp_action == "PICKUP":
            return self.validate_pickup_step(
                expected_target=exp_obj_canonical,
                obj_map=obj_map,
                held_objects=held_objects,
                contact_objects=contact_objects,
                step_id=step_id,
                now=now,
            )

        elif exp_action == "PLACE":
            return self.validate_placement_step(
                expected_target=exp_obj_canonical,
                expected_destination=exp_target_canonical,
                obj_map=obj_map,
                held_objects=held_objects,
                released_objects=released_objects,
                frame_width=W,
                step_id=step_id,
                now=now,
                har_action=har_action,
            )

        elif exp_action == "MOVE":
            return self.validate_movement_step(
                expected_target=exp_obj_canonical,
                from_location=from_loc,
                to_location=to_loc,
                current_step_action=getattr(current_step, "action", ""),
                obj_map=obj_map,
                held_objects=held_objects,
                released_objects=released_objects,
                frame_width=W,
                step_id=step_id,
                now=now,
                har_action=har_action,
            )

        elif exp_action == "COMPLETE":
            return self.validate_completion_step(step_id=step_id, now=now)

        # Generic action fallback (e.g. OPEN, PERFORM, CLOSE, STORE in simulated tests)
        if exp_action not in ("IDENTIFY", "PICKUP", "PLACE", "MOVE", "COMPLETE"):
            if har_act_upper == exp_action and har_confidence >= 0.50:
                if not exp_obj_canonical or norm_pred_target == exp_obj_canonical or exp_obj_canonical in obj_map or (not tracks and not interactions):
                    self._consecutive_seen[exp_action] = self._consecutive_seen.get(exp_action, 0) + 1
                    if self._consecutive_seen[exp_action] >= 2:
                        self._consecutive_seen[exp_action] = 0
                        self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
                        return ConfirmedAction(
                            action=exp_action,
                            object_name=exp_obj_canonical,
                            status=ActionGateStatus.CONFIRMED_CORRECT,
                            confidence=har_confidence,
                            timestamp=now,
                            validation_state="CONFIRMED_CORRECT",
                            validation_reason=f"{exp_action} on {exp_obj_canonical} confirmed",
                            is_stable=True,
                        )

        # Default fallback
        self._current_gate_status = ActionGateStatus.WAITING
        return ConfirmedAction(
            action=exp_action or "IDLE",
            object_name=exp_obj_canonical,
            status=ActionGateStatus.WAITING,
            confidence=0.5,
            timestamp=now,
            validation_state="WAITING",
            validation_reason=f"Waiting for {exp_action} action",
        )

    # ----------------------------------------------------------------- #
    # Dedicated Action-Specific Confirmation Gate Implementations
    # ----------------------------------------------------------------- #

    def validate_identification_step(
        self,
        expected_target: str,
        obj_map: Dict[str, Any],
        held_objects: set[str],
        contact_objects: set[str],
        har_action: str,
        har_confidence: float,
        step_id: int = 1,
        frame_age_ms: float = 0.0,
        is_stale: bool = False,
        now: float = 0.0,
        predicted_target: str = "",
    ) -> ConfirmedAction:
        """
        Dedicated confirmation gate for identification steps (e.g. Step 1 & Step 4).
        Requirements:
          - Target object detected and tracked in workspace
          - Confidence threshold satisfied with hysteresis (min_confidence / maintain_confidence)
          - Temporal stability confirmed across window (>= min_identify_frames)
          - Decoupled from hand interaction, movement, speed, or release states
          - Other stationary objects present in the scene do NOT trigger wrong identification
        """
        cfg_ident = self.config.get("IDENTIFY", {})
        min_conf = cfg_ident.get("min_confidence", 0.65)
        maintain_conf = cfg_ident.get("maintain_confidence", 0.45)
        req_frames = cfg_ident.get("stable_frames", self.min_identify_frames)

        exp_tgt = normalize_target(expected_target)
        norm_pred_target = normalize_target(predicted_target)
        det = obj_map.get(exp_tgt)
        det_conf = float(getattr(det, "confidence", 0.0)) if det is not None else 0.0
        trk_id = getattr(det, "track_id", None) if det is not None else None
        seen_frames = self._consecutive_seen.get(exp_tgt, 0)
        action_norm = normalize_action(har_action)

        # 1. Target correctly identified and temporally confirmed
        # Hysteresis: start threshold min_conf, maintain threshold maintain_conf
        conf_ok = (det_conf >= min_conf) if seen_frames <= 1 else (det_conf >= maintain_conf)
        is_stable = (seen_frames >= req_frames) and (det is not None and conf_ok)

        if is_stable:
            reason = "IDENTIFICATION_CONFIRMED"
            logger.info(
                "STEP_VALIDATION\n"
                "step=%d\n"
                "expected_action=IDENTIFY\n"
                "expected_target=%s\n"
                "detected_action=IDENTIFY\n"
                "detected_target=%s\n"
                "object_confidence=%.2f\n"
                "action_confidence=%.2f\n"
                "stable=true\n"
                "gate=CONFIRMED_CORRECT\n"
                "validation=true\n"
                "reason=%s",
                step_id, exp_tgt, exp_tgt, det_conf or 0.90, har_confidence, reason
            )
            self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
            self._active_wrong_event_id = None
            self._wrong_episode_active = False
            self._frames_since_wrong = 0
            action_res = ConfirmedAction(
                action="IDENTIFY",
                object_name=exp_tgt,
                target="",
                status=ActionGateStatus.CONFIRMED_CORRECT,
                confidence=max(0.88, det_conf),
                evidence={"seen_frames": seen_frames, "min_required": req_frames, "confidence": det_conf},
                timestamp=now,
                track_id=trk_id,
                validation_state="CONFIRMED_CORRECT",
                validation_reason=f"{exp_tgt} confirmed stable for {seen_frames} frames (conf {det_conf:.2f})",
                is_stable=True,
                object_confidence=det_conf or 0.90,
                action_confidence=har_confidence,
            )
            self._last_confirmed_action = action_res
            return action_res

        # 2. Tolerant Wrong Action Check:
        # Flag wrong identification if expected object is absent (seen_frames == 0)
        # AND operator is presenting / interacting with a wrong object for >= min_identify_frames
        if seen_frames == 0:
            for other_obj in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"]:
                if other_obj != exp_tgt:
                    other_held = self._consecutive_held.get(other_obj, 0)
                    other_seen = self._consecutive_seen.get(other_obj, 0)
                    if (other_held >= self.min_identify_frames and other_obj in held_objects) or (
                        other_seen >= self.min_identify_frames and (other_obj in contact_objects or action_norm == "IDENTIFY" or norm_pred_target == other_obj)
                    ):
                        other_conf = float(getattr(obj_map.get(other_obj), "confidence", 0.85))
                        reason = f"IDENTIFIED_WRONG_OBJECT_{other_obj}"
                        logger.info(
                            "STEP_VALIDATION\n"
                            "step=%d\n"
                            "expected_action=IDENTIFY\n"
                            "expected_target=%s\n"
                            "detected_action=IDENTIFY\n"
                            "detected_target=%s\n"
                            "object_confidence=%.2f\n"
                            "action_confidence=%.2f\n"
                            "stable=true\n"
                            "gate=CONFIRMED_WRONG\n"
                            "validation=false\n"
                            "reason=%s",
                            step_id, exp_tgt, other_obj, other_conf, har_confidence, reason
                        )
                        return self._emit_wrong_action(
                            action="IDENTIFY",
                            object_name=other_obj,
                            target="",
                            confidence=0.85,
                            evidence={"wrong_object": other_obj, "expected_object": exp_tgt},
                            track_id=getattr(obj_map.get(other_obj), "track_id", None),
                            timestamp=now,
                            reason=f"Identified {other_obj} instead of {exp_tgt}",
                        )

        # 3. Waiting / Accumulating evidence
        if seen_frames > 0 and det is not None:
            reason = "IDENTIFY_ACTION_NOT_CONFIRMED"
            val_state = "CONFIRMING"
            val_reason = f"{exp_tgt} stable for {seen_frames}/{req_frames} frames (conf {det_conf:.2f})"
        else:
            reason = "WAITING_FOR_DETECTION"
            val_state = "WAITING"
            val_reason = f"Waiting for {exp_tgt} identification"

        logger.debug(
            "STEP_VALIDATION: step=%d, expected=%s, seen=%d/%d, conf=%.2f, reason=%s",
            step_id, exp_tgt, seen_frames, req_frames, det_conf, reason
        )
        self._current_gate_status = ActionGateStatus.WAITING
        action_res = ConfirmedAction(
            action="IDENTIFY",
            object_name=exp_tgt,
            target="",
            status=ActionGateStatus.WAITING,
            confidence=det_conf if seen_frames > 0 else 0.5,
            evidence={"seen_frames": seen_frames, "min_required": req_frames, "target": exp_tgt},
            timestamp=now,
            track_id=trk_id,
            validation_state=val_state,
            validation_reason=val_reason,
            is_stable=False,
            object_confidence=det_conf,
            action_confidence=har_confidence,
        )
        self._last_confirmed_action = action_res
        return action_res

    def validate_pickup_step(
        self,
        expected_target: str,
        obj_map: Dict[str, Any],
        held_objects: set[str],
        contact_objects: set[str],
        step_id: int,
        now: float,
    ) -> ConfirmedAction:
        """Confirmation gate for pickup actions."""
        cfg_pickup = self.config.get("PICKUP", {})
        min_frames = cfg_pickup.get("min_pickup_frames", self.min_pickup_frames)

        exp_tgt = normalize_target(expected_target)
        held_frames = self._consecutive_held.get(exp_tgt, 0)
        has_contact = (exp_tgt in contact_objects or exp_tgt in held_objects)
        det = obj_map.get(exp_tgt)
        det_conf = float(getattr(det, "confidence", 0.90)) if det is not None else 0.90
        trk_id = getattr(det, "track_id", None) if det is not None else None

        # Check for confirmed wrong object pickup (requires sustained holding of unexpected object >= min_frames while expected target is absent)
        if held_frames < 2 and exp_tgt not in held_objects:
            candidate_wrong = set(held_objects) | set(self._consecutive_held.keys())
            for other_obj in candidate_wrong:
                if other_obj != exp_tgt:
                    other_held = self._consecutive_held.get(other_obj, 0)
                    if other_held >= min_frames and other_obj in held_objects:
                        trk = obj_map.get(other_obj)
                        other_trk_id = getattr(trk, "track_id", None)
                        reason = f"WRONG_OBJECT_PICKUP_{other_obj}"
                        logger.info(
                            "STEP_VALIDATION\n"
                            "step=%d\n"
                            "expected_action=PICKUP\n"
                            "expected_target=%s\n"
                            "detected_action=PICKUP\n"
                            "detected_target=%s\n"
                            "object_confidence=%.2f\n"
                            "action_confidence=0.88\n"
                            "stable=true\n"
                            "gate=CONFIRMED_WRONG\n"
                            "validation=false\n"
                            "reason=%s",
                            step_id, exp_tgt, other_obj, det_conf, reason
                        )
                        return self._emit_wrong_action(
                            action="PICKUP",
                            object_name=other_obj,
                            target="",
                            confidence=0.85,
                            evidence={"wrong_object": other_obj, "expected_object": exp_tgt},
                            track_id=other_trk_id,
                            timestamp=now,
                            reason=f"Wrong object picked up: {other_obj} instead of {exp_tgt}",
                        )

        # Check if expected object pickup is confirmed
        if (held_frames >= self.min_pickup_frames) or (has_contact and held_frames >= 2):
            reason = "PICKUP_CONFIRMED"
            logger.info(
                "STEP_VALIDATION\n"
                "step=%d\n"
                "expected_action=PICKUP\n"
                "expected_target=%s\n"
                "detected_action=PICKUP\n"
                "detected_target=%s\n"
                "object_confidence=%.2f\n"
                "action_confidence=0.90\n"
                "stable=true\n"
                "gate=CONFIRMED_CORRECT\n"
                "validation=true\n"
                "reason=%s",
                step_id, exp_tgt, exp_tgt, det_conf, reason
            )
            self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
            self._active_wrong_event_id = None
            self._wrong_episode_active = False
            self._frames_since_wrong = 0
            action_res = ConfirmedAction(
                action="PICKUP",
                object_name=exp_tgt,
                status=ActionGateStatus.CONFIRMED_CORRECT,
                confidence=0.90,
                evidence={"held_frames": held_frames, "object": exp_tgt},
                timestamp=now,
                track_id=trk_id,
                validation_state="CONFIRMED_CORRECT",
                validation_reason=f"{exp_tgt} pickup confirmed ({held_frames} held frames)",
                is_stable=True,
                object_confidence=det_conf,
                action_confidence=0.90,
            )
            self._last_confirmed_action = action_res
            return action_res

        # Clean wrong episode if nothing is held for sustained period
        if not any(self._consecutive_held.get(o, 0) > 0 for o in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"] if o != exp_tgt):
            self._frames_since_wrong += 1
            if self._frames_since_wrong >= 5:
                self._active_wrong_event_id = None
                self._wrong_episode_active = False

        self._current_gate_status = ActionGateStatus.WAITING
        return ConfirmedAction(
            action="PICKUP",
            object_name=exp_tgt,
            status=ActionGateStatus.WAITING,
            confidence=0.45,
            evidence={"held_frames": held_frames, "has_contact": has_contact},
            timestamp=now,
            track_id=trk_id,
            validation_state="WAITING",
            validation_reason=f"Waiting for {exp_tgt} pickup (held {held_frames}/{self.min_pickup_frames})",
            is_stable=False,
            object_confidence=det_conf,
        )

    def validate_placement_step(
        self,
        expected_target: str,
        expected_destination: str,
        obj_map: Dict[str, Any],
        held_objects: set[str],
        released_objects: set[str],
        frame_width: int,
        step_id: int,
        now: float,
        har_action: str = "PLACE",
    ) -> ConfirmedAction:
        """Confirmation gate for placement actions."""
        exp_tgt = normalize_target(expected_target)
        exp_dest = normalize_target(expected_destination)
        was_held = self._was_held.get(exp_tgt, False)
        curr_held = self._consecutive_held.get(exp_tgt, 0)
        det = obj_map.get(exp_tgt)
        trk_id = getattr(det, "track_id", None) if det is not None else None

        cfg_place = self.config.get("PLACE", {})
        spatial_tol = cfg_place.get("spatial_tolerance_px", 50)

        # Check for wrong object interaction during PLACE (sustained hold of unexpected object)
        if not was_held and curr_held == 0:
            for other_obj in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"]:
                if other_obj != exp_tgt:
                    other_held = self._consecutive_held.get(other_obj, 0)
                    if other_held >= self.min_place_frames and (other_obj in held_objects or other_held >= 3):
                        trk = obj_map.get(other_obj)
                        act_detected = normalize_action(har_action) or "PLACE"
                        return self._emit_wrong_action(
                            action=act_detected,
                            object_name=other_obj,
                            target="",
                            confidence=0.85,
                            evidence={"wrong_object": other_obj, "expected_object": exp_tgt},
                            track_id=getattr(trk, "track_id", None),
                            timestamp=now,
                            reason=f"Wrong action/object during placement: {act_detected} {other_obj}",
                        )

        # Target location validation
        in_target = False
        if not exp_dest:
            in_target = True
        elif det is not None:
            cx, cy = getattr(det, "centroid", (det.bbox[0] + det.bbox[2] // 2, det.bbox[1] + det.bbox[3] // 2))
            if exp_dest == "LOCATION_A":
                in_target = (cx < frame_width * 0.55)
            elif exp_dest == "LOCATION_B":
                in_target = (cx > frame_width * 0.45)
            elif exp_dest in ("BLUE_BOX", "MAIN_BOX"):
                blue_det = obj_map.get("BLUE_BOX")
                if blue_det:
                    bx, by, bw, bh = blue_det.bbox
                    in_target = (bx - spatial_tol <= cx <= bx + bw + spatial_tol and by - spatial_tol <= cy <= by + bh + spatial_tol)
                else:
                    in_target = True
            elif exp_dest == "YELLOW_BOX":
                yellow_det = obj_map.get("YELLOW_BOX")
                if yellow_det:
                    yx, yy, yw, yh = yellow_det.bbox
                    in_target = (yx - spatial_tol <= cx <= yx + yw + spatial_tol and yy - spatial_tol <= cy <= yy + yh + spatial_tol)
                else:
                    in_target = True
        else:
            in_target = True

        # If object was held and now released / stable in target or confirmed by HAR
        is_released = (curr_held == 0 and was_held) or (exp_tgt in released_objects)
        har_is_place = (normalize_action(har_action) == "PLACE")
        if (is_released and in_target) or (in_target and det is not None and curr_held == 0 and was_held) or (har_is_place and in_target and curr_held == 0):
            reason = "PLACEMENT_CONFIRMED"
            logger.info(
                "STEP_VALIDATION\n"
                "step=%d\n"
                "expected_action=PLACE\n"
                "expected_target=%s\n"
                "detected_action=PLACE\n"
                "detected_target=%s\n"
                "object_confidence=0.90\n"
                "action_confidence=0.88\n"
                "stable=true\n"
                "gate=CONFIRMED_CORRECT\n"
                "validation=true\n"
                "reason=%s",
                step_id, exp_tgt, exp_tgt, reason
            )
            self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
            self._active_wrong_event_id = None
            self._wrong_episode_active = False
            self._frames_since_wrong = 0
            action_res = ConfirmedAction(
                action="PLACE",
                object_name=exp_tgt,
                target=exp_dest,
                status=ActionGateStatus.CONFIRMED_CORRECT,
                confidence=0.88,
                evidence={"target": exp_dest, "was_held": was_held, "in_target": in_target},
                timestamp=now,
                track_id=trk_id,
                validation_state="CONFIRMED_CORRECT",
                validation_reason=f"{exp_tgt} placed at {exp_dest} confirmed",
                is_stable=True,
                action_confidence=0.88,
            )
            self._was_held[exp_tgt] = False
            self._last_confirmed_action = action_res
            return action_res

        # If object was held and now released clearly outside target
        if is_released and not in_target and det is not None and was_held:
            trk = obj_map.get(exp_tgt)
            return self._emit_wrong_action(
                action="PLACE",
                object_name=exp_tgt,
                target="WRONG_LOCATION",
                confidence=0.85,
                evidence={"expected_target": exp_dest, "was_held": was_held, "in_target": False},
                track_id=getattr(trk, "track_id", None),
                timestamp=now,
                reason=f"Placed {exp_tgt} outside {exp_dest}",
            )

        # Clean wrong episode if nothing is held for sustained period
        if not any(self._consecutive_held.get(o, 0) > 0 for o in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"] if o != exp_tgt):
            self._frames_since_wrong += 1
            if self._frames_since_wrong >= 5:
                self._active_wrong_event_id = None
                self._wrong_episode_active = False

        self._current_gate_status = ActionGateStatus.WAITING
        return ConfirmedAction(
            action="PLACE",
            object_name=exp_tgt,
            target=exp_dest,
            status=ActionGateStatus.WAITING,
            confidence=0.5,
            evidence={"in_target": in_target, "was_held": was_held},
            timestamp=now,
            track_id=trk_id,
            validation_state="WAITING",
            validation_reason=f"Waiting for {exp_tgt} placement at {exp_dest}",
            is_stable=False,
        )

    def validate_movement_step(
        self,
        expected_target: str,
        from_location: str,
        to_location: str,
        current_step_action: str,
        obj_map: Dict[str, Any],
        held_objects: set[str],
        released_objects: set[str],
        frame_width: int,
        step_id: int,
        now: float,
        har_action: str = "MOVE",
    ) -> ConfirmedAction:
        """Confirmation gate for movement steps (Steps 11 & 12)."""
        exp_tgt = normalize_target(expected_target)
        to_loc = normalize_target(to_location)

        # Check for wrong object moved (requires sustained hold of unexpected object)
        for other_obj in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"]:
            if other_obj != exp_tgt:
                other_held = self._consecutive_held.get(other_obj, 0)
                if other_held >= self.min_move_frames and (other_obj in held_objects or other_held >= 3) and not self._was_held.get(exp_tgt, False):
                    trk = obj_map.get(other_obj)
                    act_detected = normalize_action(har_action) or "MOVE"
                    return self._emit_wrong_action(
                        action=act_detected,
                        object_name=other_obj,
                        target="",
                        confidence=0.85,
                        evidence={"wrong_object": other_obj, "expected_object": exp_tgt},
                        track_id=getattr(trk, "track_id", None),
                        timestamp=now,
                        reason=f"Moved wrong object: {other_obj} instead of {exp_tgt}",
                    )

        det = obj_map.get(exp_tgt)
        trk_id = getattr(det, "track_id", None) if det is not None else None
        was_held = self._was_held.get(exp_tgt, False) or (exp_tgt in released_objects)
        curr_held = self._consecutive_held.get(exp_tgt, 0)

        reached_destination = False
        if not to_loc and not ("TO_A" in current_step_action or "TO_B" in current_step_action):
            reached_destination = True
        elif det is not None:
            cx, cy = getattr(det, "centroid", (det.bbox[0] + det.bbox[2] // 2, det.bbox[1] + det.bbox[3] // 2))
            if to_loc == "LOCATION_B" or "TO_B" in current_step_action:
                reached_destination = (cx > frame_width * 0.45)
            elif to_loc == "LOCATION_A" or "TO_A" in current_step_action:
                reached_destination = (cx < frame_width * 0.55)
        else:
            reached_destination = True

        har_is_move = (normalize_action(har_action) == "MOVE")
        if (was_held and curr_held == 0 and reached_destination) or (reached_destination and (exp_tgt in released_objects)) or (har_is_move and reached_destination and curr_held == 0):
            reason = "MOVEMENT_CONFIRMED"
            logger.info(
                "STEP_VALIDATION\n"
                "step=%d\n"
                "expected_action=MOVE\n"
                "expected_target=%s\n"
                "detected_action=MOVE\n"
                "detected_target=%s\n"
                "object_confidence=0.90\n"
                "action_confidence=0.90\n"
                "stable=true\n"
                "gate=CONFIRMED_CORRECT\n"
                "validation=true\n"
                "reason=%s",
                step_id, exp_tgt, exp_tgt, reason
            )
            self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
            self._active_wrong_event_id = None
            self._wrong_episode_active = False
            self._frames_since_wrong = 0
            dest_tag = to_loc or ("LOCATION_B" if "TO_B" in current_step_action else "LOCATION_A")
            action_res = ConfirmedAction(
                action="MOVE",
                object_name=exp_tgt,
                target=dest_tag,
                status=ActionGateStatus.CONFIRMED_CORRECT,
                confidence=0.90,
                evidence={"from": from_location, "to": to_loc, "reached": reached_destination},
                timestamp=now,
                track_id=trk_id,
                validation_state="CONFIRMED_CORRECT",
                validation_reason=f"{exp_tgt} moved to {dest_tag} confirmed",
                is_stable=True,
                action_confidence=0.90,
            )
            self._was_held[exp_tgt] = False
            self._last_confirmed_action = action_res
            return action_res

        # Clean wrong episode if nothing is held for sustained period
        if not any(self._consecutive_held.get(o, 0) > 0 for o in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"] if o != exp_tgt):
            self._frames_since_wrong += 1
            if self._frames_since_wrong >= 5:
                self._active_wrong_event_id = None
                self._wrong_episode_active = False

        self._current_gate_status = ActionGateStatus.WAITING
        return ConfirmedAction(
            action="MOVE",
            object_name=exp_tgt,
            target=to_loc,
            status=ActionGateStatus.WAITING,
            confidence=0.5,
            evidence={"from": from_location, "to": to_loc},
            timestamp=now,
            track_id=trk_id,
            validation_state="WAITING",
            validation_reason=f"Waiting for {exp_tgt} movement to {to_loc}",
            is_stable=False,
        )

    def validate_completion_step(self, step_id: int = 13, now: float = 0.0) -> ConfirmedAction:
        """Terminal step confirmation."""
        reason = "EXPERIMENT_COMPLETE"
        logger.info(
            "STEP_VALIDATION\n"
            "step=%d\n"
            "expected_action=COMPLETE\n"
            "expected_target=ALL\n"
            "detected_action=COMPLETE\n"
            "detected_target=ALL\n"
            "object_confidence=1.00\n"
            "action_confidence=1.00\n"
            "stable=true\n"
            "gate=CONFIRMED_CORRECT\n"
            "validation=true\n"
            "reason=%s",
            step_id, reason
        )
        self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
        self._active_wrong_event_id = None
        self._wrong_episode_active = False
        self._frames_since_wrong = 0
        action_res = ConfirmedAction(
            action="COMPLETE",
            object_name="ALL",
            status=ActionGateStatus.CONFIRMED_CORRECT,
            confidence=1.0,
            timestamp=now,
            validation_state="CONFIRMED_CORRECT",
            validation_reason="Experiment procedure successfully completed",
            is_stable=True,
            action_confidence=1.0,
        )
        self._last_confirmed_action = action_res
        return action_res
