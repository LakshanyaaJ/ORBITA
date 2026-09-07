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


class ActionConfirmationGate:
    """
    Validates physical actions against current FSM step expectations before
    authorizing any state transition.
    """

    def __init__(
        self,
        min_identify_frames: int = 3,
        min_pickup_frames: int = 3,
        min_place_frames: int = 3,
        min_move_frames: int = 4,
    ):
        self.min_identify_frames = min_identify_frames
        self.min_pickup_frames = min_pickup_frames
        self.min_place_frames = min_place_frames
        self.min_move_frames = min_move_frames

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
        logger.info("ActionConfirmationGate: Reset complete.")

    def _emit_wrong_action(
        self,
        action: str,
        object_name: str,
        target: str = "",
        confidence: float = 0.85,
        evidence: Optional[Dict[str, Any]] = None,
        track_id: Optional[int] = None,
        timestamp: float = 0.0,
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
        frame_shape: Tuple[int, int] = (720, 1280),
        fps: float = 15.0,
    ) -> ConfirmedAction:
        """
        Evaluate frame evidence against current step expectations.
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
            )

        exp_action = getattr(current_step, "expected_action", "")
        exp_obj = getattr(current_step, "expected_object", "")
        exp_target = getattr(current_step, "expected_target", "")
        from_loc = getattr(current_step, "from_location", "")
        to_loc = getattr(current_step, "to_location", "")

        # Fallback to step.action parsing if expected_action is not explicitly set
        if not exp_action and hasattr(current_step, "action"):
            act_str = current_step.action
            if act_str.startswith("IDENTIFY_"):
                exp_action = "IDENTIFY"
                exp_obj = act_str.replace("IDENTIFY_", "")
            elif act_str.startswith("PICKUP_"):
                exp_action = "PICKUP"
                exp_obj = act_str.replace("PICKUP_", "")
            elif act_str.startswith("PLACE_"):
                exp_action = "PLACE"
                parts = act_str.split("_")
                exp_obj = parts[1] + "_BOX" if len(parts) > 1 else ""
            elif act_str.startswith("MOVE_"):
                exp_action = "MOVE"
                exp_obj = "BLUE_BOX" if "BLUE" in act_str else "YELLOW_BOX"
            elif act_str == "EXPERIMENT_COMPLETE":
                exp_action = "COMPLETE"
                exp_obj = "ALL"

        # Canonicalize object aliases
        exp_obj_canonical = "BLUE_BOX" if exp_obj in ("BLUE_BOX", "MAIN_BOX") else exp_obj

        # Map detected objects to canonical identities
        obj_map: Dict[str, Any] = {}
        for obj in detected_objects:
            cls = getattr(obj, "class_name", "")
            identity = getattr(obj, "semantic_identity", "") or cls
            if identity in ("MAIN_BOX", "BLUE_BOX") or cls in ("MAIN_BOX", "BLUE_BOX"):
                canonical = "BLUE_BOX"
            elif identity == "YELLOW_BOX" or cls == "YELLOW_BOX":
                canonical = "YELLOW_BOX"
            elif identity in ("TOOL", "PEN") or cls in ("TOOL", "PEN"):
                canonical = "PEN"
            elif identity in ("SAMPLE", "WATCH") or cls in ("SAMPLE", "WATCH"):
                canonical = "WATCH"
            else:
                canonical = identity or cls
            obj_map[canonical] = obj

        # Update seen counts
        for obj_name in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"]:
            if obj_name in obj_map:
                self._consecutive_seen[obj_name] = self._consecutive_seen.get(obj_name, 0) + 1
                det = obj_map[obj_name]
                centroid = getattr(det, "centroid", (det.bbox[0] + det.bbox[2] // 2, det.bbox[1] + det.bbox[3] // 2))
                if obj_name not in self._initial_positions:
                    self._initial_positions[obj_name] = centroid
                self._last_positions[obj_name] = centroid
            else:
                self._consecutive_seen[obj_name] = 0

        # Check hand interaction states for objects
        held_objects = set()
        contact_objects = set()
        released_objects = set()

        for inter in interactions:
            cls = getattr(inter, "object_class", "")
            c_name = "BLUE_BOX" if cls in ("MAIN_BOX", "BLUE_BOX") else cls
            is_holding = getattr(inter, "is_holding", False)
            state_val = getattr(inter, "state", None)
            state_name = getattr(state_val, "name", str(state_val))

            if is_holding or state_name in ("HOLDING", "GRASPING"):
                held_objects.add(c_name)
            elif state_name in ("CONTACT", "NEAR_OBJECT"):
                contact_objects.add(c_name)
            elif state_name in ("RELEASING", "RELEASED"):
                released_objects.add(c_name)

        # Check tracks for physical states
        for trk in tracks:
            cls = getattr(trk, "class_name", "")
            identity = getattr(trk, "identity", "") or cls
            if identity in ("MAIN_BOX", "BLUE_BOX") or cls in ("MAIN_BOX", "BLUE_BOX"):
                canonical = "BLUE_BOX"
            elif identity == "YELLOW_BOX" or cls == "YELLOW_BOX":
                canonical = "YELLOW_BOX"
            elif identity in ("TOOL", "PEN") or cls in ("TOOL", "PEN"):
                canonical = "PEN"
            elif identity in ("SAMPLE", "WATCH") or cls in ("SAMPLE", "WATCH"):
                canonical = "WATCH"
            else:
                canonical = identity or cls

            if canonical not in obj_map:
                obj_map[canonical] = trk

            trk_state = getattr(trk, "state", None)
            trk_state_val = getattr(trk_state, "value", str(trk_state))
            if trk_state_val in ("BEING_HELD", "CARRIED") or getattr(trk, "hand_contact", False):
                held_objects.add(canonical)
            elif trk_state_val in ("RELEASED", "PLACED"):
                released_objects.add(canonical)

        # HAR / Movement inference
        har_act_upper = har_action.upper()
        if "TAKE" in har_act_upper or "PICK" in har_act_upper:
            if exp_obj_canonical in contact_objects or exp_obj_canonical in obj_map:
                held_objects.add(exp_obj_canonical)

        # Update held counters
        for obj_name in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"]:
            if obj_name in held_objects:
                self._consecutive_held[obj_name] = self._consecutive_held.get(obj_name, 0) + 1
                self._was_held[obj_name] = True
                if obj_name not in self._held_start_time:
                    self._held_start_time[obj_name] = now
            else:
                self._consecutive_held[obj_name] = 0

        # ------------------------------------------------------------- #
        # Step-Specific Physical Validation
        # ------------------------------------------------------------- #

        # CASE 1: IDENTIFY
        if exp_action == "IDENTIFY":
            seen_frames = self._consecutive_seen.get(exp_obj_canonical, 0)
            if seen_frames >= self.min_identify_frames:
                # Check if wrong object was identified instead
                conf = float(obj_map[exp_obj_canonical].confidence) if exp_obj_canonical in obj_map else 0.85
                self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
                self._active_wrong_event_id = None
                self._wrong_episode_active = False
                self._frames_since_wrong = 0
                action_res = ConfirmedAction(
                    action="IDENTIFY",
                    object_name=exp_obj_canonical,
                    status=ActionGateStatus.CONFIRMED_CORRECT,
                    confidence=conf,
                    evidence={"seen_frames": seen_frames, "min_required": self.min_identify_frames},
                    timestamp=now,
                )
                self._last_confirmed_action = action_res
                return action_res

            self._current_gate_status = ActionGateStatus.WAITING
            return ConfirmedAction(
                action="IDENTIFY",
                object_name=exp_obj_canonical,
                status=ActionGateStatus.WAITING,
                confidence=0.5,
                evidence={"seen_frames": seen_frames, "target": exp_obj_canonical},
                timestamp=now,
            )

        # CASE 2: PICKUP
        elif exp_action == "PICKUP":
            held_frames = self._consecutive_held.get(exp_obj_canonical, 0)
            has_contact = (exp_obj_canonical in contact_objects or exp_obj_canonical in held_objects)

            # Check for confirmed wrong object pickup
            for other_obj in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"]:
                if other_obj != exp_obj_canonical:
                    other_held = self._consecutive_held.get(other_obj, 0)
                    if other_held >= self.min_pickup_frames:
                        trk = obj_map.get(other_obj)
                        trk_id = getattr(trk, "track_id", None)
                        return self._emit_wrong_action(
                            action="PICKUP",
                            object_name=other_obj,
                            target="",
                            confidence=0.85,
                            evidence={"wrong_object": other_obj, "expected_object": exp_obj_canonical},
                            track_id=trk_id,
                            timestamp=now,
                        )

            # Check if expected object pickup is confirmed
            if (held_frames >= self.min_pickup_frames) or (has_contact and held_frames >= 2):
                self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
                self._active_wrong_event_id = None
                self._wrong_episode_active = False
                self._frames_since_wrong = 0
                action_res = ConfirmedAction(
                    action="PICKUP",
                    object_name=exp_obj_canonical,
                    status=ActionGateStatus.CONFIRMED_CORRECT,
                    confidence=0.90,
                    evidence={"held_frames": held_frames, "object": exp_obj_canonical},
                    timestamp=now,
                )
                self._last_confirmed_action = action_res
                return action_res

            # Clean wrong episode if nothing is held for sustained period
            if not any(self._consecutive_held.get(o, 0) > 0 for o in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"] if o != exp_obj_canonical):
                self._frames_since_wrong += 1
                if self._frames_since_wrong >= 5:
                    self._active_wrong_event_id = None
                    self._wrong_episode_active = False

            self._current_gate_status = ActionGateStatus.WAITING
            return ConfirmedAction(
                action="PICKUP",
                object_name=exp_obj_canonical,
                status=ActionGateStatus.WAITING,
                confidence=0.45,
                evidence={"held_frames": held_frames, "has_contact": has_contact},
                timestamp=now,
            )

        # CASE 3: PLACE
        elif exp_action == "PLACE":
            was_held = self._was_held.get(exp_obj_canonical, False)
            curr_held = self._consecutive_held.get(exp_obj_canonical, 0)
            det = obj_map.get(exp_obj_canonical)

            # Check for wrong object interaction during PLACE
            for other_obj in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"]:
                if other_obj != exp_obj_canonical:
                    other_held = self._consecutive_held.get(other_obj, 0)
                    if other_held >= self.min_pickup_frames:
                        trk = obj_map.get(other_obj)
                        trk_id = getattr(trk, "track_id", None)
                        return self._emit_wrong_action(
                            action="PLACE",
                            object_name=other_obj,
                            target="",
                            confidence=0.85,
                            evidence={"wrong_object": other_obj, "expected_object": exp_obj_canonical},
                            track_id=trk_id,
                            timestamp=now,
                        )

            # Target location validation
            in_target = False
            if det is not None:
                cx, cy = getattr(det, "centroid", (det.bbox[0] + det.bbox[2] // 2, det.bbox[1] + det.bbox[3] // 2))
                if exp_target == "LOCATION_A":
                    in_target = (cx < W * 0.55)
                elif exp_target == "LOCATION_B":
                    in_target = (cx > W * 0.45)
                elif exp_target in ("BLUE_BOX", "MAIN_BOX"):
                    # Check if object is inside or overlapping Blue Box
                    blue_det = obj_map.get("BLUE_BOX")
                    if blue_det:
                        bx, by, bw, bh = blue_det.bbox
                        in_target = (bx - 40 <= cx <= bx + bw + 40 and by - 40 <= cy <= by + bh + 40)
                    else:
                        in_target = True
                elif exp_target == "YELLOW_BOX":
                    # Check if object is inside or overlapping Yellow Box
                    yellow_det = obj_map.get("YELLOW_BOX")
                    if yellow_det:
                        yx, yy, yw, yh = yellow_det.bbox
                        in_target = (yx - 40 <= cx <= yx + yw + 40 and yy - 40 <= cy <= yy + yh + 40)
                    else:
                        in_target = True

            # If object was held and now released / stable in target
            is_released = (curr_held == 0 and was_held) or (exp_obj_canonical in released_objects)
            if (is_released and in_target) or (in_target and det is not None and curr_held == 0 and was_held):
                self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
                self._active_wrong_event_id = None
                self._wrong_episode_active = False
                self._frames_since_wrong = 0
                action_res = ConfirmedAction(
                    action="PLACE",
                    object_name=exp_obj_canonical,
                    target=exp_target,
                    status=ActionGateStatus.CONFIRMED_CORRECT,
                    confidence=0.88,
                    evidence={"target": exp_target, "was_held": was_held, "in_target": in_target},
                    timestamp=now,
                )
                self._was_held[exp_obj_canonical] = False
                self._last_confirmed_action = action_res
                return action_res

            # If object was held and now released clearly outside target
            if is_released and not in_target and det is not None and was_held:
                trk = obj_map.get(exp_obj_canonical)
                trk_id = getattr(trk, "track_id", None)
                return self._emit_wrong_action(
                    action="PLACE",
                    object_name=exp_obj_canonical,
                    target="WRONG_LOCATION",
                    confidence=0.85,
                    evidence={"expected_target": exp_target, "was_held": was_held, "in_target": False},
                    track_id=trk_id,
                    timestamp=now,
                )

            # Clean wrong episode if nothing is held for sustained period
            if not any(self._consecutive_held.get(o, 0) > 0 for o in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"] if o != exp_obj_canonical):
                self._frames_since_wrong += 1
                if self._frames_since_wrong >= 5:
                    self._active_wrong_event_id = None
                    self._wrong_episode_active = False

            self._current_gate_status = ActionGateStatus.WAITING
            return ConfirmedAction(
                action="PLACE",
                object_name=exp_obj_canonical,
                target=exp_target,
                status=ActionGateStatus.WAITING,
                confidence=0.5,
                evidence={"in_target": in_target, "was_held": was_held},
                timestamp=now,
            )

        # CASE 4: MOVE
        elif exp_action == "MOVE":
            # Check for wrong object moved
            for other_obj in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"]:
                if other_obj != exp_obj_canonical:
                    other_held = self._consecutive_held.get(other_obj, 0)
                    if other_held >= self.min_pickup_frames:
                        trk = obj_map.get(other_obj)
                        trk_id = getattr(trk, "track_id", None)
                        return self._emit_wrong_action(
                            action="MOVE",
                            object_name=other_obj,
                            target="",
                            confidence=0.85,
                            evidence={"wrong_object": other_obj, "expected_object": exp_obj_canonical},
                            track_id=trk_id,
                            timestamp=now,
                        )

            det = obj_map.get(exp_obj_canonical)
            was_held = self._was_held.get(exp_obj_canonical, False) or (exp_obj_canonical in released_objects)
            curr_held = self._consecutive_held.get(exp_obj_canonical, 0)

            reached_destination = False
            if det is not None:
                cx, cy = getattr(det, "centroid", (det.bbox[0] + det.bbox[2] // 2, det.bbox[1] + det.bbox[3] // 2))
                if to_loc == "LOCATION_B" or "TO_B" in getattr(current_step, "action", ""):
                    reached_destination = (cx > W * 0.45)
                elif to_loc == "LOCATION_A" or "TO_A" in getattr(current_step, "action", ""):
                    reached_destination = (cx < W * 0.55)

            if (was_held and curr_held == 0 and reached_destination) or (reached_destination and (exp_obj_canonical in released_objects)):
                self._current_gate_status = ActionGateStatus.CONFIRMED_CORRECT
                self._active_wrong_event_id = None
                self._wrong_episode_active = False
                self._frames_since_wrong = 0
                action_res = ConfirmedAction(
                    action="MOVE",
                    object_name=exp_obj_canonical,
                    target=to_loc or ("LOCATION_B" if "TO_B" in getattr(current_step, "action", "") else "LOCATION_A"),
                    status=ActionGateStatus.CONFIRMED_CORRECT,
                    confidence=0.90,
                    evidence={"from": from_loc, "to": to_loc, "reached": reached_destination},
                    timestamp=now,
                )
                self._was_held[exp_obj_canonical] = False
                self._last_confirmed_action = action_res
                return action_res

            # Clean wrong episode if nothing is held for sustained period
            if not any(self._consecutive_held.get(o, 0) > 0 for o in ["BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH"] if o != exp_obj_canonical):
                self._frames_since_wrong += 1
                if self._frames_since_wrong >= 5:
                    self._active_wrong_event_id = None
                    self._wrong_episode_active = False

            self._current_gate_status = ActionGateStatus.WAITING
            return ConfirmedAction(
                action="MOVE",
                object_name=exp_obj_canonical,
                target=to_loc,
                status=ActionGateStatus.WAITING,
                confidence=0.5,
                evidence={"from": from_loc, "to": to_loc},
                timestamp=now,
            )

        # Terminal step
        elif exp_action == "COMPLETE":
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
            )
            self._last_confirmed_action = action_res
            return action_res

        self._current_gate_status = ActionGateStatus.WAITING
        return ConfirmedAction(
            action=exp_action or "IDLE",
            object_name=exp_obj_canonical,
            status=ActionGateStatus.WAITING,
            confidence=0.5,
            timestamp=now,
        )
