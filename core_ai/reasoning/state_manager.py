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
from core_ai.reasoning.action_gate import ActionConfirmationGate, ActionGateStatus, ConfirmedAction
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
    confirmed_action: Optional[ConfirmedAction] = None
    voice_status: str = "IDLE"
    steps: list[dict] = field(default_factory=list)
    camera_source: str = "UNKNOWN"
    frame_index: int = 0
    total_frames: int = 0
    video_time: str = "00:00.00"
    tracks: list[Any] = field(default_factory=list)
    frame_age_ms: float = 0.0
    is_stale: bool = False
    raw_detections: Optional[list] = None
    nms_detections: Optional[list] = None
    grouped_detections: Optional[dict] = None

    def to_dict(self) -> dict:
        d = self.fsm_state.to_dict()
        d["voice_message"] = self.voice_message
        d["voice_status"] = self.voice_status
        d["hud_message"] = self.hud_message
        d["alert_level"] = self.alert_level
        d["fps"] = round(self.fps, 1)
        d["latency_ms"] = round(self.latency_ms, 2)
        d["frame_age_ms"] = round(self.frame_age_ms, 1)
        d["live_edge"] = "LIVE" if self.frame_age_ms < 250 else ("STREAM BEHIND" if self.frame_age_ms < 1500 else "CRITICAL STREAM LATENCY")
        d["is_stale"] = self.is_stale
        d["steps"] = self.steps
        val_state = getattr(self.confirmed_action, "validation_state", self.confirmed_action.status.value) if self.confirmed_action else "WAITING"
        val_reason = getattr(self.confirmed_action, "validation_reason", "") if self.confirmed_action else ""
        checklist = getattr(self.confirmed_action, "why_completed_checklist", []) if self.confirmed_action else []
        features = getattr(self.confirmed_action, "interaction_features", {}) if self.confirmed_action else {}
        step_conf = round(getattr(self.confirmed_action, "step_confidence", 0.0), 3) if self.confirmed_action else 0.0

        scores = {
            "object_confidence": round(getattr(self.confirmed_action, "object_confidence", 0.0), 3) if self.confirmed_action else 0.0,
            "hand_confidence": round(getattr(self.confirmed_action, "hand_confidence", 0.0), 3) if self.confirmed_action else 0.0,
            "tracking_confidence": round(getattr(self.confirmed_action, "tracking_confidence", 0.0), 3) if self.confirmed_action else 0.0,
            "interaction_confidence": round(getattr(self.confirmed_action, "interaction_confidence", 0.0), 3) if self.confirmed_action else 0.0,
            "action_confidence": round(getattr(self.confirmed_action, "action_confidence", 0.0), 3) if self.confirmed_action else 0.0,
            "step_confidence": step_conf,
        }

        d["validation_state"] = val_state
        d["validation_reason"] = val_reason
        d["why_completed_checklist"] = checklist
        d["interaction_features"] = features
        d["confidence_scores"] = scores
        d["step_confidence"] = step_conf

        v_debug = getattr(self.confirmed_action, "validation_debug", {}) if self.confirmed_action else {}
        d["validation_debug"] = v_debug

        grp = self.grouped_detections or {}
        d["detections"] = {
            "LOCATION_A": grp.get("LOCATION_A", []),
            "LOCATION_B": grp.get("LOCATION_B", []),
            "PEN": grp.get("PEN", []),
            "WATCH": grp.get("WATCH", []),
            "BLUE_BOX": grp.get("BLUE_BOX", []),
            "YELLOW_BOX": grp.get("YELLOW_BOX", []),
            "HAND": grp.get("HAND", []),
            "location_a": grp.get("LOCATION_A", []),
            "location_b": grp.get("LOCATION_B", []),
            "pen": grp.get("PEN", []),
            "watch": grp.get("WATCH", []),
            "blue_box": grp.get("BLUE_BOX", []),
            "yellow_box": grp.get("YELLOW_BOX", []),
            "hand": grp.get("HAND", []),
        }
        d["structured_detections"] = {
            "LOCATION_A": grp.get("LOCATION_A", []),
            "LOCATION_B": grp.get("LOCATION_B", []),
            "PEN": grp.get("PEN", []),
            "WATCH": grp.get("WATCH", []),
            "BLUE_BOX": grp.get("BLUE_BOX", []),
            "YELLOW_BOX": grp.get("YELLOW_BOX", []),
            "HAND": grp.get("HAND", []),
        }

        if self.confirmed_action:
            d["confirmed_action"] = {
                "action": self.confirmed_action.action,
                "object": self.confirmed_action.object_name,
                "target": self.confirmed_action.target,
                "status": self.confirmed_action.status.value,
                "confidence": round(self.confirmed_action.confidence, 3),
                "validation_state": val_state,
                "validation_reason": val_reason,
                "why_completed_checklist": checklist,
                "interaction_features": features,
                "confidence_scores": scores,
                "validation_debug": v_debug,
            }
        
        # Section 20 SIH Development/Debug Telemetry Panel Block
        d["debug_telemetry"] = {
            "source": self.camera_source,
            "frame": self.frame_index,
            "total_frames": self.total_frames,
            "video_time": self.video_time,
            "yolo_detections": [
                f"{o.class_name} {o.confidence:.2f}" for o in self.detected_objects if getattr(o, "source", "") == "yolo"
            ],
            "tracks": [
                f"ID {getattr(t, 'track_id', '?')} — {getattr(t, 'identity', getattr(t, 'class_name', ''))} — {getattr(getattr(t, 'state', None), 'value', getattr(t, 'state', 'ON_TABLE'))}"
                for t in self.tracks
            ],
            "action": f"{self.confirmed_action.action} {self.confirmed_action.object_name}" if self.confirmed_action else (f"{self.action_prediction.action} {self.action_prediction.target_object}" if self.action_prediction else "IDLE"),
            "action_status": self.confirmed_action.status.value if self.confirmed_action else "WAITING",
            "validation_state": val_state,
            "validation_reason": val_reason,
            "why_completed_checklist": checklist,
            "interaction_features": features,
            "confidence_scores": scores,
            "step_confidence": step_conf,
            "fsm_step": f"STEP {self.fsm_state.current_step.id}: {self.fsm_state.current_step.label}" if self.fsm_state.current_step else ("COMPLETED" if self.fsm_state.is_complete else "NONE"),
            "fsm_step_number": self.fsm_state.current_step.id if self.fsm_state.current_step else (self.fsm_state.total_steps + 1 if self.fsm_state.is_complete else 0),
            "voice_prompt": self.voice_message,
            "voice_status": self.voice_status,
        }

        # Requirement 9 Master Prompt Validation & Perception Debug Panel
        exp_step_id = self.fsm_state.current_step.id if self.fsm_state.current_step else 0
        exp_action = getattr(self.fsm_state.current_step, "expected_action", "") or (self.fsm_state.current_step.action.split("_")[0] if self.fsm_state.current_step else "IDLE")
        exp_target = getattr(self.fsm_state.current_step, "expected_object", "") or (self.fsm_state.current_step.action.split("_", 1)[1] if self.fsm_state.current_step and "_" in self.fsm_state.current_step.action else "")

        det_action = self.confirmed_action.action if self.confirmed_action else (self.action_prediction.action if self.action_prediction else "IDLE")
        det_target = self.confirmed_action.object_name if self.confirmed_action else (self.action_prediction.target_object if self.action_prediction else "")

        target_match = bool(v_debug.get("targetMatch", (det_target.upper() == exp_target.upper()))) if (exp_target and det_target) else bool(v_debug.get("targetMatch", False))
        temporal_val = str(v_debug.get("temporalConfirmation", "3 / 3" if val_state in ("CONFIRMED_CORRECT", "PASS") else "0 / 3"))
        val_status_str = "PASS" if getattr(self.confirmed_action, "status", None) == ActionGateStatus.CONFIRMED_CORRECT else (
            "WAITING" if getattr(self.confirmed_action, "status", None) == ActionGateStatus.WAITING else "CONFIRMING"
        )
        fsm_trans_str = f"STEP {exp_step_id} -> STEP {exp_step_id + 1}" if getattr(self.fsm_state, "is_transition", False) else f"STEP {exp_step_id} (ACTIVE)"
        timeline_str = f"{len(self.fsm_state.completed_step_ids)} / {self.fsm_state.total_steps} DONE"

        raw_list = [f"{d.get('raw_class', '')} {int(d.get('confidence', 0)*100)}%" for d in (self.raw_detections or [])]
        if not raw_list:
            raw_list = [f"{o.class_name} {int(o.confidence*100)}%" for o in self.detected_objects]

        nms_list = [f"{d.get('class', '')} {int(d.get('conf', 0)*100)}%" for d in (self.nms_detections or [])]
        if not nms_list:
            nms_list = [f"{o.class_name} {int(o.confidence*100)}%" for o in self.detected_objects]

        tracked_list = [
            f"{getattr(t, 'class_name', '')} -> ID {getattr(t, 'track_id', '?')}"
            for t in self.tracks if getattr(t, 'track_id', -1) > 0
        ]
        if not tracked_list:
            tracked_list = [
                f"{o.class_name} -> ID {getattr(o, 'track_id', '?')}"
                for o in self.detected_objects if getattr(o, 'track_id', -1) > 0
            ]

        d["debug_panel"] = {
            "raw_detections": raw_list,
            "after_nms": nms_list,
            "tracked": tracked_list,
            "expected": f"{exp_action} + {exp_target}",
            "detected": f"{det_action} + {det_target}",
            "target_match": target_match,
            "temporal": temporal_val,
            "validation": val_status_str,
            "fsm": fsm_trans_str,
            "timeline": timeline_str,
        }

        # Multi-Object YOLO Detections Summary (all 5 required classes)
        yolo_summary = {}
        for c in ["yellow_box", "blue_box", "pen", "watch", "hand"]:
            matching = [o for o in self.detected_objects if getattr(o, "class_name", "") == c]
            if matching:
                top = max(matching, key=lambda x: getattr(x, "confidence", 0.0))
                yolo_summary[c] = {
                    "detected": True,
                    "confidence": round(float(top.confidence), 3),
                    "box": list(top.bbox),
                }
            else:
                yolo_summary[c] = {
                    "detected": False,
                    "confidence": 0.0,
                    "box": [],
                }
        d["yolo_detections_summary"] = yolo_summary

        # Real-Time ACTION STATE dictionary for debug panel
        hand_detected = bool(self.left_hand or self.right_hand or any(getattr(o, "class_name", "") == "hand" for o in self.detected_objects))
        target_detected = bool(any(getattr(o, "class_name", "") == exp_target for o in self.detected_objects))
        dist_val = features.get("hand_object_distance", "--") if features else "--"
        motion_detected = bool(
            (features and (features.get("object_velocity", 0.0) > 10.0 or features.get("hand_velocity", 0.0) > 15.0))
            or (self.left_hand and getattr(self.left_hand, "speed", 0.0) > 15.0)
            or (self.right_hand and getattr(self.right_hand, "speed", 0.0) > 15.0)
        )
        val_prog = 100 if val_state in ("ACTION_CONFIRMED", "CONFIRMED_CORRECT") else int(step_conf * 100)
        is_step_complete = bool(val_state in ("ACTION_CONFIRMED", "CONFIRMED_CORRECT") or getattr(self.confirmed_action, "status", None) == ActionGateStatus.CONFIRMED_CORRECT)

        d["action_state"] = {
            "frame_id": self.frame_index,
            "current_step": f"{exp_step_id} / {self.fsm_state.total_steps}",
            "current_step_label": f"Step {exp_step_id}: {self.fsm_state.current_step.label if self.fsm_state.current_step else 'None'}",
            "expected": exp_action,
            "expected_action": exp_action,
            "required": exp_target or "NONE",
            "required_object": exp_target or "NONE",
            "validator": "IDENTIFY_OBJECT" if "IDENTIFY" in exp_action else f"{exp_action}_VALIDATOR",
            "object_valid": "YES" if (target_detected or target_match) else "NO",
            "consecutive": temporal_val,
            "action": det_action,
            "detected_action": det_action,
            "step": "READY TO COMPLETE" if is_step_complete else ("COMPLETED" if self.fsm_state.is_complete else "IN PROGRESS"),
            "step_status": "READY TO COMPLETE" if is_step_complete else ("COMPLETED" if self.fsm_state.is_complete else "IN PROGRESS"),
            "hand_detected": "YES" if hand_detected else "NO",
            "object_detected": "YES" if target_detected else "NO",
            "hand_object_distance": f"{dist_val}px" if isinstance(dist_val, (int, float)) and dist_val < 900 else "--",
            "spatial_relationship": "NEAR" if (isinstance(dist_val, (int, float)) and dist_val < 100) else ("CONTACT" if (features and features.get("is_contact", False)) else "SEPARATED"),
            "motion_detected": "YES" if motion_detected else "NO",
            "validation_progress": f"{val_prog}%",
            "step_complete": "YES" if is_step_complete else "NO",
        }

        d["realtime_debug_state"] = {
            "frame_id": self.frame_index,
            "detections": {
                "BLUE_BOX": {"detected": len(grp.get("BLUE_BOX", [])) > 0, "conf": max([x.get("confidence", 0.0) for x in grp.get("BLUE_BOX", [])], default=0.0)},
                "YELLOW_BOX": {"detected": len(grp.get("YELLOW_BOX", [])) > 0, "conf": max([x.get("confidence", 0.0) for x in grp.get("YELLOW_BOX", [])], default=0.0)},
                "PEN": {"detected": len(grp.get("PEN", [])) > 0, "conf": max([x.get("confidence", 0.0) for x in grp.get("PEN", [])], default=0.0)},
                "WATCH": {"detected": len(grp.get("WATCH", [])) > 0, "conf": max([x.get("confidence", 0.0) for x in grp.get("WATCH", [])], default=0.0)},
                "HAND": {"detected": len(grp.get("HAND", [])) > 0, "conf": max([x.get("confidence", 0.0) for x in grp.get("HAND", [])], default=0.0)},
            },
            "current_step": f"{exp_step_id} / {self.fsm_state.total_steps}",
            "expected": exp_action,
            "required": exp_target or "NONE",
            "validator": "IDENTIFY_OBJECT" if "IDENTIFY" in exp_action else f"{exp_action}_VALIDATOR",
            "object_valid": "YES" if (target_detected or target_match) else "NO",
            "consecutive": temporal_val,
            "action": det_action,
            "step": "READY TO COMPLETE" if is_step_complete else ("COMPLETED" if self.fsm_state.is_complete else "IN PROGRESS"),
        }

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
HAR_TO_EXPERIMENT_VERB: dict[str, str] = {
    "OPEN": "OPEN",
    "TAKE": "TAKE",
    "PICKUP": "PICKUP",
    "PICK_UP": "PICKUP",
    "PLACE": "PLACE",
    "TRANSFER": "TRANSFER",
    "MOVE": "MOVE",
    "CLOSE": "CLOSE",
    "ACTIVATE": "ACTIVATE",
    "PERFORM": "PERFORM",
    "STORE": "STORE",
    "IDENTIFY": "IDENTIFY",
    "IDENTIFICATION": "IDENTIFY",
    "COMPLETE": "COMPLETE",
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

        # Initialize FSM and Action Confirmation Gate
        self.fsm = ExperimentFSM(
            steps=config.experiment_steps,
            experiment_id=self.experiment_id,
            confidence_threshold=config.har.action_confidence_min,
            confirmation_frames_required=3,
        )
        self.action_gate = ActionConfirmationGate()

        # Robust Voice Latch & Section 25 Correction Latch
        self._last_spoken_step_idx: Optional[int] = None
        self._last_voice_message: str = ""
        self._last_status: FSMStatus = FSMStatus.WAITING
        self._last_voice_time: float = 0.0
        self._voice_active_until: float = 0.0
        self._voice_status: str = "IDLE"
        self._voice_events: List[Dict[str, Any]] = []
        self._voice_event_counter: int = 0
        self._last_wrong_action: str = ""
        self._last_wrong_step: int = -1
        self._last_wrong_track_id: Optional[int] = None
        self._last_wrong_event_id: str = ""
        self._last_wrong_voice_time: float = 0.0
        self._wrong_voice_cooldown_seconds: float = 4.0
        self._last_step_id: Optional[int] = None
        self._last_progress_voice_time: float = 0.0

        logger.info("StateManager initialized. Experiment: %s", self.experiment_id)

    @property
    def voice_status(self) -> str:
        return "PLAYING" if time.time() < self._voice_active_until else "IDLE"

    @property
    def voice_events(self) -> List[Dict[str, Any]]:
        return self._voice_events

    def speak_step(self, step_idx: int) -> None:
        """Speaks the exact instruction for step_idx once."""
        if 0 <= step_idx < len(self.config.experiment_steps):
            step = self.config.experiment_steps[step_idx]
            text = step.voice_prompt
            self._execute_voice(text, step_id=step.id, voice_type="STEP_PROMPT")

    def speak_wrong_sequence(
        self,
        message: str,
        wrong_action: str = "",
        current_step_id: int = -1,
        is_new_event: bool = True,
        wrong_event_id: str = "",
        track_id: Optional[int] = None,
    ) -> None:
        """Speaks wrong sequence error message strictly when a NEW wrong action event is confirmed."""
        now = time.time()
        if not message.startswith("Wrong sequence"):
            message = f"Wrong sequence. {message}"

        # Section 8, 9, 10 & 11: Event Deduplication is PRIMARY
        # Do not speak for ongoing frames of the same wrong action event
        if not is_new_event:
            return

        # Secondary Cooldown Guard
        if (
            current_step_id == self._last_wrong_step
            and wrong_action == self._last_wrong_action
            and (now - self._last_wrong_voice_time) < self._wrong_voice_cooldown_seconds
        ):
            return

        self._last_wrong_action = wrong_action
        self._last_wrong_step = current_step_id
        self._last_wrong_track_id = track_id
        self._last_wrong_event_id = wrong_event_id
        self._last_wrong_voice_time = now

        self._execute_voice(
            message,
            step_id=-1,
            voice_type="WRONG_ACTION",
            custom_event_id=wrong_event_id or None,
            is_error=True,
        )

    def speak_complete(self) -> None:
        """Speaks final experiment completion message once."""
        text = "Experiment complete. All steps were successfully completed."
        self._execute_voice(text, step_id=13, voice_type="COMPLETION")

    def _execute_voice(
        self,
        text: str,
        step_id: int = 0,
        voice_type: str = "STEP_PROMPT",
        custom_event_id: Optional[str] = None,
        is_error: bool = False,
    ) -> None:
        now = time.time()
        self.on_voice(text)
        self._last_voice_message = text
        self._last_voice_time = now

        words = len(text.split())
        est_duration = max(1.8, words * 0.42)
        self._voice_active_until = now + est_duration

        self._voice_event_counter += 1
        event_id = custom_event_id or f"VOICE_EVENT_{self._voice_event_counter:03d}"

        event = {
            "voice_event_id": event_id,
            "type": voice_type,
            "step": step_id,
            "text": text,
            "timestamp": now,
            "source": "state_manager",
            "status": "PLAYING",
            "formatted_time": time.strftime("%H:%M:%S", time.localtime(now)),
        }
        self._voice_events.append(event)
        logger.info("VOICE EVENT %s: type=%s step=%d text=\"%s\" timestamp=%.2f",
                    event_id, voice_type, step_id, text, now)

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
        tracks: Optional[list[Any]] = None,
        camera_source: str = "UNKNOWN",
        frame_index: int = 0,
        total_frames: int = 0,
        video_time: str = "00:00.00",
        frame_age_ms: float = 0.0,
        is_stale: bool = False,
        raw_detections: Optional[list] = None,
        nms_detections: Optional[list] = None,
        grouped_detections: Optional[dict] = None,
    ) -> ValidationResult:
        """
        Process perception & interaction evidence through ActionConfirmationGate and FSM.
        Voice guidance is strictly driven by confirmed FSM step transitions.
        If is_stale is True, action confirmation is bypassed to prevent state advancement on backlog frames.
        """
        now = time.time()
        curr_voice_stat = self.voice_status

        # Map HAR action to experiment action + object
        action_verb = HAR_TO_EXPERIMENT_VERB.get(prediction.action.upper(), "IDLE")
        detected_object = (
            prediction.target_object
            if prediction.target_object is not None
            else self._resolve_object(prediction, detected_objects, interactions)
        )

        # 1. Action Confirmation Gate (with stale frame protection)
        current_step = self.fsm.current_step
        current_step_id = getattr(current_step, "id", None) if current_step else None
        if self._last_step_id is not None and self._last_step_id != current_step_id:
            self.action_gate.reset_for_step(current_step)
        self._last_step_id = current_step_id

        if is_stale:
            # Stale frame safeguard: do not confirm actions on delayed/stale frames
            confirmed_action = ConfirmedAction(
                action="IDLE",
                object_name="",
                status=ActionGateStatus.WAITING,
                confidence=0.0,
                validation_state="WAITING",
                validation_reason="STALE_FRAME_BLOCKED",
            )
        else:
            confirmed_action = self.action_gate.evaluate(
                current_step=current_step,
                detected_objects=detected_objects,
                tracks=tracks or [],
                interactions=interactions or [],
                har_action=prediction.action,
                har_confidence=prediction.confidence,
                predicted_target=getattr(prediction, "target_object", "") or detected_object,
                fps=fps,
                frame_age_ms=frame_age_ms,
                is_stale=is_stale,
            )

        # 2. Run FSM with authoritative confirmed action
        fsm_state = self.fsm.process(
            detected_action=action_verb,
            detected_object=detected_object,
            confidence=prediction.confidence,
            confirmed_action=confirmed_action,
            frame_age_ms=frame_age_ms,
        )

        # 3. Synchronized Authoritative Voice Output
        if self._last_spoken_step_idx is None and fsm_state.current_step_idx == 0 and not fsm_state.is_complete:
            # Announce Step 1 on experiment start
            self.speak_step(0)
            self._last_spoken_step_idx = 0
        elif fsm_state.is_complete:
            # Terminal state: announce completion exactly once
            if self._last_spoken_step_idx != fsm_state.total_steps:
                self.speak_complete()
                self._last_spoken_step_idx = fsm_state.total_steps
        elif getattr(fsm_state, "is_transition", False):
            # Confirmed transition to next step
            next_step = self.fsm.current_step
            self.action_gate.reset_for_step(next_step)
            self._last_step_id = getattr(next_step, "id", None) if next_step else None
            if self._last_spoken_step_idx != fsm_state.current_step_idx:
                self.speak_step(fsm_state.current_step_idx)
                self._last_spoken_step_idx = fsm_state.current_step_idx
        elif fsm_state.status in (
            FSMStatus.WRONG_SEQUENCE,
            FSMStatus.WRONG_OBJECT,
            FSMStatus.STEP_SKIPPED,
            FSMStatus.OUT_OF_SEQUENCE,
            FSMStatus.WRONG_ACTION,
        ):
            # Dynamic Section 24 wrong action guidance
            wrong_act_name = getattr(confirmed_action, "action", "")
            wrong_obj_name = getattr(confirmed_action, "object_name", "")
            wrong_full = f"{wrong_act_name} {wrong_obj_name}".strip()
            curr_id = current_step.id if current_step else -1

            if fsm_state.recovery_message and fsm_state.status in (FSMStatus.STEP_SKIPPED, FSMStatus.OUT_OF_SEQUENCE):
                recovery = fsm_state.recovery_message
            elif current_step:
                act = getattr(current_step, "expected_action", "")
                if not act and hasattr(current_step, "action"):
                    act = current_step.action.split("_")[0]
                label = current_step.label
                if act == "IDENTIFY" or label.lower().startswith("identify"):
                    obj_name = getattr(current_step, "expected_object", "")
                    clean_obj = obj_name.replace("_", " ").title() if obj_name else label.replace("Identify the ", "").replace("Identify ", "")
                    recovery = f"Wrong sequence. You're doing the wrong step. Please identify the {clean_obj}."
                elif act == "PICKUP" or label.lower().startswith("pick up"):
                    obj_name = getattr(current_step, "expected_object", "")
                    clean_obj = obj_name.replace("_", " ").title() if obj_name else label.replace("Pick up the ", "").replace("Pick up ", "")
                    recovery = f"Wrong sequence. You're doing the wrong step. Please pick up the {clean_obj}."
                elif label.lower().startswith("place"):
                    recovery = f"Wrong sequence. Please {label[0].lower() + label[1:]}."
                elif label.lower().startswith("move"):
                    from_l = getattr(current_step, "from_location", "")
                    to_l = getattr(current_step, "to_location", "")
                    if from_l and to_l:
                        obj_n = getattr(current_step, "expected_object", "").replace("_", " ").title()
                        f_name = from_l.replace("_", " ").title()
                        t_name = to_l.replace("_", " ").title()
                        recovery = f"Wrong sequence. Please move the {obj_n} from {f_name} to {t_name}."
                    else:
                        recovery = f"Wrong sequence. Please {label[0].lower() + label[1:]}."
                else:
                    recovery = f"Wrong sequence. Please {label[0].lower() + label[1:]}."
            is_new = getattr(confirmed_action, "is_new_event", True) if confirmed_action else True
            event_id = getattr(confirmed_action, "event_id", "") if confirmed_action else ""
            trk_id = getattr(confirmed_action, "track_id", None) if confirmed_action else None

            self.speak_wrong_sequence(
                recovery,
                wrong_action=wrong_full,
                current_step_id=curr_id,
                is_new_event=is_new,
                wrong_event_id=event_id,
                track_id=trk_id,
            )
        elif getattr(confirmed_action, "validation_state", "") == "ACTION_IN_PROGRESS":
            # Section 13: "Good, continue." when operator is approaching the correct action
            if (now - self._last_progress_voice_time) > 8.0 and self.voice_status == "IDLE":
                self._last_progress_voice_time = now
                self._execute_voice("Good, continue.", step_id=current_step.id if current_step else 0, voice_type="GUIDANCE")

        # Run rule engine for telemetry
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

        # Generate HUD and voice text snapshot
        voice_msg = self._last_voice_message or (fsm_state.current_step.voice_prompt if fsm_state.current_step else "Ready.")
        _, hud_msg, alert_level = self._generate_messages(
            fsm_state, rule_matches, prediction
        )

        steps_list = [
            {
                "id": s.id,
                "action": s.action,
                "label": s.label,
                "description": s.description,
                "expected_action": getattr(s, "expected_action", ""),
                "expected_object": getattr(s, "expected_object", ""),
                "expected_target": getattr(s, "expected_target", ""),
            }
            for s in self.config.experiment_steps
        ]

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
            confirmed_action=confirmed_action,
            voice_status=curr_voice_stat,
            steps=steps_list,
            camera_source=camera_source,
            frame_index=frame_index,
            total_frames=total_frames,
            video_time=video_time,
            tracks=tracks or [],
            frame_age_ms=frame_age_ms,
            is_stale=is_stale,
            raw_detections=raw_detections,
            nms_detections=nms_detections,
            grouped_detections=grouped_detections,
        )

    def reset(self) -> None:
        """Reset experiment to step 1."""
        self.fsm.reset()
        self.action_gate.reset()
        self._last_spoken_step_idx = None
        self._last_voice_message = ""
        self._last_status = FSMStatus.WAITING
        self._voice_status = "IDLE"
        self._voice_events.clear()
        self._voice_event_counter = 0
        self._last_wrong_action = ""
        self._last_wrong_step = -1
        self._last_wrong_track_id = None
        self._last_wrong_event_id = ""
        self._last_wrong_voice_time = 0.0
        self._last_step_id = None
        ts = int(time.time()) % 100000
        self.experiment_id = f"{self.config.experiment_id_prefix}{ts:05d}"
        logger.info("StateManager reset. New experiment: %s", self.experiment_id)
        # Speak Step 1 prompt
        self.speak_step(0)
        self._last_spoken_step_idx = 0

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

            # Check for near / approaching objects
            near = [i for i in interactions if getattr(getattr(i, "state", None), "name", "") in ("NEAR_OBJECT", "NEAR", "APPROACHING")]
            if near:
                best_near = max(near, key=lambda x: getattr(x, "confidence", 0.0))
                return best_near.object_class

            # Check pointing
            pointing = [i for i in interactions if getattr(i, "is_pointing", False)]
            if pointing:
                best_pt = max(pointing, key=lambda x: getattr(x, "confidence", 0.0))
                return best_pt.object_class

        if not detected_objects:
            return ""

        # Filter out PERSON/human detections
        candidates = [o for o in detected_objects if str(o.class_name).upper() not in ("PERSON", "HUMAN")]
        if not candidates:
            return ""

        # 1. If interactions exist, prefer candidate closest to the active hand
        if interactions:
            cand_inter = [i for i in interactions if getattr(i, "object_class", "") in {str(c.class_name).upper() for c in candidates}]
            if cand_inter:
                best_by_dist = min(cand_inter, key=lambda x: getattr(x, "distance_px", 9999.0))
                if getattr(best_by_dist, "distance_px", 9999.0) < 300.0:
                    return str(best_by_dist.object_class).upper()

        # 2. If current expected step mentions specific objects, check matching candidates
        current = self.fsm.get_current_state().current_step
        if current:
            expected_objs = {str(o).upper() for o in current.expected_objects}
            matching = [o for o in candidates if str(o.class_name).upper() in expected_objs]
            if matching:
                return str(matching[0].class_name).upper()

        # 3. If predicted action specifies target object that is present, preserve it
        if prediction and getattr(prediction, "target_object", ""):
            pred_tgt = str(prediction.target_object).upper()
            if any(str(c.class_name).upper() == pred_tgt for c in candidates):
                return pred_tgt

        return str(candidates[0].class_name).upper()

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
