"""
ORBITA Procedure Step Validation Specifications
================================================
Defines strict, independent validation specifications for all experiment steps.
Perception (YOLO detection, hand detection, proximity) is explicitly decoupled
from action completion.

Every step has:
  - required_objects
  - required_action
  - required_spatial_relationship
  - required_motion
  - expected_before_state (precondition)
  - expected_after_state (postcondition)
  - minimum_confidence
  - minimum_duration
  - temporal_confirmation_frames
  - failure_conditions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepValidationSpec:
    """Formal validation specification for an experiment procedure step."""
    step_id: int
    instruction: str
    required_objects: List[str]
    required_action: str                          # "IDENTIFY" | "PICKUP" | "PLACE" | "MOVE" | "COMPLETE"
    required_spatial_relationship: str            # "POINTING_AT" | "GRASPED" | "CONTAINED_IN" | "AT_LOCATION_A" | "AT_LOCATION_B" | "RELEASED"
    required_motion: str                          # "POINTING_GESTURE" | "LIFT_AND_HOLD" | "RELEASE_MOTION" | "TRANSFER_MOTION" | "INSERT_AND_RELEASE" | "NONE"
    expected_before_state: Dict[str, Any]
    expected_after_state: Dict[str, Any]
    minimum_confidence: float = 0.70
    minimum_duration: float = 0.5                 # seconds
    temporal_confirmation_frames: int = 3         # frames (~100-200ms at 15-30 FPS)
    movement_threshold_px: float = 20.0
    failure_condition: str = ""


# Pre-defined specifications for the canonical 13-step experiment procedure
STEP_SPECS: Dict[int, StepValidationSpec] = {
    1: StepValidationSpec(
        step_id=1,
        instruction="Identify the Blue Box",
        required_objects=["BLUE_BOX"],
        required_action="IDENTIFY_BLUE_BOX",
        required_spatial_relationship="IN_WORKSPACE",
        required_motion="NONE",
        expected_before_state={"target_present": True, "object_stationary": True},
        expected_after_state={"target_identified": True, "object_not_displaced": True},
        minimum_confidence=0.35,
        minimum_duration=0.3,
        temporal_confirmation_frames=3,
        movement_threshold_px=0.0,
        failure_condition="WRONG_OBJECT_IDENTIFIED",
    ),
    2: StepValidationSpec(
        step_id=2,
        instruction="Pick up the Blue Box",
        required_objects=["BLUE_BOX"],
        required_action="PICKUP",
        required_spatial_relationship="GRASPED",
        required_motion="LIFT_AND_HOLD",
        expected_before_state={"on_table": True, "held": False},
        expected_after_state={"held": True, "displaced": True, "displacement_px_min": 20.0},
        minimum_confidence=0.70,
        minimum_duration=0.5,
        temporal_confirmation_frames=3,
        movement_threshold_px=20.0,
        failure_condition="WRONG_OBJECT_PICKED_UP",
    ),
    3: StepValidationSpec(
        step_id=3,
        instruction="Place the Blue Box at Location A",
        required_objects=["BLUE_BOX"],
        required_action="PLACE",
        required_spatial_relationship="AT_LOCATION_A",
        required_motion="RELEASE_MOTION",
        expected_before_state={"held": True},
        expected_after_state={"destination": "LOCATION_A", "released": True, "stationary": True},
        minimum_confidence=0.70,
        minimum_duration=0.5,
        temporal_confirmation_frames=3,
        movement_threshold_px=15.0,
        failure_condition="PLACED_OUTSIDE_LOCATION_A",
    ),
    4: StepValidationSpec(
        step_id=4,
        instruction="Identify the Yellow Box",
        required_objects=["YELLOW_BOX"],
        required_action="IDENTIFY_YELLOW_BOX",
        required_spatial_relationship="IN_WORKSPACE",
        required_motion="NONE",
        expected_before_state={"target_present": True, "object_stationary": True},
        expected_after_state={"target_identified": True, "object_not_displaced": True},
        minimum_confidence=0.35,
        minimum_duration=0.3,
        temporal_confirmation_frames=3,
        movement_threshold_px=0.0,
        failure_condition="WRONG_OBJECT_IDENTIFIED",
    ),
    5: StepValidationSpec(
        step_id=5,
        instruction="Pick up the Yellow Box",
        required_objects=["YELLOW_BOX"],
        required_action="PICKUP",
        required_spatial_relationship="GRASPED",
        required_motion="LIFT_AND_HOLD",
        expected_before_state={"on_table": True, "held": False},
        expected_after_state={"held": True, "displaced": True, "displacement_px_min": 20.0},
        minimum_confidence=0.70,
        minimum_duration=0.5,
        temporal_confirmation_frames=3,
        movement_threshold_px=20.0,
        failure_condition="WRONG_OBJECT_PICKED_UP",
    ),
    6: StepValidationSpec(
        step_id=6,
        instruction="Place the Yellow Box at Location B",
        required_objects=["YELLOW_BOX"],
        required_action="PLACE",
        required_spatial_relationship="AT_LOCATION_B",
        required_motion="RELEASE_MOTION",
        expected_before_state={"held": True},
        expected_after_state={"destination": "LOCATION_B", "released": True, "stationary": True},
        minimum_confidence=0.70,
        minimum_duration=0.5,
        temporal_confirmation_frames=3,
        movement_threshold_px=15.0,
        failure_condition="PLACED_OUTSIDE_LOCATION_B",
    ),
    7: StepValidationSpec(
        step_id=7,
        instruction="Pick up the Pen",
        required_objects=["PEN"],
        required_action="PICKUP",
        required_spatial_relationship="GRASPED",
        required_motion="LIFT_AND_HOLD",
        expected_before_state={"on_table": True, "held": False},
        expected_after_state={"held": True, "displaced": True, "displacement_px_min": 20.0},
        minimum_confidence=0.65,
        minimum_duration=0.5,
        temporal_confirmation_frames=3,
        movement_threshold_px=20.0,
        failure_condition="WRONG_OBJECT_PICKED_UP",
    ),
    8: StepValidationSpec(
        step_id=8,
        instruction="Place the Pen inside the Blue Box",
        required_objects=["PEN", "BLUE_BOX"],
        required_action="PLACE",
        required_spatial_relationship="CONTAINED_IN_BLUE_BOX",
        required_motion="INSERT_AND_RELEASE",
        expected_before_state={"held": True},
        expected_after_state={"target_container": "BLUE_BOX", "released": True, "stationary": True},
        minimum_confidence=0.65,
        minimum_duration=0.5,
        temporal_confirmation_frames=3,
        movement_threshold_px=15.0,
        failure_condition="PLACED_OUTSIDE_BLUE_BOX",
    ),
    9: StepValidationSpec(
        step_id=9,
        instruction="Pick up the Watch",
        required_objects=["WATCH"],
        required_action="PICKUP",
        required_spatial_relationship="GRASPED",
        required_motion="LIFT_AND_HOLD",
        expected_before_state={"on_table": True, "held": False},
        expected_after_state={"held": True, "displaced": True, "displacement_px_min": 20.0},
        minimum_confidence=0.65,
        minimum_duration=0.5,
        temporal_confirmation_frames=3,
        movement_threshold_px=20.0,
        failure_condition="WRONG_OBJECT_PICKED_UP",
    ),
    10: StepValidationSpec(
        step_id=10,
        instruction="Place the Watch inside the Yellow Box",
        required_objects=["WATCH", "YELLOW_BOX"],
        required_action="PLACE",
        required_spatial_relationship="CONTAINED_IN_YELLOW_BOX",
        required_motion="INSERT_AND_RELEASE",
        expected_before_state={"held": True},
        expected_after_state={"target_container": "YELLOW_BOX", "released": True, "stationary": True},
        minimum_confidence=0.65,
        minimum_duration=0.5,
        temporal_confirmation_frames=3,
        movement_threshold_px=15.0,
        failure_condition="PLACED_OUTSIDE_YELLOW_BOX",
    ),
    11: StepValidationSpec(
        step_id=11,
        instruction="Move Blue Box from Location A to Location B",
        required_objects=["BLUE_BOX"],
        required_action="MOVE",
        required_spatial_relationship="AT_LOCATION_B",
        required_motion="TRANSFER_MOTION",
        expected_before_state={"location": "LOCATION_A"},
        expected_after_state={"location": "LOCATION_B", "displacement_px_min": 80.0, "released": True, "stationary": True},
        minimum_confidence=0.70,
        minimum_duration=0.8,
        temporal_confirmation_frames=3,
        movement_threshold_px=80.0,
        failure_condition="MOVED_TO_WRONG_LOCATION",
    ),
    12: StepValidationSpec(
        step_id=12,
        instruction="Move Yellow Box from Location B to Location A",
        required_objects=["YELLOW_BOX"],
        required_action="MOVE",
        required_spatial_relationship="AT_LOCATION_A",
        required_motion="TRANSFER_MOTION",
        expected_before_state={"location": "LOCATION_B"},
        expected_after_state={"location": "LOCATION_A", "displacement_px_min": 80.0, "released": True, "stationary": True},
        minimum_confidence=0.70,
        minimum_duration=0.8,
        temporal_confirmation_frames=3,
        movement_threshold_px=80.0,
        failure_condition="MOVED_TO_WRONG_LOCATION",
    ),
    13: StepValidationSpec(
        step_id=13,
        instruction="Experiment Complete",
        required_objects=[],
        required_action="COMPLETE",
        required_spatial_relationship="ALL_COMPLETED",
        required_motion="NONE",
        expected_before_state={"previous_steps_completed": True},
        expected_after_state={"experiment_complete": True},
        minimum_confidence=1.0,
        minimum_duration=0.1,
        temporal_confirmation_frames=1,
        movement_threshold_px=0.0,
        failure_condition="",
    ),
}


def get_step_validation_spec(step: Any) -> StepValidationSpec:
    """Retrieve or dynamically construct validation spec for a procedure step."""
    if step is None:
        return StepValidationSpec(
            step_id=0,
            instruction="No Step",
            required_objects=[],
            required_action="IDLE",
            required_spatial_relationship="NONE",
            required_motion="NONE",
            expected_before_state={},
            expected_after_state={},
        )

    def _get(k: str, default: Any = "") -> Any:
        if isinstance(step, dict):
            return step.get(k, default)
        return getattr(step, k, default)

    s_id = _get("id", 0)
    if s_id in STEP_SPECS:
        return STEP_SPECS[s_id]

    # Dynamic fallback based on step attributes
    label = _get("label", _get("action", ""))
    act = _get("expected_action", "") or (_get("action", "").split("_")[0] if "_" in _get("action", "") else _get("action", ""))
    act_str = _get("action", "")
    obj = _get("expected_object", "") or (act_str.split("_", 1)[1] if "_" in act_str else "")
    tgt = _get("expected_target", "")
    from_l = _get("from_location", "")
    to_l = _get("to_location", "")

    return StepValidationSpec(
        step_id=s_id,
        instruction=str(label),
        required_objects=[obj] if obj else [],
        required_action=str(act).upper() if act else "IDLE",
        required_spatial_relationship=str(tgt).upper() if tgt else "NONE",
        required_motion=str(to_l).upper() if to_l else "NONE",
        expected_before_state={"from": from_l} if from_l else {},
        expected_after_state={"to": to_l} if to_l else {},
        minimum_confidence=0.65,
        minimum_duration=0.5,
        movement_threshold_px=15.0 if str(act).upper() in ("PICKUP", "TAKE", "MOVE", "PLACE") else 0.0,
        temporal_confirmation_frames=2 if str(act).upper() in ("PICKUP", "TAKE", "PLACE") else 3,
    )
