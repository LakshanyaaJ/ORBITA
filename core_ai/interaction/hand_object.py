"""
ORBITA Hand-Object Interaction Module
======================================
Determines whether and how a hand is interacting with each detected object.

Interaction is NOT classified from a single frame.
It requires temporal consistency across multiple frames.

Interaction State Machine:
  FAR → APPROACHING → NEAR → CONTACT → GRASPING → MANIPULATING → RELEASING → FAR

States:
  FAR          — hand far from object
  APPROACHING  — hand moving toward object
  NEAR         — hand within approach threshold, still moving
  CONTACT      — hand within contact distance for N frames
  GRASPING     — hand stationary/slow with object proximity (confirmed grasp)
  MANIPULATING — hand+object moving together
  RELEASING    — hand moving away from previously grasped object

Features provided to HAR:
  - interaction_state (integer encoded)
  - hand_object_distance (pixels)
  - approach_velocity (signed: - = approaching)
  - duration_in_state (frames)
  - object_displacement (total object movement while interacting)
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from core_ai.perception.object_detector import DetectedObject
from core_ai.perception.hand_tracker import HandState

logger = logging.getLogger(__name__)


class InteractionState(IntEnum):
    FAR = 0
    APPROACHING = 1
    NEAR = 2
    CONTACT = 3
    GRASPING = 4
    MANIPULATING = 5
    RELEASING = 6


@dataclass
class HandObjectInteraction:
    """Describes the current interaction between one hand and one object."""
    hand_side: str
    object_class: str
    state: InteractionState
    distance_px: float
    approach_velocity: float          # Positive = approaching, negative = receding
    duration_frames: int
    object_displacement: float        # Total object movement since contact
    confidence: float                 # Interaction confidence 0–1


@dataclass
class InteractionTrackerState:
    """Internal state for one (hand, object) pair."""
    state: InteractionState = InteractionState.FAR
    duration: int = 0
    contact_frame_count: int = 0
    last_object_centroid: Optional[np.ndarray] = None
    cumulative_object_displacement: float = 0.0
    prev_distances: deque = field(default_factory=lambda: deque(maxlen=5))


class HandObjectInteractionTracker:
    """
    Tracks interaction state between each hand and each detected object
    across frames using temporal consistency.

    Usage:
        tracker = HandObjectInteractionTracker(config)
        interactions = tracker.update(left_hand, right_hand, objects)
        # returns list[HandObjectInteraction]
    """

    def __init__(self, config):
        self.contact_distance = config.contact_distance_px
        self.grasp_frames = config.grasp_frames_required
        self.release_frames = config.release_frames_required
        # Key: (side, class_name) → InteractionTrackerState
        self._states: dict[tuple[str, str], InteractionTrackerState] = defaultdict(
            InteractionTrackerState
        )

    def update(
        self,
        left: HandState,
        right: HandState,
        objects: list[DetectedObject],
        timestamp: float = 0.0,
    ) -> list[HandObjectInteraction]:
        """
        Update interaction states and return current interactions.

        Returns a list of HandObjectInteraction for every (hand, object) pair
        that is not in state FAR.
        """
        results: list[HandObjectInteraction] = []

        for hand in (left, right):
            if not hand.is_visible:
                continue
            for obj in objects:
                if obj.class_name == "PERSON":
                    continue
                key = (hand.side, obj.class_name)
                dist = float(np.linalg.norm(
                    hand.position - np.array(obj.centroid, dtype=np.float32)
                ))
                interaction = self._transition(key, hand, obj, dist)
                if interaction.state != InteractionState.FAR:
                    results.append(interaction)

        return results

    def get_primary_interaction(
        self, interactions: list[HandObjectInteraction]
    ) -> Optional[HandObjectInteraction]:
        """Return the highest-priority interaction (MANIPULATING > GRASPING > CONTACT > ...)."""
        if not interactions:
            return None
        priority = [
            InteractionState.MANIPULATING,
            InteractionState.GRASPING,
            InteractionState.CONTACT,
            InteractionState.RELEASING,
            InteractionState.NEAR,
            InteractionState.APPROACHING,
        ]
        for p in priority:
            candidates = [i for i in interactions if i.state == p]
            if candidates:
                return max(candidates, key=lambda x: x.confidence)
        return None

    def reset(self) -> None:
        self._states.clear()

    # ----------------------------------------------------------------------- #
    # Internal FSM transition
    # ----------------------------------------------------------------------- #
    def _transition(
        self,
        key: tuple[str, str],
        hand: HandState,
        obj: DetectedObject,
        dist: float,
    ) -> HandObjectInteraction:
        s = self._states[key]

        # Compute approach velocity (positive = approaching)
        s.prev_distances.append(dist)
        if len(s.prev_distances) >= 2:
            approach_vel = float(s.prev_distances[-2] - s.prev_distances[-1])
        else:
            approach_vel = 0.0

        near_thresh = self.contact_distance * 2.5
        contact_thresh = self.contact_distance

        prev_state = s.state
        s.duration += 1

        # Object displacement tracking
        obj_centroid = np.array(obj.centroid, dtype=np.float32)
        if s.last_object_centroid is not None:
            s.cumulative_object_displacement += float(
                np.linalg.norm(obj_centroid - s.last_object_centroid)
            )
        s.last_object_centroid = obj_centroid

        # ---- FSM transitions ----
        if s.state == InteractionState.FAR:
            if dist < near_thresh:
                s.state = InteractionState.NEAR if dist < near_thresh else InteractionState.FAR
                s.duration = 0
            if dist < contact_thresh:
                s.state = InteractionState.NEAR
                s.duration = 0

        elif s.state == InteractionState.NEAR:
            if dist < contact_thresh:
                s.contact_frame_count += 1
                if s.contact_frame_count >= self.grasp_frames:
                    s.state = InteractionState.CONTACT
                    s.duration = 0
            elif dist > near_thresh * 1.5:
                s.state = InteractionState.FAR
                s.duration = 0
                s.contact_frame_count = 0

        elif s.state == InteractionState.CONTACT:
            if dist < contact_thresh:
                # Check if hand is stationary (grasping)
                if hand.speed < 5.0:
                    s.state = InteractionState.GRASPING
                    s.duration = 0
                # Check if hand+object moving (manipulating)
                elif hand.speed > 8.0:
                    s.state = InteractionState.MANIPULATING
                    s.duration = 0
            elif dist > near_thresh:
                s.state = InteractionState.RELEASING
                s.duration = 0

        elif s.state == InteractionState.GRASPING:
            if hand.speed > 8.0 and dist < contact_thresh:
                s.state = InteractionState.MANIPULATING
                s.duration = 0
            elif dist > contact_thresh:
                s.state = InteractionState.RELEASING
                s.duration = 0

        elif s.state == InteractionState.MANIPULATING:
            if dist > contact_thresh * 1.5:
                s.state = InteractionState.RELEASING
                s.duration = 0
            elif hand.speed < 3.0:
                s.state = InteractionState.GRASPING
                s.duration = 0

        elif s.state == InteractionState.RELEASING:
            if dist > near_thresh * 2:
                s.state = InteractionState.FAR
                s.duration = 0
                s.contact_frame_count = 0
                s.cumulative_object_displacement = 0.0

        # State confidence: higher when in state longer
        confidence = float(np.clip(s.duration / max(self.grasp_frames, 1), 0.0, 1.0))
        if s.state in (InteractionState.MANIPULATING, InteractionState.GRASPING):
            confidence = min(confidence + 0.3, 1.0)

        return HandObjectInteraction(
            hand_side=key[0],
            object_class=key[1],
            state=s.state,
            distance_px=dist,
            approach_velocity=approach_vel,
            duration_frames=s.duration,
            object_displacement=s.cumulative_object_displacement,
            confidence=confidence,
        )
