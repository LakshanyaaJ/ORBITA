"""
ORBITA Hand-Object Interaction Module
======================================
Determines fine-grained spatial and physical interactions between tracked hands
and detected experiment objects (SAMPLE, TOOL, MAIN_BOX, RED_BOX, YELLOW_BOX).

Key capabilities:
  - Multi-signal contact detection: index fingertip, thumb tip, and wrist proximity
    to object bounding boxes (not simple center distance).
  - Velocity coupling for HOLDING detection: compares hand velocity vector with
    object velocity vector.
  - Release detection: detects when an object becomes stationary as the hand departs.
  - Container transfer tracking: recognizes SAMPLE / TOOL being held, moved into a
    container (RED_BOX, YELLOW_BOX, MAIN_BOX), and released.
  - Explainable confidence fusion from visual evidence, kinematic similarity, and
    temporal consistency.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core_ai.perception.object_detector import DetectedObject
from core_ai.perception.hand_tracker import HandState

logger = logging.getLogger(__name__)


class InteractionState(IntEnum):
    NOT_INTERACTING = 0
    NEAR_OBJECT = 1
    CONTACT = 2
    HOLDING = 3
    RELEASING = 4
    RELEASED = 5

    # Backwards compatibility aliases
    FAR = 0
    APPROACHING = 1
    NEAR = 1
    GRASPING = 3
    MANIPULATING = 3


CONTAINER_CLASSES = {"MAIN_BOX", "RED_BOX", "YELLOW_BOX", "BLUE_BOX"}
MANIPULABLE_CLASSES = {"SAMPLE", "TOOL"}


@dataclass
class HandObjectInteraction:
    """Describes the current interaction between one hand and one object."""
    hand_side: str
    object_class: str
    state: InteractionState
    distance_px: float
    approach_velocity: float          # Positive = approaching (px/s), negative = receding
    duration_frames: int
    object_displacement: float        # Total object movement since contact
    confidence: float                 # Fused confidence 0.0 - 1.0

    # Explainable confidence sub-components
    hand_confidence: float = 0.0
    object_confidence: float = 0.0
    distance_score: float = 0.0
    velocity_score: float = 0.0
    temporal_score: float = 0.0

    # High-level semantics & kinematics
    is_fingertip_contact: bool = False
    is_holding: bool = False
    transfer_event: Optional[str] = None  # e.g. "TRANSFER(SAMPLE -> YELLOW_BOX)"
    target_container: Optional[str] = None
    hand_object_overlap: float = 0.0
    relative_hand_object_motion: float = 0.0
    is_pointing: bool = False
    hand_speed: float = 0.0
    object_speed: float = 0.0
    contact_duration: int = 0


@dataclass
class InteractionTrackerState:
    """Internal state for one (hand, object) pair across time."""
    state: InteractionState = InteractionState.NOT_INTERACTING
    duration: int = 0
    contact_frame_count: int = 0
    holding_frame_count: int = 0
    release_frame_count: int = 0
    last_object_centroid: Optional[np.ndarray] = None
    last_timestamp: float = 0.0
    object_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    cumulative_object_displacement: float = 0.0
    prev_distances: deque = field(default_factory=lambda: deque(maxlen=6))
    initial_holding_container: Optional[str] = None


def bbox_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """Computes IoU between two (x, y, w, h) bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    return float(interArea / max(unionArea, 1e-6))


def point_to_bbox_distance(pt: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Compute the minimum Euclidean distance from a point to a bounding box perimeter."""
    bx, by, bw, bh = bbox
    px, py = float(pt[0]), float(pt[1])
    rx1, ry1 = float(bx), float(by)
    rx2, ry2 = float(bx + bw), float(by + bh)

    dx = max(rx1 - px, 0.0, px - rx2)
    dy = max(ry1 - py, 0.0, py - ry2)
    return float(np.sqrt(dx * dx + dy * dy))


def bbox_contains_point(pt: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    """Check if point is strictly inside bounding box."""
    bx, by, bw, bh = bbox
    return (bx <= pt[0] <= bx + bw) and (by <= pt[1] <= by + bh)


class HandObjectInteractionTracker:
    """
    Tracks interaction states between hands and objects using multi-signal visual evidence:
    fingertip proximity, bounding box overlap, velocity coupling, and temporal persistence.
    """

    def __init__(self, config: Any):
        self.config = config
        self.contact_distance = getattr(config, "contact_distance_px", 60)
        self.grasp_frames = getattr(config, "grasp_frames_required", 4)
        self.release_frames = getattr(config, "release_frames_required", 4)

        # Key: (side, object_class) -> InteractionTrackerState
        self._states: dict[tuple[str, str], InteractionTrackerState] = defaultdict(
            InteractionTrackerState
        )

        # Global object container containment memory
        self._object_containers: dict[str, Optional[str]] = {}

    def update(
        self,
        left: HandState,
        right: HandState,
        objects: list[DetectedObject],
        timestamp: float = 0.0,
    ) -> list[HandObjectInteraction]:
        """
        Evaluate and return all active interactions between visible hands and objects.
        """
        results: list[HandObjectInteraction] = []

        # 1. Update container containment mapping for objects
        containers = [obj for obj in objects if obj.class_name in CONTAINER_CLASSES]
        for obj in objects:
            if obj.class_name in MANIPULABLE_CLASSES:
                c_name = None
                for c in containers:
                    if bbox_contains_point(np.array(obj.centroid, dtype=np.float32), c.bbox):
                        c_name = c.class_name
                        break
                self._object_containers[obj.class_name] = c_name

        # 2. Update interaction FSM per hand and per object
        for hand in (left, right):
            if hand is None or not getattr(hand, "is_visible", False) or getattr(hand, "confidence", 0.0) < 0.35:
                continue

            for obj in objects:
                if obj.class_name == "PERSON":
                    continue

                key = (hand.side, obj.class_name)
                interaction = self._transition(key, hand, obj, containers, timestamp)
                if interaction.state != InteractionState.NOT_INTERACTING:
                    results.append(interaction)

        return results

    def get_primary_interaction(
        self, interactions: list[HandObjectInteraction]
    ) -> Optional[HandObjectInteraction]:
        """Return the highest-priority interaction."""
        if not interactions:
            return None
        priority = [
            InteractionState.HOLDING,
            InteractionState.CONTACT,
            InteractionState.RELEASING,
            InteractionState.RELEASED,
            InteractionState.NEAR_OBJECT,
        ]
        for p in priority:
            candidates = [i for i in interactions if i.state == p]
            if candidates:
                return max(candidates, key=lambda x: x.confidence)
        return None

    def reset(self) -> None:
        """Reset internal interaction tracking states."""
        self._states.clear()
        self._object_containers.clear()

    # ----------------------------------------------------------------------- #
    # Live Overlay Drawing for Interactions
    # ----------------------------------------------------------------------- #
    def draw_interactions(
        self,
        frame: np.ndarray,
        interactions: list[HandObjectInteraction],
        objects: list[DetectedObject],
        left: HandState,
        right: HandState,
    ) -> np.ndarray:
        """Render interaction vectors, badges, and transfer indicators."""
        vis = frame.copy()
        if not interactions:
            return vis

        obj_map = {o.class_name: o for o in objects}
        hand_map = {"left": left, "right": right}

        state_colors = {
            InteractionState.NOT_INTERACTING: (180, 180, 180),
            InteractionState.NEAR_OBJECT: (0, 200, 255),       # Orange/Yellow
            InteractionState.CONTACT: (0, 255, 255),           # Yellow
            InteractionState.HOLDING: (0, 230, 80),            # Vibrant Green
            InteractionState.RELEASING: (180, 100, 255),       # Pink
            InteractionState.RELEASED: (200, 200, 200),        # Grey
        }

        for interaction in interactions:
            if interaction.state == InteractionState.NOT_INTERACTING:
                continue

            h = hand_map.get(interaction.hand_side)
            o = obj_map.get(interaction.object_class)
            if not h or not o or not h.is_visible:
                continue

            colour = state_colors.get(interaction.state, (0, 255, 0))

            # Connect hand index tip (or wrist) to object centroid
            h_pt = h.index_tip if h.index_tip is not None else h.position
            p1 = (int(h_pt[0]), int(h_pt[1]))
            p2 = (int(o.centroid[0]), int(o.centroid[1]))

            # Draw dashed/solid interaction link
            cv2.line(vis, p1, p2, (20, 25, 35), 3, cv2.LINE_AA)
            cv2.line(vis, p1, p2, colour, 2, cv2.LINE_AA)

            # Midpoint interaction state badge
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2

            state_name = interaction.state.name
            badge = f"{state_name} ({int(interaction.distance_px)}px | {int(interaction.confidence * 100)}%)"
            if interaction.transfer_event:
                badge = f"{badge} -> {interaction.transfer_event}"

            cv2.rectangle(vis, (mid_x - 6, mid_y - 14), (mid_x + len(badge) * 7 + 10, mid_y + 8), (15, 20, 30), -1)
            cv2.rectangle(vis, (mid_x - 6, mid_y - 14), (mid_x + len(badge) * 7 + 10, mid_y + 8), colour, 1)
            cv2.putText(vis, badge, (mid_x, mid_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)

        return vis

    # ----------------------------------------------------------------------- #
    # Internal FSM Transition Logic
    # ----------------------------------------------------------------------- #
    def _transition(
        self,
        key: tuple[str, str],
        hand: HandState,
        obj: DetectedObject,
        containers: list[DetectedObject],
        timestamp: float,
    ) -> HandObjectInteraction:
        s = self._states[key]
        dt = timestamp - s.last_timestamp if s.last_timestamp > 0.0 else 0.033
        s.last_timestamp = timestamp
        s.duration += 1

        # 1. Multi-signal spatial distances
        obj_centroid = np.array(obj.centroid, dtype=np.float32)
        dist_wrist = point_to_bbox_distance(hand.position, obj.bbox)

        dist_thumb = float("inf")
        dist_index = float("inf")
        if hand.thumb_tip is not None:
            dist_thumb = point_to_bbox_distance(hand.thumb_tip, obj.bbox)
        if hand.index_tip is not None:
            dist_index = point_to_bbox_distance(hand.index_tip, obj.bbox)

        fingertip_dists = [dist_thumb, dist_index]
        for tip in (hand.middle_tip, hand.ring_tip, hand.pinky_tip):
            if tip is not None:
                fingertip_dists.append(point_to_bbox_distance(tip, obj.bbox))

        min_fingertip_dist = min(fingertip_dists)
        effective_dist = min(min_fingertip_dist, dist_wrist)

        # Bounding box overlap between hand and object
        hand_obj_iou = 0.0
        hand_obj_overlap_px = 0.0
        if getattr(hand, "bbox", None) and getattr(obj, "bbox", None):
            hx, hy, hw, hh = hand.bbox
            ox, oy, ow, oh = obj.bbox
            ix1 = max(hx, ox)
            iy1 = max(hy, oy)
            ix2 = min(hx + hw, ox + ow)
            iy2 = min(hy + hh, oy + oh)
            if ix2 > ix1 and iy2 > iy1:
                hand_obj_overlap_px = float((ix2 - ix1) * (iy2 - iy1))
                hand_obj_iou = hand_obj_overlap_px / float(hw * hh + ow * oh - hand_obj_overlap_px + 1e-6)

        any_tip_inside = any(
            tip is not None and bbox_contains_point(tip, obj.bbox)
            for tip in (hand.thumb_tip, hand.index_tip, hand.middle_tip, hand.ring_tip, hand.pinky_tip)
        )
        any_landmark_inside = False
        if hand.finger_landmarks is not None and len(hand.finger_landmarks) == 21:
            for lm in hand.finger_landmarks:
                if bbox_contains_point(lm, obj.bbox):
                    any_landmark_inside = True
                    break

        # Fingertip contact flag
        is_fingertip_contact = (
            min_fingertip_dist < 32.0 or
            any_tip_inside or
            any_landmark_inside or
            (hand_obj_overlap_px > 150.0 and min_fingertip_dist < 60.0)
        )

        # 2. Object kinematics and displacement
        if s.last_object_centroid is not None and dt > 0.001:
            raw_obj_vel = (obj_centroid - s.last_object_centroid) / dt
            alpha = 0.5
            s.object_velocity = alpha * raw_obj_vel + (1.0 - alpha) * s.object_velocity
            disp = float(np.linalg.norm(obj_centroid - s.last_object_centroid))
            s.cumulative_object_displacement += disp
        s.last_object_centroid = obj_centroid.copy()
        obj_speed = float(np.linalg.norm(s.object_velocity))

        # 3. Approach velocity (positive = getting closer)
        s.prev_distances.append(effective_dist)
        if len(s.prev_distances) >= 2:
            approach_vel = float((s.prev_distances[-2] - s.prev_distances[-1]) / max(dt, 0.001))
        else:
            approach_vel = 0.0

        # 4. Kinematic velocity coupling for HOLDING detection
        # When hand picks up an object, both velocity vectors align
        vel_similarity = 0.0
        if hand.speed > 20.0 and obj_speed > 20.0:
            hand_v_norm = hand.velocity / (hand.speed + 1e-6)
            obj_v_norm = s.object_velocity / (obj_speed + 1e-6)
            dot = float(np.dot(hand_v_norm, obj_v_norm))
            vel_similarity = max(0.0, dot)

        # 5. Threshold definitions
        near_thresh = self.contact_distance * 2.2
        contact_thresh = self.contact_distance

        prev_state = s.state
        transfer_event = None

        # ---- FSM STATE TRANSITIONS ----
        if s.state == InteractionState.NOT_INTERACTING:
            if is_fingertip_contact or effective_dist < contact_thresh:
                s.state = InteractionState.CONTACT
                s.contact_frame_count = 1
                s.duration = 0
            elif effective_dist < near_thresh:
                s.state = InteractionState.NEAR_OBJECT
                s.duration = 0

        elif s.state == InteractionState.NEAR_OBJECT:
            if is_fingertip_contact or effective_dist < contact_thresh:
                s.contact_frame_count += 1
                if s.contact_frame_count >= 2:
                    s.state = InteractionState.CONTACT
                    s.duration = 0
            elif effective_dist > near_thresh * 1.3:
                s.state = InteractionState.NOT_INTERACTING
                s.duration = 0
                s.contact_frame_count = 0

        elif s.state == InteractionState.CONTACT:
            if (is_fingertip_contact or effective_dist < contact_thresh):
                s.contact_frame_count += 1
                # Confirm HOLDING if:
                # (a) Object follows hand (similar velocity and small distance) OR
                # (b) Fingertips in contact for required grasp frames
                if (vel_similarity > 0.60 and s.cumulative_object_displacement > 15.0) or (s.contact_frame_count >= self.grasp_frames):
                    s.state = InteractionState.HOLDING
                    s.holding_frame_count = 1
                    s.duration = 0
                    # Remember starting container
                    s.initial_holding_container = self._object_containers.get(obj.class_name)
            elif effective_dist > contact_thresh * 1.4:
                s.state = InteractionState.RELEASING
                s.duration = 0

        elif s.state == InteractionState.HOLDING:
            s.holding_frame_count += 1
            # Still holding if distance remains small AND (either moving together or both stationary)
            if effective_dist <= contact_thresh * 1.4:
                # Check for container entrance during hold
                current_container = self._object_containers.get(obj.class_name)
                # maintain holding
                pass
            else:
                # Separation: hand moves away
                s.state = InteractionState.RELEASING
                s.duration = 0
                s.release_frame_count = 1

        elif s.state == InteractionState.RELEASING:
            s.release_frame_count += 1
            # Release confirmed if hand separates AND object becomes stationary
            if obj_speed < 30.0 and effective_dist > contact_thresh:
                s.state = InteractionState.RELEASED
                s.duration = 0
                # Check if this release completed a container transfer!
                target_container = self._object_containers.get(obj.class_name)
                if target_container is not None and s.initial_holding_container != target_container:
                    transfer_event = f"TRANSFER({obj.class_name} -> {target_container})"
                    logger.info("High-level event detected: %s", transfer_event)
            elif effective_dist < contact_thresh and (is_fingertip_contact or hand.grasp_confidence > 0.6):
                # Re-contacted
                s.state = InteractionState.HOLDING
                s.duration = 0

        elif s.state == InteractionState.RELEASED:
            if s.duration > self.release_frames or effective_dist > near_thresh * 1.5:
                s.state = InteractionState.NOT_INTERACTING
                s.duration = 0
                s.contact_frame_count = 0
                s.holding_frame_count = 0
                s.release_frame_count = 0
                s.cumulative_object_displacement = 0.0
                s.initial_holding_container = None

        # 6. Explainable Weighted Confidence Fusion
        hand_c = float(hand.confidence)
        obj_c = float(obj.confidence)
        dist_score = float(np.clip(1.0 - effective_dist / max(near_thresh, 1.0), 0.0, 1.0))
        vel_score = float(np.clip(vel_similarity, 0.0, 1.0)) if s.state == InteractionState.HOLDING else 0.5
        temp_score = float(np.clip(s.duration / max(self.grasp_frames, 1), 0.0, 1.0))

        w_hand = 0.25
        w_obj = 0.20
        w_dist = 0.25
        w_vel = 0.15
        w_temp = 0.15

        fused_conf = float(np.clip(
            w_hand * hand_c +
            w_obj * obj_c +
            w_dist * dist_score +
            w_vel * vel_score +
            w_temp * temp_score,
            0.0, 1.0
        ))

        # Overlap and Pointing gesture computation
        hand_overlap = 0.0
        if getattr(hand, "bbox", None) and getattr(obj, "bbox", None):
            hand_overlap = bbox_iou(hand.bbox, obj.bbox)

        is_pointing = False
        if dist_index != float("inf") and dist_wrist != float("inf"):
            # Index tip closer than wrist by at least 15px and within 220px of object
            if dist_index < 220.0 and (dist_wrist - dist_index) >= 15.0:
                is_pointing = True

        return HandObjectInteraction(
            hand_side=key[0],
            object_class=key[1],
            state=s.state,
            distance_px=effective_dist,
            approach_velocity=approach_vel,
            duration_frames=s.duration,
            object_displacement=s.cumulative_object_displacement,
            confidence=fused_conf,
            hand_confidence=hand_c,
            object_confidence=obj_c,
            distance_score=dist_score,
            velocity_score=vel_score,
            temporal_score=temp_score,
            is_fingertip_contact=is_fingertip_contact,
            is_holding=(s.state == InteractionState.HOLDING),
            transfer_event=transfer_event,
            target_container=self._object_containers.get(obj.class_name),
            hand_object_overlap=hand_overlap,
            relative_hand_object_motion=vel_similarity,
            is_pointing=is_pointing,
            hand_speed=float(getattr(hand, "speed", 0.0)),
            object_speed=obj_speed,
            contact_duration=s.contact_frame_count,
        )

