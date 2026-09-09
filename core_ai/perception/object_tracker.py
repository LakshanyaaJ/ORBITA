"""
ORBITA Multi-Object Tracker
============================
Maintains consistent identity, velocity, and state for detected experiment objects
and human operators across video frames.

Key capabilities:
  1. Persistent track IDs across temporary occlusions / dropped frames (grace period).
  2. Velocity estimation (dx/dt, dy/dt) for interaction and trajectory analysis.
  3. Bounding-box spatial-temporal smoothing to eliminate jitter.
  4. Explicit primary operator identification when multiple persons appear.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


from enum import Enum

class ObjectPhysicalState(str, Enum):
    ON_TABLE = "ON_TABLE"
    HAND_CONTACT = "HAND_CONTACT"
    BEING_HELD = "BEING_HELD"
    CARRIED = "CARRIED"
    RELEASED = "RELEASED"
    PLACED = "PLACED"


@dataclass
class TrackedState:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[int, int]        # cx, cy
    confidence: float
    raw_label: str = ""
    source: str = "detector"
    identity: str = ""
    raw_class: str = ""
    semantic_identity: str = ""
    velocity: tuple[float, float] = (0.0, 0.0)  # px/sec (vx, vy)
    state: str = "ON_TABLE"
    first_seen: float = 0.0
    last_seen: float = 0.0
    hits: int = 1
    time_since_update: int = 0
    hand_contact: bool = False
    held_duration: float = 0.0
    initial_position: tuple[int, int] = (0, 0)
    history: List[tuple[int, int]] = field(default_factory=list)
    velocity_history: List[tuple[float, float]] = field(default_factory=list)
    hand_contact_history: List[bool] = field(default_factory=list)
    physical_state_history: List[str] = field(default_factory=list)
    pickup_time: Optional[float] = None
    release_time: Optional[float] = None
    placement_location: str = ""

    def __post_init__(self):
        if not self.raw_class:
            self.raw_class = self.raw_label or self.class_name
        if not self.semantic_identity:
            self.semantic_identity = self.identity or self.class_name
        if not self.identity:
            self.identity = self.semantic_identity
        if not self.raw_label:
            self.raw_label = self.raw_class
        if self.initial_position == (0, 0) and self.centroid != (0, 0):
            self.initial_position = self.centroid
        if not self.history and self.centroid != (0, 0):
            self.history.append(self.centroid)
        if not self.velocity_history:
            self.velocity_history.append(self.velocity)
        if not self.hand_contact_history:
            self.hand_contact_history.append(self.hand_contact)
        if not self.physical_state_history:
            state_str = self.state.value if hasattr(self.state, "value") else str(self.state)
            self.physical_state_history.append(state_str)

    @property
    def center(self) -> tuple[int, int]:
        return self.centroid

    @property
    def physical_state(self) -> str:
        return self.state.value if hasattr(self.state, "value") else str(self.state)

    @physical_state.setter
    def physical_state(self, val: Any) -> None:
        self.state = val

    @property
    def current_position(self) -> tuple[int, int]:
        return self.centroid

    @property
    def position_history(self) -> List[tuple[int, int]]:
        return self.history


def compute_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """Computes IoU between two (x, y, w, h) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea


class MultiObjectTracker:
    """
    Multi-object tracker using IoU + spatial centroid matching with velocity estimation.
    """

    def __init__(
        self,
        max_age: int = 15,
        min_hits: int = 1,
        iou_threshold: float = 0.25,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._next_id: int = 1
        self._tracks: Dict[int, TrackedState] = {}
        self._primary_operator_id: Optional[int] = None

    @property
    def tracks(self) -> Dict[int, TrackedState]:
        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._primary_operator_id = None

    def update(
        self,
        detections: List[Any],
        timestamp: float = 0.0,
    ) -> List[Any]:
        """
        Updates tracks with new detections from current frame.
        Mutates or attaches track_id and velocity to each detection.
        Returns the updated list of detections including track metadata.
        """
        if timestamp <= 0.0:
            timestamp = time.time()

        # Increment age for existing tracks
        for track in self._tracks.values():
            track.time_since_update += 1

        matched_tracks = set()
        matched_detections = set()

        # Match detections to existing tracks by IoU and class compatibility
        if self._tracks and detections:
            track_ids = list(self._tracks.keys())
            
            # Compute cost matrix (1.0 - IoU)
            cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for t_idx, t_id in enumerate(track_ids):
                track = self._tracks[t_id]
                for d_idx, det in enumerate(detections):
                    # Class matching preference
                    class_match = (det.class_name == track.class_name)
                    iou = compute_iou(track.bbox, det.bbox)
                    
                    if not class_match:
                        # Soft penalty if raw labels match, hard penalty if totally different
                        cost_matrix[t_idx, d_idx] = 1.0 - (iou * 0.4)
                    else:
                        cost_matrix[t_idx, d_idx] = 1.0 - iou

            # Greedy assignment based on best matches
            for _ in range(min(len(track_ids), len(detections))):
                min_val = float(cost_matrix.min())
                if min_val > (1.0 - self.iou_threshold):
                    break
                t_idx, d_idx = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
                t_id = track_ids[t_idx]

                if t_id not in matched_tracks and d_idx not in matched_detections:
                    track = self._tracks[t_id]
                    det = detections[d_idx]

                    # Calculate velocity (px/sec)
                    dt = max(0.01, timestamp - track.last_seen)
                    vx = (det.centroid[0] - track.centroid[0]) / dt
                    vy = (det.centroid[1] - track.centroid[1]) / dt

                    # Smooth velocity with EMA
                    alpha = 0.4
                    smooth_vx = alpha * vx + (1 - alpha) * track.velocity[0]
                    smooth_vy = alpha * vy + (1 - alpha) * track.velocity[1]

                    # Smooth bounding box
                    s_alpha = 0.75
                    new_x = int(s_alpha * det.bbox[0] + (1 - s_alpha) * track.bbox[0])
                    new_y = int(s_alpha * det.bbox[1] + (1 - s_alpha) * track.bbox[1])
                    new_w = int(s_alpha * det.bbox[2] + (1 - s_alpha) * track.bbox[2])
                    new_h = int(s_alpha * det.bbox[3] + (1 - s_alpha) * track.bbox[3])

                    track.bbox = (new_x, new_y, new_w, new_h)
                    track.centroid = (new_x + new_w // 2, new_y + new_h // 2)
                    track.confidence = det.confidence
                    track.last_seen = timestamp
                    track.hits += 1
                    track.time_since_update = 0
                    track.velocity = (smooth_vx, smooth_vy)
                    track.history.append(track.centroid)
                    if len(track.history) > 30:
                        track.history.pop(0)
                    track.velocity_history.append(track.velocity)
                    if len(track.velocity_history) > 30:
                        track.velocity_history.pop(0)
                    track.hand_contact_history.append(track.hand_contact)
                    if len(track.hand_contact_history) > 30:
                        track.hand_contact_history.pop(0)
                    track.physical_state_history.append(track.physical_state)
                    if len(track.physical_state_history) > 30:
                        track.physical_state_history.pop(0)

                    # Update detection object
                    det.bbox = track.bbox
                    det.centroid = track.centroid
                    det.track_id = t_id
                    det.velocity = track.velocity
                    det.state = track.state
                    det.identity = track.identity or det.class_name

                    matched_tracks.add(t_id)
                    matched_detections.add(d_idx)

                # Set used row/col to infinity
                cost_matrix[t_idx, :] = 10.0
                cost_matrix[:, d_idx] = 10.0

        # Create new tracks for unmatched detections with duplicate rejection safeguard
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                # SAFEGUARD: Do not spawn a new track if another active track of the same canonical class overlaps
                overlapping_existing = any(
                    (t.class_name == det.class_name or {t.class_name, det.class_name} <= {"BLUE_BOX", "MAIN_BOX"})
                    and compute_iou(t.bbox, det.bbox) > 0.35
                    and t.time_since_update <= 5
                    for t in self._tracks.values()
                )
                if overlapping_existing:
                    continue

                new_id = self._next_id
                self._next_id += 1
                new_track = TrackedState(
                    track_id=new_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    centroid=det.centroid,
                    confidence=det.confidence,
                    raw_label=getattr(det, "raw_label", ""),
                    source=getattr(det, "source", "detector"),
                    identity=getattr(det, "semantic_identity", "") or det.class_name,
                    raw_class=getattr(det, "raw_label", "") or det.class_name,
                    semantic_identity=getattr(det, "semantic_identity", "") or det.class_name,
                    velocity=(0.0, 0.0),
                    state="ON_TABLE",
                    first_seen=timestamp,
                    last_seen=timestamp,
                    hits=1,
                    time_since_update=0,
                    initial_position=det.centroid,
                    history=[det.centroid],
                )
                self._tracks[new_id] = new_track
                det.track_id = new_id
                det.velocity = (0.0, 0.0)
                det.state = "ON_TABLE"
                det.identity = new_track.identity

        # Remove dead tracks that exceeded max_age
        dead_ids = [t_id for t_id, t in self._tracks.items() if t.time_since_update > self.max_age]
        for t_id in dead_ids:
            del self._tracks[t_id]

        # Deduplicate active tracks of the same class that heavily overlap (IoU > 0.40)
        track_ids_list = list(self._tracks.keys())
        for i in range(len(track_ids_list)):
            tid1 = track_ids_list[i]
            if tid1 not in self._tracks:
                continue
            trk1 = self._tracks[tid1]
            for j in range(i + 1, len(track_ids_list)):
                tid2 = track_ids_list[j]
                if tid2 not in self._tracks:
                    continue
                trk2 = self._tracks[tid2]
                same_cls = (trk1.class_name == trk2.class_name) or ({trk1.class_name, trk2.class_name} <= {"BLUE_BOX", "MAIN_BOX"})
                if same_cls and compute_iou(trk1.bbox, trk2.bbox) > 0.40:
                    if trk1.hits >= trk2.hits:
                        del self._tracks[tid2]
                    else:
                        del self._tracks[tid1]
                        break

        # Determine primary operator if persons are present
        persons = [t for t in self._tracks.values() if t.class_name == "PERSON" and t.time_since_update == 0]
        if persons:
            primary = max(persons, key=lambda p: p.bbox[2] * p.bbox[3])
            self._primary_operator_id = primary.track_id
        else:
            self._primary_operator_id = None

        # Return only detections with valid, active tracks and no duplicate track IDs
        seen_track_ids = set()
        canonical_detections = []
        for det in detections:
            tid = getattr(det, "track_id", -1)
            if tid > 0 and tid in self._tracks and tid not in seen_track_ids:
                seen_track_ids.add(tid)
                canonical_detections.append(det)

        return canonical_detections

    def update_interaction_states(self, interactions: list[Any], dt: float = 0.033) -> None:
        """Update physical tracking states (ON_TABLE, HAND_CONTACT, BEING_HELD, CARRIED, RELEASED, PLACED)."""
        now = time.time()
        active_objects = set()
        for inter in interactions:
            obj_cls = getattr(inter, "object_class", "")
            is_holding = getattr(inter, "is_holding", False)
            state_val = getattr(inter, "state", None)
            state_name = getattr(state_val, "name", str(state_val))

            for track in self._tracks.values():
                if track.class_name == obj_cls or track.identity == obj_cls:
                    active_objects.add(track.track_id)
                    speed = (track.velocity[0] ** 2 + track.velocity[1] ** 2) ** 0.5
                    if is_holding or state_name in ("HOLDING", "GRASPING"):
                        track.hand_contact = True
                        track.held_duration += dt
                        track.state = ObjectPhysicalState.CARRIED if speed > 30.0 else ObjectPhysicalState.BEING_HELD
                        if track.pickup_time is None:
                            track.pickup_time = now
                    elif state_name in ("CONTACT", "NEAR_OBJECT"):
                        track.hand_contact = True
                        track.state = ObjectPhysicalState.HAND_CONTACT
                    elif state_name in ("RELEASING", "RELEASED"):
                        track.hand_contact = False
                        track.held_duration = 0.0
                        track.state = ObjectPhysicalState.PLACED
                        track.release_time = now
                        track.placement_location = "LOCATION_A" if track.centroid[0] < 640 else "LOCATION_B"

        # Reset objects not currently interacting
        for track in self._tracks.values():
            if track.track_id not in active_objects:
                track.hand_contact = False
                track.held_duration = 0.0
                track.state = ObjectPhysicalState.ON_TABLE

    def get_track(self, track_id: int) -> Optional[TrackedState]:
        return self._tracks.get(track_id)

    def get_active_tracks(self, max_staleness: int = 3) -> List[TrackedState]:
        return [t for t in self._tracks.values() if t.time_since_update <= max_staleness]

