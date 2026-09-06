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


@dataclass
class TrackedState:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[int, int]        # cx, cy
    confidence: float
    raw_label: str
    source: str
    velocity: tuple[float, float] = (0.0, 0.0)  # px/sec (vx, vy)
    first_seen: float = 0.0
    last_seen: float = 0.0
    hits: int = 1
    time_since_update: int = 0
    history: List[tuple[int, int]] = field(default_factory=list)


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

                    # Update detection object
                    det.bbox = track.bbox
                    det.centroid = track.centroid
                    if hasattr(det, "track_id"):
                        det.track_id = t_id
                    if hasattr(det, "velocity"):
                        det.velocity = track.velocity

                    matched_tracks.add(t_id)
                    matched_detections.add(d_idx)

                # Set used row/col to infinity
                cost_matrix[t_idx, :] = 10.0
                cost_matrix[:, d_idx] = 10.0

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
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
                    velocity=(0.0, 0.0),
                    first_seen=timestamp,
                    last_seen=timestamp,
                    hits=1,
                    time_since_update=0,
                    history=[det.centroid],
                )
                self._tracks[new_id] = new_track
                if hasattr(det, "track_id"):
                    det.track_id = new_id
                if hasattr(det, "velocity"):
                    det.velocity = (0.0, 0.0)

        # Remove dead tracks that exceeded max_age
        dead_ids = [t_id for t_id, t in self._tracks.items() if t.time_since_update > self.max_age]
        for t_id in dead_ids:
            del self._tracks[t_id]

        # Determine primary operator if persons are present
        persons = [t for t in self._tracks.values() if t.class_name == "PERSON" and t.time_since_update == 0]
        if persons:
            # Primary operator: person closest to experiment workspace (or largest bounding box area)
            primary = max(persons, key=lambda p: p.bbox[2] * p.bbox[3])
            self._primary_operator_id = primary.track_id
        else:
            self._primary_operator_id = None

        return detections

    def get_track(self, track_id: int) -> Optional[TrackedState]:
        return self._tracks.get(track_id)

    def get_active_tracks(self) -> List[TrackedState]:
        return [t for t in self._tracks.values() if t.time_since_update == 0]
