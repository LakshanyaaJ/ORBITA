"""
ORBITA Feature Fusion
======================
Combines perception outputs into a fixed-size temporal feature vector
that serves as input to the GRU/LSTM action recognition model.

Feature vector composition (64 dimensions total):

  [0:34]   Pose keypoints (17 joints × 2 normalized coords)
  [34:38]  Left hand (x, y, vx, vy) — torso-normalized
  [38:42]  Right hand (x, y, vx, vy)
  [42:46]  Left-hand interaction features (state, dist, vel, duration)
  [46:50]  Right-hand interaction features
  [50:60]  Object one-hot presence (10 classes)
  [60:62]  Primary interaction object centroid (x, y) normalized
  [62:64]  Frame meta (grasp_confidence_left, grasp_confidence_right)

The window manager maintains a rolling FIFO of 30 frames,
returning shape (30, 64) as input to the temporal model.

DESIGN NOTE:
  Object identity is explicitly encoded (one-hot) so the temporal model
  does not need to infer object class from motion alone.
  This is architecturally important: the LSTM should learn WHEN and HOW
  not WHAT (that is the detector's job).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

from core_ai.perception.object_detector import DetectedObject
from core_ai.perception.pose_estimator import PoseResult
from core_ai.perception.hand_tracker import HandState
from core_ai.interaction.hand_object import HandObjectInteraction, InteractionState

logger = logging.getLogger(__name__)

FEATURE_DIM = 64

OBJECT_CLASSES = [
    "PERSON", "MAIN_BOX", "RED_BOX", "YELLOW_BOX",
    "SAMPLE", "TOOL", "CHAMBER", "UNKNOWN1", "UNKNOWN2", "UNKNOWN3",
]
OBJECT_CLASS_IDX = {c: i for i, c in enumerate(OBJECT_CLASSES)}


def build_feature_vector(
    pose: Optional[PoseResult],
    left_hand: Optional[HandState],
    right_hand: Optional[HandState],
    objects: list[DetectedObject],
    interactions: list[HandObjectInteraction],
    frame_shape: tuple[int, int],
) -> np.ndarray:
    """
    Build a single 64-dim feature vector from one frame's perception outputs.

    Args:
        pose: Pose estimation result (or None if no person detected).
        left_hand: Left hand state.
        right_hand: Right hand state.
        objects: Detected objects in frame.
        interactions: Current hand-object interactions.
        frame_shape: (height, width) of the frame for normalizing.

    Returns:
        np.ndarray of shape (64,)
    """
    fv = np.zeros(FEATURE_DIM, dtype=np.float32)
    fh, fw = frame_shape

    # --- [0:34] Pose keypoints (torso-normalized, already in [-1.5, 1.5]) ---
    if pose is not None:
        kp_norm = pose.keypoints_norm.flatten()[:34]  # 17 joints × 2
        fv[:len(kp_norm)] = kp_norm

    # --- [34:42] Hand positions and velocities (frame-normalized) ---
    for i, hand in enumerate([left_hand, right_hand]):
        base = 34 + i * 4
        if hand is not None and hand.is_visible:
            fv[base]     = hand.position[0] / (fw + 1e-6)
            fv[base + 1] = hand.position[1] / (fh + 1e-6)
            fv[base + 2] = np.clip(hand.velocity[0] / 20.0, -1.0, 1.0)
            fv[base + 3] = np.clip(hand.velocity[1] / 20.0, -1.0, 1.0)

    # --- [42:50] Interaction features per hand ---
    for i, side in enumerate(["left", "right"]):
        base = 42 + i * 4
        hand_interactions = [x for x in interactions if x.hand_side == side]
        if hand_interactions:
            # Use highest-state interaction
            best = max(hand_interactions, key=lambda x: x.state)
            fv[base]     = float(best.state) / 6.0             # Normalized state
            fv[base + 1] = np.clip(best.distance_px / 200.0, 0.0, 1.0)
            fv[base + 2] = np.clip(best.approach_velocity / 20.0, -1.0, 1.0)
            fv[base + 3] = np.clip(best.duration_frames / 30.0, 0.0, 1.0)

    # --- [50:60] Object one-hot presence ---
    for obj in objects:
        idx = OBJECT_CLASS_IDX.get(obj.class_name, len(OBJECT_CLASSES) - 1)
        if idx < 10:
            fv[50 + idx] = max(fv[50 + idx], obj.confidence)

    # --- [60:62] Primary interacting object centroid ---
    # Find the most-active interaction object
    if interactions:
        primary = max(interactions, key=lambda x: x.state * 10 + x.confidence)
        # Find the object
        for obj in objects:
            if obj.class_name == primary.object_class:
                fv[60] = obj.centroid[0] / (fw + 1e-6)
                fv[61] = obj.centroid[1] / (fh + 1e-6)
                break

    # --- [62:64] Grasp confidence per hand ---
    if left_hand is not None:
        fv[62] = left_hand.grasp_confidence
    if right_hand is not None:
        fv[63] = right_hand.grasp_confidence

    return fv


class TemporalFeatureWindow:
    """
    Maintains a sliding FIFO window of feature vectors.
    Returns the window as a (window_size, feature_dim) array
    ready for input to the GRU.
    """

    def __init__(self, window_size: int = 30, feature_dim: int = FEATURE_DIM):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        # Pad with zeros on init
        for _ in range(window_size):
            self._buffer.append(np.zeros(feature_dim, dtype=np.float32))

    def push(self, fv: np.ndarray) -> None:
        """Add a new feature vector to the window."""
        assert fv.shape == (self.feature_dim,), f"Expected ({self.feature_dim},), got {fv.shape}"
        self._buffer.append(fv.copy())

    def get_window(self) -> np.ndarray:
        """Return current window as (window_size, feature_dim)."""
        return np.stack(list(self._buffer), axis=0)

    def is_full(self) -> bool:
        return len(self._buffer) == self.window_size

    def reset(self) -> None:
        self._buffer.clear()
        for _ in range(self.window_size):
            self._buffer.append(np.zeros(self.feature_dim, dtype=np.float32))
