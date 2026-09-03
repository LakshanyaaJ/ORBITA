"""
ORBITA Hand Tracker
===================
Tracks left and right hands (position, velocity, trajectory) per frame.

Primary source: wrist keypoints from PoseEstimator (always available).
Optional enhancement: MediaPipe Hands for finger-level landmarks.

Output: HandState per hand per frame, including:
  - position (pixel)
  - velocity (pixel/frame)
  - trajectory history
  - grasping confidence (heuristic based on wrist velocity + finger curl)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from orbita.perception.pose_estimator import PoseResult

logger = logging.getLogger(__name__)

# History length for velocity computation
HISTORY_LEN = 10


@dataclass
class HandState:
    """State of one hand at a single frame."""
    side: str                            # "left" | "right"
    position: np.ndarray                 # (x, y) in pixels
    velocity: np.ndarray                 # (vx, vy) pixels/frame
    speed: float                         # scalar speed
    is_visible: bool
    grasp_confidence: float              # 0.0 – 1.0 heuristic
    finger_landmarks: Optional[np.ndarray] = None  # (21, 2) if MediaPipe available
    timestamp: float = 0.0


class HandTracker:
    """
    Extracts hand state from pose keypoints, optionally enhanced with
    MediaPipe Hands for finger-level detail.

    Usage:
        tracker = HandTracker(config)
        left, right = tracker.track(pose_result, frame, timestamp)
    """

    def __init__(self, config):
        self.config = config
        self._mp_hands = None
        self._left_history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
        self._right_history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)

        if config.backend == "mediapipe":
            self._try_load_mediapipe()

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def track(
        self,
        pose: Optional[PoseResult],
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> tuple[HandState, HandState]:
        """
        Returns (left_hand, right_hand).
        If no pose is available, returns invisible HandState objects.
        """
        if pose is None:
            invisible = self._invisible_hand
            return invisible("left", timestamp), invisible("right", timestamp)

        left_px = pose.left_wrist.copy()
        right_px = pose.right_wrist.copy()

        # Finger landmarks from MediaPipe (optional enhancement)
        left_fingers: Optional[np.ndarray] = None
        right_fingers: Optional[np.ndarray] = None
        if self._mp_hands is not None:
            left_fingers, right_fingers = self._mediapipe_fingers(frame)

        # Update histories
        self._left_history.append(left_px)
        self._right_history.append(right_px)

        left_vel = self._velocity(self._left_history)
        right_vel = self._velocity(self._right_history)

        left_speed = float(np.linalg.norm(left_vel))
        right_speed = float(np.linalg.norm(right_vel))

        left_state = HandState(
            side="left",
            position=left_px,
            velocity=left_vel,
            speed=left_speed,
            is_visible=bool(pose.keypoints_conf[9] > 0.25),
            grasp_confidence=self._grasp_heuristic(left_speed, left_fingers),
            finger_landmarks=left_fingers,
            timestamp=timestamp,
        )
        right_state = HandState(
            side="right",
            position=right_px,
            velocity=right_vel,
            speed=right_speed,
            is_visible=bool(pose.keypoints_conf[10] > 0.25),
            grasp_confidence=self._grasp_heuristic(right_speed, right_fingers),
            finger_landmarks=right_fingers,
            timestamp=timestamp,
        )
        return left_state, right_state

    def reset(self) -> None:
        """Clear velocity history (call on experiment reset)."""
        self._left_history.clear()
        self._right_history.clear()

    def draw(
        self,
        frame: np.ndarray,
        left: HandState,
        right: HandState,
    ) -> np.ndarray:
        """Draw hand indicators on frame."""
        import cv2
        vis = frame.copy()
        for hand, colour in [(left, (0, 200, 255)), (right, (255, 180, 0))]:
            if hand.is_visible:
                cx, cy = int(hand.position[0]), int(hand.position[1])
                cv2.circle(vis, (cx, cy), 8, colour, -1)
                cv2.circle(vis, (cx, cy), 12, colour, 2)
                label = f"{hand.side[0].upper()} hand"
                cv2.putText(vis, label, (cx + 14, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
                # Velocity arrow
                vx, vy = hand.velocity
                end = (int(cx + vx * 3), int(cy + vy * 3))
                cv2.arrowedLine(vis, (cx, cy), end, colour, 1, tipLength=0.3)
        return vis

    # ----------------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _velocity(history: deque) -> np.ndarray:
        """Compute smoothed velocity from position history."""
        if len(history) < 2:
            return np.zeros(2, dtype=np.float32)
        # Use last 3 frames for velocity smoothing
        n = min(3, len(history))
        recent = list(history)[-n:]
        deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return np.mean(deltas, axis=0).astype(np.float32)

    @staticmethod
    def _grasp_heuristic(
        speed: float,
        finger_landmarks: Optional[np.ndarray],
    ) -> float:
        """
        Estimate grasping probability from wrist speed and finger geometry.
        Low speed + fingers curled → high grasp confidence.
        This is a heuristic; real grasp detection requires learned models.
        """
        # Speed component: slow wrist → more likely holding something
        speed_score = float(np.clip(1.0 - speed / 20.0, 0.0, 1.0))

        if finger_landmarks is not None and len(finger_landmarks) == 21:
            # Finger curl: compare tip distance to palm vs straight-finger distance
            palm = finger_landmarks[0]
            tips = finger_landmarks[[4, 8, 12, 16, 20]]
            mids = finger_landmarks[[2, 6, 10, 14, 18]]
            tip_dists = np.linalg.norm(tips - palm, axis=1)
            mid_dists = np.linalg.norm(mids - palm, axis=1)
            curl_ratio = np.mean(tip_dists / (mid_dists + 1e-6))
            curl_score = float(np.clip(1.5 - curl_ratio, 0.0, 1.0))
            return 0.4 * speed_score + 0.6 * curl_score

        return speed_score

    @staticmethod
    def _invisible_hand(side: str, timestamp: float) -> HandState:
        return HandState(
            side=side,
            position=np.zeros(2, dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            speed=0.0,
            is_visible=False,
            grasp_confidence=0.0,
            timestamp=timestamp,
        )

    def _mediapipe_fingers(
        self, frame: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        import cv2
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._mp_hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None, None
            h, w = frame.shape[:2]
            left_lm: Optional[np.ndarray] = None
            right_lm: Optional[np.ndarray] = None
            for hand_info, hand_lm in zip(
                res.multi_handedness, res.multi_hand_landmarks
            ):
                label = hand_info.classification[0].label.lower()
                pts = np.array(
                    [[lm.x * w, lm.y * h] for lm in hand_lm.landmark],
                    dtype=np.float32,
                )
                if label == "left":
                    left_lm = pts
                else:
                    right_lm = pts
            return left_lm, right_lm
        except Exception as exc:
            logger.warning("MediaPipe hands error: %s", exc)
            return None, None

    def _try_load_mediapipe(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
            self._mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config.mediapipe_max_hands,
                min_detection_confidence=self.config.mediapipe_detection_confidence,
                min_tracking_confidence=self.config.mediapipe_tracking_confidence,
            )
            logger.info("MediaPipe Hands loaded.")
        except ImportError:
            logger.warning("mediapipe not installed — wrist-only hand tracking.")
        except Exception as exc:
            logger.warning("MediaPipe Hands init failed: %s", exc)
