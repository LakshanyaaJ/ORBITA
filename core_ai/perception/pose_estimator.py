"""
ORBITA Pose Estimator
=====================
Extracts 17 body keypoints from each frame.

Backends:
  "yolo"      → YOLOv8n-pose (preferred, offline)
  "mediapipe" → MediaPipe Pose (fallback)
  "mock"      → Returns zeroed keypoints (unit tests / no model)

Keypoint indices follow the COCO17 convention:
  0  nose
  1  left_eye  2  right_eye
  3  left_ear  4  right_ear
  5  left_shoulder  6  right_shoulder
  7  left_elbow  8  right_elbow
  9  left_wrist  10 right_wrist
  11 left_hip  12 right_hip
  13 left_knee  14 right_knee
  15 left_ankle  16 right_ankle

DESIGN NOTE:
  Coordinates are returned in two forms:
    raw_px: pixel (x, y) as detected
    normalized: torso-relative, scale-invariant (x, y in [-1, 1] roughly)

  Torso normalization uses the mid-shoulder to mid-hip distance as the
  reference unit. This makes the feature representation robust to camera
  distance and subject body size.

SCIENTIFIC HONESTY:
  This is 2D pose estimation. It does NOT provide true 3D body orientation.
  Replacing this module with a 3D HMR system is explicitly supported by
  the architecture — the module interface is intentionally minimal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 17

# COCO keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Indices of "important" keypoints for downstream features
IDX_NOSE = 0
IDX_L_SHOULDER = 5
IDX_R_SHOULDER = 6
IDX_L_ELBOW = 7
IDX_R_ELBOW = 8
IDX_L_WRIST = 9
IDX_R_WRIST = 10
IDX_L_HIP = 11
IDX_R_HIP = 12


@dataclass
class PoseResult:
    """Pose estimation result for one person in one frame."""
    keypoints_px: np.ndarray           # Shape (17, 2) — pixel coords
    keypoints_conf: np.ndarray         # Shape (17,)  — per-keypoint confidence
    keypoints_norm: np.ndarray         # Shape (17, 2) — torso-normalized
    bbox: tuple[int, int, int, int]    # (x, y, w, h) person bounding box
    overall_confidence: float
    timestamp: float = 0.0

    @property
    def left_wrist(self) -> np.ndarray:
        return self.keypoints_px[IDX_L_WRIST]

    @property
    def right_wrist(self) -> np.ndarray:
        return self.keypoints_px[IDX_R_WRIST]

    @property
    def mid_shoulder(self) -> np.ndarray:
        return (self.keypoints_px[IDX_L_SHOULDER] + self.keypoints_px[IDX_R_SHOULDER]) / 2

    @property
    def mid_hip(self) -> np.ndarray:
        return (self.keypoints_px[IDX_L_HIP] + self.keypoints_px[IDX_R_HIP]) / 2

    @property
    def torso_length(self) -> float:
        return float(np.linalg.norm(self.mid_shoulder - self.mid_hip) + 1e-6)


class PoseEstimator:
    """
    Estimates human body pose from a video frame.

    Usage:
        estimator = PoseEstimator(config)
        results = estimator.estimate(frame, timestamp=time.time())
        # results is a list[PoseResult] (one per person detected)
    """

    def __init__(self, config):
        self.config = config
        self.backend = config.backend
        self._model = None
        self._mp_pose = None

        if self.backend == "yolo":
            self._try_load_yolo(config.yolo_model_path)
            if self._model is None:
                logger.warning("YOLOv8-pose unavailable — falling back to mediapipe.")
                self.backend = "mediapipe"

        if self.backend == "mediapipe":
            self._try_load_mediapipe()
            if self._mp_pose is None:
                logger.warning("MediaPipe pose unavailable — using mock backend.")
                self.backend = "mock"

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def estimate(
        self, frame: np.ndarray, timestamp: float = 0.0
    ) -> list[PoseResult]:
        if self.backend == "yolo":
            return self._estimate_yolo(frame, timestamp)
        if self.backend == "mediapipe":
            return self._estimate_mediapipe(frame, timestamp)
        return self._estimate_mock(frame, timestamp)

    def draw(self, frame: np.ndarray, results: list[PoseResult]) -> np.ndarray:
        """Draw skeleton on frame."""
        vis = frame.copy()
        # COCO skeleton connections
        skeleton = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6),
        ]
        for pose in results:
            kp = pose.keypoints_px
            conf = pose.keypoints_conf
            # Draw joints
            for i, (x, y) in enumerate(kp):
                if conf[i] > 0.3:
                    cv_xy = (int(x), int(y))
                    color = (0, 255, 150) if i in (9, 10) else (100, 220, 255)
                    import cv2
                    cv2.circle(vis, cv_xy, 4, color, -1)
            # Draw bones
            for i, j in skeleton:
                if conf[i] > 0.3 and conf[j] > 0.3:
                    import cv2
                    p1 = (int(kp[i][0]), int(kp[i][1]))
                    p2 = (int(kp[j][0]), int(kp[j][1]))
                    cv2.line(vis, p1, p2, (80, 180, 255), 1, cv2.LINE_AA)
        return vis

    # ----------------------------------------------------------------------- #
    # YOLO backend
    # ----------------------------------------------------------------------- #
    def _estimate_yolo(
        self, frame: np.ndarray, timestamp: float
    ) -> list[PoseResult]:
        try:
            results = self._model.predict(frame, verbose=False, stream=False)
            poses: list[PoseResult] = []
            for r in results:
                if r.keypoints is None:
                    continue
                for person_idx in range(len(r.keypoints.xy)):
                    kp_xy = r.keypoints.xy[person_idx].cpu().numpy()    # (17, 2)
                    kp_conf = r.keypoints.conf[person_idx].cpu().numpy()  # (17,)

                    # Person bbox
                    if r.boxes is not None and person_idx < len(r.boxes):
                        b = r.boxes[person_idx]
                        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        overall_conf = float(b.conf[0])
                    else:
                        h, w = frame.shape[:2]
                        bbox = (0, 0, w, h)
                        overall_conf = float(kp_conf.mean())

                    kp_norm = self._normalize_keypoints(kp_xy, kp_conf)
                    poses.append(PoseResult(
                        keypoints_px=kp_xy,
                        keypoints_conf=kp_conf,
                        keypoints_norm=kp_norm,
                        bbox=bbox,
                        overall_confidence=overall_conf,
                        timestamp=timestamp,
                    ))
            return poses
        except Exception as exc:
            logger.warning("YOLO pose inference error: %s", exc)
            return []

    # ----------------------------------------------------------------------- #
    # MediaPipe backend
    # ----------------------------------------------------------------------- #
    def _estimate_mediapipe(
        self, frame: np.ndarray, timestamp: float
    ) -> list[PoseResult]:
        import cv2
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._mp_pose.process(rgb)
            if res.pose_landmarks is None:
                return []

            h, w = frame.shape[:2]
            lm = res.pose_landmarks.landmark

            # MediaPipe has 33 landmarks; we map to the 17 COCO subset
            mp_to_coco = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            kp_xy = np.zeros((17, 2), dtype=np.float32)
            kp_conf = np.zeros(17, dtype=np.float32)

            for coco_idx, mp_idx in enumerate(mp_to_coco):
                if mp_idx < len(lm):
                    kp_xy[coco_idx] = [lm[mp_idx].x * w, lm[mp_idx].y * h]
                    kp_conf[coco_idx] = lm[mp_idx].visibility

            bbox = (0, 0, w, h)
            kp_norm = self._normalize_keypoints(kp_xy, kp_conf)
            return [PoseResult(
                keypoints_px=kp_xy,
                keypoints_conf=kp_conf,
                keypoints_norm=kp_norm,
                bbox=bbox,
                overall_confidence=float(kp_conf.mean()),
                timestamp=timestamp,
            )]
        except Exception as exc:
            logger.warning("MediaPipe pose error: %s", exc)
            return []

    # ----------------------------------------------------------------------- #
    # Mock backend (unit tests / no model)
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _estimate_mock(
        frame: np.ndarray, timestamp: float
    ) -> list[PoseResult]:
        h, w = frame.shape[:2]
        # Place skeleton at a plausible center-screen location
        cx, cy = w // 2, h // 2
        kp_xy = np.array([
            [cx, cy - 80],          # nose
            [cx - 10, cy - 90],     # left_eye
            [cx + 10, cy - 90],     # right_eye
            [cx - 20, cy - 85],     # left_ear
            [cx + 20, cy - 85],     # right_ear
            [cx - 40, cy - 50],     # left_shoulder
            [cx + 40, cy - 50],     # right_shoulder
            [cx - 55, cy - 10],     # left_elbow
            [cx + 55, cy - 10],     # right_elbow
            [cx - 65, cy + 30],     # left_wrist
            [cx + 65, cy + 30],     # right_wrist
            [cx - 30, cy + 50],     # left_hip
            [cx + 30, cy + 50],     # right_hip
            [cx - 30, cy + 110],    # left_knee
            [cx + 30, cy + 110],    # right_knee
            [cx - 30, cy + 170],    # left_ankle
            [cx + 30, cy + 170],    # right_ankle
        ], dtype=np.float32)
        kp_conf = np.full(17, 0.80, dtype=np.float32)
        kp_norm = PoseEstimator._normalize_keypoints(kp_xy, kp_conf)
        return [PoseResult(
            keypoints_px=kp_xy,
            keypoints_conf=kp_conf,
            keypoints_norm=kp_norm,
            bbox=(cx - 80, cy - 100, 160, 280),
            overall_confidence=0.80,
            timestamp=timestamp,
        )]

    # ----------------------------------------------------------------------- #
    # Torso normalization
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _normalize_keypoints(
        kp_xy: np.ndarray, kp_conf: np.ndarray
    ) -> np.ndarray:
        """
        Normalize keypoints relative to torso center and torso length.
        Output range: approximately [-1.5, 1.5] for typical body poses.
        Keypoints with confidence < 0.2 are zeroed.
        """
        # Torso reference
        ls = kp_xy[IDX_L_SHOULDER]
        rs = kp_xy[IDX_R_SHOULDER]
        lh = kp_xy[IDX_L_HIP]
        rh = kp_xy[IDX_R_HIP]

        mid_shoulder = (ls + rs) / 2
        mid_hip = (lh + rh) / 2
        torso_center = (mid_shoulder + mid_hip) / 2
        torso_length = float(np.linalg.norm(mid_shoulder - mid_hip) + 1e-6)

        normalized = (kp_xy - torso_center) / torso_length
        # Zero out low-confidence keypoints
        normalized[kp_conf < 0.2] = 0.0
        return normalized.astype(np.float32)

    # ----------------------------------------------------------------------- #
    # Model loading helpers
    # ----------------------------------------------------------------------- #
    def _try_load_yolo(self, model_path: str) -> None:
        import os
        if not os.path.exists(model_path):
            logger.info("YOLOv8-pose model not found at %s", model_path)
            return
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(model_path)
            logger.info("YOLOv8-pose loaded from %s", model_path)
        except ImportError:
            logger.warning("ultralytics not installed.")
        except Exception as exc:
            logger.warning("Failed to load YOLOv8-pose: %s", exc)

    def _try_load_mediapipe(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=self.config.confidence_threshold,
                min_tracking_confidence=self.config.confidence_threshold,
            )
            logger.info("MediaPipe pose loaded.")
        except ImportError:
            logger.warning("mediapipe not installed.")
        except Exception as exc:
            logger.warning("Failed to load MediaPipe pose: %s", exc)
