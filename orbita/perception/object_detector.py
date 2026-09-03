"""
ORBITA Object Detector
======================
Hybrid detection pipeline:
  1. Colour-chroma HSV segmentation (fast, zero-model, always available)
  2. Optional YOLOv8 semantic overlay (when model file present)

Returns DetectedObject dataclass per object per frame.

DESIGN NOTE:
  The chroma detector is the reliable baseline for demo/prototype.
  YOLO adds class confidence when a trained model is available.
  The two can be fused by class label matching.

SCIENTIFIC HONESTY:
  Colour thresholds are tuned for controlled lab/demo conditions.
  Real spaceflight illumination will require dataset-driven retraining
  of the YOLO model and potential illumination normalization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x, y, w, h  (top-left corner + size)
    centroid: tuple[int, int]
    timestamp: float = 0.0
    source: str = "chroma"            # "chroma" | "yolo" | "fused"

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def x2(self) -> int:
        return self.bbox[0] + self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[1] + self.bbox[3]


# --------------------------------------------------------------------------- #
# Colour ranges for HSV chroma detection
# Keys must match class names in config.
# Stored as (lower, upper) numpy arrays for cv2.inRange.
# --------------------------------------------------------------------------- #
DEFAULT_HSV_RANGES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "RED_BOX": (
        np.array([0,   120,  80]),
        np.array([10,  255, 255]),
    ),
    "RED_BOX_HIGH": (           # Red wraps around H=180
        np.array([170, 120,  80]),
        np.array([180, 255, 255]),
    ),
    "YELLOW_BOX": (
        np.array([20,  120,  80]),
        np.array([35,  255, 255]),
    ),
    "MAIN_BOX": (
        np.array([100, 30,   30]),
        np.array([140, 255, 200]),
    ),
    "SAMPLE": (
        np.array([60,  80,   80]),
        np.array([90,  255, 255]),
    ),
}


# --------------------------------------------------------------------------- #
# Object Detector class
# --------------------------------------------------------------------------- #
class ObjectDetector:
    """
    Detects experiment objects in each video frame.

    Usage:
        detector = ObjectDetector(config)
        objects = detector.detect(frame, timestamp=time.time())
    """

    def __init__(self, config, hsv_ranges: dict | None = None):
        """
        Args:
            config: DetectionConfig from orbita.app.config
            hsv_ranges: Override HSV colour ranges (optional)
        """
        self.config = config
        self.min_area = config.min_area_px
        self.confidence_threshold = config.confidence_threshold

        # Build HSV range lookup (merge defaults with config overrides)
        self._hsv_ranges = self._build_hsv_ranges(
            config.color_ranges if hasattr(config, "color_ranges") else {},
            hsv_ranges or {},
        )

        # Optional YOLO model
        self._yolo = None
        if config.use_yolo:
            self._try_load_yolo(config.yolo_model_path)

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def detect(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> list[DetectedObject]:
        """
        Run detection on a single BGR frame.

        Returns:
            List of DetectedObject (may be empty).
        """
        objects: list[DetectedObject] = []

        # 1. Chroma-based detection (always runs)
        objects.extend(self._detect_chroma(frame, timestamp))

        # 2. YOLO overlay (if model available)
        if self._yolo is not None:
            yolo_objs = self._detect_yolo(frame, timestamp)
            objects = self._fuse(objects, yolo_objs)

        return objects

    def draw(self, frame: np.ndarray, objects: list[DetectedObject]) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of the frame."""
        vis = frame.copy()
        colours = {
            "RED_BOX": (0, 0, 220),
            "YELLOW_BOX": (0, 220, 220),
            "MAIN_BOX": (200, 100, 0),
            "SAMPLE": (0, 200, 80),
            "PERSON": (200, 200, 200),
            "TOOL": (200, 150, 50),
            "CHAMBER": (150, 200, 255),
        }
        for obj in objects:
            colour = colours.get(obj.class_name, (180, 180, 180))
            x, y, w, h = obj.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)
            label = f"{obj.class_name} {obj.confidence:.2f}"
            cv2.putText(vis, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, colour, 1, cv2.LINE_AA)
        return vis

    # ----------------------------------------------------------------------- #
    # Chroma detection
    # ----------------------------------------------------------------------- #
    def _detect_chroma(
        self, frame: np.ndarray, timestamp: float
    ) -> list[DetectedObject]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: list[DetectedObject] = []

        seen_classes: set[str] = set()

        for class_name, (lower, upper) in self._hsv_ranges.items():
            # Strip the "_HIGH" suffix used for red-hue wrap-around
            canonical = class_name.replace("_HIGH", "")

            mask = cv2.inRange(hsv, lower, upper)

            # Merge red-hue wrap-around into single RED_BOX mask
            if class_name == "RED_BOX" and "RED_BOX_HIGH" in self._hsv_ranges:
                mask2 = cv2.inRange(
                    hsv,
                    self._hsv_ranges["RED_BOX_HIGH"][0],
                    self._hsv_ranges["RED_BOX_HIGH"][1],
                )
                mask = cv2.bitwise_or(mask, mask2)

            if canonical in seen_classes:
                continue   # Already processed (avoid double from HIGH variant)

            # Morphological clean-up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2

                # Confidence heuristic: larger & more circular → higher confidence
                (_, _), radius = cv2.minEnclosingCircle(cnt)
                circularity = area / (np.pi * radius ** 2 + 1e-6)
                confidence = float(np.clip(0.5 + 0.3 * circularity, 0.5, 0.92))

                results.append(DetectedObject(
                    class_name=canonical,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    centroid=(cx, cy),
                    timestamp=timestamp,
                    source="chroma",
                ))

            seen_classes.add(canonical)

        return results

    # ----------------------------------------------------------------------- #
    # YOLO detection
    # ----------------------------------------------------------------------- #
    def _detect_yolo(
        self, frame: np.ndarray, timestamp: float
    ) -> list[DetectedObject]:
        if self._yolo is None:
            return []
        try:
            results = self._yolo.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                stream=False,
            )
            objs: list[DetectedObject] = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names.get(cls_id, f"CLASS_{cls_id}").upper()
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w // 2, y1 + h // 2
                    objs.append(DetectedObject(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, w, h),
                        centroid=(cx, cy),
                        timestamp=timestamp,
                        source="yolo",
                    ))
            return objs
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            return []

    # ----------------------------------------------------------------------- #
    # Fusion
    # ----------------------------------------------------------------------- #
    def _fuse(
        self,
        chroma_objs: list[DetectedObject],
        yolo_objs: list[DetectedObject],
    ) -> list[DetectedObject]:
        """
        Simple fusion: if YOLO confirms a chroma detection (IoU > 0.3),
        promote confidence and mark source as "fused".
        Otherwise keep all chroma + YOLO detections.
        """
        used_yolo: set[int] = set()
        for co in chroma_objs:
            for i, yo in enumerate(yolo_objs):
                if i in used_yolo:
                    continue
                if yo.class_name == co.class_name and self._iou(co, yo) > 0.3:
                    co.confidence = max(co.confidence, yo.confidence)
                    co.source = "fused"
                    used_yolo.add(i)
                    break

        # Add YOLO-only detections (novel classes not in chroma)
        for i, yo in enumerate(yolo_objs):
            if i not in used_yolo:
                chroma_objs.append(yo)

        return chroma_objs

    # ----------------------------------------------------------------------- #
    # Utilities
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _iou(a: DetectedObject, b: DetectedObject) -> float:
        """Intersection over Union for two DetectedObjects."""
        ix = max(0, min(a.x2, b.x2) - max(a.x, b.x))
        iy = max(0, min(a.y2, b.y2) - max(a.y, b.y))
        inter = ix * iy
        union = a.area + b.area - inter
        return inter / (union + 1e-6)

    @staticmethod
    def _build_hsv_ranges(
        config_ranges: dict,
        override_ranges: dict,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        merged = dict(DEFAULT_HSV_RANGES)
        for name, vals in config_ranges.items():
            if isinstance(vals, (list, tuple)) and len(vals) == 6:
                merged[name] = (
                    np.array(vals[:3], dtype=np.uint8),
                    np.array(vals[3:], dtype=np.uint8),
                )
        for name, (lo, hi) in override_ranges.items():
            merged[name] = (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        return merged

    def _try_load_yolo(self, model_path: str) -> None:
        import os
        if not os.path.exists(model_path):
            logger.info(
                "YOLO model not found at %s — falling back to chroma detection.",
                model_path,
            )
            return
        try:
            from ultralytics import YOLO  # type: ignore
            self._yolo = YOLO(model_path)
            logger.info("YOLO model loaded from %s", model_path)
        except ImportError:
            logger.warning("ultralytics not installed — chroma detection only.")
        except Exception as exc:
            logger.warning("Failed to load YOLO model: %s", exc)
