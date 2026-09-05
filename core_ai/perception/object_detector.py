"""
ORBITA Object Detector
======================
High-precision hybrid detection and tracking pipeline:
  1. YOLOv8 Semantic Object Detection (real-world object detection + tracking)
  2. Bounding-box Color-Chroma Semantic Mapping (classifies physical items into ORBITA ontology)
  3. Targeted Chroma Fallback (for unclassified synthetic lab components)
  4. Temporal Object Stabilization (eliminates bounding box jitter and flickering)

Target Ontology:
  - PERSON      -> Human operator / astronaut
  - SAMPLE      -> Samples, vials, bottles, biological specimens
  - TOOL        -> Scissors, tweezers, knives, instruments, pens
  - MAIN_BOX    -> Primary experiment container / workstation chamber
  - RED_BOX     -> Red sample container
  - YELLOW_BOX  -> Yellow reagent / auxiliary container
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    raw_label: str = ""               # original YOLO COCO label

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
# --------------------------------------------------------------------------- #
DEFAULT_HSV_RANGES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "RED_BOX": (
        np.array([0,   100,  70]),
        np.array([12,  255, 255]),
    ),
    "RED_BOX_HIGH": (           # Red wraps around H=180
        np.array([168, 100,  70]),
        np.array([180, 255, 255]),
    ),
    "YELLOW_BOX": (
        np.array([18,  90,   70]),
        np.array([38,  255, 255]),
    ),
    "MAIN_BOX": (
        np.array([95,  30,   30]),
        np.array([140, 255, 200]),
    ),
    "SAMPLE": (
        np.array([45,  70,   70]),
        np.array([88,  255, 255]),
    ),
}


# --------------------------------------------------------------------------- #
# Object Detector class
# --------------------------------------------------------------------------- #
class ObjectDetector:
    """
    Detects experiment objects in each video frame with YOLO + color intelligence.
    """

    def __init__(self, config: Any, hsv_ranges: dict | None = None):
        self.config = config
        self.min_area = getattr(config, "min_area_px", 400)
        # Sensitive confidence threshold for real-world handheld objects
        self.confidence_threshold = min(0.30, getattr(config, "confidence_threshold", 0.35))

        # Build HSV range lookup
        self._hsv_ranges = self._build_hsv_ranges(
            config.color_ranges if hasattr(config, "color_ranges") else {},
            hsv_ranges or {},
        )

        # YOLO model initialization
        self._yolo = None
        self._device = "cpu"
        if getattr(config, "use_yolo", True):
            self._try_load_yolo(getattr(config, "yolo_model_path", "models/yolov8n.pt"))

        # Temporal smoother to eliminate jitter & brief occlusions
        self._tracked_objects: Dict[str, Dict[str, Any]] = {}

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def detect(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> list[DetectedObject]:
        """
        Run robust detection on a single BGR frame.
        """
        objects: list[DetectedObject] = []

        if self._yolo is not None:
            # 1. Primary: High-accuracy YOLO detection with semantic mapping
            yolo_objs = self._detect_yolo(frame, timestamp)
            objects.extend(yolo_objs)

            # 2. Secondary: Supplemental chroma check for missed experiment boxes
            found_classes = {o.class_name for o in yolo_objs}
            missing_boxes = {"RED_BOX", "YELLOW_BOX", "MAIN_BOX"} - found_classes
            if missing_boxes:
                chroma_objs = self._detect_chroma(
                    frame,
                    timestamp,
                    filter_classes=missing_boxes,
                    min_area_override=3000,
                    min_confidence_override=0.72,
                )
                # Keep chroma detections only if non-overlapping with existing YOLO detections
                for co in chroma_objs:
                    if not any(self._iou(co, yo) > 0.20 for yo in yolo_objs):
                        objects.append(co)
        else:
            # Fallback if YOLO not available
            objects.extend(self._detect_chroma(frame, timestamp))

        # 3. Apply temporal smoothing to stabilize bounding boxes
        return self._smooth_detections(objects, timestamp)

    def draw(self, frame: np.ndarray, objects: list[DetectedObject]) -> np.ndarray:
        """Draw bounding boxes and formatted labels onto a copy of the frame."""
        vis = frame.copy()
        colours = {
            "RED_BOX": (0, 0, 220),       # Red
            "YELLOW_BOX": (0, 220, 220),   # Yellow
            "MAIN_BOX": (200, 100, 0),     # Cyan-Blue
            "SAMPLE": (0, 200, 80),        # Green
            "PERSON": (220, 220, 220),     # White
            "TOOL": (220, 140, 20),        # Orange
            "CHAMBER": (150, 200, 255),    # Light blue
        }

        for obj in objects:
            colour = colours.get(obj.class_name, (180, 180, 180))
            x, y, w, h = obj.bbox

            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)

            # Construct transparent label
            if obj.raw_label and obj.raw_label != obj.class_name:
                label = f"{obj.class_name} [{obj.raw_label}] {obj.confidence:.2f}"
            else:
                label = f"{obj.class_name} {obj.confidence:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty1 = max(0, y - th - 8)

            # Solid background pill for text contrast
            cv2.rectangle(vis, (x, ty1), (x + tw + 6, ty1 + th + 6), colour, -1)
            cv2.putText(
                vis,
                label,
                (x + 3, ty1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )

        return vis

    # ----------------------------------------------------------------------- #
    # YOLO detection with Semantic & Color Mapping
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
                device=self._device,
                verbose=False,
                stream=False,
            )
            objs: list[DetectedObject] = []
            for r in results:
                if not r.boxes:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    raw_name = r.names.get(cls_id, f"CLASS_{cls_id}").upper()
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    w, h = max(1, x2 - x1), max(1, y2 - y1)
                    cx, cy = x1 + w // 2, y1 + h // 2

                    # Map raw YOLO/COCO object to ORBITA semantic experiment class
                    mapped_class, adj_conf = self._classify_yolo_object(
                        frame, raw_name, conf, x1, y1, w, h
                    )

                    objs.append(DetectedObject(
                        class_name=mapped_class,
                        confidence=adj_conf,
                        bbox=(x1, y1, w, h),
                        centroid=(cx, cy),
                        timestamp=timestamp,
                        source="yolo",
                        raw_label=raw_name.lower(),
                    ))
            return objs
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            return []

    def _classify_yolo_object(
        self,
        frame: np.ndarray,
        raw_name: str,
        conf: float,
        x1: int,
        y1: int,
        w: int,
        h: int,
    ) -> tuple[str, float]:
        """
        Maps standard COCO object classes to ORBITA experiment ontology:
          - PERSON      -> person
          - SAMPLE      -> bottle, cup, bowl, vase, apple, etc.
          - TOOL        -> scissors, knife, fork, spoon, remote, phone, toothbrush
          - MAIN_BOX    -> suitcase, backpack, book, laptop, large container
          - RED_BOX     -> any container/box/book with dominant red hue
          - YELLOW_BOX  -> any container/box/book with dominant yellow hue
        """
        raw_lower = raw_name.lower().strip()

        # 1. Exact match with ORBITA schema
        if raw_name in ("PERSON", "MAIN_BOX", "RED_BOX", "YELLOW_BOX", "SAMPLE", "TOOL", "CHAMBER"):
            return raw_name, conf

        if raw_lower == "person":
            return "PERSON", conf

        # 2. Extract central crop color characteristics
        frame_h, frame_w = frame.shape[:2]
        cx1 = max(0, min(frame_w - 1, x1))
        cy1 = max(0, min(frame_h - 1, y1))
        cx2 = max(0, min(frame_w, x1 + w))
        cy2 = max(0, min(frame_h, y1 + h))

        color = "UNKNOWN"
        if cx2 > cx1 and cy2 > cy1:
            crop = frame[cy1:cy2, cx1:cx2]
            ih, iw = crop.shape[:2]
            if ih > 10 and iw > 10:
                inner = crop[int(ih * 0.15):int(ih * 0.85), int(iw * 0.15):int(iw * 0.85)]
            else:
                inner = crop

            hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
            h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            total = h_c.size
            if total > 0:
                red_px = np.count_nonzero(((h_c <= 12) | (h_c >= 168)) & (s_c > 50) & (v_c > 40))
                yellow_px = np.count_nonzero((h_c >= 16) & (h_c <= 38) & (s_c > 50) & (v_c > 50))
                green_px = np.count_nonzero((h_c >= 45) & (h_c <= 88) & (s_c > 45) & (v_c > 40))
                blue_px = np.count_nonzero((h_c >= 95) & (h_c <= 140) & (v_c > 35))

                if red_px / total > 0.12:
                    color = "RED"
                elif yellow_px / total > 0.12:
                    color = "YELLOW"
                elif green_px / total > 0.15:
                    color = "GREEN"
                elif blue_px / total > 0.20:
                    color = "BLUE"

        # 3. Categorize by semantic class and color
        # Tools & instruments
        if raw_lower in ("scissors", "knife", "fork", "spoon", "toothbrush", "hair drier", "remote", "cell phone"):
            return "TOOL", max(conf, 0.85)

        # Samples (small containers, vials, biological specimens)
        if raw_lower in ("bottle", "cup", "wine glass", "bowl", "vase", "apple", "orange", "banana"):
            return "SAMPLE", max(conf, 0.85)

        # Color-specific boxes
        if color == "RED":
            return "RED_BOX", max(conf, 0.88)
        if color == "YELLOW":
            return "YELLOW_BOX", max(conf, 0.88)
        if color == "GREEN":
            return "SAMPLE", max(conf, 0.80)

        # Boxes & Containers (suitcases, backpacks, books, laptops, boxes)
        if raw_lower in ("suitcase", "backpack", "handbag", "book", "laptop", "box", "microwave", "refrigerator", "tv"):
            if (w * h > 30000) or color == "BLUE":
                return "MAIN_BOX", max(conf, 0.90)
            return "MAIN_BOX", max(conf, 0.82)

        # Fallback by dominant color
        if color == "RED":
            return "RED_BOX", max(conf, 0.80)
        if color == "YELLOW":
            return "YELLOW_BOX", max(conf, 0.80)

        return raw_name, conf

    # ----------------------------------------------------------------------- #
    # Chroma detection
    # ----------------------------------------------------------------------- #
    def _detect_chroma(
        self,
        frame: np.ndarray,
        timestamp: float,
        filter_classes: Optional[Set[str]] = None,
        min_area_override: Optional[int] = None,
        min_confidence_override: Optional[float] = None,
    ) -> list[DetectedObject]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: list[DetectedObject] = []
        seen_classes: set[str] = set()
        frame_area = frame.shape[0] * frame.shape[1]
        eff_min_area = min_area_override or self.min_area
        eff_min_conf = min_confidence_override or 0.55

        for class_name, (lower, upper) in self._hsv_ranges.items():
            canonical = class_name.replace("_HIGH", "")
            if filter_classes and canonical not in filter_classes:
                continue

            mask = cv2.inRange(hsv, lower, upper)

            # Merge red wrap-around
            if class_name == "RED_BOX" and "RED_BOX_HIGH" in self._hsv_ranges:
                mask2 = cv2.inRange(
                    hsv,
                    self._hsv_ranges["RED_BOX_HIGH"][0],
                    self._hsv_ranges["RED_BOX_HIGH"][1],
                )
                mask = cv2.bitwise_or(mask, mask2)

            if canonical in seen_classes:
                continue

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Ignore tiny specks or huge background areas (>40% screen)
                if area < eff_min_area or area > (frame_area * 0.40):
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                # Ignore extreme aspect ratios (lines, borders)
                if w / (h + 1e-5) > 4.5 or h / (w + 1e-5) > 4.5:
                    continue

                cx, cy = x + w // 2, y + h // 2

                # Confidence heuristic
                (_, _), radius = cv2.minEnclosingCircle(cnt)
                circularity = area / (np.pi * radius ** 2 + 1e-6)
                confidence = float(np.clip(0.55 + 0.35 * circularity, 0.55, 0.90))

                if confidence < eff_min_conf:
                    continue

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
    # Temporal Smoothing
    # ----------------------------------------------------------------------- #
    def _smooth_detections(
        self,
        current_objs: list[DetectedObject],
        timestamp: float,
    ) -> list[DetectedObject]:
        """
        Smooths bounding box coordinates across frames to eliminate jitter
        and bridges brief detection dropouts (up to 2 frames).
        """
        alpha = 0.70  # EMA smoothing factor for current frame
        updated_tracks: Dict[str, Dict[str, Any]] = {}

        for obj in current_objs:
            key = obj.class_name
            if key in self._tracked_objects:
                prev = self._tracked_objects[key]
                px, py, pw, ph = prev["bbox"]
                cx, cy, cw, ch = obj.bbox

                # Smooth coordinates
                sx = int(alpha * cx + (1 - alpha) * px)
                sy = int(alpha * cy + (1 - alpha) * py)
                sw = int(alpha * cw + (1 - alpha) * pw)
                sh = int(alpha * ch + (1 - alpha) * ph)
                smoothed_bbox = (sx, sy, sw, sh)
                smoothed_centroid = (sx + sw // 2, sy + sh // 2)

                obj.bbox = smoothed_bbox
                obj.centroid = smoothed_centroid
                obj.confidence = float(alpha * obj.confidence + (1 - alpha) * prev["confidence"])

                updated_tracks[key] = {
                    "object": obj,
                    "bbox": smoothed_bbox,
                    "confidence": obj.confidence,
                    "missed_count": 0,
                    "timestamp": timestamp,
                }
            else:
                updated_tracks[key] = {
                    "object": obj,
                    "bbox": obj.bbox,
                    "confidence": obj.confidence,
                    "missed_count": 0,
                    "timestamp": timestamp,
                }

        # Keep objects that disappeared for only 1-2 frames (retains continuity)
        for key, prev in self._tracked_objects.items():
            if key not in updated_tracks and prev["missed_count"] < 2:
                prev["missed_count"] += 1
                prev["confidence"] *= 0.85
                recovered_obj = prev["object"]
                recovered_obj.confidence = prev["confidence"]
                recovered_obj.timestamp = timestamp
                updated_tracks[key] = prev

        self._tracked_objects = updated_tracks
        return [t["object"] for t in self._tracked_objects.values()]

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
        try:
            import os
            import torch
            from ultralytics import YOLO  # type: ignore

            candidates = [
                Path(model_path),
                Path("models/yolov8n.pt"),
                Path(__file__).parent.parent.parent / "models" / "yolov8n.pt",
                Path("yolov8n.pt"),
            ]
            valid_path = None
            for p in candidates:
                if p.exists():
                    valid_path = p
                    break

            if valid_path:
                self._yolo = YOLO(str(valid_path))
                logger.info("YOLO model loaded from %s", valid_path)
            else:
                logger.info("YOLO model not found locally; auto-initializing from Ultralytics...")
                self._yolo = YOLO("yolov8n.pt")
                target = Path("models") / "yolov8n.pt"
                target.parent.mkdir(exist_ok=True)
                if Path("yolov8n.pt").exists() and not target.exists():
                    Path("yolov8n.pt").rename(target)
                logger.info("YOLO model auto-downloaded and initialized.")

            # Hardware acceleration
            if torch.cuda.is_available():
                self._device = "cuda"
                self._yolo.to("cuda")
                logger.info("YOLO accelerated on NVIDIA CUDA GPU.")
            else:
                self._device = "cpu"
                logger.info("YOLO running on CPU.")

        except ImportError:
            logger.warning("ultralytics not installed — fallback to chroma detection.")
        except Exception as exc:
            logger.warning("Failed to load YOLO model: %s — fallback to chroma detection.", exc)
