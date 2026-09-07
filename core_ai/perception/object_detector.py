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
    track_id: int = -1
    velocity: tuple[float, float] = (0.0, 0.0)
    semantic_identity: str = ""

    @property
    def is_yolo(self) -> bool:
        """True if detection originated from neural YOLO model."""
        return self.source == "yolo"

    @property
    def is_hybrid(self) -> bool:
        """True if detection originated from or was assisted by chroma/heuristic rules."""
        return self.source in ("chroma", "fused")

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
    "BLUE_BOX": (               # Expanded blue container
        np.array([95,  55,   30]),
        np.array([135, 255, 255]),
    ),
    "MAIN_BOX": (               # Backward compatible alias for blue container
        np.array([95,  55,   30]),
        np.array([135, 255, 255]),
    ),
    "YELLOW_BOX": (
        np.array([18,  90,   70]),
        np.array([38,  255, 255]),
    ),
    "RED_BOX": (
        np.array([0,   140,  70]),
        np.array([10,  255, 255]),
    ),
    "RED_BOX_HIGH": (           # Red wraps around H=180
        np.array([170, 140,  70]),
        np.array([180, 255, 255]),
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

        # Multi-object tracker for persistent identity and trajectory velocity
        from core_ai.perception.object_tracker import MultiObjectTracker
        self.tracker = MultiObjectTracker(max_age=15, min_hits=1, iou_threshold=0.25)

        # Temporal smoother to eliminate jitter & brief occlusions
        self._tracked_objects: Dict[str, Dict[str, Any]] = {}
        self._frame_count: int = 0
        self.current_ai_source: str = "PRIMARY_AI" if self._yolo is not None else "FALLBACK_AI"

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def detect(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        hands: Optional[tuple[Any, Any]] = None,
    ) -> list[DetectedObject]:
        """
        Run robust detection and multi-object tracking on a single BGR frame.
        """
        self._frame_count += 1
        objects: list[DetectedObject] = []

        if self._yolo is not None:
            # 1. Primary: High-accuracy YOLO detection with semantic mapping
            yolo_objs = self._detect_yolo(frame, timestamp)
            objects.extend(yolo_objs)

            # 2. Secondary: Supplemental chroma check for missed experiment boxes
            found_classes = {o.class_name for o in yolo_objs}
            missing_boxes = {"RED_BOX", "YELLOW_BOX", "MAIN_BOX"} - found_classes
            
            # Optimization: If missing boxes are already actively tracked with high confidence, skip expensive chroma
            if missing_boxes and hasattr(self, "tracker") and hasattr(self.tracker, "tracks"):
                tracked_classes = {
                    t.class_name for t in self.tracker.tracks.values()
                    if getattr(t, "time_since_update", 0) <= 2
                }
                missing_boxes = missing_boxes - tracked_classes

            # Auxiliary chroma scheduling:
            # - Immediate recovery if a missing box was tracked recently (<=15 frames)
            # - Low-cadence periodic discovery (every 6 frames) for new boxes
            recently_seen_classes = set()
            if hasattr(self, "tracker") and hasattr(self.tracker, "tracks"):
                recently_seen_classes = {
                    t.class_name for t in self.tracker.tracks.values()
                    if getattr(t, "time_since_update", 999) <= 15
                }
            should_run_chroma = (
                bool(missing_boxes & recently_seen_classes)
                or (self._frame_count % 6 == 0)
            )

            if missing_boxes and should_run_chroma:
                chroma_objs = self._detect_chroma(
                    frame,
                    timestamp,
                    filter_classes=missing_boxes,
                    min_area_override=3000,
                    min_confidence_override=0.72,
                    hands=hands,
                )
                # Keep chroma detections only if non-overlapping with existing YOLO detections
                for co in chroma_objs:
                    if not any(self._iou(co, yo) > 0.20 for yo in yolo_objs):
                        objects.append(co)
        else:
            # Fallback if YOLO not available
            objects.extend(self._detect_chroma(frame, timestamp, hands=hands))

        # 3. Classify AI detection source telemetry
        has_yolo = any(getattr(o, "source", "") == "yolo" for o in objects)
        has_chroma = any(getattr(o, "source", "") == "chroma" for o in objects)
        if has_yolo and has_chroma:
            self.current_ai_source = "HYBRID"
        elif has_yolo:
            self.current_ai_source = "PRIMARY_AI"
        elif has_chroma:
            self.current_ai_source = "FALLBACK_AI"
        else:
            self.current_ai_source = "UNCERTAIN"

        # 4. Apply temporal smoothing to stabilize bounding boxes
        smoothed = self._smooth_detections(objects, timestamp)

        # 5. Apply multi-object tracking: persistent track IDs, velocity vectors
        tracked = self.tracker.update(smoothed, timestamp)
        return tracked

    def get_ai_source(self) -> str:
        """Return the current AI source category: PRIMARY_AI | HYBRID | FALLBACK_AI | UNCERTAIN."""
        return self.current_ai_source

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
                imgsz=getattr(self.config, "yolo_imgsz", 480),
                verbose=False,
                stream=False,
            )
            objs: list[DetectedObject] = []
            raw_telemetry: list[dict] = []
            mapped_telemetry: list[dict] = []

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

                    raw_telemetry.append({
                        "raw_class": raw_name,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, w, h]
                    })

                    # Map raw YOLO/COCO object to ORBITA semantic experiment class
                    mapped_class, adj_conf = self._classify_yolo_object(
                        frame, raw_name, conf, x1, y1, w, h
                    )

                    mapped_telemetry.append({
                        "semantic_class": mapped_class,
                        "raw_class": raw_name,
                        "confidence": round(adj_conf, 3),
                        "bbox": [x1, y1, w, h]
                    })

                    objs.append(DetectedObject(
                        class_name=mapped_class,
                        confidence=adj_conf,
                        bbox=(x1, y1, w, h),
                        centroid=(cx, cy),
                        timestamp=timestamp,
                        source="yolo",
                        raw_label=raw_name.lower(),
                        semantic_identity=mapped_class,
                    ))

            self.last_raw_yolo_detections = raw_telemetry
            self.last_mapped_detections = mapped_telemetry
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
        Transparently separates RAW YOLO detections from SECONDARY PERCEPTION
        and maps to ORBITA official semantic ontology:
          - BLUE_BOX    -> Blue container / main box / book/laptop with blue hue
          - YELLOW_BOX  -> Yellow container / box with yellow hue
          - PEN         -> Pen / pencil / stylus / tool / scissors / elongated instrument
          - WATCH       -> Watch / clock / small circular sample
          - PERSON      -> Human operator
        """
        raw_lower = raw_name.lower().strip()

        # 1. Exact match with ORBITA official ontology
        if raw_name in ("BLUE_BOX", "YELLOW_BOX", "PEN", "WATCH", "PERSON"):
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
                red_px = np.count_nonzero(((h_c <= 10) | (h_c >= 170)) & (s_c > 130) & (v_c > 50))
                yellow_px = np.count_nonzero((h_c >= 16) & (h_c <= 38) & (s_c > 70) & (v_c > 50))
                blue_px = np.count_nonzero((h_c >= 95) & (h_c <= 135) & (s_c > 55) & (v_c > 35))

                if red_px / total > 0.20:
                    color = "RED"
                elif yellow_px / total > 0.15:
                    color = "YELLOW"
                elif blue_px / total > 0.20:
                    color = "BLUE"

        # 3. Categorize by semantic class and color
        # Pen / Tools
        if raw_lower in ("pen", "pencil", "marker", "stylus", "scissors", "knife", "fork", "spoon", "toothbrush", "remote", "cell phone") or raw_name == "TOOL":
            return "PEN", max(conf, 0.85)

        # Watch / Clock / Small circular specimens
        if raw_lower in ("clock", "watch", "wristwatch", "timer", "stopwatch") or (raw_name == "SAMPLE" and w * h < 18000):
            return "WATCH", max(conf, 0.85)

        # Containers & Boxes
        if raw_name == "MAIN_BOX" or color == "BLUE":
            return "BLUE_BOX", max(conf, 0.88)
        if raw_name == "YELLOW_BOX" or color == "YELLOW":
            return "YELLOW_BOX", max(conf, 0.88)
        if raw_name == "RED_BOX" or color == "RED":
            return "RED_BOX", max(conf, 0.88)

        if raw_lower in ("box", "suitcase", "backpack", "handbag", "book", "package", "laptop", "container"):
            if color == "YELLOW":
                return "YELLOW_BOX", max(conf, 0.88)
            if color == "BLUE":
                return "BLUE_BOX", max(conf, 0.88)
            if color == "RED":
                return "RED_BOX", max(conf, 0.88)
            return "BLUE_BOX", max(conf, 0.82)

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
        hands: Optional[tuple[Any, Any]] = None,
    ) -> list[DetectedObject]:
        H, W = frame.shape[:2]
        downscale = max(1, int(round(W / 480.0))) if (W > 640 and H > 360) else 1
        if downscale > 1:
            frame_work = cv2.resize(frame, (W // downscale, H // downscale), interpolation=cv2.INTER_LINEAR)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        else:
            frame_work = frame
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        hsv = cv2.cvtColor(frame_work, cv2.COLOR_BGR2HSV)
        results: list[DetectedObject] = []
        seen_classes: set[str] = set()
        frame_area = H * W
        eff_min_area_scaled = (min_area_override or self.min_area) / (downscale * downscale)
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

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates: list[tuple[float, DetectedObject]] = []
            for cnt in contours:
                raw_area = cv2.contourArea(cnt)
                # Ignore tiny specks or huge background areas (>40% screen)
                if raw_area < eff_min_area_scaled or raw_area > ((frame_area / (downscale * downscale)) * 0.40):
                    continue

                area = float(raw_area * (downscale * downscale))
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                x, y, w, h = rx * downscale, ry * downscale, rw * downscale, rh * downscale

                # Ignore table fixtures / top bezel (top 8% of frame)
                if y < int(H * 0.08):
                    continue

                # Ignore extreme aspect ratios (lines, borders, elongated arms)
                aspect = w / (h + 1e-5)
                if aspect > 2.5 or aspect < 0.40:
                    continue

                # Reject arms entering from left/right image borders
                if (x <= 5 or x + w >= W - 5) and (aspect > 1.8 or aspect < 0.55):
                    continue

                cx, cy = x + w // 2, y + h // 2

                # 1. Geometric Shape Validation: Container boxes are solid planar rectangles
                rect = cv2.minAreaRect(cnt)
                box_area = max(1.0, float(rect[1][0] * rect[1][1]))
                rectangularity = float(raw_area / box_area)
                solidity = float(raw_area / (rw * rh + 1e-5))

                if canonical in ("RED_BOX", "YELLOW_BOX", "MAIN_BOX"):
                    # True experiment containers possess high rectangularity and solidity
                    # Knuckles, palm lines, and finger contours have low rectangularity (<0.50)
                    if rectangularity < 0.52 or solidity < 0.42:
                        continue

                # 2. Organic Skin Chrominance Rejection for RED_BOX
                # Industrial plastic containers exhibit high red saturation and red channel dominance.
                # Human skin and palm creases have lower red purity and high green/blue values.
                if canonical == "RED_BOX":
                    crop_bgr = frame[y:y+h, x:x+w]
                    if crop_bgr.size > 0:
                        b_mean = float(np.mean(crop_bgr[:, :, 0]))
                        g_mean = float(np.mean(crop_bgr[:, :, 1]))
                        r_mean = float(np.mean(crop_bgr[:, :, 2]))
                        # Ensure red dominance over green/blue and minimum red intensity
                        if r_mean < 100.0 or (r_mean / max(1.0, g_mean)) < 1.30 or (r_mean / max(1.0, b_mean)) < 1.30:
                            continue

                # 3. Hand-Object Disambiguation & Coexistence
                # If hand tracking data is available, verify candidate is not a sub-contour of a hand.
                if hands and canonical == "RED_BOX":
                    left_h, right_h = hands
                    is_subsumed_by_hand = False
                    for hand in (left_h, right_h):
                        if hand is not None and getattr(hand, "is_visible", False):
                            hx, hy, hw, hh = getattr(hand, "bbox", (0, 0, 0, 0))
                            ix1 = max(x, hx)
                            iy1 = max(y, hy)
                            ix2 = min(x + w, hx + hw)
                            iy2 = min(y + h, hy + hh)
                            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            # If candidate is mostly inside the hand bounding box:
                            if inter_area / (w * h + 1e-5) > 0.55:
                                # Keep as RED_BOX only if it has strong standalone evidence
                                # of being an actual container (large area outside or high rectangularity)
                                if rectangularity < 0.68 or area < 4500:
                                    is_subsumed_by_hand = True
                                    break
                    if is_subsumed_by_hand:
                        continue

                # Score confidence from rectangularity and solidity
                confidence = float(np.clip(0.60 + 0.22 * rectangularity + 0.15 * solidity, 0.60, 0.95))

                if confidence < eff_min_conf:
                    continue

                candidates.append((area, DetectedObject(
                    class_name=canonical,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    centroid=(cx, cy),
                    timestamp=timestamp,
                    source="chroma",
                )))

            # If multiple candidates found for a canonical box class, prioritize by area / confidence
            if candidates:
                candidates.sort(key=lambda c: c[0], reverse=True)
                kept: list[DetectedObject] = []
                for _, obj in candidates:
                    if not any(self._iou(obj, k) > 0.25 for k in kept):
                        kept.append(obj)
                        if len(kept) >= 2:
                            break
                results.extend(kept)

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
        Matches detections to existing tracks by spatial proximity, preventing
        distinct objects of the same class from collapsing into each other.
        """
        alpha = 0.70  # EMA smoothing factor for current frame
        matched_prev_keys: set[str] = set()
        smoothed_results: list[DetectedObject] = []

        for obj in current_objs:
            cx, cy, cw, ch = obj.bbox
            best_match_key = None
            best_match_dist = float("inf")

            # Search existing tracks of the same class for spatial proximity
            for key, prev in self._tracked_objects.items():
                if prev["class_name"] != obj.class_name or key in matched_prev_keys:
                    continue
                px, py, pw, ph = prev["bbox"]
                # Spatial IoU or centroid distance check
                iou = self._iou_bbox((px, py, pw, ph), (cx, cy, cw, ch))
                dist = float(np.hypot((cx + cw // 2) - (px + pw // 2), (cy + ch // 2) - (py + ph // 2)))

                if (iou > 0.15 or dist < 100.0) and dist < best_match_dist:
                    best_match_key = key
                    best_match_dist = dist

            if best_match_key is not None:
                matched_prev_keys.add(best_match_key)
                prev = self._tracked_objects[best_match_key]
                px, py, pw, ph = prev["bbox"]

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

                prev["bbox"] = smoothed_bbox
                prev["confidence"] = obj.confidence
                prev["missed_count"] = 0
                prev["timestamp"] = timestamp
                prev["object"] = obj
                smoothed_results.append(obj)
            else:
                # New distinct object track
                track_id = f"{obj.class_name}_{len(self._tracked_objects) + len(smoothed_results)}"
                self._tracked_objects[track_id] = {
                    "class_name": obj.class_name,
                    "object": obj,
                    "bbox": obj.bbox,
                    "confidence": obj.confidence,
                    "missed_count": 0,
                    "timestamp": timestamp,
                }
                smoothed_results.append(obj)

        # Retain objects that disappeared for only 1 frame (prevents flicker)
        for key in list(self._tracked_objects.keys()):
            if key not in matched_prev_keys:
                prev = self._tracked_objects[key]
                prev["missed_count"] += 1
                if prev["missed_count"] <= 1:
                    # Bridge 1-frame flicker by preserving previous smoothed detection
                    smoothed_results.append(prev["object"])
                else:
                    del self._tracked_objects[key]

        return smoothed_results

    # ----------------------------------------------------------------------- #
    # Utilities
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _iou_bbox(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
        """Intersection over Union for two (x, y, w, h) bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        union = float(boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter)
        return inter / (union + 1e-6) if union > 0 else 0.0

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
                Path("models/orbita_yolo_detector_v3.pt"),
                Path(__file__).parent.parent.parent / "models" / "orbita_yolo_detector_v3.pt",
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
                self._active_model_path = str(valid_path.resolve())
                logger.info("YOLO model loaded from %s", valid_path)
            else:
                logger.info("YOLO model not found locally; auto-initializing from Ultralytics...")
                self._yolo = YOLO("yolov8n.pt")
                self._active_model_path = "yolov8n.pt"
                target = Path("models") / "yolov8n.pt"
                target.parent.mkdir(exist_ok=True)
                if Path("yolov8n.pt").exists() and not target.exists():
                    Path("yolov8n.pt").rename(target)
                    self._active_model_path = str(target.resolve())
                logger.info("YOLO model auto-downloaded and initialized.")

            # Hardware acceleration
            if torch.cuda.is_available():
                self._device = "cuda"
                self._yolo.to("cuda")
                logger.info("YOLO accelerated on NVIDIA CUDA GPU.")
            else:
                self._device = "cpu"
                logger.info("YOLO running on CPU.")

            # Section 11 debug telemetry
            classes_list = list(self._yolo.names.values()) if hasattr(self._yolo, "names") else []
            logger.info(
                "\n================ MODEL DEBUG ================\n"
                "MODEL:\n"
                "path=%s\n"
                "classes=%s\n"
                "device=%s\n"
                "imgsz=%s\n"
                "conf=%.2f\n"
                "iou=%.2f\n"
                "=============================================",
                self._active_model_path,
                classes_list,
                self._device,
                getattr(self.config, "yolo_imgsz", 480),
                self.confidence_threshold,
                getattr(self.config, "iou_threshold", 0.45),
            )

        except ImportError:
            logger.warning("ultralytics not installed — fallback to chroma detection.")
        except Exception as exc:
            logger.warning("Failed to load YOLO model: %s — fallback to chroma detection.", exc)

    @property
    def active_model_path(self) -> str:
        return getattr(self, "_active_model_path", "models/orbita_yolo_detector_v3.pt")

    def get_model_telemetry(self) -> dict:
        """Returns structured model telemetry for debugging."""
        classes_list = list(self._yolo.names.values()) if (self._yolo and hasattr(self._yolo, "names")) else []
        return {
            "path": getattr(self, "_active_model_path", "unknown"),
            "classes": classes_list,
            "device": getattr(self, "_device", "cpu"),
            "imgsz": getattr(self.config, "yolo_imgsz", 480),
            "conf": self.confidence_threshold,
            "iou": getattr(self.config, "iou_threshold", 0.45),
        }

    def load_model(self, model_path: str) -> bool:
        """Dynamically reload YOLO detector with a new version weights checkpoint."""
        try:
            from ultralytics import YOLO
            p = Path(model_path)
            if not p.exists():
                logger.warning("Cannot load YOLO model from non-existent path: %s", model_path)
                return False
            new_yolo = YOLO(str(p))
            if getattr(self, "_device", "cpu") == "cuda":
                new_yolo.to("cuda")
            self._yolo = new_yolo
            self._active_model_path = str(p.resolve())
            logger.info("Successfully hot-reloaded YOLO detector from %s", p)
            classes_list = list(new_yolo.names.values()) if hasattr(new_yolo, "names") else []
            logger.info(
                "\n================ MODEL DEBUG ================\n"
                "MODEL:\npath=%s\nclasses=%s\ndevice=%s\nimgsz=%s\nconf=%.2f\niou=%.2f\n=============================================",
                self._active_model_path,
                classes_list,
                self._device,
                getattr(self.config, "yolo_imgsz", 480),
                self.confidence_threshold,
                getattr(self.config, "iou_threshold", 0.45),
            )
            return True
        except Exception as exc:
            logger.error("Failed to hot-reload YOLO detector from %s: %s", model_path, exc)
            return False

    def draw(self, frame: np.ndarray, detections: list[DetectedObject]) -> np.ndarray:
        """
        Draws professional bounding boxes, labels, and confidence tags on frame.
        """
        if frame is None or not detections:
            return frame

        vis = frame.copy()
        class_colors = {
            "PERSON": (80, 220, 100),       # Vibrant Green
            "MAIN_BOX": (240, 180, 40),     # Azure / Sky Blue (BGR)
            "RED_BOX": (40, 50, 235),       # Vibrant Red
            "YELLOW_BOX": (20, 215, 255),   # Vibrant Yellow
            "SAMPLE": (230, 80, 210),       # Magenta / Violet
            "TOOL": (255, 140, 0),          # Cyan / Blue
        }

        for det in detections:
            x, y, w, h = det.bbox
            color = class_colors.get(det.class_name, (180, 190, 200))

            # Bounding box with clean sharp aerospace lines
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

            # Centroid point
            cx, cy = getattr(det, "centroid", (x + w // 2, y + h // 2))
            cv2.circle(vis, (cx, cy), 3, color, -1, cv2.LINE_AA)

            # Optional velocity motion indicator
            vx, vy = getattr(det, "velocity", (0.0, 0.0))
            if abs(vx) + abs(vy) > 15.0:
                end_x = int(cx + np.clip(vx * 0.2, -40, 40))
                end_y = int(cy + np.clip(vy * 0.2, -40, 40))
                cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), color, 1, cv2.LINE_AA, tipLength=0.3)

            # Label text with Track ID and Confidence
            conf_pct = int(det.confidence * 100)
            track_prefix = f"#{det.track_id} " if getattr(det, "track_id", -1) > 0 else ""
            source_tag = " [HYBRID]" if getattr(det, "source", "") == "chroma" else " [YOLO]"
            label = f"{track_prefix}{det.class_name}{source_tag} {conf_pct}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # Header badge above box
            badge_y1 = max(0, y - th - 6)
            badge_y2 = y
            badge_x2 = min(vis.shape[1], x + tw + 8)
            cv2.rectangle(vis, (x, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                vis,
                label,
                (x + 4, badge_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (10, 15, 20),
                1,
                cv2.LINE_AA,
            )

        return vis

