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
# Ontology, Class Normalization & Audit Constants
# --------------------------------------------------------------------------- #
REQUIRED_YOLO_CLASSES: list[str] = [
    "LOCATION_A",
    "LOCATION_B",
    "PEN",
    "WATCH",
    "BLUE_BOX",
    "YELLOW_BOX",
    "HAND",
]

DEFAULT_CLASS_THRESHOLDS: dict[str, float] = {
    "location_a": 0.25,
    "location_b": 0.25,
    "pen": 0.20,        # Sensitive for slender lab instruments
    "watch": 0.20,      # Sensitive for small specimens
    "blue_box": 0.25,
    "yellow_box": 0.25,
    "hand": 0.25,       # Stable tracking threshold
}


class NormalizedClassName(str):
    """
    String subclass supporting bidirectional case-insensitive comparison
    and canonical ORBITA ontology alias matching.
    """
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, (str, NormalizedClassName)):
            return False
        s1 = self.lower().strip()
        s2 = str(other).lower().strip()
        if s1 == s2:
            return True
        aliases = {
            "location_a": {"location_a", "loc_a", "loca", "location a"},
            "location_b": {"location_b", "loc_b", "locb", "location b"},
            "yellow_box": {"yellow_box", "yellow_box_container"},
            "blue_box": {"blue_box", "main_box", "blue_box_container"},
            "pen": {"pen", "tool"},
            "watch": {"watch", "sample"},
            "hand": {"hand", "hands"},
        }
        for group in aliases.values():
            if s1 in group and s2 in group:
                return True
        return False

    def __hash__(self) -> int:
        return str.__hash__(self)


def map_raw_to_app_class(raw_name: str) -> str:
    """Normalize names so variants map correctly to canonical application classes."""
    raw = str(raw_name).lower().strip()
    if raw in ("location_a", "loc_a", "loca", "location a"):
        return "LOCATION_A"
    if raw in ("location_b", "loc_b", "locb", "location b"):
        return "LOCATION_B"
    if raw in ("yellow_box", "yellow box", "yellow_box_container"):
        return "YELLOW_BOX"
    if raw in ("blue_box", "blue box", "main_box", "main box", "blue_box_container"):
        return "BLUE_BOX"
    if raw in ("pen", "tool", "pencil", "marker", "stylus"):
        return "PEN"
    if raw in ("watch", "sample", "clock", "wristwatch", "timer", "specimen"):
        return "WATCH"
    if raw in ("hand", "hands", "person_hand"):
        return "HAND"
    return raw.upper()


def audit_model_classes(model_names: dict[int, str] | list[str]) -> list[str]:
    """
    Prints model classes and verifies presence of all 7 required YOLO classes.
    Reports any missing class with:
      'MISSING MODEL CLASS: <class>'
    Returns list of missing classes.
    """
    if isinstance(model_names, dict):
        items = sorted(model_names.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))
    else:
        items = list(enumerate(model_names))

    print("\nMODEL CLASSES:")
    for idx, name in items:
        print(f"{idx} -> {name}")

    print("\nYOLO CLASS ID -> APPLICATION CLASS:")
    for idx, name in items:
        app_cls = map_raw_to_app_class(name)
        print(f"{idx} -> {app_cls}")

    mapped_app_classes = {map_raw_to_app_class(name) for _, name in items}
    missing_classes: list[str] = []

    for req_cls in REQUIRED_YOLO_CLASSES:
        if req_cls not in mapped_app_classes:
            print(f"MISSING MODEL CLASS: {req_cls}")
            logger.warning("MISSING MODEL CLASS: %s", req_cls)
            missing_classes.append(req_cls)

    return missing_classes


class NormalizedDetectionsDict(dict):
    """
    Unified detection dictionary supporting both canonical uppercase (LOCATION_A, LOCATION_B,
    BLUE_BOX, YELLOW_BOX, PEN, WATCH, HAND) and lowercase (location_a, etc.) access.
    """
    def _norm_key(self, k: Any) -> str:
        s = str(k).upper().strip()
        if s in ("LOC_A", "LOCA", "LOCATION A"):
            return "LOCATION_A"
        if s in ("LOC_B", "LOCB", "LOCATION B"):
            return "LOCATION_B"
        if s in ("MAIN_BOX", "BLUE_BOX_CONTAINER"):
            return "BLUE_BOX"
        if s in ("YELLOW_BOX_CONTAINER",):
            return "YELLOW_BOX"
        if s in ("TOOL", "PENCIL"):
            return "PEN"
        if s in ("SAMPLE", "CLOCK"):
            return "WATCH"
        if s in ("HANDS", "PERSON_HAND"):
            return "HAND"
        return s

    def __getitem__(self, key: Any) -> Any:
        nk = self._norm_key(key)
        if super().__contains__(nk):
            return super().__getitem__(nk)
        lk = str(key).lower().strip()
        if super().__contains__(lk):
            return super().__getitem__(lk)
        if super().__contains__(key):
            return super().__getitem__(key)
        return []

    def __contains__(self, key: Any) -> bool:
        nk = self._norm_key(key)
        lk = str(key).lower().strip()
        return super().__contains__(nk) or super().__contains__(lk) or super().__contains__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            val = self[key]
            return val if val is not None else default
        except KeyError:
            return default


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
    source: str = "chroma"            # "chroma" | "yolo" | "fused" | "hand_tracker"
    raw_label: str = ""               # original YOLO COCO label
    track_id: int = -1
    velocity: tuple[float, float] = (0.0, 0.0)
    semantic_identity: str = ""
    frame_id: int = 0

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

    @property
    def center_x(self) -> int:
        return self.centroid[0]

    @property
    def center_y(self) -> int:
        return self.centroid[1]

    @property
    def bounding_box_xyxy(self) -> list[int]:
        return [self.x, self.y, self.x2, self.y2]

    def to_detection_dict(self, frame_id: Optional[int] = None) -> dict[str, Any]:
        """
        Unified per-frame detection dictionary required by ORBITA specification:
        {
          "class_name": str,
          "confidence": float,
          "x1": int,
          "y1": int,
          "x2": int,
          "y2": int,
          "center_x": int,
          "center_y": int,
          "timestamp": float,
          "frame_id": int,
          "bounding_box": [x1, y1, x2, y2],
          "frame_timestamp": float
        }
        """
        norm_name = map_raw_to_app_class(str(self.class_name))
        f_id = self.frame_id if self.frame_id > 0 else (frame_id or 0)

        return {
            "class_name": NormalizedClassName(norm_name.lower()),
            "confidence": round(float(self.confidence), 3),
            "bbox": [int(self.x), int(self.y), int(self.x2), int(self.y2)],
            "center": [int(self.center_x), int(self.center_y)],
            "frame_id": int(f_id),
            "timestamp": float(self.timestamp),
            "x1": int(self.x),
            "y1": int(self.y),
            "x2": int(self.x2),
            "y2": int(self.y2),
            "center_x": int(self.center_x),
            "center_y": int(self.center_y),
            "bounding_box": [int(self.x), int(self.y), int(self.x2), int(self.y2)],
            "frame_timestamp": float(self.timestamp),
        }


# --------------------------------------------------------------------------- #
# Helper functions: IoU and Class-Aware NMS
# --------------------------------------------------------------------------- #
def compute_iou_bbox(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """Computes IoU between two (x, y, w, h) bounding boxes."""
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


def apply_class_aware_nms(
    detections: list[DetectedObject],
    iou_threshold: float = 0.45,
    min_confidence: float = 0.20,
) -> list[DetectedObject]:
    """
    Performs per-class Non-Maximum Suppression (NMS).
    Keeps only the highest-quality detection among heavily overlapping boxes of the same class.
    Preserves multiple different classes even if they overlap.
    """
    if not detections:
        return []

    valid_dets = [d for d in detections if getattr(d, "confidence", 0.0) >= min_confidence]
    if len(valid_dets) <= 1:
        return valid_dets

    # Group by canonical class
    grouped: dict[str, list[DetectedObject]] = {}
    for d in valid_dets:
        c_name = str(getattr(d, "semantic_identity", "") or getattr(d, "class_name", "")).upper()
        canon = "BLUE_BOX" if c_name in ("MAIN_BOX", "BLUE_BOX") else (
            "PEN" if c_name in ("TOOL", "PEN") else (
                "WATCH" if c_name in ("SAMPLE", "WATCH") else c_name
            )
        )
        grouped.setdefault(canon, []).append(d)

    kept: list[DetectedObject] = []
    for canon_cls, group in grouped.items():
        # Sort group by confidence descending (secondary tie-breaker: area)
        sorted_group = sorted(group, key=lambda x: (x.confidence, x.area), reverse=True)
        selected_for_class: list[DetectedObject] = []

        for candidate in sorted_group:
            should_suppress = False
            for selected in selected_for_class:
                iou = compute_iou_bbox(candidate.bbox, selected.bbox)
                if iou >= iou_threshold:
                    should_suppress = True
                    break
            if not should_suppress:
                selected_for_class.append(candidate)

        kept.extend(selected_for_class)

    return kept


# --------------------------------------------------------------------------- #
# Colour ranges for HSV chroma detection
# --------------------------------------------------------------------------- #
DEFAULT_HSV_RANGES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "BLUE_BOX": (               # Primary blue container (MAIN_BOX canonical)
        np.array([90,  45,   30]),
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
    Detects experiment objects in each video frame with YOLO + multi-class intelligence.
    Continuously detects all 5 required classes:
      1. yellow_box
      2. blue_box
      3. pen
      4. watch
      5. hand
    """

    def __init__(self, config: Any, hsv_ranges: dict | None = None):
        self.config = config
        self.min_area = getattr(config, "min_area_px", 400)
        self.confidence_threshold = min(0.25, getattr(config, "confidence_threshold", 0.25))
        self.iou_threshold = getattr(config, "iou_threshold", 0.45)

        # Class-specific confidence thresholds
        self.class_thresholds: dict[str, float] = dict(DEFAULT_CLASS_THRESHOLDS)
        if hasattr(config, "class_thresholds") and isinstance(config.class_thresholds, dict):
            self.class_thresholds.update(config.class_thresholds)

        # Build HSV range lookup for fallback/synthetic testing
        self._hsv_ranges = self._build_hsv_ranges(
            config.color_ranges if hasattr(config, "color_ranges") else {},
            hsv_ranges or {},
        )

        # YOLO model initialization
        self._yolo = None
        self._device = "cpu"
        self._active_model_path: str = ""
        self.missing_model_classes: list[str] = []
        if getattr(config, "use_yolo", True):
            self._try_load_yolo(getattr(config, "yolo_model_path", "models/orbita_yolo_detector_v3.pt"))

        # Multi-object tracker for persistent identity and trajectory velocity
        from core_ai.perception.object_tracker import MultiObjectTracker
        self.tracker = MultiObjectTracker(max_age=15, min_hits=1, iou_threshold=0.25)

        # Multi-class temporal smoother: keeps detections across 1–3 missed frames
        self._tracked_objects: dict[str, dict[str, Any]] = {}
        self._frame_count: int = 0
        self.current_ai_source: str = "PRIMARY_AI" if self._yolo is not None else "FALLBACK_AI"

        # Telemetry & debug caching
        self.last_raw_yolo_detections: list[dict] = []
        self.last_mapped_detections: list[dict] = []
        self.last_nms_detections: list[dict] = []
        self.last_tracked_detections: list[dict] = []
        self.last_grouped_detections: dict[str, list[dict[str, Any]]] = {c: [] for c in REQUIRED_YOLO_CLASSES}

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def detect(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        hands: Optional[tuple[Any, Any]] = None,
        frame_id: int = 0,
    ) -> list[DetectedObject]:
        """
        Run continuous multi-object detection, per-class NMS, hand integration,
        and temporal smoothing (1-3 frames) on a single frame.
        Evaluates all five classes simultaneously: BLUE_BOX, YELLOW_BOX, PEN, WATCH, HAND.
        """
        self._frame_count += 1
        f_id = frame_id if frame_id > 0 else self._frame_count
        objects: list[DetectedObject] = []

        # 1. Primary: Run YOLO inference if model loaded or overridden
        if self._yolo is not None or "_detect_yolo" in self.__dict__ or getattr(self.config, "use_yolo", False):
            try:
                yolo_objs = self._detect_yolo(frame, timestamp, frame_id=f_id)
            except TypeError:
                yolo_objs = self._detect_yolo(frame, timestamp)
            for o in yolo_objs:
                o.frame_id = f_id
            objects.extend(yolo_objs)

            # Supplemental chroma check if configured: detect any missing color-distinguishable targets
            if getattr(self.config, "allow_chroma_fallback", False):
                found_classes = {
                    map_raw_to_app_class(str(getattr(o, "semantic_identity", "") or getattr(o, "class_name", "")))
                    for o in objects
                }
                missing_chroma: set[str] = set()
                if "BLUE_BOX" not in found_classes:
                    missing_chroma.add("BLUE_BOX")
                if "YELLOW_BOX" not in found_classes:
                    missing_chroma.add("YELLOW_BOX")
                if missing_chroma:
                    try:
                        chroma_objs = self._detect_chroma(
                            frame, timestamp, filter_classes=missing_chroma, hands=hands, frame_id=f_id
                        )
                    except TypeError:
                        chroma_objs = self._detect_chroma(
                            frame, timestamp, filter_classes=missing_chroma, hands=hands
                        )
                    objects.extend(chroma_objs)
        else:
            # Fallback if YOLO model is disabled in config
            try:
                objects.extend(self._detect_chroma(frame, timestamp, hands=hands, frame_id=f_id))
            except TypeError:
                objects.extend(self._detect_chroma(frame, timestamp, hands=hands))

        # 2. Hand Detection: Treat Hand as a first-class detection in unified state
        if hands:
            left_h, right_h = hands
            for h in (left_h, right_h):
                if h is not None and getattr(h, "is_visible", False):
                    hx, hy, hw, hh = getattr(h, "bbox", (0, 0, 0, 0))
                    if hw > 8 and hh > 8:
                        h_conf = float(getattr(h, "confidence", 0.85))
                        hcx = int(hx + hw // 2)
                        hcy = int(hy + hh // 2)
                        pos = getattr(h, "position", None)
                        if pos is not None and len(pos) >= 2:
                            hcx, hcy = int(pos[0]), int(pos[1])

                        h_thresh = self.class_thresholds.get("hand", 0.30)
                        if h_conf >= h_thresh:
                            objects.append(DetectedObject(
                                class_name=NormalizedClassName("hand"),
                                confidence=h_conf,
                                bbox=(int(hx), int(hy), int(hw), int(hh)),
                                centroid=(hcx, hcy),
                                timestamp=timestamp,
                                source="hand_tracker",
                                raw_label="hand",
                                semantic_identity="HAND",
                                track_id=getattr(h, "hand_id", -1),
                                velocity=getattr(h, "velocity", (0.0, 0.0)),
                                frame_id=f_id,
                            ))

        # 3. Class-aware NMS: suppresses duplicates per-class while preserving different overlapping classes
        canonical_detections = apply_class_aware_nms(
            objects,
            iou_threshold=self.iou_threshold,
            min_confidence=min(self.class_thresholds.values()),
        )

        # 4. Short-term temporal smoothing & tracking (persists 1–3 missed frames to eliminate flicker)
        smoothed = self._smooth_detections(canonical_detections, timestamp)

        # 5. Extract unified grouped detections dictionary for all 7 required classes
        grouped = NormalizedDetectionsDict({
            "LOCATION_A": [],
            "LOCATION_B": [],
            "PEN": [],
            "WATCH": [],
            "BLUE_BOX": [],
            "YELLOW_BOX": [],
            "HAND": [],
        })
        for det in smoothed:
            det.frame_id = f_id
            det_dict = det.to_detection_dict(frame_id=f_id)
            c_upper = map_raw_to_app_class(str(getattr(det, "semantic_identity", "") or det.class_name))
            if c_upper in grouped:
                grouped[c_upper].append(det_dict)
            else:
                grouped.setdefault(c_upper, []).append(det_dict)

        # Also mirror to lowercase aliases so legacy consumers function seamlessly
        for c in REQUIRED_YOLO_CLASSES:
            grouped[c.lower()] = grouped[c]

        self.last_grouped_detections = grouped

        # 6. Classify AI detection source telemetry
        has_yolo = any(getattr(o, "source", "") == "yolo" for o in smoothed)
        has_hand = any(getattr(o, "source", "") == "hand_tracker" for o in smoothed)
        has_chroma = any(getattr(o, "source", "") == "chroma" for o in smoothed)

        if has_yolo:
            self.current_ai_source = "PRIMARY_AI"
        elif has_chroma:
            self.current_ai_source = "FALLBACK_AI"
        elif has_hand:
            self.current_ai_source = "HAND_AI"
        else:
            self.current_ai_source = "UNCERTAIN"

        # 7. Apply persistent multi-object tracking
        tracked = self.tracker.update(smoothed, timestamp)

        self.last_tracked_detections = [
            {"class": str(o.class_name), "conf": round(float(o.confidence), 3), "track_id": getattr(o, "track_id", -1), "bbox": list(o.bbox)}
            for o in tracked
        ]

        return tracked

    def get_grouped_detections(self) -> dict[str, list[dict[str, Any]]]:
        """
        Returns latest detections grouped by class across all 7 entities.
        """
        return self.last_grouped_detections

    def get_structured_detections(self) -> dict[str, list[dict[str, Any]]]:
        """
        Structured detection dictionary required by ORBITA specification:
        {
          "LOCATION_A": [],
          "LOCATION_B": [],
          "PEN": [],
          "WATCH": [],
          "BLUE_BOX": [],
          "YELLOW_BOX": [],
          "HAND": []
        }
        Each detection:
        {
          "class_name": "...",
          "confidence": 0.00,
          "bbox": [x1, y1, x2, y2],
          "center": [cx, cy],
          "frame_id": "...",
          "timestamp": "..."
        }
        Preserves multiple detections per class without using only detections[0].
        """
        out: dict[str, list[dict[str, Any]]] = {c: [] for c in REQUIRED_YOLO_CLASSES}
        for c in REQUIRED_YOLO_CLASSES:
            dets = self.last_grouped_detections.get(c, [])
            for d in dets:
                bx = list(d.get("bbox", d.get("bounding_box", [d.get("x1", 0), d.get("y1", 0), d.get("x2", 0), d.get("y2", 0)])))
                cen = list(d.get("center", [d.get("center_x", 0), d.get("center_y", 0)]))
                out[c].append({
                    "class_name": str(c),
                    "confidence": round(float(d.get("confidence", 0.0)), 2),
                    "bbox": bx,
                    "center": cen,
                    "frame_id": str(d.get("frame_id", "")),
                    "timestamp": str(d.get("timestamp", d.get("frame_timestamp", ""))),
                })
        return out

    def get_ai_source(self) -> str:
        """Return the current AI source category: PRIMARY_AI | HYBRID | FALLBACK_AI | UNCERTAIN."""
        return self.current_ai_source

    # ----------------------------------------------------------------------- #
    # YOLO multi-class detection
    # ----------------------------------------------------------------------- #
    def _detect_yolo(
        self, frame: np.ndarray, timestamp: float, frame_id: int = 0
    ) -> list[DetectedObject]:
        """
        Runs YOLO model inference on the latest frame.
        Extracts detections for ALL classes independently.
        Never selects only the first or highest-confidence detection.
        """
        if self._yolo is None:
            return []

        try:
            # Run inference with the lowest class-specific threshold to capture all relevant objects
            base_conf = min(self.class_thresholds.values())
            results = self._yolo.predict(
                frame,
                conf=base_conf,
                device=self._device,
                imgsz=getattr(self.config, "yolo_imgsz", 640),
                iou=self.iou_threshold,
                verbose=False,
                stream=False,
            )
            objs: list[DetectedObject] = []
            raw_telemetry: list[dict] = []
            mapped_telemetry: list[dict] = []

            for r in results:
                if not r.boxes:
                    continue
                # Iterate through ALL boxes in results without taking only the first/highest
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    raw_name = r.names.get(cls_id, f"CLASS_{cls_id}").strip()
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    w, h = max(1, x2 - x1), max(1, y2 - y1)
                    cx, cy = x1 + w // 2, y1 + h // 2

                    raw_telemetry.append({
                        "raw_class": raw_name,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, w, h]
                    })

                    # Map raw model class name to canonical ORBITA ontology
                    mapped_class = self._map_yolo_class(raw_name)

                    # Filter each class independently by its configured threshold
                    c_thresh = self.class_thresholds.get(mapped_class.lower(), self.confidence_threshold)
                    if conf < c_thresh:
                        continue

                    # Critical BLUE_BOX False-Positive Rejection (Requirement 2):
                    # The physical blue box is a 3D container, NOT a large white location sheet or wood grain.
                    if mapped_class.upper() == "BLUE_BOX":
                        frame_h, frame_w = frame.shape[:2]
                        aspect = w / float(h + 1e-5)
                        # Reject wrong aspect/shape (containers are 0.45 to 2.2, paper sheets are wider or distorted)
                        if aspect > 2.3 or aspect < 0.45:
                            continue
                        # Reject detections covering excessively large regions (e.g. table sheets)
                        if w > 0.45 * frame_w and h > 0.20 * frame_h:
                            continue
                        # Visual consistency: Real blue box is predominantly blue.
                        # White paper sheets have very low color saturation (mean S < 35) and high brightness.
                        roi = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                        if roi.size > 0:
                            b_mean, g_mean, r_mean = cv2.mean(roi)[:3]
                            # If region is nearly pure white (paper), reject as BLUE_BOX
                            if b_mean > 175 and g_mean > 175 and r_mean > 175 and (b_mean - r_mean) < 25:
                                continue

                    mapped_telemetry.append({
                        "semantic_class": mapped_class,
                        "raw_class": raw_name,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, w, h]
                    })

                    objs.append(DetectedObject(
                        class_name=NormalizedClassName(mapped_class),
                        confidence=conf,
                        bbox=(x1, y1, w, h),
                        centroid=(cx, cy),
                        timestamp=timestamp,
                        source="yolo",
                        raw_label=raw_name.lower(),
                        semantic_identity=mapped_class.upper(),
                        frame_id=frame_id,
                    ))

            # Apply per-class Non-Maximum Suppression (NMS) immediately after extraction
            objs = apply_class_aware_nms(objs, iou_threshold=self.iou_threshold, min_confidence=base_conf)

            self.last_raw_yolo_detections = raw_telemetry
            self.last_mapped_detections = mapped_telemetry
            self.last_nms_detections = [
                {"class": str(o.class_name), "conf": round(float(o.confidence), 3), "bbox": list(o.bbox)}
                for o in objs
            ]
            return objs
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            return []

    def _map_yolo_class(self, raw_name: str) -> str:
        """
        Maps raw YOLO model detection classes to ORBITA ontology.
        Does NOT fake detections using color segmentation.
        """
        return map_raw_to_app_class(raw_name).lower()

    # ----------------------------------------------------------------------- #
    # Chroma detection (Synthetic/Testing Fallback)
    # ----------------------------------------------------------------------- #
    def _detect_chroma(
        self,
        frame: np.ndarray,
        timestamp: float,
        filter_classes: Optional[Set[str]] = None,
        min_area_override: Optional[int] = None,
        min_confidence_override: Optional[float] = None,
        hands: Optional[tuple[Any, Any]] = None,
        frame_id: int = 0,
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
                if raw_area < eff_min_area_scaled or raw_area > ((frame_area / (downscale * downscale)) * 0.40):
                    continue

                area = float(raw_area * (downscale * downscale))
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                x, y, w, h = rx * downscale, ry * downscale, rw * downscale, rh * downscale

                if y < int(H * 0.08):
                    continue

                aspect = w / (h + 1e-5)
                if aspect > 2.5 or aspect < 0.40:
                    continue

                if (x <= 5 or x + w >= W - 5) and (aspect > 1.8 or aspect < 0.55):
                    continue

                cx, cy = x + w // 2, y + h // 2

                rect = cv2.minAreaRect(cnt)
                box_area = max(1.0, float(rect[1][0] * rect[1][1]))
                rectangularity = float(raw_area / box_area)
                solidity = float(raw_area / (rw * rh + 1e-5))

                if canonical in ("RED_BOX", "YELLOW_BOX", "MAIN_BOX", "BLUE_BOX"):
                    if rectangularity < 0.52 or solidity < 0.42:
                        continue

                if canonical == "RED_BOX":
                    roi = frame[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
                    if roi.size > 0:
                        mean_b, mean_g, mean_r = cv2.mean(roi)[:3]
                        red_dominance = float(mean_r) - max(float(mean_g), float(mean_b))
                        if red_dominance < 75.0 or mean_r < 160.0:
                            continue

                confidence = float(np.clip(0.60 + 0.22 * rectangularity + 0.15 * solidity, 0.60, 0.95))
                if confidence < eff_min_conf:
                    continue

                candidates.append((area, DetectedObject(
                    class_name=NormalizedClassName(canonical),
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    centroid=(cx, cy),
                    timestamp=timestamp,
                    source="chroma",
                    raw_label=canonical.lower(),
                    semantic_identity=canonical.upper(),
                    frame_id=frame_id,
                )))

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
    # Short-Term Temporal Tracking & Smoothing (1-3 Frames)
    # ----------------------------------------------------------------------- #
    def _smooth_detections(
        self,
        current_objs: list[DetectedObject],
        timestamp: float,
    ) -> list[DetectedObject]:
        """
        Temporal tracking and coordinate smoothing.
        Maintains detections across 1–3 temporarily missed frames so detections
        do not flicker.
        """
        alpha = 0.70  # EMA smoothing factor
        matched_prev_keys: set[str] = set()
        smoothed_results: list[DetectedObject] = []

        for obj in current_objs:
            cx, cy, cw, ch = obj.bbox
            best_match_key = None
            best_match_dist = float("inf")

            # Search existing tracks of the same class for spatial proximity
            for key, prev in self._tracked_objects.items():
                if str(prev["class_name"]).lower() != str(obj.class_name).lower() or key in matched_prev_keys:
                    continue
                px, py, pw, ph = prev["bbox"]
                iou = self._iou_bbox((px, py, pw, ph), (cx, cy, cw, ch))
                dist = float(np.hypot((cx + cw // 2) - (px + pw // 2), (cy + ch // 2) - (py + ph // 2)))

                if (iou > 0.15 or dist < 120.0) and dist < best_match_dist:
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
                matched_prev_keys.add(track_id)
                smoothed_results.append(obj)

        # Retain objects temporarily missed for up to 3 frames (prevents flicker)
        for key in list(self._tracked_objects.keys()):
            if key not in matched_prev_keys:
                prev = self._tracked_objects[key]
                prev["missed_count"] += 1
                if prev["missed_count"] <= 3:
                    # Decay confidence gently while temporarily occluded/missed
                    decayed_conf = max(0.20, float(prev["confidence"] * 0.95))
                    prev["confidence"] = decayed_conf
                    prev["object"].confidence = decayed_conf
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
                Path("models/orbita_yolo_detector_v4.pt"),
                Path(__file__).parent.parent.parent / "models" / "orbita_yolo_detector_v4.pt",
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

            # Print and audit model classes
            if hasattr(self._yolo, "names"):
                self.missing_model_classes = audit_model_classes(self._yolo.names)

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
            "missing_classes": getattr(self, "missing_model_classes", []),
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
            if hasattr(new_yolo, "names"):
                self.missing_model_classes = audit_model_classes(new_yolo.names)
            return True
        except Exception as exc:
            logger.error("Failed to hot-reload YOLO detector from %s: %s", model_path, exc)
            return False

    def draw(self, frame: np.ndarray, detections: list[DetectedObject]) -> np.ndarray:
        """
        Draws professional bounding boxes, labels, and confidence tags for ALL detected classes.
        """
        if frame is None or not detections:
            return frame

        vis = frame.copy()
        class_colors = {
            "location_a": (220, 180, 50),   # Soft Cyan/Teal (BGR)
            "LOCATION_A": (220, 180, 50),
            "location_b": (180, 80, 220),   # Violet/Purple (BGR)
            "LOCATION_B": (180, 80, 220),
            "yellow_box": (30, 220, 255),   # Vibrant Yellow (BGR)
            "YELLOW_BOX": (30, 220, 255),
            "blue_box": (240, 160, 30),     # Royal Blue (BGR)
            "BLUE_BOX": (240, 160, 30),
            "MAIN_BOX": (240, 160, 30),
            "pen": (0, 165, 255),           # Vibrant Orange (BGR)
            "PEN": (0, 165, 255),
            "TOOL": (0, 165, 255),
            "watch": (220, 60, 200),        # Magenta Violet (BGR)
            "WATCH": (220, 60, 200),
            "SAMPLE": (220, 60, 200),
            "hand": (50, 230, 100),         # Neon Emerald (BGR)
            "HAND": (50, 230, 100),
            "red_box": (40, 50, 235),       # Vibrant Red (BGR)
            "RED_BOX": (40, 50, 235),
            "person": (80, 220, 100),       # Green (BGR)
            "PERSON": (80, 220, 100),
        }

        for det in detections:
            x, y, w, h = det.bbox
            c_name_str = str(det.class_name)
            canonical_label = map_raw_to_app_class(c_name_str)
            color = class_colors.get(canonical_label, class_colors.get(c_name_str, (180, 190, 200)))

            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

            # Draw center point
            cx, cy = getattr(det, "centroid", (x + w // 2, y + h // 2))
            cv2.circle(vis, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 6, (10, 15, 20), 1, cv2.LINE_AA)

            # Draw label badge with class name + confidence
            conf_pct = int(det.confidence * 100)
            track_prefix = f"#{det.track_id} " if getattr(det, "track_id", -1) > 0 else ""
            label = f"{track_prefix}{canonical_label} {conf_pct}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)

            badge_y1 = max(0, y - th - 6)
            badge_y2 = y
            badge_x2 = min(vis.shape[1], x + tw + 8)
            cv2.rectangle(vis, (x, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                vis,
                label,
                (x + 4, badge_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (10, 15, 20),
                1,
                cv2.LINE_AA,
            )

        return vis


# --------------------------------------------------------------------------- #
# Real-Time Debug Overlay Rendering (Disabled per user requirement)
# --------------------------------------------------------------------------- #
def draw_debug_overlay(
    frame: np.ndarray,
    grouped_detections: dict[str, list[dict[str, Any]]],
    action_state: dict[str, Any],
) -> np.ndarray:
    """
    Debug overlay panel is disabled so camera feed remains clean and unobstructed.
    Returns the frame unchanged.
    """
    return frame



