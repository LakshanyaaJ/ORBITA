"""
ORBITA Configuration System

All tunable parameters are centralized here.
The experiment-specific step definitions live in config/experiment_config.json
and are loaded at runtime, keeping business logic decoupled from code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Camera configuration
# --------------------------------------------------------------------------- #
@dataclass
class CameraConfig:
    source: str | int = 0           # 0 = default webcam, path = video file
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip: bool = False              # Flip horizontal for mirror-mode demos
    buffer_size: int = 1
    jpeg_quality: int = 70
    drop_old_frames: bool = True
    prefer_hardware_acceleration: bool = True
    low_latency: bool = True
    ai_fps: int = 15
    sequential: bool = False        # True = synchronous frame-by-frame read without dropping


# --------------------------------------------------------------------------- #
# Object detection configuration
# --------------------------------------------------------------------------- #
@dataclass
class ZoneDefinition:
    """Configurable physical experiment zone (e.g. LOCATION_A, LOCATION_B, TABLE_ROI)."""
    name: str
    zone_type: str = "polygon"      # "polygon" | "rectangle"
    coordinates: list[Any] = field(default_factory=list)
    label: str = ""
    color: list[int] = field(default_factory=lambda: [255, 255, 255])

    def to_pixel_polygon(self, frame_w: int, frame_h: int) -> list[tuple[int, int]]:
        """Converts normalized or pixel coordinates into pixel integer vertices [(x, y), ...]."""
        if not self.coordinates:
            return []
        
        pts: list[tuple[int, int]] = []
        if self.zone_type == "rectangle" and len(self.coordinates) == 4 and not isinstance(self.coordinates[0], (list, tuple)):
            # [x1, y1, x2, y2]
            x1, y1, x2, y2 = [float(v) for v in self.coordinates]
            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                px1, py1 = int(x1 * frame_w), int(y1 * frame_h)
                px2, py2 = int(x2 * frame_w), int(y2 * frame_h)
            else:
                px1, py1, px2, py2 = int(x1), int(y1), int(x2), int(y2)
            return [(px1, py1), (px2, py1), (px2, py2), (px1, py2)]

        for item in self.coordinates:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                x, y = float(item[0]), float(item[1])
                if x <= 1.0 and y <= 1.0:
                    pts.append((int(x * frame_w), int(y * frame_h)))
                else:
                    pts.append((int(x), int(y)))
        return pts

    def contains_point(self, pt: tuple[float, float], frame_w: int, frame_h: int) -> bool:
        """Ray-casting algorithm for point-in-polygon testing."""
        poly = self.to_pixel_polygon(frame_w, frame_h)
        if len(poly) < 3:
            return False
        px, py = pt[0], pt[1]
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or px <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def bbox_overlap(self, bbox: tuple[int, int, int, int], frame_w: int, frame_h: int) -> float:
        """Returns approximate overlap or centroid containment."""
        bx, by, bw, bh = bbox
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        if self.contains_point((cx, cy), frame_w, frame_h):
            return 1.0
        # Check corners
        corners = [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)]
        contained_corners = sum(1 for c in corners if self.contains_point(c, frame_w, frame_h))
        return contained_corners / 4.0


# --------------------------------------------------------------------------- #
# Object detection configuration
# --------------------------------------------------------------------------- #
@dataclass
class DetectionConfig:
    # Object classes the system handles (7 canonical ORBITA experiment entities)
    classes: list[str] = field(default_factory=lambda: [
        "LOCATION_A", "LOCATION_B", "PEN", "WATCH",
        "BLUE_BOX", "YELLOW_BOX", "HAND"
    ])

    # HSV colour ranges for secondary color validation only
    # Format: [H_low, S_low, V_low, H_high, S_high, V_high]
    color_ranges: dict[str, list[int]] = field(default_factory=lambda: {
        "YELLOW_BOX": [18, 90, 70, 38, 255, 255],     # Yellow hue
        "BLUE_BOX":   [90, 80, 50, 135, 255, 255],    # Blue hue
    })

    yolo_model_path: str = (
        str(MODELS_DIR / "orbita_yolo_detector_v4.pt") if (MODELS_DIR / "orbita_yolo_detector_v4.pt").exists()
        else (str(MODELS_DIR / "orbita_yolo_detector_v3.pt") if (MODELS_DIR / "orbita_yolo_detector_v3.pt").exists()
        else str(MODELS_DIR / "yolov8n.pt"))
    )
    use_yolo: bool = True           # Primary neural detector
    confidence_threshold: float = 0.25
    min_area_px: int = 400          # Ignore tiny detections
    yolo_imgsz: int = 640           # 640x640 matches training resolution for small objects
    yolo_interval: int = 1          # Process YOLO every N frames (1=every frame, 2=alternate frames with tracking)
    half_precision: bool = True     # Use FP16 on CUDA
    performance_mode: str = "BALANCED"  # "QUALITY" | "BALANCED" | "LOW_LATENCY"
    allow_chroma_fallback: bool = False

    # Configurable Physical Zone ROIs (Section 2)
    zones: dict[str, ZoneDefinition] = field(default_factory=dict)

    # Temporal confirmation thresholds (Section 8)
    detection_confirm_frames: int = 3
    object_lost_frames: int = 3
    contact_confirm_frames: int = 3
    zone_confirm_frames: int = 3

    # Per-class confidence thresholds (Section 11)
    class_thresholds: dict[str, float] = field(default_factory=lambda: {
        "pen": 0.18,
        "watch": 0.20,
        "blue_box": 0.25,
        "yellow_box": 0.12,
        "hand": 0.25,
        "location_a": 0.25,
        "location_b": 0.25,
    })


# --------------------------------------------------------------------------- #
# Pose estimation configuration
# --------------------------------------------------------------------------- #
@dataclass
class PoseConfig:
    backend: str = "yolo"           # "yolo" | "mediapipe"
    yolo_model_path: str = str(MODELS_DIR / "yolov8n-pose.pt")
    confidence_threshold: float = 0.40
    normalize_to_torso: bool = True
    cadence: int = 3
    hmr_interval: int = 2           # Run HMR every N pose passes
    imgsz: int = 480


# --------------------------------------------------------------------------- #
# Hand tracking configuration
# --------------------------------------------------------------------------- #
@dataclass
class HandConfig:
    backend: str = "mediapipe"      # "mediapipe" (direct hands) | "pose" (wrist from pose)
    mediapipe_max_hands: int = 2
    mediapipe_detection_confidence: float = 0.35
    mediapipe_tracking_confidence: float = 0.35
    mediapipe_model_path: str = str(MODELS_DIR / "hand_landmarker.task")


# --------------------------------------------------------------------------- #
# Interaction configuration
# --------------------------------------------------------------------------- #
@dataclass
class InteractionConfig:
    contact_distance_px: int = 60   # Pixel distance to consider hand near object
    grasp_frames_required: int = 5  # Consecutive frames to confirm grasp
    release_frames_required: int = 4


# --------------------------------------------------------------------------- #
# Temporal HAR configuration
# --------------------------------------------------------------------------- #
@dataclass
class HARConfig:
    feature_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    window_frames: int = 30
    action_confidence_min: float = 0.55
    checkpoint_path: str = str(MODELS_DIR / "orbita_gru.pth")


# --------------------------------------------------------------------------- #
# Voice / TTS configuration
# --------------------------------------------------------------------------- #
@dataclass
class VoiceConfig:
    enabled: bool = True
    rate: int = 175                 # Words per minute
    volume: float = 0.95
    voice_id: str | None = "female" # "female", "zira", "hazel", or specific voice ID


# --------------------------------------------------------------------------- #
# Video recording / streaming configuration
# --------------------------------------------------------------------------- #
@dataclass
class VideoConfig:
    record: bool = True
    output_dir: str = str(EXPERIMENTS_DIR)
    fourcc: str = "mp4v"
    stream_port: int = 8080         # Local MJPEG stream port (separate from API)
    jpeg_quality: int = 70
    drop_old_frames: bool = True
    low_latency: bool = True


# --------------------------------------------------------------------------- #
# Backend / API configuration
# --------------------------------------------------------------------------- #
@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_dist: str = str(PROJECT_ROOT / "frontend" / "dist")
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


# --------------------------------------------------------------------------- #
# Experiment step (loaded from JSON)
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentStep:
    id: int
    action: str
    label: str
    description: str
    expected_objects: list[str]
    voice_prompt: str
    completion_voice: str
    timeout_seconds: int
    expected_action: str = ""
    expected_object: str = ""
    expected_target: str = ""
    from_location: str = ""
    to_location: str = ""


# --------------------------------------------------------------------------- #
# Master config
# --------------------------------------------------------------------------- #
@dataclass
class OrbitaConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    har: HARConfig = field(default_factory=HARConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    api: APIConfig = field(default_factory=APIConfig)
    experiment_steps: list[ExperimentStep] = field(default_factory=list)
    experiment_name: str = "Sample Box Experiment"
    experiment_id_prefix: str = "EXP"

    def set_performance_mode(self, mode: str) -> dict[str, Any]:
        """Dynamically configure pipeline trade-offs (Section 50)."""
        mode_upper = str(mode).upper()
        if mode_upper == "QUALITY":
            self.detection.yolo_imgsz = 640
            self.detection.yolo_interval = 1
            self.pose.cadence = 2
            self.pose.hmr_interval = 1
            self.detection.performance_mode = "QUALITY"
        elif mode_upper == "LOW_LATENCY":
            self.detection.yolo_imgsz = 384
            self.detection.yolo_interval = 2
            self.pose.cadence = 4
            self.pose.hmr_interval = 3
            self.detection.performance_mode = "LOW_LATENCY"
        else:
            self.detection.yolo_imgsz = 480
            self.detection.yolo_interval = 1
            self.pose.cadence = 3
            self.pose.hmr_interval = 2
            self.detection.performance_mode = "BALANCED"

        return {
            "mode": self.detection.performance_mode,
            "yolo_imgsz": self.detection.yolo_imgsz,
            "yolo_interval": self.detection.yolo_interval,
            "pose_cadence": self.pose.cadence,
            "hmr_interval": self.pose.hmr_interval,
        }


def load_config(experiment_config_path: str | None = None) -> OrbitaConfig:
    """Load master config, merging experiment steps from JSON."""
    cfg = OrbitaConfig()

    config_path = experiment_config_path or str(CONFIG_DIR / "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        cfg.experiment_name = data.get("experiment_name", cfg.experiment_name)
        cfg.experiment_id_prefix = data.get("experiment_id_prefix", cfg.experiment_id_prefix)

        # Merge thresholds
        thresholds = data.get("thresholds", {})
        if "action_confidence_min" in thresholds:
            cfg.har.action_confidence_min = thresholds["action_confidence_min"]
        if "interaction_frames_required" in thresholds:
            cfg.interaction.grasp_frames_required = thresholds["interaction_frames_required"]
        if "hand_object_contact_distance_px" in thresholds:
            cfg.interaction.contact_distance_px = thresholds["hand_object_contact_distance_px"]
        if "temporal_window_frames" in thresholds:
            cfg.har.window_frames = thresholds["temporal_window_frames"]
        if "detection_confirm_frames" in thresholds:
            cfg.detection.detection_confirm_frames = thresholds["detection_confirm_frames"]
        if "object_lost_frames" in thresholds:
            cfg.detection.object_lost_frames = thresholds["object_lost_frames"]
        if "contact_confirm_frames" in thresholds:
            cfg.detection.contact_confirm_frames = thresholds["contact_confirm_frames"]
        if "zone_confirm_frames" in thresholds:
            cfg.detection.zone_confirm_frames = thresholds["zone_confirm_frames"]
        if "class_confidence_thresholds" in thresholds:
            cfg.detection.class_thresholds.update(thresholds["class_confidence_thresholds"])

        # Load physical zone definitions
        zones_data = data.get("zones", {})
        for z_name, z_info in zones_data.items():
            if isinstance(z_info, dict):
                cfg.detection.zones[z_name.upper()] = ZoneDefinition(
                    name=z_name.upper(),
                    zone_type=z_info.get("type", "polygon"),
                    coordinates=z_info.get("coordinates", []),
                    label=z_info.get("label", z_name),
                    color=z_info.get("color", [255, 255, 255]),
                )

        # Load experiment steps
        cfg.experiment_steps = [
            ExperimentStep(
                id=s["id"],
                action=s["action"],
                label=s["label"],
                description=s["description"],
                expected_objects=s.get("expected_objects", []),
                voice_prompt=s["voice_prompt"],
                completion_voice=s.get("completion_voice", ""),
                timeout_seconds=s.get("timeout_seconds", 30),
                expected_action=s.get("expected_action", ""),
                expected_object=s.get("expected_object", ""),
                expected_target=s.get("expected_target", ""),
                from_location=s.get("from_location", ""),
                to_location=s.get("to_location", ""),
            )
            for s in data.get("steps", [])
        ]

    return cfg
