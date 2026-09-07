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
class DetectionConfig:
    # Object classes the system handles (must match simulator & YOLO labels)
    classes: list[str] = field(default_factory=lambda: [
        "PERSON", "MAIN_BOX", "RED_BOX", "YELLOW_BOX",
        "SAMPLE", "TOOL", "CHAMBER"
    ])

    # HSV colour ranges for the fallback chroma detector
    # Format: [H_low, S_low, V_low, H_high, S_high, V_high]
    color_ranges: dict[str, list[int]] = field(default_factory=lambda: {
        "RED_BOX":    [0, 120, 80, 10, 255, 255],     # Red hue
        "YELLOW_BOX": [20, 120, 80, 35, 255, 255],    # Yellow hue
        "MAIN_BOX":   [100, 30, 30, 140, 255, 200],   # Blue-grey
        "SAMPLE":     [60, 80, 80, 90, 255, 255],      # Green
    })

    yolo_model_path: str = str(MODELS_DIR / "orbita_yolo_detector_v3.pt") if (MODELS_DIR / "orbita_yolo_detector_v3.pt").exists() else str(MODELS_DIR / "yolov8n.pt")
    use_yolo: bool = True           # Falls back to chroma-only if False or model absent
    confidence_threshold: float = 0.35
    min_area_px: int = 400          # Ignore tiny detections
    yolo_imgsz: int = 480


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
    voice_id: str | None = None     # None = OS default; set to a specific voice ID


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
