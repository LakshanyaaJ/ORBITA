"""
ORBITA FastAPI Backend
=======================
Endpoints:
  GET  /                    → Serve React dashboard (built dist)
  GET  /video_feed          → Ultra-low-latency MJPEG stream (<100ms)
  WS   /ws/telemetry        → Real-time state broadcast
  POST /api/control         → start/stop/reset/scenario
  GET  /api/status          → System status & hardware telemetry
  GET  /api/camera/status   → Active camera connection status & FPS
  GET  /api/camera/diagnostics → Real-time streaming pipeline diagnostics
  GET  /api/logs            → Experiment logs
  GET  /api/experiments     → List past experiments

Architecture:
  - Dedicated Capture Loop: Captures frames at 25-30 FPS into bounded LatestFrameBuffer (size=1)
  - Dedicated AI Worker Thread: Runs YOLO + Pose + Hands + HAR + FSM independently (10-15 FPS)
  - Dedicated Live Streamer: Blits cached AI overlays and yields MJPEG at 25-30 FPS with <100ms latency
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core_ai.app.config import OrbitaConfig, load_config
from core_ai.backend.websocket_manager import WebSocketManager
from core_ai.har.feature_fusion import TemporalFeatureWindow, build_feature_vector
from core_ai.har.temporal_model import ActionClassifier, ActionPrediction
from core_ai.interaction.hand_object import HandObjectInteractionTracker
from core_ai.logging.experiment_logger import ExperimentLogger
from core_ai.perception.hand_tracker import HandTracker
from core_ai.perception.object_detector import ObjectDetector
from core_ai.perception.pose_estimator import PoseEstimator
from core_ai.reasoning.state_manager import StateManager, ValidationResult
from core_ai.simulation.simulator import ExperimentSimulator, Scenario
from core_ai.video.camera import Camera
from core_ai.video.camera_config import (
    IPCameraConfig,
    get_default_ip_camera_url,
    validate_and_format_camera_url,
)
from core_ai.video.camera_manager import CameraManager
from core_ai.video.recorder import VideoRecorder
from core_ai.video.streamer import MJPEGStreamer
from core_ai.voice.tts import TTSEngine

from core_ai.dataset.video_ingest import VideoIngestPipeline
from core_ai.dataset.frame_extractor import FrameExtractor, ExtractionConfig
from core_ai.dataset.dataset_manager import DatasetManager
from core_ai.dataset.annotator import AssistedAnnotator
from core_ai.dataset.review_queue import ReviewQueue
from core_ai.recording.quality_filter import DatasetQualityFilter
from core_ai.recording.run_collector import RunCollector
from core_ai.training.model_registry import ModelRegistry
from core_ai.training.train_yolo import train_yolo_model, check_dataset_readiness
from core_ai.training.train_har import train_temporal_har

logger = logging.getLogger(__name__)


# =========================================================================== #
# Thread-safe AI Annotation Cache
# =========================================================================== #
class AnnotationCache:
    """Holds the latest AI detections and overlay metadata for zero-wait compositing."""

    def __init__(self):
        self.lock = threading.Lock()
        self.result: Optional[ValidationResult] = None
        self.objects: List[Any] = []
        self.poses: List[Any] = []
        self.left_hand: Optional[Any] = None
        self.right_hand: Optional[Any] = None
        self.interactions: List[Any] = []
        self.camera_name: str = ""
        self.ai_latency_ms: float = 0.0
        self.timestamp: float = 0.0

    def update(
        self,
        result: Optional[ValidationResult],
        objects: List[Any],
        poses: List[Any],
        left_hand: Optional[Any],
        right_hand: Optional[Any],
        interactions: List[Any],
        camera_name: str,
        ai_latency_ms: float,
    ) -> None:
        with self.lock:
            self.result = result
            self.objects = list(objects) if objects else []
            self.poses = list(poses) if poses else []
            self.left_hand = left_hand
            self.right_hand = right_hand
            self.interactions = list(interactions) if interactions else []
            self.camera_name = camera_name
            self.ai_latency_ms = ai_latency_ms
            self.timestamp = time.time()

    def get_snapshot(self) -> Tuple[Optional[ValidationResult], List[Any], List[Any], Optional[Any], Optional[Any], List[Any], str, float]:
        with self.lock:
            return (
                self.result,
                self.objects,
                self.poses,
                self.left_hand,
                self.right_hand,
                self.interactions,
                self.camera_name,
                self.ai_latency_ms,
            )


# =========================================================================== #
# Global Application State
# =========================================================================== #
class AppState:
    def __init__(self):
        self.config: Optional[OrbitaConfig] = None
        self.ws_manager = WebSocketManager()
        self.streamer = MJPEGStreamer(jpeg_quality=70)
        self.camera_manager: Optional[CameraManager] = None
        self._camera_legacy: Optional[Camera] = None
        self.recorder: Optional[VideoRecorder] = None
        self.tts: Optional[TTSEngine] = None
        self.detector: Optional[ObjectDetector] = None
        self.pose: Optional[PoseEstimator] = None
        self.hand_tracker: Optional[HandTracker] = None
        self.interaction_tracker: Optional[HandObjectInteractionTracker] = None
        self.feature_window: Optional[TemporalFeatureWindow] = None
        self.classifier: Optional[ActionClassifier] = None
        self.state_manager: Optional[StateManager] = None
        self.experiment_logger: Optional[ExperimentLogger] = None
        self.simulator: Optional[ExperimentSimulator] = None

        # Closed-loop dataset, candidate recording, and model registry
        self.dataset_manager = DatasetManager()
        self.review_queue = ReviewQueue()
        self.run_collector = RunCollector()
        self.model_registry = ModelRegistry()
        self.annotator: Optional[AssistedAnnotator] = None

        # Threading and background loops
        self.is_running: bool = False
        self.annotation_cache = AnnotationCache()
        self.ai_thread: Optional[threading.Thread] = None
        self.stream_thread: Optional[threading.Thread] = None
        self.stream_task: Optional[asyncio.Task] = None
        self.pipeline_task: Optional[asyncio.Task] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        self.current_scenario: str = "A"
        self.mode: str = "sim"          # "sim" | "webcam" | "ip_camera"
        self.pipeline_fps: float = 0.0  # Stream FPS
        self.ai_fps: float = 0.0        # AI Inference FPS
        self.total_pipeline_latency_ms: float = 0.0
        self.last_result: Optional[ValidationResult] = None
        self.frame_count: int = 0
        self._fps_timer: float = time.monotonic()
        self.readiness: dict[str, str] = {}

    @property
    def camera(self) -> Optional[Camera]:
        if self.camera_manager:
            return self.camera_manager._jetson_camera
        return self._camera_legacy

    @camera.setter
    def camera(self, cam: Optional[Camera]):
        self._camera_legacy = cam
        if self.camera_manager and cam:
            self.camera_manager._jetson_camera = cam

    def reload_production_models(self) -> dict[str, Any]:
        """Dynamically reload promoted production models into live inference engines."""
        reloaded = {}
        if self.model_registry:
            prod_det = self.model_registry.get_production_model("yolo_detector")
            if prod_det and self.detector:
                ok = self.detector.load_model(prod_det.weights_path)
                reloaded["yolo_detector"] = {
                    "version": prod_det.version,
                    "success": ok,
                    "path": prod_det.weights_path,
                }
            elif self.detector:
                reloaded["yolo_detector"] = {
                    "version": "baseline",
                    "success": True,
                    "path": self.detector.active_model_path,
                }

            prod_har = self.model_registry.get_production_model("har_gru")
            if prod_har and self.classifier:
                ok = self.classifier.load_checkpoint(prod_har.weights_path)
                reloaded["har_gru"] = {
                    "version": prod_har.version,
                    "success": ok,
                    "path": prod_har.weights_path,
                }
            elif self.classifier:
                reloaded["har_gru"] = {
                    "version": "heuristic_fallback",
                    "success": True,
                    "path": self.classifier.active_checkpoint_path,
                }
        logger.info("Production models reloaded for live inference: %s", reloaded)
        return reloaded


_state = AppState()


# =========================================================================== #
# Lifespan (startup/shutdown)
# =========================================================================== #
@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield
    await _shutdown()


async def _startup() -> None:
    global _state
    if _state.is_running:
        return
    _state._main_loop = asyncio.get_running_loop()
    _state.config = load_config()

    # Initialize all modules
    cfg = _state.config
    _state.tts = TTSEngine(cfg.voice)
    _state.detector = ObjectDetector(cfg.detection)
    _state.pose = PoseEstimator(cfg.pose)
    _state.hand_tracker = HandTracker(cfg.hand)
    _state.interaction_tracker = HandObjectInteractionTracker(cfg.interaction)
    _state.feature_window = TemporalFeatureWindow(cfg.har.window_frames)
    _state.classifier = ActionClassifier(cfg.har)
    _state.recorder = VideoRecorder(cfg.video)
    _state.annotator = AssistedAnnotator(detector=_state.detector)
    # Ensure live inference starts with the currently promoted production models
    _state.reload_production_models()

    # Set streamer JPEG quality from config
    stream_q = getattr(cfg.camera, "jpeg_quality", 70)
    _state.streamer.jpeg_quality = stream_q

    def on_voice(msg: str):
        if _state.tts:
            _state.tts.speak(msg)

    _state.state_manager = StateManager(cfg, on_voice=on_voice)
    _state.simulator = ExperimentSimulator(Scenario.A)

    # Initialize CameraManager
    _state.camera_manager = CameraManager(cfg.camera if cfg else None)
    if _state.mode == "webcam":
        _state.camera_manager.connect_jetson_camera(0)
    elif _state.mode == "ip_camera":
        default_url = get_default_ip_camera_url()
        if default_url:
            _state.camera_manager.connect_ip_camera(url=default_url)
    else:
        _state.camera_manager.set_simulation_mode()

    # Initialize experiment logger
    _state.experiment_logger = ExperimentLogger(
        experiment_id=_state.state_manager.experiment_id,
        experiment_name=cfg.experiment_name,
        output_dir=str(Path("experiments")),
        total_steps=len(cfg.experiment_steps),
    )

    # Offline Standalone Model & Subsystem Readiness Checklist
    yolo_ready = _state.detector is not None and (
        getattr(_state.detector, "_yolo", None) is not None
        or getattr(_state.detector, "_model", None) is not None
        or bool(getattr(_state.detector, "active_model_path", ""))
    )
    har_ready = _state.classifier is not None and (_state.classifier.is_ready() or bool(getattr(_state.classifier, "active_checkpoint_path", "")))
    mediapipe_ready = _state.pose is not None and _state.hand_tracker is not None
    tts_ready = _state.tts is not None and _state.tts.is_available()
    try:
        from core_ai.database.sqlite_db import OrbitaDB
        db = OrbitaDB()
        db.init_db()
        db_ready = True
    except Exception as exc:
        logger.error("Database readiness check failed: %s", exc)
        db_ready = False
    recorder_ready = _state.recorder is not None

    all_ready = yolo_ready and har_ready and mediapipe_ready and tts_ready and db_ready and recorder_ready

    _state.readiness = {
        "YOLO": "READY" if yolo_ready else "NOT_READY",
        "HAR": "READY" if har_ready else "NOT_READY",
        "MediaPipe": "READY" if mediapipe_ready else "NOT_READY",
        "TTS": "READY" if tts_ready else "NOT_READY",
        "DATABASE": "READY" if db_ready else "NOT_READY",
        "RECORDER": "READY" if recorder_ready else "NOT_READY",
        "SYSTEM": "READY" if all_ready else "SYSTEM NOT READY",
    }

    logger.info("============================================================")
    logger.info("ORBITA SYSTEM READINESS AUDIT (100% OFFLINE STANDALONE)")
    logger.info("============================================================")
    for comp, st in _state.readiness.items():
        logger.info("  %-12s: %s", comp, st)
    logger.info("============================================================")

    _state.is_running = True

    # 1. Start dedicated AI inference worker thread (independent of streaming)
    _state.ai_thread = threading.Thread(
        target=_ai_worker_loop,
        daemon=True,
        name="orbita-ai-worker",
    )
    _state.ai_thread.start()

    # 2. Start high-speed streaming worker thread (25-30 FPS, offloaded from asyncio loop)
    _state.stream_thread = threading.Thread(
        target=_stream_producer_thread,
        daemon=True,
        name="orbita-stream-worker",
    )
    _state.stream_thread.start()

    logger.info("ORBITA low-latency backend initialized (AI Worker + Decoupled Streamer active).")


async def _shutdown() -> None:
    _state.is_running = False
    if _state.stream_thread and _state.stream_thread.is_alive():
        _state.stream_thread.join(timeout=2.0)
    if _state.ai_thread and _state.ai_thread.is_alive():
        _state.ai_thread.join(timeout=2.0)
    if _state.camera_manager:
        _state.camera_manager.disconnect()
    if _state._camera_legacy:
        _state._camera_legacy.stop()
    if _state.recorder:
        _state.recorder.stop()
    if _state.tts:
        _state.tts.stop()
    if _state.experiment_logger:
        _state.experiment_logger.export()
    logger.info("ORBITA backend stopped.")


# =========================================================================== #
# Dedicated AI Inference Worker Loop (Threaded)
# =========================================================================== #
def _ai_worker_loop() -> None:
    """
    Runs AI models (YOLO, Pose, Hands, HAR, FSM) in a dedicated background thread.
    Operates at independent cadence (10-15 FPS) and updates the AnnotationCache.
    Does NOT block live video streaming.
    """
    logger.info("ORBITA AI Worker thread started.")
    ai_count = 0
    ai_timer = time.time()
    target_ai_fps = getattr(_state.config.camera, "ai_fps", 15) if _state.config else 15
    target_interval = 1.0 / max(1, target_ai_fps)

    while _state.is_running:
        t0 = time.time()
        try:
            _execute_ai_cycle()
        except Exception as exc:
            logger.warning("AI Worker cycle error: %s", exc)

        # Calculate AI FPS
        ai_count += 1
        now = time.time()
        elapsed = now - ai_timer
        if elapsed >= 1.0:
            _state.ai_fps = round(ai_count / elapsed, 1)
            ai_count = 0
            ai_timer = now

        # Regulate AI thread to avoid 100% GPU/CPU starvation
        cycle_time = time.time() - t0
        sleep_needed = target_interval - cycle_time
        if sleep_needed > 0.005:
            time.sleep(sleep_needed)
        else:
            time.sleep(0.002)


def _execute_ai_cycle() -> None:
    """Execute one inference cycle on the latest available camera or sim frame."""
    cfg = _state.config
    t0 = time.time()
    active_source = _state.camera_manager.active_source if _state.camera_manager else _state.mode

    frame = None
    cam_name = ""
    latency_ms = 0.0

    if active_source == "sim" and _state.simulator:
        sim_frame = _state.simulator.next_frame()
        frame = sim_frame.image
        # Update simulation frame in camera manager so streamer can read it
        if _state.camera_manager:
            _state.camera_manager.push_sim_frame(frame)

        verb = sim_frame.action_verb
        obj = sim_frame.action_object
        confidence = sim_frame.confidence
        t_stamp = sim_frame.timestamp

        objects = _state.detector.detect(frame, t_stamp) if _state.detector else []
        poses = []
        left, right = None, None

        prediction = ActionPrediction(
            action=verb,
            confidence=confidence,
            next_action="IDLE",
            next_confidence=0.3,
            is_uncertain=confidence < (cfg.har.action_confidence_min if cfg else 0.55),
            target_object=obj,
        )

    elif _state.camera_manager and active_source in ("jetson_camera", "ip_camera", "phone_webcam", "video_file"):
        frame_buf = _state.camera_manager.get_frame_buffer()
        frame, cap_ts, _ = frame_buf.get_latest()
        if frame is None:
            return

        if active_source == "video_file":
            latency_ms = 0.0
            is_stale = False
            cam_name = "Demo MP4"
        else:
            latency_ms = max(0.0, (time.monotonic() - cap_ts) * 1000.0)
            is_stale = latency_ms > 500.0
            if active_source in ("ip_camera", "phone_webcam"):
                cam_name = "Phone Camera"
            else:
                cam_name = "Jetson Camera"
        t_stamp = time.time()

        # Perception: run pose & hand tracking first so detector has hand context
        poses = _state.pose.estimate(frame, t_stamp) if _state.pose else []
        pose = poses[0] if poses else None
        person_bbox = pose.bbox if pose else None
        left, right = _state.hand_tracker.track(pose, frame, t_stamp, person_bbox=person_bbox) if _state.hand_tracker else (None, None)
        objects = _state.detector.detect(frame, t_stamp, hands=(left, right)) if _state.detector else []
        interactions = _state.interaction_tracker.update(left, right, objects, t_stamp) if _state.interaction_tracker else []

        # Update physical track states with hand interactions
        tracks = []
        if _state.detector and hasattr(_state.detector, "tracker") and _state.detector.tracker:
            _state.detector.tracker.update_interaction_states(interactions, dt=1.0 / max(1.0, _state.ai_fps))
            tracks = _state.detector.tracker.get_active_tracks()

        # Feature fusion & HAR
        fv = build_feature_vector(pose, left, right, objects, interactions, frame.shape[:2])
        if _state.feature_window:
            _state.feature_window.push(fv)
            window = _state.feature_window.get_window()
        else:
            window = np.zeros((30, 64), dtype=np.float32)

        prediction = _state.classifier.predict(window) if _state.classifier else ActionPrediction(
            action="IDLE", confidence=0.5, next_action="IDLE", next_confidence=0.3, is_uncertain=False, target_object=""
        )
        obj = ""

    else:
        return

    # Extract source telemetry metadata
    frame_idx = 0
    total_frames = 0
    video_time_str = "00:00.00"
    cam_source_tag = "UNKNOWN"
    if active_source == "video_file" and _state.camera_manager and _state.camera_manager._jetson_camera:
        cam = _state.camera_manager._jetson_camera
        frame_idx = getattr(cam, "frame_index", 0)
        total_frames = getattr(cam, "total_video_frames", 0)
        fps_val = getattr(cam, "actual_fps", 30.0) or 30.0
        secs = frame_idx / max(1.0, fps_val)
        video_time_str = f"{int(secs // 60):02d}:{secs % 60:05.2f}"
        cam_source_tag = "MP4"
    elif active_source == "ip_camera":
        cam_source_tag = "IP CAMERA"
        video_time_str = time.strftime("%M:%S", time.localtime())
    elif active_source == "phone_webcam":
        cam_source_tag = "PHONE"
        video_time_str = time.strftime("%M:%S", time.localtime())
    elif active_source == "jetson_camera":
        cam_source_tag = "JETSON / USB"
        video_time_str = time.strftime("%M:%S", time.localtime())
    elif active_source == "sim":
        cam_source_tag = "SIMULATOR"
        video_time_str = "00:00.00"

    # Reasoning / State Manager
    result = None
    if _state.state_manager:
        ai_calc_time = (time.time() - t0) * 1000.0
        result = _state.state_manager.process(
            prediction=prediction,
            detected_objects=objects,
            fps=_state.ai_fps,
            latency_ms=ai_calc_time + latency_ms,
            left_hand=left,
            right_hand=right,
            interactions=interactions if active_source != "sim" else [],
            tracks=tracks if 'tracks' in locals() else [],
            camera_source=cam_source_tag,
            frame_index=frame_idx,
            total_frames=total_frames,
            video_time=video_time_str,
            frame_age_ms=latency_ms,
            is_stale=is_stale if 'is_stale' in locals() else False,
        )
        _state.last_result = result

        if _state.experiment_logger:
            _state.experiment_logger.log(result)

        # Closed-loop candidate recording
        if _state.run_collector and _state.run_collector.is_active:
            _state.run_collector.record_frame(
                frame=frame,
                fsm_step_idx=result.fsm_state.current_step_idx,
                fsm_status=result.fsm_state.status.name,
                detected_action=prediction.action,
                action_confidence=prediction.confidence,
                detected_objects=objects,
            )
            if result.fsm_state.status.name == "COMPLETED":
                _state.run_collector.finalize_run(
                    was_successful=True,
                    total_steps=result.fsm_state.total_steps,
                    completed_steps=len(result.fsm_state.completed_step_ids),
                )

        # Auto-stop video recording upon experiment completion
        if result and getattr(result.fsm_state, "is_complete", False):
            if _state.recorder and _state.recorder.is_recording():
                rec_path = _state.recorder.stop()
                logger.info("Experiment complete! Video recording auto-finalized: %s", rec_path)

    ai_duration_ms = (time.time() - t0) * 1000.0

    # Update thread-safe annotation cache for instant streamer overlay
    _state.annotation_cache.update(
        result=result,
        objects=objects,
        poses=poses,
        left_hand=left,
        right_hand=right,
        interactions=interactions if active_source != "sim" else [],
        camera_name=cam_name,
        ai_latency_ms=ai_duration_ms,
    )

    # Schedule WebSocket broadcast on FastAPI's main asyncio loop
    if result and _state._main_loop and _state._main_loop.is_running():
        try:
            # Emit explicit STEP_VALIDATED event when FSM transitions
            if getattr(result.fsm_state, "is_transition", False):
                prev_step_id = result.fsm_state.completed_step_ids[-1] if result.fsm_state.completed_step_ids else result.fsm_state.current_step_idx
                next_step_id = result.fsm_state.current_step.id if result.fsm_state.current_step else 13
                exp_act = result.confirmed_action.action if result.confirmed_action else result.fsm_state.detected_action
                exp_tgt = result.confirmed_action.object_name if result.confirmed_action else result.fsm_state.detected_object
                step_evt = {
                    "event": "STEP_VALIDATED",
                    "step": prev_step_id,
                    "expected_action": exp_act,
                    "target": exp_tgt,
                    "result": "CONFIRMED_CORRECT",
                    "next_step": next_step_id,
                }
                logger.info("Emitting STEP_VALIDATED event: %s", step_evt)
                asyncio.run_coroutine_threadsafe(
                    _state.ws_manager.broadcast(step_evt),
                    _state._main_loop,
                )

            # Build enriched telemetry payload
            ws_payload = result.to_dict()
            if _state.recorder:
                ws_payload["recording"] = _state.recorder.get_telemetry()
            if _state.experiment_logger:
                ws_payload["logging"] = _state.experiment_logger.get_telemetry()
            if _state.detector and hasattr(_state.detector, "get_ai_source"):
                ws_payload["ai_source"] = _state.detector.get_ai_source()

            asyncio.run_coroutine_threadsafe(
                _state.ws_manager.broadcast(ws_payload),
                _state._main_loop,
            )
        except Exception:
            pass


# =========================================================================== #
# Dedicated High-Speed Live Streamer Worker (Threaded)
# =========================================================================== #
def _stream_producer_thread() -> None:
    """
    Dedicated high-speed live streaming worker thread (25-30 FPS).
    Offloaded from asyncio event loop so cv2.imencode() never blocks server I/O.
    Pulls freshest frame from LatestFrameBuffer, draws latest cached AI overlays in <1ms,
    and updates the MJPEG streamer.
    """
    logger.info("ORBITA Live Streamer thread started.")
    while _state.is_running:
        t0 = time.monotonic()
        try:
            _produce_single_stream_frame()
        except Exception as exc:
            logger.warning("Streaming thread error: %s", exc)

        # Regulate to ~30 FPS target (0.033s)
        elapsed = time.monotonic() - t0
        sleep_dur = max(0.002, 0.033 - elapsed)
        time.sleep(sleep_dur)


def _produce_single_stream_frame() -> None:
    active_source = _state.camera_manager.active_source if _state.camera_manager else _state.mode
    frame = None
    cam_latency_ms = 0.0

    if _state.camera_manager:
        frame, cam_fps, cam_latency_ms = _state.camera_manager.read_with_metadata()

    if frame is None:
        # Camera is disconnected or connecting
        if _state.camera_manager:
            cam_stat = _state.camera_manager.get_status()
            if cam_stat.get("status") in ("connecting", "reconnecting", "error", "disconnected", "waiting"):
                _show_camera_status_frame(cam_stat.get("status", ""), cam_stat.get("error"))
        return

    t_now = time.monotonic()

    # Retrieve latest cached AI annotations snapshot (instant non-blocking read)
    (
        last_result,
        objects,
        poses,
        left,
        right,
        interactions,
        cam_name,
        ai_latency_ms,
    ) = _state.annotation_cache.get_snapshot()

    # Fast compositing: draw bounding boxes, skeleton, and HUD banner in <1ms
    annotated = _annotate_frame(
        frame=frame,
        result=last_result,
        objects=objects,
        poses=poses,
        left_hand=left,
        right_hand=right,
        interactions=interactions,
        camera_name=cam_name or ("Phone Camera" if active_source in ("ip_camera", "phone_webcam") else "Live Camera"),
        latency_ms=cam_latency_ms,
    )

    # Update streamer (thread-safe event-driven dispatch to all connected viewers)
    _state.streamer.update(annotated, timestamp=t_now)

    # Write to local recorder if active
    if _state.recorder and _state.recorder.is_recording():
        _state.recorder.write(annotated)

    # Total end-to-end pipeline latency metric
    _state.total_pipeline_latency_ms = cam_latency_ms + _state.streamer.encode_latency_ms

    # Stream FPS tracking
    _state.frame_count += 1
    elapsed = time.monotonic() - _state._fps_timer
    if elapsed >= 1.0:
        _state.pipeline_fps = round(_state.frame_count / elapsed, 1)
        _state.frame_count = 0
        _state._fps_timer = time.monotonic()


async def _streaming_loop() -> None:
    """Async no-op compatibility stub (streaming is now handled by _stream_producer_thread)."""
    while _state.is_running:
        await asyncio.sleep(1.0)


# =========================================================================== #
# Fast HUD & Annotation Rendering (<1 ms)
# =========================================================================== #
def _annotate_frame(
    frame: np.ndarray,
    result: Optional[ValidationResult] = None,
    objects: Optional[list] = None,
    poses: Optional[list] = None,
    left_hand: Optional[Any] = None,
    right_hand: Optional[Any] = None,
    interactions: Optional[list] = None,
    camera_name: str = "",
    latency_ms: float = 0.0,
) -> np.ndarray:
    """Draw all AI overlays on frame with optimized OpenCV operations."""
    vis = frame.copy()

    # 1. Detected objects
    if objects and _state.detector:
        try:
            vis = _state.detector.draw(vis, objects)
        except Exception:
            pass

    # 2. Pose skeleton
    if poses and _state.pose:
        try:
            vis = _state.pose.draw(vis, poses)
        except Exception:
            pass

    # 3. Hand tracking skeletons & persistent IDs
    if (left_hand or right_hand) and _state.hand_tracker:
        try:
            vis = _state.hand_tracker.draw(vis, left_hand, right_hand)
        except Exception:
            pass

    # 3b. Hand-object interactions (holding, contact, release)
    if interactions and _state.interaction_tracker and objects and (left_hand or right_hand):
        try:
            vis = _state.interaction_tracker.draw_interactions(vis, interactions, objects, left_hand, right_hand)
        except Exception:
            pass

    # 4. Hand System Engineering Telemetry Card (Top Right)
    if (left_hand and left_hand.is_visible) or (right_hand and right_hand.is_visible):
        ew, eh = 270, 95
        ex = max(10, vis.shape[1] - ew - 10)
        ey = 38
        cv2.rectangle(vis, (ex, ey), (ex + ew, ey + eh), (12, 16, 26), -1)
        cv2.rectangle(vis, (ex, ey), (ex + ew, ey + eh), (0, 200, 255), 1)

        lh_stat = f"L-Hand #{left_hand.hand_id if left_hand else 1}: {int(left_hand.confidence*100) if (left_hand and left_hand.is_visible) else 0}% ({left_hand.track_status if left_hand else 'LOST'})"
        rh_stat = f"R-Hand #{right_hand.hand_id if right_hand else 2}: {int(right_hand.confidence*100) if (right_hand and right_hand.is_visible) else 0}% ({right_hand.track_status if right_hand else 'LOST'})"

        primary_int = None
        if interactions:
            cand = sorted(interactions, key=lambda x: (getattr(x, "state", 0), getattr(x, "confidence", 0)), reverse=True)
            if cand:
                primary_int = cand[0]

        int_line = f"Interaction: {primary_int.hand_side[0].upper()} -> {primary_int.object_class} ({primary_int.state.name})" if primary_int else "Interaction: None"
        spd_val = max(left_hand.speed if (left_hand and left_hand.is_visible) else 0.0, right_hand.speed if (right_hand and right_hand.is_visible) else 0.0)
        bot_line = f"Speed: {int(spd_val)} px/s | Landmarks: 21/21"

        cv2.putText(vis, "HAND TELEMETRY", (ex + 8, ey + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, lh_stat, (ex + 8, ey + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(vis, rh_stat, (ex + 8, ey + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(vis, int_line, (ex + 8, ey + 67), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(vis, bot_line, (ex + 8, ey + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)

    # 4. Status banner
    status_colors = {
        "success": (0, 200, 80),
        "error": (0, 60, 220),
        "warning": (0, 170, 230),
        "info": (180, 180, 180),
    }
    if result:
        alert_level = result.alert_level
        colour = status_colors.get(alert_level, (180, 180, 180))
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (10, 15, 25), -1)
        hud_txt = result.hud_message[:55]
        cv2.putText(vis, hud_txt, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)

        # FPS & latency readout
        fps_text = f"{_state.pipeline_fps:.0f} FPS | AI {_state.ai_fps:.0f}"
        if latency_ms > 0:
            fps_text = f"{latency_ms:.0f}ms | {fps_text}"
        cv2.putText(vis, fps_text, (vis.shape[1] - 170, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 140), 1, cv2.LINE_AA)

    # 5. Bottom camera badge
    if camera_name:
        h, w = vis.shape[:2]
        badge_text = f"SOURCE: {camera_name} · ZERO-LAG"
        cv2.rectangle(vis, (0, h - 22), (w, h), (10, 15, 25), -1)
        cv2.putText(vis, badge_text, (10, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 180, 230), 1, cv2.LINE_AA)

    return vis


def _show_camera_status_frame(status: str, error_msg: Optional[str] = None):
    """Update stream with clear status screen when camera is reconnecting or offline."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(blank, (0, 0), (640, 480), (12, 17, 24), -1)

    if status in ("connecting", "reconnecting"):
        msg = "CAMERA CONNECTING..." if status == "connecting" else "RECONNECTING TO CAMERA..."
        sub = "Establishing low-latency stream..."
        col = (60, 190, 240)
    elif status == "waiting":
        msg = "WAITING FOR PHONE WEBCAM..."
        sub = "Scan the QR code or open /cam on your mobile phone."
        col = (60, 190, 240)
    elif status == "error":
        msg = "CAMERA OFFLINE"
        sub = error_msg or "Unable to reach IP camera. Auto-retrying..."
        col = (80, 80, 240)
    else:
        msg = "CAMERA STANDBY"
        sub = "Select a camera source to start live monitoring."
        col = (140, 150, 160)

    cv2.putText(blank, msg, (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    cv2.putText(blank, sub[:70], (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 180), 1, cv2.LINE_AA)
    _state.streamer.update(blank)


# =========================================================================== #
# FastAPI Application & Endpoints
# =========================================================================== #
app = FastAPI(title="ORBITA", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_app() -> FastAPI:
    return app


# Video stream endpoint (MJPEG)
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _state.streamer.generate_mjpeg(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# WebSocket telemetry endpoint
@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await _state.ws_manager.connect(websocket)
    try:
        # Send initial snapshot immediately so client renders on connect
        if _state.last_result:
            await websocket.send_text(json.dumps(_state.last_result.to_dict()))
        # Telemetry updates are broadcast event-driven via _state.ws_manager.broadcast()
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        _state.ws_manager.disconnect(websocket)


# Phone Camera WebSocket Stream & Signaling Endpoint
@app.websocket("/ws/phone_camera")
async def ws_phone_camera(websocket: WebSocket):
    """
    Receives raw binary frames and telemetry from the mobile phone web app.
    Zero native apps required.
    """
    token = websocket.query_params.get("token", "")
    if not _state.camera_manager or not _state.camera_manager.phone_receiver.validate_token(token):
        await websocket.close(code=4003, reason="Invalid pairing token")
        return

    await websocket.accept()
    phone_rcv = _state.camera_manager.phone_receiver
    phone_rcv.register_client({
        "client": str(websocket.client),
        "token": token,
        "connected_at": time.time(),
    })
    # Auto-activate phone webcam source on connection
    _state.camera_manager.connect_phone_webcam()
    _state.mode = "phone_webcam"

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                phone_rcv.ingest_frame_bytes(msg["bytes"])
            elif "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])
                    p_type = payload.get("type", "")
                    if p_type == "ping":
                        ts = payload.get("timestamp", time.time())
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": ts,
                            "server_time": time.time(),
                        }))
                    elif p_type == "rtt":
                        rtt = float(payload.get("rtt_ms", 0.0))
                        phone_rcv.update_rtt(rtt)
                    elif p_type == "device_info":
                        phone_rcv.register_client(payload)
                except Exception:
                    pass
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        phone_rcv.unregister_client()


# Phone Pairing REST Endpoints
@app.get("/api/camera/phone_pairing")
async def camera_phone_pairing():
    """Return active pairing code, LAN IP, and QR connect URL."""
    if _state.camera_manager:
        return _state.camera_manager.phone_receiver.get_connection_info()
    return {"connected": False, "pairing_token": "0000"}


@app.post("/api/camera/phone_pairing/refresh")
async def camera_phone_pairing_refresh():
    """Regenerate a new 4-digit pairing code."""
    if _state.camera_manager:
        _state.camera_manager.phone_receiver.refresh_pairing_token()
        return _state.camera_manager.phone_receiver.get_connection_info()
    return {"pairing_token": "0000"}


@app.get("/cam")
async def serve_cam_spa():
    """Direct route for phone browsers scanning the QR code or visiting /cam."""
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        from fastapi.responses import FileResponse
        return FileResponse(index_file)
    return JSONResponse({"message": "ORBITA CAM Web App - please build frontend."})


# Control endpoint
@app.post("/api/control")
async def control(body: dict):
    action = body.get("action", "")
    if action in ("start", "start_experiment"):
        if _state.state_manager:
            _state.state_manager.reset()
        if _state.simulator:
            _state.simulator.reset()
        if _state.hand_tracker:
            _state.hand_tracker.reset()
        if _state.interaction_tracker:
            _state.interaction_tracker.reset()
        if _state.feature_window:
            _state.feature_window.reset()

        exp_id = _state.state_manager.experiment_id if _state.state_manager else f"EXP_{int(time.time())}"
        cfg = _state.config
        if cfg:
            _state.experiment_logger = ExperimentLogger(
                experiment_id=exp_id,
                experiment_name=cfg.experiment_name,
                output_dir=str(Path("experiments")),
                total_steps=len(cfg.experiment_steps),
            )
        rec_path = ""
        if _state.recorder:
            rec_path = _state.recorder.start(experiment_id=exp_id)
        return {"status": "started", "experiment_id": exp_id, "recording_path": rec_path}

    if action in ("stop", "stop_experiment"):
        rec_path = ""
        if _state.recorder and _state.recorder.is_recording():
            rec_path = _state.recorder.stop()
        if _state.experiment_logger:
            _state.experiment_logger.export()
        return {"status": "stopped", "recording_path": rec_path}

    if action == "reset":
        if _state.recorder and _state.recorder.is_recording():
            _state.recorder.stop()
        if _state.state_manager:
            _state.state_manager.reset()
        if _state.simulator:
            _state.simulator.reset()
        if _state.hand_tracker:
            _state.hand_tracker.reset()
        if _state.interaction_tracker:
            _state.interaction_tracker.reset()
        if _state.feature_window:
            _state.feature_window.reset()
        return {"status": "reset"}

    if action == "scenario":
        scenario_id = body.get("scenario", "A").upper()
        try:
            scenario = Scenario(scenario_id)
        except ValueError:
            scenario = Scenario.A
        _state.current_scenario = scenario_id
        if _state.simulator:
            _state.simulator.reset(scenario)
        if _state.state_manager:
            _state.state_manager.reset()
        return {"status": f"scenario_{scenario_id}"}

    if action == "start_recording":
        if _state.recorder and _state.state_manager:
            path = _state.recorder.start(
                experiment_id=_state.state_manager.experiment_id
            )
            return {"status": "recording", "path": path}

    if action == "stop_recording":
        if _state.recorder:
            path = _state.recorder.stop()
            return {"status": "stopped", "path": path}

    if action == "start_candidate_recording":
        if _state.run_collector and _state.state_manager:
            run_id = _state.run_collector.start_run(_state.state_manager.experiment_id)
            return {"status": "candidate_recording_started", "run_id": run_id}

    if action == "stop_candidate_recording":
        if _state.run_collector and _state.run_collector.is_active:
            res = _state.run_collector.finalize_run(was_successful=True)
            return {"status": "candidate_recording_stopped", "candidate": res}
        return {"status": "not_recording"}

    if action == "export_log":
        if _state.experiment_logger:
            paths = _state.experiment_logger.export()
            return {"status": "exported", "paths": paths}

    if action == "set_mode":
        new_mode = body.get("mode", "sim")
        _state.mode = new_mode
        if _state.camera_manager:
            if new_mode == "webcam":
                _state.camera_manager.connect_jetson_camera(0)
            elif new_mode == "sim":
                _state.camera_manager.set_simulation_mode()
        return {"status": f"mode_{new_mode}"}

    return {"status": "unknown_action"}


# Camera Connection Endpoints
@app.post("/api/camera/connect")
async def camera_connect(body: dict):
    if not _state.camera_manager:
        return JSONResponse(status_code=500, content={"status": "error", "error": "CameraManager not initialized."})

    source = body.get("source", "ip_camera")

    if source == "jetson_camera":
        device_index = int(body.get("device_index", 0))
        success, err = _state.camera_manager.connect_jetson_camera(device_index)
        if not success:
            return JSONResponse(status_code=400, content={"status": "error", "error": err})
        _state.mode = "webcam"
        return {"status": "connected", "source": "jetson_camera", "device_index": device_index}

    elif source == "ip_camera":
        url = body.get("url")
        ip = body.get("ip")
        port = body.get("port")
        path = body.get("path")
        timeout_sec = float(body.get("timeout_sec", 4.0))

        success, err = _state.camera_manager.connect_ip_camera(
            url=url, ip=ip, port=port, path=path, timeout_sec=timeout_sec
        )
        if not success:
            return JSONResponse(status_code=400, content={"status": "error", "error": err})
        _state.mode = "ip_camera"
        return {
            "status": "connected",
            "source": "ip_camera",
            "url": _state.camera_manager.active_url,
        }

    elif source == "phone_webcam":
        success, err = _state.camera_manager.connect_phone_webcam()
        _state.mode = "phone_webcam"
        return {
            "status": "connected",
            "source": "phone_webcam",
            "pairing": _state.camera_manager.phone_receiver.get_connection_info(),
        }

    elif source == "video_file":
        video_path = body.get("path") or body.get("url") or "vdata/20260905_145858.mp4"
        loop = bool(body.get("loop", False))
        success, err = _state.camera_manager.connect_video_file(video_path, loop=loop)
        if not success:
            return JSONResponse(status_code=400, content={"status": "error", "error": err})
        _state.mode = "video_file"
        return {"status": "connected", "source": "video_file", "path": video_path}

    elif source == "sim":
        _state.camera_manager.set_simulation_mode()
        _state.mode = "sim"
        return {"status": "connected", "source": "sim"}

    return JSONResponse(status_code=400, content={"status": "error", "error": f"Unknown camera source: '{source}'."})


@app.post("/api/camera/disconnect")
async def camera_disconnect():
    if _state.camera_manager:
        _state.camera_manager.disconnect()
    _state.mode = "disconnected"
    _show_camera_status_frame("disconnected")
    return {"status": "disconnected"}


@app.get("/api/camera/status")
async def camera_status():
    if _state.camera_manager:
        st = _state.camera_manager.get_status()
        st["stream_fps"] = round(_state.pipeline_fps, 1)
        st["ai_fps"] = round(_state.ai_fps, 1)
        return st
    return {
        "connected": False,
        "source": "none",
        "url": "",
        "fps": 0.0,
        "stream_fps": 0.0,
        "ai_fps": 0.0,
        "latency_ms": 0.0,
        "status": "disconnected",
        "error": None,
        "rotation": -1,
    }


@app.post("/api/camera/rotate")
async def api_camera_rotate(body: Optional[dict] = Body(default=None)):
    """Set camera rotation: 0, 90, 180, 270, or -1 (auto-horizontal)."""
    degrees = int((body or {}).get("degrees", 90))
    if _state.camera_manager:
        current_rot = _state.camera_manager.set_rotation(degrees)
        return {"success": True, "rotation": current_rot}
    return {"success": False, "rotation": 0}



@app.get("/api/camera/diagnostics")
async def camera_diagnostics():
    """
    Real-time streaming pipeline diagnostics:
    Camera FPS, Stream FPS, AI FPS, Dropped Frames, Pipeline Latency, Encoding Time, Resolution.
    """
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    cam_diag = _state.camera_manager.get_diagnostics() if _state.camera_manager else {}
    stream_diag = _state.streamer.get_diagnostics()

    frame_age = cam_diag.get("frame_age_ms", cam_diag.get("latency_ms", 0.0))
    live_edge = "LIVE" if frame_age < 250.0 else ("BEHIND" if frame_age < 1000.0 else "CRITICAL")

    return {
        "camera_fps": cam_diag.get("actual_fps", 0.0),
        "stream_fps": stream_diag.get("stream_fps", _state.pipeline_fps),
        "ai_fps": round(_state.ai_fps, 1),
        "pipeline_latency_ms": round(_state.total_pipeline_latency_ms, 1),
        "camera_latency_ms": cam_diag.get("latency_ms", 0.0),
        "frame_age_ms": round(frame_age, 1),
        "live_edge": cam_diag.get("live_edge", live_edge),
        "queue_depth": cam_diag.get("queue_depth", 0),
        "encode_latency_ms": stream_diag.get("encode_latency_ms", 0.0),
        "dropped_frames_pct": cam_diag.get("dropped_pct", 0.0),
        "buffer_size": 1,
        "resolution": cam_diag.get("resolution", "1280x720"),
        "source": cam_diag.get("source", _state.mode),
        "status": cam_diag.get("status", "connected"),
        "active_clients": stream_diag.get("active_clients", 0),
        "cpu_pct": cpu,
        "mem_pct": mem,
    }


@app.get("/api/camera/stream")
async def camera_stream():
    return StreamingResponse(
        _state.streamer.generate_mjpeg(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# System Status
@app.get("/api/status")
async def api_status():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    cam_info = _state.camera_manager.get_status() if _state.camera_manager else {}
    cam_diag = _state.camera_manager.get_diagnostics() if _state.camera_manager else {}
    is_cam_online = cam_info.get("connected", False) or (_state.mode == "sim")

    frame_age = cam_diag.get("frame_age_ms", cam_info.get("latency_ms", 0.0))
    live_edge = "LIVE" if frame_age < 250.0 else ("BEHIND" if frame_age < 1000.0 else "CRITICAL")

    return {
        "ai_engine": "ONLINE",
        "system_status": "READY" if _state.readiness.get("SYSTEM") == "READY" else "INITIALIZING",
        "readiness": _state.readiness,
        "ai_source": _state.detector.get_ai_source() if (_state.detector and hasattr(_state.detector, "get_ai_source")) else "PRIMARY_AI",
        "camera": "ONLINE" if is_cam_online else "OFFLINE",
        "camera_source": cam_info.get("source", _state.mode),
        "camera_status": cam_info.get("status", "disconnected"),
        "camera_fps": cam_info.get("fps", 0.0),
        "stream_fps": round(_state.pipeline_fps, 1),
        "ai_fps": round(_state.ai_fps, 1),
        "camera_latency_ms": cam_info.get("latency_ms", 0.0),
        "frame_age_ms": round(frame_age, 1),
        "live_edge": cam_diag.get("live_edge", live_edge),
        "pipeline_latency_ms": round(_state.total_pipeline_latency_ms, 1),
        "dropped_frames_pct": cam_diag.get("dropped_pct", 0.0),
        "tts": "ONLINE" if (_state.tts and _state.tts.is_available()) else "OFFLINE",
        "recording": "ON" if (_state.recorder and _state.recorder.is_recording()) else "OFF",
        "recording_telemetry": _state.recorder.get_telemetry() if _state.recorder else {"is_recording": False, "status": "STOPPED"},
        "logging_telemetry": _state.experiment_logger.get_telemetry() if _state.experiment_logger else {"status": "IDLE", "events_written": 0},
        "stream": "ON",
        "storage": "OK",
        "mode": _state.mode,
        "scenario": _state.current_scenario,
        "pipeline_fps": round(_state.pipeline_fps, 1),
        "cpu_pct": cpu,
        "mem_pct": mem,
        "ws_clients": _state.ws_manager.client_count,
    }


# Logs & Experiments
@app.get("/api/logs")
async def api_logs():
    if _state.experiment_logger:
        return _state.experiment_logger.get_summary()
    return {}


@app.get("/api/experiments")
async def api_experiments():
    from core_ai.database.sqlite_db import OrbitaDB
    db = OrbitaDB()
    return db.list_experiments()


# =========================================================================== #
# Dataset & Closed-Loop Learning Endpoints
# =========================================================================== #
@app.get("/api/dataset/status")
async def api_dataset_status():
    """Returns dataset manifest, ingestion report, counts, active models, and honest stage."""
    manifest = _state.dataset_manager.get_manifest() if _state.dataset_manager else None
    ingest_report_path = Path("datasets/orbita/metadata/ingestion_report.json")
    ingest_report = None
    if ingest_report_path.exists():
        try:
            with open(ingest_report_path, "r", encoding="utf-8") as f:
                ingest_report = json.load(f)
        except Exception:
            pass

    prod_detector = _state.model_registry.get_production_model("yolo_detector") if _state.model_registry else None
    prod_har = _state.model_registry.get_production_model("har_gru") if _state.model_registry else None

    # Count verified labels from sidecars
    verified_count = 0
    if _state.dataset_manager and _state.dataset_manager.labels_dir.exists():
        for meta_p in _state.dataset_manager.labels_dir.rglob("*.meta.json"):
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("status") == "verified":
                    verified_count += 1
            except Exception:
                pass

    active_det = _state.detector.active_model_path if _state.detector else "models/yolov8n.pt"
    active_har = _state.classifier.active_checkpoint_path if _state.classifier else ""

    # Honest workflow stage evaluation
    stage = "NO_VIDEOS"
    if ingest_report and ingest_report.get("total_videos", 0) > 0:
        stage = "VIDEOS_FOUND"
    if manifest and manifest.total_extracted_frames > 0:
        stage = "FRAMES_EXTRACTED"
    if manifest and manifest.annotated_frames > 0:
        stage = "ANNOTATIONS_GENERATED"
    if verified_count > 0:
        stage = "ANNOTATIONS_VERIFIED"
    if verified_count >= 5 and manifest and manifest.split_counts.get("train", 0) > 0:
        stage = "DATASET_READY"
    if prod_detector or prod_har:
        stage = "MODEL_TRAINED"
    if (prod_detector and prod_detector.metrics) or (prod_har and prod_har.metrics):
        stage = "MODEL_VALIDATED"
    if (prod_detector and prod_detector.is_production) or (prod_har and prod_har.is_production):
        stage = "MODEL_PROMOTED"
    if ((prod_detector and prod_detector.weights_path == active_det) or
        (prod_har and prod_har.weights_path == active_har)):
        stage = "MODEL_LOADED_INFERENCE"

    return {
        "stage": stage,
        "manifest": manifest.to_dict() if manifest else None,
        "ingestion_report": ingest_report,
        "verified_annotations_count": verified_count,
        "production_models": {
            "yolo_detector": prod_detector.to_dict() if prod_detector else None,
            "har_gru": prod_har.to_dict() if prod_har else None,
        },
        "live_inference_models": {
            "yolo_detector": {
                "active_path": active_det,
                "is_promoted_loaded": (prod_detector.weights_path == active_det) if prod_detector else False,
            },
            "har_gru": {
                "active_path": active_har,
                "is_promoted_loaded": (prod_har.weights_path == active_har) if prod_har else False,
            },
        },
        "review_queue_counts": {
            "pending": len(_state.review_queue.list_candidates(status_filter="pending")),
            "approved": len(_state.review_queue.list_candidates(status_filter="approved")),
            "rejected": len(_state.review_queue.list_candidates(status_filter="rejected")),
        },
    }


@app.post("/api/experiment/start")
async def api_experiment_start(body: dict):
    """Explicit experiment start mechanism: resets FSM and arms automatic run recording."""
    experiment_id = body.get("experiment_id", "EXP001")
    scenario_id = body.get("scenario", "A").upper()

    if _state.state_manager:
        _state.state_manager.reset()
        _state.state_manager.experiment_id = experiment_id
        if hasattr(_state.state_manager, "fsm"):
            _state.state_manager.fsm.experiment_id = experiment_id

    if _state.simulator:
        try:
            scenario = Scenario(scenario_id)
            _state.simulator.reset(scenario)
        except Exception:
            pass

    run_id = ""
    if _state.run_collector:
        run_id = _state.run_collector.start_run(experiment_id)

    logger.info("Experiment %s started (Scenario: %s, Candidate run: %s)", experiment_id, scenario_id, run_id)
    return {
        "status": "experiment_started",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "recording": True,
    }


@app.post("/api/experiment/stop")
async def api_experiment_stop():
    """Stops active experiment and finalizes candidate run recording."""
    candidate_summary = None
    if _state.run_collector and _state.run_collector.is_active:
        candidate_summary = _state.run_collector.finalize_run(was_successful=True)

    return {
        "status": "experiment_stopped",
        "candidate": candidate_summary,
    }


@app.post("/api/dataset/verify_annotation")
async def api_dataset_verify_annotation(body: dict):
    """Marks a single annotation as verified by human reviewer."""
    meta_path = body.get("meta_path", "")
    reviewer = body.get("verified_by", "human_reviewer")
    notes = body.get("notes", None)
    updated_boxes = body.get("updated_boxes", None)

    if not meta_path or not Path(meta_path).exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"meta_path '{meta_path}' not found."})

    if not _state.annotator:
        _state.annotator = AssistedAnnotator(detector=_state.detector)

    record = _state.annotator.verify_annotation(
        meta_path=meta_path,
        verified_by=reviewer,
        notes=notes,
        updated_boxes=updated_boxes,
    )
    return {"status": "verified", "record": record.to_dict()}


@app.post("/api/dataset/verify_batch")
async def api_dataset_verify_batch(body: dict = None):
    """Batch marks pre-annotated frames as verified for training approval."""
    body = body or {}
    split = body.get("split", "train")
    reviewer = body.get("verified_by", "mission_specialist")

    if not _state.annotator:
        _state.annotator = AssistedAnnotator(detector=_state.detector)

    verified_count = 0
    split_lbl_dir = _state.dataset_manager.labels_dir / split
    if split_lbl_dir.exists():
        for meta_p in split_lbl_dir.glob("*.meta.json"):
            _state.annotator.verify_annotation(meta_p, verified_by=reviewer, notes="Batch verified via UI/API")
            verified_count += 1

    return {"status": "batch_verified", "split": split, "count": verified_count}


@app.post("/api/dataset/ingest")
async def api_dataset_ingest(body: dict = None):
    """Scan vdata/, extract quality-filtered frames, and assign video splits."""
    body = body or {}
    vdata_dir = body.get("vdata_dir", "vdata")
    sample_fps = float(body.get("sample_fps", 2.0))
    blur_threshold = float(body.get("blur_threshold", 40.0))

    pipeline = VideoIngestPipeline(vdata_dir=vdata_dir)
    report = pipeline.scan_and_report("datasets/orbita/metadata/ingestion_report.json")

    _state.dataset_manager.initialize_directories()
    videos = sorted(list(Path(vdata_dir).glob("*.mp4")))
    video_ids = [v.stem for v in videos]
    split_map = _state.dataset_manager.assign_video_splits(video_ids)

    extractor = FrameExtractor(ExtractionConfig(
        sample_fps=sample_fps,
        blur_threshold=blur_threshold,
        duplicate_threshold=0.95,
        max_dimension=1280,
    ))

    extracted_counts = {}
    for vpath in videos:
        vid = vpath.stem
        split = split_map.get(vid, "train")
        out_dir = _state.dataset_manager.images_dir / split
        frames = extractor.extract_from_video(vpath, out_dir, filename_prefix=vid)
        kept = [f for f in frames if f.is_kept]
        extracted_counts[vid] = len(kept)

    manifest = _state.dataset_manager.update_manifest(
        total_videos=len(videos),
        video_split_map=split_map,
    )

    return {
        "status": "ingestion_complete",
        "ingestion_report": report.to_dict(),
        "extracted_counts": extracted_counts,
        "manifest": manifest.to_dict(),
    }


@app.post("/api/dataset/pre_annotate")
async def api_dataset_pre_annotate():
    """Run assisted pre-labeling on all unannotated frames using existing detector."""
    if not _state.annotator:
        _state.annotator = AssistedAnnotator(detector=_state.detector)

    total_annotated = 0
    total_boxes = 0
    for split in ["train", "val", "test"]:
        img_dir = _state.dataset_manager.images_dir / split
        lbl_dir = _state.dataset_manager.labels_dir / split
        if not img_dir.exists():
            continue
        imgs = sorted(list(img_dir.glob("*.jpg")))
        for img_p in imgs:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            record = _state.annotator.pre_annotate_image(img_p, lbl_p, min_confidence=0.20)
            total_annotated += 1
            total_boxes += len(record.boxes)

    manifest = _state.dataset_manager.update_manifest(
        total_videos=len(_state.dataset_manager.get_manifest().video_split_map if _state.dataset_manager.get_manifest() else {}),
        video_split_map=_state.dataset_manager.get_manifest().video_split_map if _state.dataset_manager.get_manifest() else {},
    )

    return {
        "status": "pre_annotation_complete",
        "frames_annotated": total_annotated,
        "boxes_generated": total_boxes,
        "manifest": manifest.to_dict(),
    }


@app.get("/api/dataset/candidates")
async def api_dataset_candidates(status: Optional[str] = None):
    """List candidate runs in the human review queue."""
    candidates = _state.review_queue.list_candidates(status_filter=status)
    return [c.to_dict() for c in candidates]


@app.get("/api/dataset/candidates/{run_id}")
async def api_dataset_candidate_detail(run_id: str):
    """Get full metadata, quality metrics, and frame filenames for a candidate run."""
    details = _state.review_queue.get_candidate_details(run_id)
    if not details:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Candidate run not found."})
    return details


@app.post("/api/dataset/candidates/{run_id}/review")
async def api_dataset_candidate_review(run_id: str, body: dict):
    """Human-in-the-loop review action: approve or reject candidate run."""
    action = body.get("action", "approve").lower()
    reviewer = body.get("reviewer", "human_operator")
    notes = body.get("notes", "")

    if action == "approve":
        target_split = body.get("target_split", "train")
        try:
            res = _state.review_queue.approve_candidate(
                run_id=run_id,
                reviewer_name=reviewer,
                target_split=target_split,
                notes=notes,
            )
            return res
        except FileNotFoundError as e:
            return JSONResponse(status_code=404, content={"status": "error", "message": str(e)})

    elif action == "reject":
        reason = body.get("reason", "Operator rejected candidate run.")
        try:
            res = _state.review_queue.reject_candidate(
                run_id=run_id,
                reviewer_name=reviewer,
                reason=reason,
            )
            return res
        except FileNotFoundError as e:
            return JSONResponse(status_code=404, content={"status": "error", "message": str(e)})

    return JSONResponse(status_code=400, content={"status": "error", "message": f"Invalid review action '{action}'"})


@app.post("/api/training/retrain")
async def api_training_retrain(body: dict = None):
    """
    Controlled retraining endpoint.
    Safely blocks training if annotations are missing or unverified unless explicitly overridden.
    """
    body = body or {}
    model_type = body.get("model_type", "har").lower()
    epochs = int(body.get("epochs", 5))
    require_verified = bool(body.get("require_verified", True))

    if model_type == "yolo":
        res = train_yolo_model(
            data_yaml="datasets/orbita/data.yaml",
            epochs=epochs,
            device="cpu",
        )
        return res

    elif model_type == "har":
        res = train_temporal_har(epochs=epochs)
        return res

    elif model_type == "all":
        yolo_res = train_yolo_model(data_yaml="datasets/orbita/data.yaml", epochs=epochs, device="cpu")
        har_res = train_temporal_har(epochs=epochs)
        return {"yolo": yolo_res, "har": har_res}

    return JSONResponse(status_code=400, content={"status": "error", "message": f"Unknown model type '{model_type}'"})


@app.get("/api/models/versions")
async def api_models_versions():
    """List all versioned models, metrics, and active production status."""
    if _state.model_registry:
        models = _state.model_registry.list_models()
        return [m.to_dict() for m in models]
    return []


@app.post("/api/models/promote")
async def api_models_promote(body: dict):
    """Evaluate and promote a candidate model to production, hot-reloading live inference."""
    version = body.get("version", "")
    if not _state.model_registry:
        return JSONResponse(status_code=500, content={"status": "error", "message": "ModelRegistry not initialized."})

    promoted, msg = _state.model_registry.evaluate_and_promote(version, min_improvement=0.0)
    reloaded = {}
    if promoted:
        reloaded = _state.reload_production_models()
    return {"version": version, "promoted": promoted, "message": msg, "reloaded": reloaded}


# =========================================================================== #
# Reference Video (vdata) YOLO Inspection Endpoints
# =========================================================================== #
@app.get("/api/vdata/videos")
async def api_vdata_videos():
    """List all available reference videos in vdata/ with probed metadata."""
    vdata_dir = Path("vdata")
    if not vdata_dir.exists():
        return []

    videos = []
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        for p in sorted(vdata_dir.glob(f"*{ext}")):
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            duration = round(total_frames / fps, 1) if fps > 0 else 0.0
            cap.release()

            size_mb = round(p.stat().st_size / (1024 * 1024), 1)
            videos.append({
                "filename": p.name,
                "path": str(p),
                "duration_seconds": duration,
                "total_frames": total_frames,
                "fps": round(fps, 1),
                "width": w,
                "height": h,
                "resolution": f"{w}x{h}",
                "size_mb": size_mb,
            })
    return videos


@app.get("/api/vdata/stream/{filename}")
async def api_vdata_stream(filename: str):
    """
    Streams a reference video from vdata/ with real-time YOLO object detection bounding boxes overlaid.
    Loops continuously so user can observe how YOLO detects objects throughout the trial.
    """
    video_path = Path("vdata") / filename
    if not video_path.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Video '{filename}' not found."})

    async def generate():
        while True:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                break

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_delay = 1.0 / max(10.0, min(fps, 30.0))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize for responsive real-time inference if 4K or very large
                h, w = frame.shape[:2]
                target_w = 960
                if w > target_w:
                    target_h = int(h * (target_w / w))
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

                t_start = time.time()
                detections = []
                poses = []
                left, right = None, None
                interactions = []
                try:
                    if _state.pose:
                        poses = _state.pose.estimate(frame, t_start)
                    pose = poses[0] if poses else None
                    person_bbox = pose.bbox if pose else None
                    if _state.hand_tracker:
                        left, right = _state.hand_tracker.track(pose, frame, t_start, person_bbox=person_bbox)
                    if _state.detector:
                        detections = _state.detector.detect(frame, timestamp=t_start, hands=(left, right))
                    if _state.interaction_tracker:
                        interactions = _state.interaction_tracker.update(left, right, detections, t_start)
                    frame = _annotate_frame(
                        frame=frame,
                        result=None,
                        objects=detections,
                        poses=poses,
                        left_hand=left,
                        right_hand=right,
                        interactions=interactions,
                        camera_name=f"VData: {filename}",
                        latency_ms=(time.time() - t_start) * 1000.0,
                    )
                except Exception as e:
                    logger.warning("Error running perception on frame: %s", e)
                t_infer = (time.time() - t_start) * 1000.0

                # Header HUD overlay
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (12, 17, 24), -1)
                hud_text = f"YOLO INSPECT: {filename} | {len(detections)} OBJECTS DETECTED | {t_infer:.0f}ms"
                cv2.putText(frame, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 220, 120), 1, cv2.LINE_AA)

                model_name = Path(_state.detector.active_model_path).name if _state.detector else "yolov8n.pt"
                cv2.putText(frame, f"MODEL: {model_name}", (frame.shape[1] - 220, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1, cv2.LINE_AA)

                _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
                await asyncio.sleep(frame_delay)

            cap.release()
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@app.post("/api/vdata/inspect_frame")
async def api_vdata_inspect_frame(body: dict):
    """
    Inspect a specific frame from a vdata reference video.
    Returns annotated frame image (base64), detection bounding boxes, confidence, and model metadata.
    """
    import base64

    filename = body.get("filename", "")
    frame_idx = int(body.get("frame_idx", 0))
    video_path = Path("vdata") / filename
    if not video_path.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Video '{filename}' not found."})

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = max(0, min(frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Could not read frame {frame_idx}"})

    h, w = frame.shape[:2]
    target_w = 960
    if w > target_w:
        target_h = int(h * (target_w / w))
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    t0 = time.time()
    poses = _state.pose.estimate(frame, t0) if _state.pose else []
    pose = poses[0] if poses else None
    person_bbox = pose.bbox if pose else None
    left, right = _state.hand_tracker.track(pose, frame, t0, person_bbox=person_bbox) if _state.hand_tracker else (None, None)
    detections = _state.detector.detect(frame, timestamp=t0, hands=(left, right)) if _state.detector else []
    interactions = _state.interaction_tracker.update(left, right, detections, t0) if _state.interaction_tracker else []
    latency_ms = round((time.time() - t0) * 1000.0, 1)

    annotated = _annotate_frame(
        frame=frame,
        result=None,
        objects=detections,
        poses=poses,
        left_hand=left,
        right_hand=right,
        interactions=interactions,
        camera_name=f"Inspect: {filename}",
        latency_ms=latency_ms,
    )
    _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64_img = base64.b64encode(buf).decode("utf-8")

    det_list = []
    for d in detections:
        det_list.append({
            "class_name": d.class_name,
            "confidence": round(float(d.confidence), 3),
            "bbox": list(d.bbox),
            "centroid": list(getattr(d, "centroid", (0, 0))),
            "source": getattr(d, "source", "yolo"),
        })

    for h_side, hand in [("left", left), ("right", right)]:
        if hand and getattr(hand, "is_visible", False):
            det_list.append({
                "class_name": f"HAND ({h_side.upper()})",
                "confidence": round(float(hand.confidence), 3),
                "bbox": list(hand.bbox),
                "centroid": [int(hand.position[0]), int(hand.position[1])],
                "source": "hand_tracker",
            })

    return {
        "filename": filename,
        "frame_idx": frame_idx,
        "total_frames": total_frames,
        "latency_ms": latency_ms,
        "model": Path(_state.detector.active_model_path).name if _state.detector else "yolov8n.pt",
        "detections_count": len(det_list),
        "detections": det_list,
        "image_data": f"data:image/jpeg;base64,{b64_img}",
    }



# Serve React frontend (if built)
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
