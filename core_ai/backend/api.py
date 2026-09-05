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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
        camera_name: str,
        ai_latency_ms: float,
    ) -> None:
        with self.lock:
            self.result = result
            self.objects = list(objects) if objects else []
            self.poses = list(poses) if poses else []
            self.left_hand = left_hand
            self.right_hand = right_hand
            self.camera_name = camera_name
            self.ai_latency_ms = ai_latency_ms
            self.timestamp = time.time()

    def get_snapshot(self) -> Tuple[Optional[ValidationResult], List[Any], List[Any], Optional[Any], Optional[Any], str, float]:
        with self.lock:
            return (
                self.result,
                self.objects,
                self.poses,
                self.left_hand,
                self.right_hand,
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

        # Threading and background loops
        self.is_running: bool = False
        self.annotation_cache = AnnotationCache()
        self.ai_thread: Optional[threading.Thread] = None
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
        self._fps_timer: float = time.time()

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

    _state.is_running = True

    # 1. Start dedicated AI inference worker thread (independent of streaming)
    _state.ai_thread = threading.Thread(
        target=_ai_worker_loop,
        daemon=True,
        name="orbita-ai-worker",
    )
    _state.ai_thread.start()

    # 2. Start high-speed streaming task (25-30 FPS)
    _state.stream_task = asyncio.create_task(_streaming_loop())
    _state.pipeline_task = _state.stream_task  # Maintain reference

    logger.info("ORBITA low-latency backend initialized (AI Worker + Decoupled Streamer active).")


async def _shutdown() -> None:
    _state.is_running = False
    if _state.stream_task:
        _state.stream_task.cancel()
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

    elif _state.camera_manager and active_source in ("jetson_camera", "ip_camera"):
        frame_buf = _state.camera_manager.get_frame_buffer()
        frame, cap_ts, _ = frame_buf.get_latest()
        if frame is None:
            return

        latency_ms = max(0.0, (time.time() - cap_ts) * 1000.0)
        cam_name = "Phone IP Camera" if active_source == "ip_camera" else "Jetson Camera"
        t_stamp = time.time()

        # Perception
        objects = _state.detector.detect(frame, t_stamp) if _state.detector else []
        poses = _state.pose.estimate(frame, t_stamp) if _state.pose else []
        pose = poses[0] if poses else None
        left, right = _state.hand_tracker.track(pose, frame, t_stamp) if _state.hand_tracker else (None, None)
        interactions = _state.interaction_tracker.update(left, right, objects, t_stamp) if _state.interaction_tracker else []

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

    # Reasoning / State Manager
    result = None
    if _state.state_manager:
        if active_source != "sim":
            detected_obj = _state.state_manager._resolve_object(prediction, objects)
        else:
            detected_obj = obj

        ai_calc_time = (time.time() - t0) * 1000.0
        result = _state.state_manager.process(
            prediction,
            objects,
            fps=_state.ai_fps,
            latency_ms=ai_calc_time + latency_ms,
        )
        _state.last_result = result

        if _state.experiment_logger:
            _state.experiment_logger.log(result)

    ai_duration_ms = (time.time() - t0) * 1000.0

    # Update thread-safe annotation cache for instant streamer overlay
    _state.annotation_cache.update(
        result=result,
        objects=objects,
        poses=poses,
        left_hand=left,
        right_hand=right,
        camera_name=cam_name,
        ai_latency_ms=ai_duration_ms,
    )

    # Schedule WebSocket broadcast on FastAPI's main asyncio loop
    if result and _state._main_loop and _state._main_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(
                _state.ws_manager.broadcast(result.to_dict()),
                _state._main_loop,
            )
        except Exception:
            pass


# =========================================================================== #
# High-Speed Live Streamer Loop (Async)
# =========================================================================== #
async def _streaming_loop() -> None:
    """
    Live streaming loop: runs at camera frame rate (~30 FPS).
    Pulls newest frame from LatestFrameBuffer, draws latest cached AI overlays in <1ms,
    and pushes to the MJPEG streamer with zero queue backlog.
    """
    logger.info("ORBITA Live Streaming loop started.")
    while _state.is_running:
        t0 = time.time()
        try:
            await _stream_single_frame()
        except Exception as exc:
            logger.warning("Streaming loop error: %s", exc)

        # Regulate to ~30 FPS target (0.033s)
        elapsed = time.time() - t0
        sleep_dur = max(0.005, 0.033 - elapsed)
        await asyncio.sleep(sleep_dur)


async def _stream_single_frame() -> None:
    active_source = _state.camera_manager.active_source if _state.camera_manager else _state.mode
    frame = None
    cam_latency_ms = 0.0

    if _state.camera_manager:
        frame, cam_fps, cam_latency_ms = _state.camera_manager.read_with_metadata()

    if frame is None:
        # Camera is disconnected or connecting
        if _state.camera_manager:
            cam_stat = _state.camera_manager.get_status()
            if cam_stat.get("status") in ("connecting", "reconnecting", "error", "disconnected"):
                _show_camera_status_frame(cam_stat.get("status", ""), cam_stat.get("error"))
        return

    t_now = time.time()

    # Retrieve latest cached AI annotations snapshot (instant non-blocking read)
    (
        last_result,
        objects,
        poses,
        left,
        right,
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
        camera_name=cam_name or ("Phone IP Camera" if active_source == "ip_camera" else "Live Camera"),
        latency_ms=cam_latency_ms,
    )

    # Update streamer (event-driven dispatch to all connected viewers)
    _state.streamer.update(annotated, timestamp=t_now)

    # Write to local recorder if active
    if _state.recorder and _state.recorder.is_recording():
        _state.recorder.write(annotated)

    # Total end-to-end pipeline latency metric
    _state.total_pipeline_latency_ms = cam_latency_ms + _state.streamer.encode_latency_ms

    # Stream FPS tracking
    _state.frame_count += 1
    elapsed = time.time() - _state._fps_timer
    if elapsed >= 1.0:
        _state.pipeline_fps = round(_state.frame_count / elapsed, 1)
        _state.frame_count = 0
        _state._fps_timer = time.time()


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

    # 3. Hand tracking points
    if (left_hand or right_hand) and _state.hand_tracker:
        try:
            vis = _state.hand_tracker.draw(vis, left_hand, right_hand)
        except Exception:
            pass

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
        while True:
            if _state.last_result:
                await websocket.send_text(json.dumps(_state.last_result.to_dict()))
            await asyncio.sleep(0.08)
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
    if action == "reset":
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
    }


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

    return {
        "camera_fps": cam_diag.get("actual_fps", 0.0),
        "stream_fps": stream_diag.get("stream_fps", _state.pipeline_fps),
        "ai_fps": round(_state.ai_fps, 1),
        "pipeline_latency_ms": round(_state.total_pipeline_latency_ms, 1),
        "camera_latency_ms": cam_diag.get("latency_ms", 0.0),
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

    return {
        "ai_engine": "ONLINE",
        "camera": "ONLINE" if is_cam_online else "OFFLINE",
        "camera_source": cam_info.get("source", _state.mode),
        "camera_status": cam_info.get("status", "disconnected"),
        "camera_fps": cam_info.get("fps", 0.0),
        "stream_fps": round(_state.pipeline_fps, 1),
        "ai_fps": round(_state.ai_fps, 1),
        "camera_latency_ms": cam_info.get("latency_ms", 0.0),
        "pipeline_latency_ms": round(_state.total_pipeline_latency_ms, 1),
        "dropped_frames_pct": cam_diag.get("dropped_pct", 0.0),
        "tts": "ONLINE" if (_state.tts and _state.tts.is_available()) else "OFFLINE",
        "recording": "ON" if (_state.recorder and _state.recorder.is_recording()) else "OFF",
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


# Serve React frontend (if built)
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
