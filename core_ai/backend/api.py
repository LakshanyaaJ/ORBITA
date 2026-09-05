"""
ORBITA FastAPI Backend
=======================
Endpoints:
  GET  /                    → Serve React dashboard (built dist)
  GET  /video_feed           → MJPEG stream
  WS   /ws/telemetry         → Real-time state broadcast
  POST /api/control          → start/stop/reset/scenario
  GET  /api/status           → System status
  GET  /api/logs             → Experiment logs
  GET  /api/experiments      → List past experiments

The AI pipeline loop runs as a background asyncio task.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

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
# Global application state
# =========================================================================== #
class AppState:
    def __init__(self):
        self.config: Optional[OrbitaConfig] = None
        self.ws_manager = WebSocketManager()
        self.streamer = MJPEGStreamer()
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
        self.pipeline_task: Optional[asyncio.Task] = None

        self.current_scenario: str = "A"
        self.mode: str = "sim"          # "sim" | "webcam" | "ip_camera"
        self.is_running: bool = False
        self.pipeline_fps: float = 0.0
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

    # Start pipeline
    _state.is_running = True
    _state.pipeline_task = asyncio.create_task(_pipeline_loop())
    logger.info("ORBITA backend started.")


async def _shutdown() -> None:
    _state.is_running = False
    if _state.pipeline_task:
        _state.pipeline_task.cancel()
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
# AI Pipeline Loop (async background task)
# =========================================================================== #
async def _pipeline_loop() -> None:
    """Main AI processing loop — runs at camera FPS."""
    while _state.is_running:
        t0 = time.time()
        try:
            await _process_frame()
        except Exception as exc:
            logger.warning("Pipeline error: %s", exc)
        elapsed = time.time() - t0
        await asyncio.sleep(max(0, 0.033 - elapsed))  # Target ~30 fps


async def _process_frame() -> None:
    cfg = _state.config
    t0 = time.time()

    # --- 1. Acquire frame ---
    frame = None
    objects = []
    poses = []
    left = None
    right = None
    cam_name = ""
    latency_ms = 0.0

    active_source = _state.camera_manager.active_source if _state.camera_manager else _state.mode

    if active_source == "sim" and _state.simulator:
        sim_frame = _state.simulator.next_frame()
        frame = sim_frame.image

        # In sim mode, use simulator's ground-truth action (bypasses HAR for cleaner demo)
        verb = sim_frame.action_verb
        obj = sim_frame.action_object
        confidence = sim_frame.confidence

        # Still run object detector on sim frame for visualization
        t_stamp = sim_frame.timestamp
        objects = _state.detector.detect(frame, t_stamp) if _state.detector else []

        # Create a synthetic prediction matching the simulator
        prediction = ActionPrediction(
            action=verb,
            confidence=confidence,
            next_action="IDLE",
            next_confidence=0.3,
            is_uncertain=confidence < (cfg.har.action_confidence_min if cfg else 0.55),
            target_object=obj,
        )

    elif _state.camera_manager and active_source in ("jetson_camera", "ip_camera"):
        frame, cam_fps, cam_latency = _state.camera_manager.read_with_metadata()
        latency_ms = cam_latency
        cam_name = "Phone IP Camera" if active_source == "ip_camera" else "Jetson Camera"

        if frame is None:
            # If camera is connecting or disconnected, update status frame on streamer
            cam_stat = _state.camera_manager.get_status()
            if cam_stat.get("status") in ("connecting", "reconnecting", "error"):
                _show_camera_status_frame(cam_stat.get("status", ""), cam_stat.get("error"))
            return

        t_stamp = time.time()

        # --- 2. Perception ---
        objects = _state.detector.detect(frame, t_stamp) if _state.detector else []
        poses = _state.pose.estimate(frame, t_stamp) if _state.pose else []
        pose = poses[0] if poses else None
        left, right = _state.hand_tracker.track(pose, frame, t_stamp) if _state.hand_tracker else (None, None)
        interactions = _state.interaction_tracker.update(left, right, objects, t_stamp) if _state.interaction_tracker else []

        # --- 3. Feature fusion ---
        fv = build_feature_vector(pose, left, right, objects, interactions, frame.shape[:2])
        if _state.feature_window:
            _state.feature_window.push(fv)
            window = _state.feature_window.get_window()
        else:
            window = np.zeros((30, 64), dtype=np.float32)

        # --- 4. Action classification ---
        prediction = _state.classifier.predict(window) if _state.classifier else ActionPrediction(
            action="IDLE", confidence=0.5, next_action="IDLE", next_confidence=0.3, is_uncertain=False, target_object=""
        )
        obj = ""

    elif _state.camera:
        frame = _state.camera.read()
        if frame is None:
            return

        t_stamp = time.time()
        cam_name = "Jetson Camera"

        # Perception
        objects = _state.detector.detect(frame, t_stamp) if _state.detector else []
        poses = _state.pose.estimate(frame, t_stamp) if _state.pose else []
        pose = poses[0] if poses else None
        left, right = _state.hand_tracker.track(pose, frame, t_stamp) if _state.hand_tracker else (None, None)
        interactions = _state.interaction_tracker.update(left, right, objects, t_stamp) if _state.interaction_tracker else []

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

    # --- 5. Procedural reasoning ---
    if _state.state_manager:
        if active_source != "sim":
            detected_obj = _state.state_manager._resolve_object(prediction, objects)
        else:
            detected_obj = obj

        result = _state.state_manager.process(
            prediction,
            objects,
            fps=_state.pipeline_fps,
            latency_ms=((time.time() - t0) * 1000) + latency_ms,
        )
        _state.last_result = result

        # Log to experiment logger
        if _state.experiment_logger:
            _state.experiment_logger.log(result)

    # --- 6. Annotate frame ---
    if frame is not None:
        annotated = _annotate_frame(
            frame=frame,
            result=result if _state.last_result else None,
            objects=objects,
            poses=poses,
            left_hand=left,
            right_hand=right,
            camera_name=cam_name,
            latency_ms=latency_ms,
        )
        _state.streamer.update(annotated)

        # Write to recorder
        if _state.recorder and _state.recorder.is_recording():
            _state.recorder.write(annotated)

    # --- 7. Broadcast to dashboard ---
    if _state.last_result:
        await _state.ws_manager.broadcast(_state.last_result.to_dict())

    # --- FPS tracking ---
    _state.frame_count += 1
    elapsed = time.time() - _state._fps_timer
    if elapsed >= 1.0:
        _state.pipeline_fps = _state.frame_count / elapsed
        _state.frame_count = 0
        _state._fps_timer = time.time()


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
    """Draw all AI overlays (bboxes, pose, hands, HUD banner) on frame."""
    import cv2
    vis = frame.copy()

    # 1. Draw detected objects (YOLO / chroma)
    if objects and _state.detector:
        try:
            vis = _state.detector.draw(vis, objects)
        except Exception:
            pass

    # 2. Draw pose skeleton
    if poses and _state.pose:
        try:
            vis = _state.pose.draw(vis, poses)
        except Exception:
            pass

    # 3. Draw hand points
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

        # FPS & latency tag
        fps_text = f"{_state.pipeline_fps:.1f} FPS"
        if latency_ms > 0:
            fps_text = f"{latency_ms:.0f}ms | {fps_text}"
        cv2.putText(vis, fps_text, (vis.shape[1] - 145, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 140), 1, cv2.LINE_AA)

    # 5. Bottom camera badge for live cameras
    if camera_name:
        h, w = vis.shape[:2]
        badge_text = f"SOURCE: {camera_name}"
        cv2.rectangle(vis, (0, h - 22), (w, h), (10, 15, 25), -1)
        cv2.putText(vis, badge_text, (10, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 180, 230), 1, cv2.LINE_AA)

    return vis


def _show_camera_status_frame(status: str, error_msg: Optional[str] = None):
    """Update stream with placeholder when camera is connecting or experiencing an error."""
    import cv2
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background pattern
    cv2.rectangle(blank, (0, 0), (640, 480), (12, 17, 24), -1)
    
    if status in ("connecting", "reconnecting"):
        msg = "CAMERA CONNECTING..." if status == "connecting" else "RECONNECTING TO PHONE CAMERA..."
        sub = "Establishing network stream..."
        col = (60, 190, 240)
    elif status == "error":
        msg = "CAMERA CONNECTION ERROR"
        sub = error_msg or "Unable to reach IP camera. Check network and IP."
        col = (80, 80, 240)
    else:
        msg = "CAMERA DISCONNECTED"
        sub = "Select a camera source to start streaming."
        col = (140, 150, 160)

    cv2.putText(blank, msg, (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    cv2.putText(blank, sub[:70], (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 180), 1, cv2.LINE_AA)
    _state.streamer.update(blank)


# =========================================================================== #
# FastAPI App
# =========================================================================== #
app = FastAPI(title="ORBITA", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_app() -> FastAPI:
    """Return the FastAPI application instance."""
    return app


# Video stream
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _state.streamer.generate_mjpeg(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# WebSocket
@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await _state.ws_manager.connect(websocket)
    try:
        while True:
            # Send current state immediately upon connect
            if _state.last_result:
                await websocket.send_text(json.dumps(_state.last_result.to_dict()))
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        _state.ws_manager.disconnect(websocket)


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


# =========================================================================== #
# Camera Management Endpoints
# =========================================================================== #
@app.post("/api/camera/connect")
async def camera_connect(body: dict):
    """
    Connect to a camera source:
      - Jetson / Local camera: {"source": "jetson_camera", "device_index": 0}
      - Phone IP Camera: {"source": "ip_camera", "url": "http://...", "ip": "...", "port": 8080, "path": "/video"}
      - Simulation: {"source": "sim"}
    """
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

    elif source == "sim":
        _state.camera_manager.set_simulation_mode()
        _state.mode = "sim"
        return {"status": "connected", "source": "sim"}

    return JSONResponse(status_code=400, content={"status": "error", "error": f"Unknown camera source: '{source}'."})


@app.post("/api/camera/disconnect")
async def camera_disconnect():
    """Disconnect the active camera and return to standby."""
    if _state.camera_manager:
        _state.camera_manager.disconnect()
    _state.mode = "disconnected"
    _show_camera_status_frame("disconnected")
    return {"status": "disconnected"}


@app.get("/api/camera/status")
async def camera_status():
    """Get active camera connection state, FPS, and latency metrics."""
    if _state.camera_manager:
        return _state.camera_manager.get_status()
    return {
        "connected": False,
        "source": "none",
        "url": "",
        "fps": 0.0,
        "latency_ms": 0.0,
        "status": "disconnected",
        "error": None,
    }


@app.get("/api/camera/stream")
async def camera_stream():
    """Alias for /video_feed to provide standard camera streaming endpoint."""
    return StreamingResponse(
        _state.streamer.generate_mjpeg(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# Status
@app.get("/api/status")
async def api_status():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    cam_info = _state.camera_manager.get_status() if _state.camera_manager else {}
    is_cam_online = cam_info.get("connected", False) or (_state.mode == "sim")

    return {
        "ai_engine": "ONLINE",
        "camera": "ONLINE" if is_cam_online else "OFFLINE",
        "camera_source": cam_info.get("source", _state.mode),
        "camera_status": cam_info.get("status", "disconnected"),
        "camera_fps": cam_info.get("fps", 0.0),
        "camera_latency_ms": cam_info.get("latency_ms", 0.0),
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


# Logs
@app.get("/api/logs")
async def api_logs():
    if _state.experiment_logger:
        return _state.experiment_logger.get_summary()
    return {}


# Past experiments
@app.get("/api/experiments")
async def api_experiments():
    from core_ai.database.sqlite_db import OrbitaDB
    db = OrbitaDB()
    return db.list_experiments()


# Serve React frontend (if built)
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
