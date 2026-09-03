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

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from orbita.app.config import OrbitaConfig, load_config
from orbita.backend.websocket_manager import WebSocketManager
from orbita.har.feature_fusion import TemporalFeatureWindow, build_feature_vector
from orbita.har.temporal_model import ActionClassifier, ActionPrediction
from orbita.interaction.hand_object import HandObjectInteractionTracker
from orbita.logging.experiment_logger import ExperimentLogger
from orbita.perception.hand_tracker import HandTracker
from orbita.perception.object_detector import ObjectDetector
from orbita.perception.pose_estimator import PoseEstimator
from orbita.reasoning.state_manager import StateManager, ValidationResult
from orbita.simulation.simulator import ExperimentSimulator, Scenario
from orbita.video.camera import Camera
from orbita.video.recorder import VideoRecorder
from orbita.video.streamer import MJPEGStreamer
from orbita.voice.tts import TTSEngine

logger = logging.getLogger(__name__)

# =========================================================================== #
# Global application state
# =========================================================================== #
class AppState:
    def __init__(self):
        self.config: Optional[OrbitaConfig] = None
        self.ws_manager = WebSocketManager()
        self.streamer = MJPEGStreamer()
        self.camera: Optional[Camera] = None
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
        self.mode: str = "sim"          # "sim" | "webcam"
        self.is_running: bool = False
        self.pipeline_fps: float = 0.0
        self.last_result: Optional[ValidationResult] = None
        self.frame_count: int = 0
        self._fps_timer: float = time.time()


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
    if _state.camera:
        _state.camera.stop()
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
    if _state.mode == "sim" and _state.simulator:
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

    elif _state.camera:
        frame = _state.camera.read()
        if frame is None:
            return

        t_stamp = time.time()

        # --- 2. Perception ---
        objects = _state.detector.detect(frame, t_stamp)
        poses = _state.pose.estimate(frame, t_stamp)
        pose = poses[0] if poses else None
        left, right = _state.hand_tracker.track(pose, frame, t_stamp)
        interactions = _state.interaction_tracker.update(left, right, objects, t_stamp)

        # --- 3. Feature fusion ---
        fv = build_feature_vector(pose, left, right, objects, interactions, frame.shape[:2])
        _state.feature_window.push(fv)
        window = _state.feature_window.get_window()

        # --- 4. Action classification ---
        prediction = _state.classifier.predict(window)
        obj = ""

    else:
        return

    # --- 5. Procedural reasoning ---
    if _state.state_manager:
        detected_obj = ""
        if _state.mode != "sim":
            from orbita.reasoning.state_manager import StateManager
            detected_obj = _state.state_manager._resolve_object(prediction, objects)
        else:
            detected_obj = obj

        result = _state.state_manager.process(
            prediction,
            objects if 'objects' in dir() else [],
            fps=_state.pipeline_fps,
            latency_ms=(time.time() - t0) * 1000,
        )
        _state.last_result = result

        # Log to experiment logger
        if _state.experiment_logger:
            _state.experiment_logger.log(result)

    # --- 6. Annotate frame ---
    if frame is not None:
        annotated = _annotate_frame(frame, result if _state.last_result else None)
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


def _annotate_frame(frame, result: Optional[ValidationResult]):
    """Draw all AI overlays on frame."""
    import cv2
    vis = frame.copy()

    # Status banner
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
        cv2.putText(vis, result.hud_message[:60], (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

        # FPS
        fps_text = f"{_state.pipeline_fps:.1f} FPS"
        cv2.putText(vis, fps_text, (vis.shape[1] - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 100), 1, cv2.LINE_AA)

    return vis


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
    except WebSocketDisconnect:
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
        if new_mode == "webcam":
            if _state.camera is None:
                cfg = _state.config
                if cfg:
                    cam = Camera(cfg.camera)
                    if cam.start():
                        _state.camera = cam
        return {"status": f"mode_{new_mode}"}

    return {"status": "unknown_action"}


# Status
@app.get("/api/status")
async def api_status():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    return {
        "ai_engine": "ONLINE",
        "camera": "ONLINE" if (_state.camera or _state.mode == "sim") else "OFFLINE",
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
    from orbita.database.sqlite_db import OrbitaDB
    db = OrbitaDB()
    return db.list_experiments()


# Serve React frontend (if built)
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
