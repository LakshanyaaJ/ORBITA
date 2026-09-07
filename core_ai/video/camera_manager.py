"""
ORBITA Camera Manager
=====================
Unified controller for video input sources:
  - Jetson / Local Camera (CSI / USB via OpenCV/GStreamer)
  - Phone IP Camera (IP Webcam / RTSP / HTTP)
  - Phone Web App (Direct browser stream, no app required)
  - Simulation Mode

Guarantees clean resource handoff between camera sources without restarting ORBITA.
Provides multi-consumer access to bounded LatestFrameBuffer for zero-lag streaming.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core_ai.app.config import CameraConfig
from core_ai.video.camera import Camera
from core_ai.video.camera_config import (
    IPCameraConfig,
    get_default_ip_camera_url,
    validate_and_format_camera_url,
)
from core_ai.video.frame_buffer import LatestFrameBuffer
from core_ai.video.ip_camera import IPCamera
from core_ai.video.phone_stream_receiver import PhoneStreamReceiver

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages active camera sources and ensures exclusive hardware/network stream access.
    Exposes unified diagnostic metrics for real-time monitoring.
    """

    def __init__(self, default_config: Optional[CameraConfig] = None):
        self.default_config = default_config or CameraConfig()

        self._active_source: str = "sim"  # "sim" | "jetson_camera" | "ip_camera" | "phone_webcam"
        self._active_url: str = get_default_ip_camera_url()
        self._jetson_device_index: int = 0

        self._jetson_camera: Optional[Camera] = None
        self._ip_camera: Optional[IPCamera] = None
        self._phone_receiver: PhoneStreamReceiver = PhoneStreamReceiver(port=8000)

        # Fallback buffer for simulation frames
        self._sim_buffer = LatestFrameBuffer(name="sim-buffer")

        self._lock = threading.Lock()
        self._status: str = "disconnected"
        self._last_error: Optional[str] = None
        self._rotation: int = -1  # -1 = auto-horizontal (guarantees horizontal view), 0, 90, 180, 270

    @property
    def active_source(self) -> str:
        return self._active_source

    @property
    def active_url(self) -> str:
        return self._active_url

    @property
    def phone_receiver(self) -> PhoneStreamReceiver:
        return self._phone_receiver

    @property
    def rotation(self) -> int:
        return self._rotation

    def set_rotation(self, degrees: int) -> int:
        """Set camera rotation: 0, 90, 180, 270, or -1 (auto-horizontal)."""
        with self._lock:
            self._rotation = degrees
            if self._phone_receiver:
                self._phone_receiver.rotation = degrees
            if self._ip_camera:
                self._ip_camera.rotation = degrees
            if self._jetson_camera:
                self._jetson_camera.rotation = degrees
            logger.info("CameraManager: rotation set to %d", degrees)
            return self._rotation

    def get_frame_buffer(self) -> LatestFrameBuffer:
        """Return the active frame buffer for non-blocking multi-consumer reads."""
        with self._lock:
            if self._active_source == "phone_webcam":
                return self._phone_receiver.frame_buffer
            elif self._active_source == "ip_camera" and self._ip_camera:
                return self._ip_camera.frame_buffer
            elif self._active_source == "jetson_camera" and self._jetson_camera:
                return self._jetson_camera.frame_buffer
            return self._sim_buffer

    def get_status(self) -> Dict[str, Any]:
        """Return standardized status dictionary for FastAPI and Frontend."""
        with self._lock:
            connected = False
            fps = 0.0
            latency_ms = 0.0
            status = self._status
            error = self._last_error

            if self._active_source == "phone_webcam":
                connected = self._phone_receiver.is_connected
                fps = self._phone_receiver.actual_fps
                latency_ms = self._phone_receiver.latency_ms
                status = "connected" if connected else "waiting"
            elif self._active_source == "ip_camera" and self._ip_camera:
                connected = self._ip_camera.is_connected()
                fps = self._ip_camera.actual_fps
                latency_ms = self._ip_camera.latency_ms
                status = self._ip_camera.status
                error = self._ip_camera.error_message or error
            elif self._active_source == "jetson_camera" and self._jetson_camera:
                connected = self._jetson_camera.is_running()
                fps = self._jetson_camera.actual_fps
                status = "connected" if connected else "disconnected"
                latency_ms = self._jetson_camera.latency_ms
            elif self._active_source == "sim":
                connected = True
                status = "connected"
                fps = 30.0
                latency_ms = 0.0

            return {
                "connected": connected,
                "source": self._active_source,
                "url": self._active_url if self._active_source == "ip_camera" else "",
                "device_index": self._jetson_device_index if self._active_source == "jetson_camera" else None,
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "status": status,
                "error": error if error else None,
                "rotation": self._rotation,
            }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return detailed low-latency telemetry diagnostics."""
        with self._lock:
            active_src = self._active_source
            if active_src == "phone_webcam":
                diag = self._phone_receiver.get_diagnostics()
            elif active_src == "ip_camera" and self._ip_camera:
                diag = self._ip_camera.get_diagnostics()
            elif active_src == "jetson_camera" and self._jetson_camera:
                diag = self._jetson_camera.frame_buffer.stats
                diag["status"] = "connected" if self._jetson_camera.is_running() else "disconnected"
                diag["actual_fps"] = self._jetson_camera.actual_fps
            else:
                diag = self._sim_buffer.stats
                diag["status"] = "connected"
                diag["actual_fps"] = 30.0

            diag["source"] = active_src
            diag["target_fps"] = self.default_config.fps
            diag["buffer_size"] = 1
            diag["low_latency_mode"] = True
            return diag

    def connect_phone_webcam(self) -> Tuple[bool, str]:
        """Switch to Phone Web App camera mode."""
        with self._lock:
            logger.info("CameraManager: Switching to Phone Web App camera")
            self._release_all(keep_phone_receiver=True)
            self._active_source = "phone_webcam"
            self._status = "waiting"
            self._last_error = None
            return True, ""

    def connect_ip_camera(
        self,
        url: Optional[str] = None,
        ip: Optional[str] = None,
        port: Optional[int | str] = None,
        path: Optional[str] = None,
        timeout_sec: float = 4.0,
    ) -> Tuple[bool, str]:
        """
        Switch to Phone IP Camera stream with optimized low-latency settings.
        """
        valid, formatted_url, err = validate_and_format_camera_url(
            url=url, ip=ip, port=port, path=path
        )
        if not valid:
            self._last_error = err
            self._status = "error"
            return False, err

        with self._lock:
            logger.info("CameraManager: Switching to IP Camera: %s", formatted_url)
            self._status = "connecting"
            self._last_error = None

            # Stop existing sources cleanly
            self._release_all()

            # Initialize new IP Camera with low-latency parameters
            cfg = IPCameraConfig(
                url=formatted_url,
                timeout_sec=timeout_sec,
                width=self.default_config.width,
                height=self.default_config.height,
                target_fps=self.default_config.fps,
                jpeg_quality=getattr(self.default_config, "jpeg_quality", 70),
                drop_old_frames=True,
                prefer_hardware_acceleration=getattr(self.default_config, "prefer_hardware_acceleration", True),
                low_latency=True,
            )
            ip_cam = IPCamera(cfg)
            success = ip_cam.connect(timeout_sec=timeout_sec)

            if success:
                self._ip_camera = ip_cam
                self._active_source = "ip_camera"
                self._active_url = formatted_url
                self._status = "connected"
                self._last_error = None
                return True, ""
            else:
                self._ip_camera = None
                self._active_source = "ip_camera"
                self._active_url = formatted_url
                self._status = "error"
                self._last_error = ip_cam.error_message
                return False, ip_cam.error_message

    def connect_jetson_camera(self, device_index: int = 0) -> Tuple[bool, str]:
        """
        Switch to local Jetson USB/CSI Camera.
        """
        with self._lock:
            logger.info("CameraManager: Switching to Jetson Camera (index=%d)", device_index)
            self._status = "connecting"
            self._last_error = None

            # Stop existing sources cleanly
            self._release_all()

            cfg = CameraConfig(
                source=device_index,
                width=self.default_config.width,
                height=self.default_config.height,
                fps=self.default_config.fps,
                flip=self.default_config.flip,
            )
            cam = Camera(cfg)
            if cam.start():
                self._jetson_camera = cam
                self._active_source = "jetson_camera"
                self._jetson_device_index = device_index
                self._status = "connected"
                return True, ""
            else:
                self._jetson_camera = None
                self._active_source = "jetson_camera"
                self._status = "error"
                self._last_error = f"Failed to initialize Jetson camera at index {device_index}."
                return False, self._last_error

    def set_simulation_mode(self) -> None:
        """Switch to offline simulation mode."""
        with self._lock:
            logger.info("CameraManager: Switching to Simulation Mode")
            self._release_all()
            self._active_source = "sim"
            self._status = "connected"
            self._last_error = None

    def push_sim_frame(self, frame: np.ndarray) -> None:
        """Store simulation frame in the buffer."""
        self._sim_buffer.push(frame, time.time())

    def disconnect(self) -> None:
        """Disconnect active camera and return to standby."""
        with self._lock:
            logger.info("CameraManager: Disconnecting active camera")
            self._release_all()
            self._active_source = "disconnected"
            self._status = "disconnected"
            self._last_error = None

    def read(self) -> Optional[np.ndarray]:
        """
        Read the latest frame from the currently active camera (non-blocking).
        Returns None if no frame is available or camera is disconnected.
        """
        if self._active_source == "phone_webcam":
            return self._phone_receiver.read()
        elif self._active_source == "ip_camera" and self._ip_camera:
            return self._ip_camera.read()
        elif self._active_source == "jetson_camera" and self._jetson_camera:
            return self._jetson_camera.read()
        elif self._active_source == "sim":
            frame, _, _ = self._sim_buffer.get_latest()
            return frame
        return None

    def read_with_metadata(self) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Returns (frame, actual_fps, latency_ms).
        """
        if self._active_source == "phone_webcam":
            return self._phone_receiver.read_with_metadata()
        elif self._active_source == "ip_camera" and self._ip_camera:
            return self._ip_camera.read_with_metadata()
        elif self._active_source == "jetson_camera" and self._jetson_camera:
            return self._jetson_camera.read_with_metadata()
        elif self._active_source == "sim":
            frame, ts, _ = self._sim_buffer.get_latest()
            lat = max(0.0, (time.time() - ts) * 1000.0) if ts else 0.0
            return frame, 30.0, lat
        return None, 0.0, 0.0

    def _release_all(self, keep_phone_receiver: bool = False) -> None:
        """Ensure all camera resources are freed."""
        if self._ip_camera:
            try:
                self._ip_camera.stop()
            except Exception as e:
                logger.warning("Error stopping IP camera: %s", e)
            self._ip_camera = None

        if self._jetson_camera:
            try:
                self._jetson_camera.stop()
            except Exception as e:
                logger.warning("Error stopping Jetson camera: %s", e)
            self._jetson_camera = None

        self._sim_buffer.clear()
