"""
ORBITA Camera Manager
=====================
Unified controller for video input sources:
  - Jetson / Local Camera (CSI / USB via OpenCV/GStreamer)
  - Phone IP Camera (IP Webcam / RTSP / HTTP)
  - Simulation Mode

Guarantees clean resource handoff between camera sources without restarting ORBITA.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

from orbita.app.config import CameraConfig
from orbita.video.camera import Camera
from orbita.video.camera_config import (
    IPCameraConfig,
    get_default_ip_camera_url,
    validate_and_format_camera_url,
)
from orbita.video.ip_camera import IPCamera

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages active camera sources and ensures exclusive hardware/network stream access.
    """

    def __init__(self, default_config: Optional[CameraConfig] = None):
        self.default_config = default_config or CameraConfig()
        
        self._active_source: str = "sim"  # "sim" | "jetson_camera" | "ip_camera"
        self._active_url: str = get_default_ip_camera_url()
        self._jetson_device_index: int = 0
        
        self._jetson_camera: Optional[Camera] = None
        self._ip_camera: Optional[IPCamera] = None
        
        self._lock = threading.Lock()
        self._status: str = "disconnected"
        self._last_error: Optional[str] = None

    @property
    def active_source(self) -> str:
        return self._active_source

    @property
    def active_url(self) -> str:
        return self._active_url

    def get_status(self) -> Dict[str, Any]:
        """Return standardized status dictionary for FastAPI and Frontend."""
        with self._lock:
            connected = False
            fps = 0.0
            latency_ms = 0.0
            status = self._status
            error = self._last_error

            if self._active_source == "ip_camera" and self._ip_camera:
                connected = self._ip_camera.is_connected()
                fps = self._ip_camera.actual_fps
                latency_ms = self._ip_camera.latency_ms
                status = self._ip_camera.status
                error = self._ip_camera.error_message or error
            elif self._active_source == "jetson_camera" and self._jetson_camera:
                connected = self._jetson_camera.is_running()
                fps = self._jetson_camera.actual_fps
                status = "connected" if connected else "disconnected"
            elif self._active_source == "sim":
                connected = True
                status = "connected"

            return {
                "connected": connected,
                "source": self._active_source,
                "url": self._active_url if self._active_source == "ip_camera" else "",
                "device_index": self._jetson_device_index if self._active_source == "jetson_camera" else None,
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "status": status,
                "error": error if error else None,
            }

    def connect_ip_camera(
        self,
        url: Optional[str] = None,
        ip: Optional[str] = None,
        port: Optional[int | str] = None,
        path: Optional[str] = None,
        timeout_sec: float = 5.0,
    ) -> Tuple[bool, str]:
        """
        Switch to Phone IP Camera stream.
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

            # Initialize new IP Camera
            cfg = IPCameraConfig(
                url=formatted_url,
                timeout_sec=timeout_sec,
                width=self.default_config.width,
                height=self.default_config.height,
                target_fps=self.default_config.fps,
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
        Read the latest frame from the currently active camera.
        Returns None if no frame is available or camera is disconnected.
        """
        if self._active_source == "ip_camera" and self._ip_camera:
            return self._ip_camera.read()
        elif self._active_source == "jetson_camera" and self._jetson_camera:
            return self._jetson_camera.read()
        return None

    def read_with_metadata(self) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Returns (frame, actual_fps, latency_ms).
        """
        if self._active_source == "ip_camera" and self._ip_camera:
            return self._ip_camera.read_with_metadata()
        elif self._active_source == "jetson_camera" and self._jetson_camera:
            frame = self._jetson_camera.read()
            return frame, self._jetson_camera.actual_fps, 0.0
        return None, 0.0, 0.0

    def _release_all(self) -> None:
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
