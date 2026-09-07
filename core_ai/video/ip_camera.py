"""
ORBITA IP Camera Module
=======================
Ultra-low-latency video frame capture for Phone IP Cameras and RTSP/HTTP streams.

Optimized for edge deployment (NVIDIA Jetson Orin Nano / Host PC):
  - Uses LatestFrameBuffer (bounded size=1) with zero queue accumulation
  - Drops stale frames immediately: latest frame > every frame
  - Low-latency FFmpeg capture options (nobuffer, low_delay, max_delay=0)
  - Hardware-accelerated GStreamer decoding on Jetson (nvv4l2decoder / nvjpegdec)
  - Continuous socket drain without artificial sleep delays
  - Computes real-time streaming latency (ms), frame drop rate, and actual FPS
  - Non-blocking connection check with timeout protection
  - Automatic reconnection with exponential backoff
  - Clean resource acquisition and release
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core_ai.video.camera_config import IPCameraConfig
from core_ai.video.frame_buffer import LatestFrameBuffer

logger = logging.getLogger(__name__)


def _build_gstreamer_pipeline(url: str, config: IPCameraConfig) -> Optional[str]:
    """
    Construct an ultra-low-latency GStreamer pipeline for NVIDIA Jetson hardware acceleration.
    """
    if url.startswith("rtsp://"):
        return (
            f'rtspsrc location="{url}" latency=0 buffer-mode=none drop-on-latency=true ! '
            f'rtph264depay ! h264parse ! '
            f'nvv4l2decoder enable-max-performance=1 ! '
            f'nvvidconv ! video/x-raw, format=BGRx ! '
            f'videoconvert ! video/x-raw, format=BGR ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )
    elif url.startswith("http://") or url.startswith("https://"):
        # For HTTP MJPEG stream (e.g. Android IP Webcam)
        return (
            f'souphttpsrc location="{url}" is-live=true ! '
            f'jpegparse ! nvjpegdec ! '
            f'nvvidconv ! video/x-raw, format=BGRx ! '
            f'videoconvert ! video/x-raw, format=BGR ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )
    return None


def _open_capture_device(url: str, config: IPCameraConfig) -> Optional[cv2.VideoCapture]:
    """
    Open video stream using optimal low-latency flags for FFmpeg / GStreamer.
    """
    # 1. Set low-latency FFmpeg demuxer environment options
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "fflags;nobuffer|flags;low_delay|max_delay;0|analyzeduration;0|probesize;32"
    )

    # 2. Try GStreamer hardware acceleration if supported and preferred
    if config.prefer_hardware_acceleration and hasattr(cv2, "videoio_registry"):
        try:
            if cv2.videoio_registry.hasBackend(cv2.CAP_GSTREAMER):
                gst_pipeline = _build_gstreamer_pipeline(url, config)
                if gst_pipeline:
                    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                    if cap and cap.isOpened():
                        logger.info("IPCamera: GStreamer hardware accelerated pipeline opened: %s", url)
                        return cap
                    if cap:
                        cap.release()
        except Exception as e:
            logger.debug("IPCamera: GStreamer open attempt failed: %s; falling back to standard backend", e)

    # 3. Standard OpenCV VideoCapture with low-latency properties
    cap = cv2.VideoCapture(url)
    if cap and cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)
        except Exception:
            pass
        try:
            timeout_ms = int(config.timeout_sec * 1000)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        except Exception:
            pass
        return cap

    return None


class IPCamera:
    """
    Dedicated IP / Phone Camera capture worker with ultra-low latency.
    """

    def __init__(self, config: IPCameraConfig):
        self.config = config
        self.url = config.url
        self._cap: Optional[cv2.VideoCapture] = None

        # Bounded LatestFrameBuffer (size 1) for zero queue lag
        self._frame_buffer = LatestFrameBuffer(name="ipcam-buffer")

        # Kept for backward compatibility with unit tests
        self._frame_queue: queue.Queue[Tuple[np.ndarray, float]] = queue.Queue(maxsize=2)

        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._status: str = "disconnected"  # "disconnected" | "connecting" | "connected" | "reconnecting" | "error"
        self._error_message: str = ""

        self._actual_fps: float = 0.0
        self._latency_ms: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.time()
        self._last_frame_time: float = 0.0

        self._lock = threading.Lock()
        self.rotation: int = getattr(config, "rotation", -1)  # -1 = auto-horizontal

    @property
    def frame_buffer(self) -> LatestFrameBuffer:
        return self._frame_buffer

    @property
    def status(self) -> str:
        return self._status

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    def is_connected(self) -> bool:
        return self._status == "connected" and self._running

    def connect(self, timeout_sec: Optional[float] = None) -> bool:
        """
        Initiate connection to the IP camera stream.
        Attempts initial connection synchronously up to timeout_sec,
        then launches the background worker loop.
        """
        timeout = timeout_sec or self.config.timeout_sec
        self._status = "connecting"
        self._error_message = ""

        # Release any existing resources first
        self._stop_internal()

        logger.info("IPCamera: Connecting to %s (timeout=%.1fs, low_latency=True)...", self.url, timeout)

        connect_success = [False]
        error_msg = [
            f"Connection timed out reaching {self.url}. Ensure: 1) Phone and PC are on the same Wi-Fi network, "
            f"2) IP Webcam app is started on the phone, and 3) IP address and port match."
        ]
        cap_holder = [None]

        def _try_open():
            try:
                cap = _open_capture_device(self.url, self.config)
                if cap and cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        cap_holder[0] = cap
                        connect_success[0] = True
                    else:
                        cap.release()
                        error_msg[0] = "Camera stream opened but returned no video frames."
                else:
                    error_msg[0] = "Unable to connect. Ensure phone and Jetson/PC are on the same network and IP Webcam is running."
            except Exception as e:
                error_msg[0] = f"Connection error: {str(e)}"

        init_thread = threading.Thread(target=_try_open, daemon=True, name="ipcam-init")
        init_thread.start()
        init_thread.join(timeout=timeout)

        if not connect_success[0]:
            self._status = "error"
            self._error_message = error_msg[0]
            logger.error("IPCamera connection failed: %s", self._error_message)
            return False

        self._cap = cap_holder[0]
        self._status = "connected"
        self._running = True
        self._last_frame_time = time.time()
        self._fps_timer = time.time()
        self._frame_count = 0

        # Start persistent low-latency capture loop
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="ipcam-capture-worker",
        )
        self._thread.start()
        logger.info("IPCamera: Connected successfully to %s", self.url)
        return True

    def read(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the camera (non-blocking).
        Returns None if no frame is currently available.
        """
        # Primary: check LatestFrameBuffer
        frame, timestamp, _ = self._frame_buffer.get_latest()
        if frame is not None:
            self._latency_ms = max(0.0, (time.time() - timestamp) * 1000.0)
            return frame

        # Secondary: fallback to legacy queue (used in unit tests)
        try:
            f, t = self._frame_queue.get_nowait()
            self._latency_ms = max(0.0, (time.time() - t) * 1000.0)
            return f
        except queue.Empty:
            return None

    def read_with_metadata(self) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Returns (frame, actual_fps, latency_ms).
        """
        frame = self.read()
        return frame, self._actual_fps, self._latency_ms

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return full telemetry metrics for monitoring."""
        stats = self._frame_buffer.stats
        stats["status"] = self._status
        stats["url"] = self.url
        stats["error"] = self._error_message or None
        stats["actual_fps"] = self._actual_fps
        return stats

    def stop(self) -> None:
        """Stop capture and release all camera resources."""
        logger.info("IPCamera: Stopping capture for %s", self.url)
        self._stop_internal()
        self._status = "disconnected"
        self._error_message = ""

    def _stop_internal(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._frame_buffer.clear()

        # Clear remaining queue items
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def _capture_loop(self) -> None:
        """
        Background capture loop:
        Pulls frames continuously with ZERO sleep delays on live video.
        Drains network socket immediately, replaces stale frames, and computes FPS.
        """
        reconnect_attempts = 0

        while self._running:
            if not self._cap or not self._cap.isOpened():
                if not self._handle_reconnect(reconnect_attempts):
                    break
                reconnect_attempts += 1
                continue

            t_capture = time.time()
            ret, frame = self._cap.read()

            if not ret or frame is None or frame.size == 0:
                # Frame dropped or interrupted
                if time.time() - self._last_frame_time > 4.0:
                    self._status = "reconnecting"
                    self._error_message = "Stream interrupted. Reconnecting..."
                    if not self._handle_reconnect(reconnect_attempts):
                        break
                    reconnect_attempts += 1
                else:
                    time.sleep(0.01)
                continue

            # Frame successfully acquired
            reconnect_attempts = 0
            self._status = "connected"
            self._last_frame_time = time.time()

            # Apply orientation transformation: guarantee horizontal landscape view
            fh, fw = frame.shape[:2]
            rot = getattr(self, "rotation", -1)
            if rot == 90 or (rot == -1 and fh > fw):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rot == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Preserve aspect ratio: scale proportionally if frame exceeds configured dimensions
            fh, fw = frame.shape[:2]
            target_w, target_h = self.config.width, self.config.height
            if (target_w > 0 and target_h > 0) and (fw > target_w or fh > target_h):
                scale = min(target_w / fw, target_h / fh)
                new_w = max(1, int(round(fw * scale)))
                new_h = max(1, int(round(fh * scale)))
                if new_w != fw or new_h != fh:
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Push newest frame to LatestFrameBuffer (drops stale frame instantly)
            self._frame_buffer.push(frame, t_capture)

            # Maintain queue for legacy tests
            try:
                if self._frame_queue.full():
                    self._frame_queue.get_nowait()
                self._frame_queue.put_nowait((frame, t_capture))
            except (queue.Empty, queue.Full):
                pass

            # FPS calculation
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._actual_fps = round(self._frame_count / elapsed, 1)
                self._frame_count = 0
                self._fps_timer = now

            # NOTE: We DO NOT time.sleep() here!
            # Network camera streams (RTSP / HTTP) push at camera frame rate.
            # Reading continuously drains the socket buffer and ensures lowest latency.

    def _handle_reconnect(self, attempt: int) -> bool:
        """Attempt reconnection with backoff up to max attempts."""
        if attempt >= self.config.max_reconnect_attempts:
            self._status = "error"
            self._error_message = f"Reconnection failed after {attempt} attempts."
            logger.error("IPCamera: Max reconnection attempts reached for %s", self.url)
            return False

        self._status = "reconnecting"
        self._error_message = f"Connection lost. Reconnecting... (Attempt {attempt + 1}/{self.config.max_reconnect_attempts})"
        logger.info("IPCamera: %s", self._error_message)

        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        time.sleep(min(self.config.reconnect_interval_sec * (1.2 ** attempt), 8.0))

        if not self._running:
            return False

        try:
            cap = _open_capture_device(self.url, self.config)
            if cap and cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    self._cap = cap
                    self._status = "connected"
                    self._error_message = ""
                    self._last_frame_time = time.time()
                    logger.info("IPCamera: Successfully reconnected to %s", self.url)
                    return True
                cap.release()
        except Exception as e:
            logger.warning("IPCamera: Reconnect attempt failed: %s", e)

        return True
