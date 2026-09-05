"""
ORBITA IP Camera Module
=======================
Low-latency video frame capture for Phone IP Cameras and RTSP/HTTP streams.

Optimized for edge deployment (NVIDIA Jetson Orin Nano / Host PC):
  - Drops stale frames via 1-frame queue to eliminate buffer lag
  - Computes real-time streaming latency (ms) and actual FPS
  - Non-blocking connection check with timeout protection
  - Auto-reconnection logic on temporary network interruption
  - Clean resource acquisition and release
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from orbita.video.camera_config import IPCameraConfig

logger = logging.getLogger(__name__)


class IPCamera:
    """
    Dedicated IP / Phone Camera capture worker.
    """

    def __init__(self, config: IPCameraConfig):
        self.config = config
        self.url = config.url
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Buffer of size 1 ensures zero queued lag
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

        logger.info("IPCamera: Connecting to %s (timeout=%.1fs)...", self.url, timeout)

        # Test connection in helper thread with timeout
        connect_success = [False]
        error_msg = [
            f"Connection timed out reaching {self.url}. Ensure: 1) Phone and PC are on the same Wi-Fi network, "
            f"2) IP Webcam app is started on the phone, and 3) IP address and port match."
        ]
        cap_holder = [None]

        def _try_open():
            try:
                # OpenCV VideoCapture on URL
                cap = cv2.VideoCapture(self.url)
                if cap and cap.isOpened():
                    # Set buffer size to 1 for real-time streaming
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                    except Exception:
                        pass
                    
                    # Read a test frame to ensure stream is active
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        cap_holder[0] = cap
                        connect_success[0] = True
                    else:
                        cap.release()
                        error_msg[0] = "Camera stream opened but returned no video frames."
                else:
                    error_msg[0] = "Unable to connect. Ensure phone and Jetson are on the same network and IP Webcam is running."
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

        # Start persistent capture loop
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
        Get the latest frame from the camera queue (non-blocking).
        Returns None if no frame is currently available.
        """
        try:
            frame, capture_timestamp = self._frame_queue.get_nowait()
            # Calculate pipeline pickup latency
            now = time.time()
            self._latency_ms = max(0.0, (now - capture_timestamp) * 1000.0)
            return frame
        except queue.Empty:
            return None

    def read_with_metadata(self) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Returns (frame, actual_fps, latency_ms).
        """
        frame = self.read()
        return frame, self._actual_fps, self._latency_ms

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

        # Clear remaining queue items
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def _capture_loop(self) -> None:
        """
        Background capture loop:
        Pulls frames continuously, discards older frames, tracks FPS, and handles auto-reconnect.
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
                logger.warning("IPCamera: Dropped frame from stream: %s", self.url)
                # Check for stream timeout
                if time.time() - self._last_frame_time > 5.0:
                    self._status = "reconnecting"
                    self._error_message = "Stream interrupted. Reconnecting..."
                    if not self._handle_reconnect(reconnect_attempts):
                        break
                    reconnect_attempts += 1
                else:
                    time.sleep(0.02)
                continue

            # Frame successfully acquired
            reconnect_attempts = 0
            self._status = "connected"
            self._last_frame_time = time.time()

            # Resize if dimensions specified and different
            fh, fw = frame.shape[:2]
            target_w, target_h = self.config.width, self.config.height
            if (target_w > 0 and target_h > 0) and (fw != target_w or fh != target_h):
                frame = cv2.resize(frame, (target_w, target_h))

            # Push newest frame to queue; drop stale frame if full
            try:
                if self._frame_queue.full():
                    self._frame_queue.get_nowait()
                self._frame_queue.put_nowait((frame, t_capture))
            except queue.Empty:
                pass
            except Exception:
                pass

            # FPS calculation
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._actual_fps = round(self._frame_count / elapsed, 1)
                self._frame_count = 0
                self._fps_timer = now

            # Sleep tiny interval if camera is pushing faster than target fps
            if self.config.target_fps > 0:
                min_interval = 1.0 / self.config.target_fps
                processing_time = time.time() - t_capture
                sleep_time = min_interval - processing_time
                if sleep_time > 0.005:
                    time.sleep(sleep_time)

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

        time.sleep(min(self.config.reconnect_interval_sec * (1.2 ** attempt), 10.0))

        if not self._running:
            return False

        try:
            cap = cv2.VideoCapture(self.url)
            if cap and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                except Exception:
                    pass
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
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
