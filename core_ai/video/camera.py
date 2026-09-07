"""
ORBITA Camera Module
====================
Manages video capture from webcam, CSI/USB cameras, video files, or simulator.

Provides low-latency single-frame buffer (bounded size=1) so the pipeline
never blocks waiting for a frame, and old frames are automatically dropped.

Supports:
  - Webcam / CSI / USB (index integer or path)
  - Video file (path string)
  - Simulator frames (injected externally via push_frame)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from core_ai.video.frame_buffer import LatestFrameBuffer

logger = logging.getLogger(__name__)


class Camera:
    """
    Video frame producer with zero-latency bounded buffering.

    Usage:
        cam = Camera(config)
        cam.start()
        frame = cam.read()   # Returns latest frame or None
        cam.stop()
    """

    def __init__(self, config):
        self.config = config
        self.source = config.source
        self.width = getattr(config, "width", 1280)
        self.height = getattr(config, "height", 720)
        self.fps_target = getattr(config, "fps", 30)
        self.flip = getattr(config, "flip", False)
        self.rotation = getattr(config, "rotation", -1)  # -1 = auto-horizontal

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_buffer = LatestFrameBuffer(name="local-cam-buffer")
        self._frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._is_simulator = False  # Set True when using push_frame

        self._actual_fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.time()
        self._latency_ms: float = 0.0

    @property
    def frame_buffer(self) -> LatestFrameBuffer:
        return self._frame_buffer

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def start(self) -> bool:
        """Start the capture thread. Returns True if successful."""
        if isinstance(self.source, str) and self.source == "sim":
            self._is_simulator = True
            self._running = True
            logger.info("Camera: simulator mode.")
            return True

        try:
            src = int(self.source) if str(self.source).isdigit() else self.source
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                logger.error("Failed to open camera source: %s", self.source)
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps_target)
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="orbita-camera"
            )
            self._thread.start()
            logger.info("Camera started: source=%s %dx%d @%dfps",
                        self.source, self.width, self.height, self.fps_target)
            return True
        except Exception as exc:
            logger.error("Camera start failed: %s", exc)
            return False

    def read(self) -> Optional[np.ndarray]:
        """Get latest frame (non-blocking). Returns None if no frame available."""
        frame, ts, _ = self._frame_buffer.get_latest()
        if frame is not None:
            self._latency_ms = max(0.0, (time.time() - ts) * 1000.0)
            return frame

        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def read_with_metadata(self) -> Tuple[Optional[np.ndarray], float, float]:
        """Returns (frame, actual_fps, latency_ms)."""
        frame = self.read()
        return frame, self._actual_fps, self._latency_ms

    def push_frame(self, frame: np.ndarray) -> None:
        """Push an externally generated frame into the buffer."""
        if not self._running:
            return
        t = time.time()
        self._frame_buffer.push(frame, t)
        try:
            if self._frame_queue.full():
                self._frame_queue.get_nowait()
            self._frame_queue.put_nowait(frame.copy())
        except (queue.Empty, queue.Full):
            pass

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_buffer.clear()
        logger.info("Camera stopped.")

    def is_running(self) -> bool:
        return self._running

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    # ----------------------------------------------------------------------- #
    # Capture loop (background thread)
    # ----------------------------------------------------------------------- #
    def _capture_loop(self) -> None:
        while self._running and self._cap is not None:
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self.source, str):  # Video file looping
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("Camera read failed.")
                time.sleep(0.05)
                continue

            if self.flip:
                frame = cv2.flip(frame, 1)

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
            if self.width > 0 and self.height > 0 and (fw > self.width or fh > self.height):
                scale = min(self.width / fw, self.height / fh)
                new_w = max(1, int(round(fw * scale)))
                new_h = max(1, int(round(fh * scale)))
                if new_w != fw or new_h != fh:
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Store in LatestFrameBuffer (drops old frame instantly)
            self._frame_buffer.push(frame, t0)

            # Mirror to queue for backward compatibility
            try:
                if self._frame_queue.full():
                    self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

            # FPS tracking
            self._frame_count += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self._actual_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.time()
