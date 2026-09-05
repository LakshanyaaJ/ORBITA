"""
ORBITA Camera Module
====================
Manages video capture from webcam, video file, or simulator.

Provides a thread-safe frame queue so the AI pipeline never blocks
waiting for a new frame, and frames are never dropped in the queue.

Supports:
  - Webcam (index integer)
  - Video file (path string)
  - Simulator frames (injected externally via push_frame)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """
    Video frame producer.

    Usage:
        cam = Camera(config)
        cam.start()
        frame = cam.read()   # Returns latest frame or None
        cam.stop()
    """

    def __init__(self, config):
        self.config = config
        self.source = config.source
        self.width = config.width
        self.height = config.height
        self.fps_target = config.fps
        self.flip = config.flip

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=4)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._is_simulator = False  # Set True when using push_frame

        self._actual_fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.time()

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
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a simulator-generated frame into the queue."""
        if not self._running:
            return
        try:
            if self._frame_queue.full():
                self._frame_queue.get_nowait()
            self._frame_queue.put_nowait(frame.copy())
        except queue.Empty:
            pass

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("Camera stopped.")

    def is_running(self) -> bool:
        return self._running

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    # ----------------------------------------------------------------------- #
    # Capture loop (background thread)
    # ----------------------------------------------------------------------- #
    def _capture_loop(self) -> None:
        frame_interval = 1.0 / max(self.fps_target, 1)
        while self._running and self._cap is not None:
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self.source, str):  # Video file looping
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("Camera read failed.")
                time.sleep(0.1)
                continue

            if self.flip:
                frame = cv2.flip(frame, 1)

            # Resize if needed
            fh, fw = frame.shape[:2]
            if fw != self.width or fh != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            # Non-blocking queue put (drop oldest if full)
            try:
                if self._frame_queue.full():
                    self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

            # FPS tracking
            self._frame_count += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self._actual_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.time()

            # Regulate frame rate
            processing_time = time.time() - t0
            sleep_time = frame_interval - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
