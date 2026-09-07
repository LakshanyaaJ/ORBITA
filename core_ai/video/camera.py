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
        self.rotation = getattr(config, "rotation", 0)
        self.loop_video = getattr(config, "loop_video", False)
        self.sequential = getattr(config, "sequential", False)

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_buffer = LatestFrameBuffer(name="local-cam-buffer")
        self._frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._threaded = False
        self._is_simulator = False  # Set True when using push_frame
        self._is_eof = False
        self._frame_index: int = 0
        self._total_video_frames: int = 0
        self._frame_width: int = 0
        self._frame_height: int = 0
        self._video_fps: float = float(self.fps_target)
        self._start_time: float = 0.0

        self._actual_fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.time()
        self._latency_ms: float = 0.0

    @property
    def is_eof(self) -> bool:
        return self._is_eof

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def total_video_frames(self) -> int:
        return self._total_video_frames

    @property
    def frame_buffer(self) -> LatestFrameBuffer:
        return self._frame_buffer

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def start(self, threaded: Optional[bool] = None) -> bool:
        """Start the camera or video source. Returns True if successful."""
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

            self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._video_fps = raw_fps if raw_fps > 0 else float(self.fps_target)
            raw_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._total_video_frames = max(0, raw_count)

            dur_sec = (self._total_video_frames / max(1.0, self._video_fps)) if self._total_video_frames > 0 else 0.0
            print(f"VIDEO:\npath={self.source}\nopened={self._cap.isOpened()}\nfps={self._video_fps:.2f}\nwidth={self._frame_width}\nheight={self._frame_height}\nframe_count={self._total_video_frames}\nduration={dur_sec:.2f}s")
            logger.info(
                "\nVIDEO:\npath=%s\nopened=%s\nfps=%.2f\nwidth=%d\nheight=%d\nframe_count=%d\nduration=%.2fs",
                self.source,
                self._cap.isOpened(),
                self._video_fps,
                self._frame_width,
                self._frame_height,
                self._total_video_frames,
                dur_sec,
            )

            self._running = True
            self._is_eof = False
            self._frame_index = 0
            self._start_time = time.time()

            should_thread = threaded if threaded is not None else (not self.sequential)
            self._threaded = should_thread
            if should_thread:
                self._thread = threading.Thread(
                    target=self._capture_loop, daemon=True, name="orbita-camera"
                )
                self._thread.start()
            logger.info("Camera started: source=%s %dx%d @%dfps (threaded=%s)",
                        self.source, self.width, self.height, self.fps_target, should_thread)
            return True
        except Exception as exc:
            logger.error("Camera start failed: %s", exc)
            return False

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.flip:
            frame = cv2.flip(frame, 1)

        rot = getattr(self, "rotation", 0)
        if rot == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rot == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        fh, fw = frame.shape[:2]
        if self.width > 0 and self.height > 0 and (fw != self.width or fh != self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return frame

    def _read_sequential_frame(self) -> Optional[np.ndarray]:
        if not self._running or self._cap is None or self._is_eof:
            return None
        t0 = time.time()
        ret, frame = self._cap.read()
        if not ret:
            if isinstance(self.source, str) and getattr(self, "loop_video", False):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_index = 0
                ret, frame = self._cap.read()
                if not ret:
                    self._is_eof = True
                    return None
            else:
                self._is_eof = True
                dur = time.time() - self._start_time
                print(f"EOF_REACHED\nframes_processed={self._frame_index}\nduration_processed={dur:.2f}s")
                logger.info(
                    "\nEOF_REACHED\nframes_processed=%d\nduration_processed=%.2fs",
                    self._frame_index, dur
                )
                return None

        self._frame_index += 1
        frame = self._preprocess_frame(frame)
        self._frame_buffer.push(frame, t0)
        self._latency_ms = 0.0
        return frame

    def read(self) -> Optional[np.ndarray]:
        """Get latest frame (non-blocking if threaded, sequential if non-threaded)."""
        if not self._running:
            return None

        if not self._threaded:
            return self._read_sequential_frame()

        frame, ts, _ = self._frame_buffer.get_latest()
        if frame is not None:
            self._latency_ms = max(0.0, (time.monotonic() - ts) * 1000.0)
            return frame

        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def read_with_metadata(self) -> Tuple[Optional[np.ndarray], float, float]:
        """Returns (frame, actual_fps, frame_age_ms)."""
        frame = self.read()
        return frame, self._actual_fps, self._latency_ms

    def push_frame(self, frame: np.ndarray) -> None:
        """Push an externally generated frame into the buffer."""
        if not self._running:
            return
        t = time.monotonic()
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
            self._thread = None
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
        from pathlib import Path
        is_file = isinstance(self.source, str) and Path(self.source).is_file()
        target_dt = 1.0 / max(1.0, self._video_fps)

        while self._running and self._cap is not None:
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self.source, str):
                    if getattr(self, "loop_video", False):
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._frame_index = 0
                        continue
                    else:
                        self._is_eof = True
                        dur = time.time() - self._start_time
                        print(f"EOF_REACHED\nframes_processed={self._frame_index}\nduration_processed={dur:.2f}s")
                        logger.info(
                            "\nEOF_REACHED\nframes_processed=%d\nduration_processed=%.2fs",
                            self._frame_index, dur
                        )
                        time.sleep(0.05)
                        continue
                logger.warning("Camera read failed.")
                time.sleep(0.05)
                continue

            self._frame_index += 1
            frame = self._preprocess_frame(frame)

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

            # Rate pacing for video files
            if is_file:
                elapsed_frame = time.time() - t0
                if elapsed_frame < target_dt:
                    time.sleep(target_dt - elapsed_frame)
