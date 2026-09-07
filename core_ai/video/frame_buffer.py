"""
ORBITA Latest Frame Buffer
==========================
Thread-safe, ultra-low-latency single-frame buffer for real-time video streaming.

Key Principles:
  - Buffer size = 1 (drops stale frames immediately)
  - Zero frame queue: fresh frame > every frame
  - Non-blocking multi-consumer reads (AI and Streamer both access the latest frame without consuming it)
  - Real-time timestamp tracking and dropped frame analytics
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FrameData:
    """Wrapper holding a frame, its metadata, and monotonic capture timestamp."""
    __slots__ = ("frame", "frame_id", "timestamp", "width", "height", "source_timestamp")

    def __init__(self, frame: np.ndarray, frame_id: int, timestamp: float, source_timestamp: float = 0.0):
        self.frame = frame
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.source_timestamp = source_timestamp
        h, w = frame.shape[:2]
        self.width = w
        self.height = h

    @property
    def age_ms(self) -> float:
        return max(0.0, (time.monotonic() - self.timestamp) * 1000.0)


class LatestFrameBuffer:
    """
    Lock-minimized single-frame buffer with notification for new arrivals.
    Multiple consumers can read the latest frame independently.
    """

    def __init__(self, name: str = "orbita-buffer"):
        self.name = name
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        
        self._current_frame: Optional[FrameData] = None
        self._frame_id_counter: int = 0
        
        # Metrics
        self._frames_received: int = 0
        self._frames_dropped: int = 0
        self._actual_fps: float = 0.0
        self._fps_counter: int = 0
        self._fps_timer: float = time.monotonic()
        self._last_push_time: float = 0.0

    def push(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        source_timestamp: float = 0.0,
    ) -> int:
        """
        Push a new frame into the buffer.
        Replaces any existing frame immediately (drop-oldest strategy).
        Timestamp should be time.monotonic().
        """
        if frame is None or frame.size == 0:
            return -1

        t = timestamp if timestamp is not None else time.monotonic()
        with self._lock:
            if self._current_frame is not None:
                self._frames_dropped += 1

            self._frame_id_counter += 1
            frame_id = self._frame_id_counter
            self._current_frame = FrameData(frame, frame_id, t, source_timestamp)
            self._frames_received += 1
            self._last_push_time = t

            # FPS calculation using monotonic clock
            self._fps_counter += 1
            now = time.monotonic()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._actual_fps = round(self._fps_counter / elapsed, 1)
                self._fps_counter = 0
                self._fps_timer = now

        # Signal waiting consumers that a fresh frame is ready
        self._new_frame_event.set()
        return frame_id

    def get_latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        """
        Non-blocking read of the latest frame.
        Returns (frame, capture_timestamp, frame_id).
        Does NOT remove the frame from the buffer so other consumers can also read it.
        """
        with self._lock:
            if self._current_frame is None:
                return None, 0.0, 0
            return self._current_frame.frame, self._current_frame.timestamp, self._current_frame.frame_id

    def get_latest_with_metadata(self) -> Tuple[Optional[np.ndarray], float, int, float]:
        """
        Returns (frame, capture_monotonic_timestamp, frame_id, frame_age_ms).
        """
        with self._lock:
            if self._current_frame is None:
                return None, 0.0, 0, 0.0
            return (
                self._current_frame.frame,
                self._current_frame.timestamp,
                self._current_frame.frame_id,
                self._current_frame.age_ms,
            )

    def wait_for_frame(self, timeout: float = 0.1) -> Tuple[Optional[np.ndarray], float, int]:
        """
        Wait until a new frame arrives or timeout expires.
        Clears the event after waking.
        """
        self._new_frame_event.wait(timeout=timeout)
        self._new_frame_event.clear()
        return self.get_latest()

    def clear(self) -> None:
        """Reset the buffer."""
        with self._lock:
            self._current_frame = None
            self._new_frame_event.clear()

    @property
    def fps(self) -> float:
        with self._lock:
            return self._actual_fps

    @property
    def frame_age_ms(self) -> float:
        """Age of the latest frame in milliseconds based on monotonic time."""
        with self._lock:
            if self._current_frame is None:
                return 0.0
            return self._current_frame.age_ms

    @property
    def latency_ms(self) -> float:
        """Latency between camera capture of the latest frame and current monotonic time."""
        return self.frame_age_ms

    @property
    def stats(self) -> dict:
        """Return diagnostic metrics."""
        with self._lock:
            total = self._frames_received
            dropped = self._frames_dropped
            drop_pct = round((dropped / total * 100.0), 1) if total > 0 else 0.0
            cur_id = self._current_frame.frame_id if self._current_frame else 0
            age = round(self._current_frame.age_ms, 1) if self._current_frame else 0.0
            w = self._current_frame.width if self._current_frame else 0
            h = self._current_frame.height if self._current_frame else 0

            return {
                "fps": self._actual_fps,
                "latency_ms": age,
                "frame_age_ms": age,
                "frames_received": total,
                "frames_dropped": dropped,
                "dropped_pct": drop_pct,
                "latest_frame_id": cur_id,
                "buffer_size": 1 if self._current_frame else 0,
                "resolution": f"{w}x{h}" if w > 0 else "0x0",
            }

