"""
ORBITA Ultra-Low-Latency MJPEG Streamer
======================================
Provides an optimized local-network MJPEG stream of the video feed.
Guarantees zero frame queue backlog: clients always receive the latest frame.

Features:
  - Fast baseline JPEG compression (cv2.IMWRITE_JPEG_OPTIMIZE=0, quality 65-75)
  - Single-encode frame caching for multiple concurrent viewers
  - Asynchronous event notification for instant frame dispatch (<1 ms dispatch lag)
  - Automatic frame dropping if a client network falls behind
  - Telemetry tracking: stream FPS, encoding time (ms), active client count
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, Optional, Set

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MJPEGStreamer:
    """
    Holds the latest frame, encodes to JPEG with minimal latency,
    and serves it to HTTP StreamingResponse clients.
    """

    def __init__(self, jpeg_quality: int = 70):
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()

        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_id: int = 0
        self._latest_timestamp: float = 0.0
        self._encode_latency_ms: float = 0.0

        # Blank placeholder frame
        self._blank_frame: bytes = self._make_blank_jpeg()

        # Stream FPS tracking
        self._stream_fps: float = 0.0
        self._stream_frame_count: int = 0
        self._fps_timer: float = time.time()

        # Event notification for async streaming generators
        self._new_frame_events: Set[asyncio.Event] = set()
        self._client_count: int = 0

    @property
    def client_count(self) -> int:
        return self._client_count

    @property
    def stream_fps(self) -> float:
        return self._stream_fps

    @property
    def encode_latency_ms(self) -> float:
        return self._encode_latency_ms

    def update(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Compress and update the latest frame (thread-safe).
        Called by the streaming producer loop at camera FPS.
        """
        if frame is None or frame.size == 0:
            return

        t0 = time.time()

        # Fast JPEG encoding: OPTIMIZE=0 disables multi-pass Huffman for minimum CPU overhead
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,
        ]
        ret, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            return

        encoded_bytes = buf.tobytes()
        t_enc = (time.time() - t0) * 1000.0

        with self._lock:
            self._latest_jpeg = encoded_bytes
            self._latest_frame_id += 1
            self._latest_timestamp = timestamp or time.time()
            self._encode_latency_ms = round(t_enc, 2)

            # Stream FPS counter
            self._stream_frame_count += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._stream_fps = round(self._stream_frame_count / elapsed, 1)
                self._stream_frame_count = 0
                self._fps_timer = now

            # Wake up all waiting async streaming generators
            for event in list(self._new_frame_events):
                event.set()

    def get_latest_jpeg(self) -> bytes:
        """Get latest JPEG bytes."""
        with self._lock:
            return self._latest_jpeg or self._blank_frame

    async def generate_mjpeg(self):
        """
        Async generator for FastAPI StreamingResponse.
        Drops stale frames automatically if client is slow.
        """
        event = asyncio.Event()
        with self._lock:
            self._new_frame_events.add(event)
            self._client_count += 1

        last_sent_frame_id = -1
        boundary_header = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        boundary_footer = b"\r\n"

        try:
            while True:
                # Wait for next frame or brief timeout (max 100ms)
                try:
                    await asyncio.wait_for(event.wait(), timeout=0.08)
                except asyncio.TimeoutError:
                    pass
                finally:
                    event.clear()

                with self._lock:
                    current_id = self._latest_frame_id
                    frame_bytes = self._latest_jpeg or self._blank_frame

                # Only yield if this is a fresh frame or heartbeat
                if current_id != last_sent_frame_id or current_id == 0:
                    last_sent_frame_id = current_id
                    yield boundary_header + frame_bytes + boundary_footer

                # Tiny yield to allow event loop cooperative scheduling without throttle
                await asyncio.sleep(0.001)

        finally:
            with self._lock:
                self._new_frame_events.discard(event)
                self._client_count = max(0, self._client_count - 1)

    def get_diagnostics(self) -> Dict[str, float]:
        """Return streaming diagnostics."""
        with self._lock:
            return {
                "stream_fps": self._stream_fps,
                "encode_latency_ms": self._encode_latency_ms,
                "active_clients": self._client_count,
                "jpeg_quality": self.jpeg_quality,
            }

    @staticmethod
    def _make_blank_jpeg() -> bytes:
        """Create a lightweight placeholder frame."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank, "ORBITA — Initializing Stream...",
            (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2
        )
        _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 65, cv2.IMWRITE_JPEG_OPTIMIZE, 0])
        return buf.tobytes()
