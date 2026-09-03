"""
ORBITA MJPEG Streamer
=====================
Provides a local-network MJPEG stream of the annotated video feed.
No cloud or external infrastructure required.

The stream is served by the FastAPI backend at /video_feed.
This module maintains the latest frame for serving.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MJPEGStreamer:
    """
    Holds the latest annotated frame and serves it as MJPEG.

    Usage:
        streamer = MJPEGStreamer()
        streamer.update(frame)                    # Call each pipeline iteration
        # FastAPI endpoint calls streamer.generate()
    """

    def __init__(self, jpeg_quality: int = 80):
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._blank_frame: bytes = self._make_blank_jpeg()

    def update(self, frame: np.ndarray) -> None:
        """Update the latest frame (thread-safe)."""
        ret, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if ret:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    def get_latest_jpeg(self) -> bytes:
        """Get latest JPEG bytes."""
        with self._lock:
            return self._latest_jpeg or self._blank_frame

    async def generate_mjpeg(self):
        """
        Async generator for MJPEG multipart response.
        Use as FastAPI StreamingResponse source.
        """
        import asyncio
        while True:
            frame_bytes = self.get_latest_jpeg()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            await asyncio.sleep(0.033)  # ~30 fps cap for MJPEG

    @staticmethod
    def _make_blank_jpeg() -> bytes:
        """Create a black placeholder frame."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "ORBITA — Waiting for stream...",
                    (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2)
        _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()
