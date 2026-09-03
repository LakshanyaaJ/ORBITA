"""
ORBITA Video Recorder
======================
Records annotated experiment video to local MP4.
Completely offline — no cloud storage.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """
    Records video frames to a local MP4 file.

    Usage:
        rec = VideoRecorder(config)
        rec.start(experiment_id="EXP001")
        rec.write(frame)
        rec.stop()
    """

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.fourcc = cv2.VideoWriter_fourcc(*config.fourcc)
        self._writer: Optional[cv2.VideoWriter] = None
        self._experiment_id: str = ""
        self._frame_count: int = 0
        self._filepath: str = ""

    def start(self, experiment_id: str, width: int = 640, height: int = 480, fps: int = 20) -> str:
        """Start recording. Returns the output file path."""
        self.stop()  # Stop any previous recording

        self._experiment_id = experiment_id
        exp_dir = self.output_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{timestamp}.mp4"
        self._filepath = str(exp_dir / filename)

        self._writer = cv2.VideoWriter(
            self._filepath,
            self.fourcc,
            fps,
            (width, height),
        )
        if not self._writer.isOpened():
            logger.error("VideoWriter failed to open: %s", self._filepath)
            self._writer = None
            return ""

        self._frame_count = 0
        logger.info("Recording started: %s", self._filepath)
        return self._filepath

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(frame)
            self._frame_count += 1

    def stop(self) -> str:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Recording stopped: %s (%d frames)", self._filepath, self._frame_count)
        return self._filepath

    def is_recording(self) -> bool:
        return self._writer is not None and self._writer.isOpened()

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def frame_count(self) -> int:
        return self._frame_count
