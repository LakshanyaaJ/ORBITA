"""
ORBITA Video Recorder
======================
Records annotated experiment video to local MP4.
Completely offline — dynamic resolution, adaptive FPS, and frame integrity.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """
    Records video frames to a local MP4 file with dynamic configuration
    matching the live camera stream resolution and FPS.

    Usage:
        rec = VideoRecorder(config)
        rec.start(experiment_id="EXP001")  # Dimensions auto-adapt on 1st frame
        rec.write(frame)
        rec.stop()
    """

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        fourcc_str = getattr(config, "fourcc", "mp4v")
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self._writer: Optional[cv2.VideoWriter] = None
        self._experiment_id: str = ""
        self._frame_count: int = 0
        self._dropped_frames: int = 0
        self._filepath: str = ""
        self._canonical_filepath: str = ""
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._fps: float = 30.0
        self._start_time: Optional[float] = None
        self._duration: float = 0.0
        self._has_error: bool = False
        self._is_started: bool = False

    def start(
        self,
        experiment_id: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> str:
        """
        Start recording for an experiment.
        If width/height are None, initialization is deferred until the first frame
        so dimensions exactly match the incoming camera capture.
        """
        self.stop()  # Stop and finalize any previous recording

        self._experiment_id = experiment_id
        exp_dir = self.output_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{timestamp}.mp4"
        self._filepath = str(exp_dir / filename)
        self._canonical_filepath = str(exp_dir / "experiment.mp4")

        self._frame_count = 0
        self._dropped_frames = 0
        self._duration = 0.0
        self._has_error = False
        self._is_started = True
        self._start_time = time.time()

        if fps and fps > 0:
            self._fps = float(fps)
        else:
            self._fps = 30.0

        if width and height and width > 0 and height > 0:
            self._width = int(width)
            self._height = int(height)
            self._init_writer()

        logger.info(
            "VideoRecorder: Session initiated for %s (deferred=%s, target_fps=%.1f)",
            experiment_id,
            self._writer is None,
            self._fps,
        )
        return self._filepath

    def _init_writer(self) -> bool:
        """Initialize OpenCV VideoWriter with current width, height, and FPS."""
        if not self._width or not self._height:
            return False

        try:
            self._writer = cv2.VideoWriter(
                self._filepath,
                self.fourcc,
                self._fps,
                (self._width, self._height),
            )
            if not self._writer.isOpened():
                # Try fallback codec if mp4v fails on current OS
                logger.warning("VideoWriter fourcc failed. Trying fallback 'avc1' / 'XVID'...")
                for fallback in ("avc1", "XVID", "MJPG"):
                    fb_fourcc = cv2.VideoWriter_fourcc(*fallback)
                    self._writer = cv2.VideoWriter(
                        self._filepath,
                        fb_fourcc,
                        self._fps,
                        (self._width, self._height),
                    )
                    if self._writer.isOpened():
                        logger.info("VideoWriter opened with fallback codec '%s'", fallback)
                        break

            if not self._writer or not self._writer.isOpened():
                logger.error("VideoWriter failed to open: %s", self._filepath)
                self._writer = None
                self._has_error = True
                return False

            self._has_error = False
            logger.info("VideoWriter initialized: %s (%dx%d @ %.1f FPS)", self._filepath, self._width, self._height, self._fps)
            return True
        except Exception as exc:
            logger.error("Exception during VideoWriter initialization: %s", exc)
            self._writer = None
            self._has_error = True
            return False

    def write(self, frame: np.ndarray) -> None:
        """Write frame to MP4 ensuring dimensional integrity."""
        if not self._is_started or frame is None:
            return

        h, w = frame.shape[:2]

        # Lazy initialization on first frame if dimensions weren't predefined
        if self._writer is None and not self._has_error:
            self._width = w
            self._height = h
            if not self._init_writer():
                self._dropped_frames += 1
                return

        if self._writer is not None and self._writer.isOpened():
            try:
                # Frame integrity: resize safely if camera stream changes resolution
                if w != self._width or h != self._height:
                    frame_to_write = cv2.resize(frame, (self._width, self._height))
                else:
                    frame_to_write = frame

                self._writer.write(frame_to_write)
                self._frame_count += 1
            except Exception as exc:
                logger.error("Failed to write video frame: %s", exc)
                self._dropped_frames += 1
        else:
            self._dropped_frames += 1

    def stop(self) -> str:
        """Finalize and release recording, creating canonical experiment.mp4 copy."""
        if self._start_time:
            self._duration = time.time() - self._start_time

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(
                "Recording finalized: %s (%d frames, duration=%.1fs, dropped=%d)",
                self._filepath,
                self._frame_count,
                self._duration,
                self._dropped_frames,
            )

            # Ensure canonical experiment.mp4 exists in experiment directory
            if self._filepath and os.path.exists(self._filepath) and self._frame_count > 0:
                try:
                    shutil.copyfile(self._filepath, self._canonical_filepath)
                    logger.info("Canonical experiment.mp4 created: %s", self._canonical_filepath)
                except Exception as exc:
                    logger.warning("Could not copy canonical experiment.mp4: %s", exc)

        self._is_started = False
        return self._filepath

    def is_recording(self) -> bool:
        return self._is_started and not self._has_error

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def canonical_filepath(self) -> str:
        return self._canonical_filepath

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def get_telemetry(self) -> dict:
        """Return structured recording telemetry."""
        cur_duration = (time.time() - self._start_time) if (self._start_time and self._is_started) else self._duration
        status = "ACTIVE" if self.is_recording() else ("ERROR" if self._has_error else "STOPPED")
        return {
            "status": status,
            "recorded_frames": self._frame_count,
            "recording_duration": round(cur_duration, 1),
            "recording_resolution": f"{self._width}x{self._height}" if (self._width and self._height) else "N/A",
            "recording_fps": round(self._fps, 1),
            "recording_path": self._filepath,
            "canonical_path": self._canonical_filepath,
            "dropped_recording_frames": self._dropped_frames,
        }
