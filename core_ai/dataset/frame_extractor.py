"""
ORBITA Intelligent Frame Extractor & Quality Filter
===================================================
Converts reference and candidate videos into diverse, high-quality image frames.
Applies:
  1. Configurable temporal subsampling (sample_fps / min_frame_gap)
  2. Laplacian sharpness quality filtering (rejects motion blur)
  3. Histogram duplicate suppression (rejects nearly identical consecutive frames)
  4. Resolution normalization (preserves aspect ratio while constraining dimensions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for frame sampling and quality filtering."""
    sample_fps: float = 2.5                 # Extract ~2-3 frames per second of video
    min_frame_gap: int = 5                  # Absolute minimum frames between samples
    blur_threshold: float = 60.0            # Minimum Laplacian variance for sharpness
    duplicate_threshold: float = 0.95       # Max histogram correlation before dropping
    max_dimension: int = 1280               # Max width or height (preserves aspect ratio)
    jpeg_quality: int = 92                  # Compression quality for saved frames
    quality_filter_enabled: bool = True     # Toggle blur and duplicate rejection


@dataclass
class ExtractedFrame:
    """Metadata and status for an extracted frame."""
    frame_index: int
    timestamp_seconds: float
    sharpness: float
    similarity_to_prev: float
    is_kept: bool
    rejection_reason: Optional[str] = None
    saved_path: Optional[str] = None


class FrameExtractor:
    """
    Intelligently samples and filters frames from experiment videos.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()

    def extract_from_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        filename_prefix: Optional[str] = None,
    ) -> list[ExtractedFrame]:
        """
        Extract sampled frames from video, apply quality filters, and save to output_dir.

        Returns list of ExtractedFrame records (both kept and rejected for full transparency).
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = filename_prefix or video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # Step size in raw frames to achieve target sample_fps
        frame_step = max(self.config.min_frame_gap, int(round(fps / self.config.sample_fps)))

        results: list[ExtractedFrame] = []
        prev_hist: Optional[np.ndarray] = None
        raw_idx = 0
        kept_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if raw_idx % frame_step != 0:
                    raw_idx += 1
                    continue

                timestamp = raw_idx / fps if fps > 0 else 0.0

                # Downsample first: computing Laplacian and histogram on 720p is 10x faster than 4K
                normalized_frame = self._normalize_resolution(frame, self.config.max_dimension)

                # 1. Measure sharpness (Laplacian variance on grayscale)
                gray = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                # 2. Measure similarity to previous frame (HSV color histogram)
                hsv = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                sim = 0.0
                if prev_hist is not None:
                    sim = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))

                # 3. Quality gating
                is_kept = True
                rejection_reason = None

                if self.config.quality_filter_enabled:
                    if sharpness < self.config.blur_threshold:
                        is_kept = False
                        rejection_reason = f"Blurry (sharpness {sharpness:.1f} < {self.config.blur_threshold})"
                    elif prev_hist is not None and sim > self.config.duplicate_threshold:
                        is_kept = False
                        rejection_reason = f"Duplicate (similarity {sim:.3f} > {self.config.duplicate_threshold})"

                saved_path = None
                if is_kept:
                    frame_filename = f"{prefix}_f{raw_idx:06d}.jpg"
                    frame_file_path = output_dir / frame_filename
                    cv2.imwrite(
                        str(frame_file_path),
                        normalized_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
                    )
                    saved_path = str(frame_file_path.resolve())
                    prev_hist = hist
                    kept_count += 1

                results.append(ExtractedFrame(
                    frame_index=raw_idx,
                    timestamp_seconds=round(timestamp, 3),
                    sharpness=round(sharpness, 2),
                    similarity_to_prev=round(sim, 3),
                    is_kept=is_kept,
                    rejection_reason=rejection_reason,
                    saved_path=saved_path,
                ))

                raw_idx += 1

        finally:
            cap.release()

        logger.info(
            "Extracted %d / %d sampled frames from %s (Total video frames: %d)",
            kept_count, len(results), video_path.name, total_frames,
        )
        return results

    def _normalize_resolution(self, frame: np.ndarray, max_dim: int) -> np.ndarray:
        """Resize frame preserving aspect ratio if larger than max_dim."""
        h, w = frame.shape[:2]
        if max(h, w) <= max_dim:
            return frame

        scale = max_dim / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
