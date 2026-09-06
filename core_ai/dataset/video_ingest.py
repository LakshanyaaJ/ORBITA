"""
ORBITA Video Ingest Pipeline
============================
Discovers, probes, and catalogs reference experiment videos from `vdata/`.
Extracts video metadata (duration, FPS, resolution, codec, experiment association)
and produces a structured Ingestion Report without modifying original sources.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class VideoMetadata:
    """Detailed metadata for a single video file."""
    video_id: str
    file_name: str
    file_path: str
    file_size_bytes: int
    file_size_mb: float
    experiment_id: str
    duration_seconds: float
    total_frames: int
    fps: float
    width: int
    height: int
    aspect_ratio: str
    codec: str
    has_annotations: bool = False
    annotation_format: Optional[str] = None


@dataclass
class IngestionReport:
    """Summary of ingested reference videos across experiments."""
    total_videos: int
    unique_experiments: int
    experiment_ids: list[str]
    total_duration_seconds: float
    total_frames: int
    average_fps: float
    resolutions: list[str]
    codecs: list[str]
    has_annotations: bool
    videos: list[VideoMetadata] = field(default_factory=list)

    def summary_text(self) -> str:
        """Human-readable text summary of the ingested videos."""
        minutes = int(self.total_duration_seconds // 60)
        seconds = int(self.total_duration_seconds % 60)
        hours = minutes // 60
        rem_min = minutes % 60
        duration_fmt = f"{hours}h {rem_min}m {seconds}s" if hours > 0 else f"{rem_min}m {seconds}s"

        res_summary = ", ".join(self.resolutions) if self.resolutions else "Unknown"
        annot_summary = "Present" if self.has_annotations else "None"

        lines = [
            "=" * 50,
            "ORBITA REFERENCE VIDEO INGESTION REPORT",
            "=" * 50,
            f"Videos found:    {self.total_videos}",
            f"Experiments:     {self.unique_experiments} ({', '.join(self.experiment_ids)})",
            f"Total duration:  {duration_fmt} ({self.total_duration_seconds:.1f}s)",
            f"Total frames:    {self.total_frames:,}",
            f"Average FPS:     {self.average_fps:.1f}",
            f"Resolutions:     {res_summary}",
            f"Codecs:          {', '.join(self.codecs)}",
            f"Annotations:     {annot_summary}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VideoIngestPipeline:
    """
    Ingests and probes reference experiment video directories.

    Usage:
        pipeline = VideoIngestPipeline(vdata_dir="vdata")
        report = pipeline.scan_and_report()
        print(report.summary_text())
    """

    def __init__(self, vdata_dir: str | Path = "vdata"):
        self.vdata_dir = Path(vdata_dir)

    def scan_and_report(self, output_json_path: Optional[str | Path] = None) -> IngestionReport:
        """
        Scan directory recursively for video files, probe technical parameters,
        and generate an IngestionReport.
        """
        if not self.vdata_dir.exists():
            logger.warning("vdata directory '%s' does not exist.", self.vdata_dir)
            report = IngestionReport(
                total_videos=0,
                unique_experiments=0,
                experiment_ids=[],
                total_duration_seconds=0.0,
                total_frames=0,
                average_fps=0.0,
                resolutions=[],
                codecs=[],
                has_annotations=False,
                videos=[],
            )
            return report

        video_files = [
            p for p in self.vdata_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        video_files.sort()

        videos_meta: list[VideoMetadata] = []
        experiment_set: set[str] = set()
        resolutions_set: set[str] = set()
        codecs_set: set[str] = set()
        any_annotated = False

        for vpath in video_files:
            meta = self._probe_video(vpath)
            if meta:
                videos_meta.append(meta)
                experiment_set.add(meta.experiment_id)
                resolutions_set.add(f"{meta.width}x{meta.height}")
                codecs_set.add(meta.codec)
                if meta.has_annotations:
                    any_annotated = True

        total_videos = len(videos_meta)
        total_duration = sum(v.duration_seconds for v in videos_meta)
        total_frames = sum(v.total_frames for v in videos_meta)
        avg_fps = (sum(v.fps for v in videos_meta) / total_videos) if total_videos > 0 else 0.0

        report = IngestionReport(
            total_videos=total_videos,
            unique_experiments=len(experiment_set),
            experiment_ids=sorted(list(experiment_set)),
            total_duration_seconds=round(total_duration, 2),
            total_frames=total_frames,
            average_fps=round(avg_fps, 2),
            resolutions=sorted(list(resolutions_set)),
            codecs=sorted(list(codecs_set)),
            has_annotations=any_annotated,
            videos=videos_meta,
        )

        if output_json_path:
            out_path = Path(output_json_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info("Ingestion report written to %s", out_path)

        return report

    def _probe_video(self, video_path: Path) -> Optional[VideoMetadata]:
        """Probe technical properties of a single video using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Failed to open video file: %s", video_path)
            return None

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

            codec = ""
            for i in range(4):
                c = chr((fourcc_int >> (8 * i)) & 0xFF)
                if c.isprintable():
                    codec += c
            codec = codec.strip() or "unknown"

            duration = total_frames / fps if fps > 0 else 0.0

            file_size_bytes = video_path.stat().st_size
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

            # Detect experiment ID
            rel_parts = video_path.relative_to(self.vdata_dir).parts
            if len(rel_parts) > 1:
                experiment_id = rel_parts[0]
            else:
                # Default to root or standard experiment ID
                experiment_id = "EXP_REF_01"

            # Check if matching annotations exist
            has_annotations, annot_fmt = self._check_annotations(video_path)

            ar_ratio = f"{width}:{height}" if width and height else "unknown"

            video_id = video_path.stem

            return VideoMetadata(
                video_id=video_id,
                file_name=video_path.name,
                file_path=str(video_path.resolve()),
                file_size_bytes=file_size_bytes,
                file_size_mb=file_size_mb,
                experiment_id=experiment_id,
                duration_seconds=round(duration, 2),
                total_frames=total_frames,
                fps=round(fps, 2),
                width=width,
                height=height,
                aspect_ratio=ar_ratio,
                codec=codec,
                has_annotations=has_annotations,
                annotation_format=annot_fmt,
            )
        finally:
            cap.release()

    def _check_annotations(self, video_path: Path) -> tuple[bool, Optional[str]]:
        """Check for corresponding annotation files in same or labels/ directory."""
        stem = video_path.stem
        parent = video_path.parent

        candidate_extensions = [
            (".json", "json"),
            (".txt", "yolo_txt"),
            (".csv", "csv"),
            (".xml", "voc_xml"),
        ]

        # Check adjacent file
        for ext, fmt in candidate_extensions:
            if (parent / f"{stem}{ext}").exists():
                return True, fmt

        # Check sibling labels directory
        labels_dir = parent / "labels"
        if labels_dir.exists():
            for ext, fmt in candidate_extensions:
                if (labels_dir / f"{stem}{ext}").exists():
                    return True, fmt

        return False, None
