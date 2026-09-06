"""
ORBITA Dataset & Learning System
================================
Provides video ingestion, adaptive frame extraction, dataset management,
assisted annotation, candidate run recording, quality gating, and human review queue.
"""

from core_ai.dataset.video_ingest import VideoIngestPipeline, VideoMetadata
from core_ai.dataset.frame_extractor import FrameExtractor, ExtractionConfig, ExtractedFrame
from core_ai.dataset.dataset_manager import DatasetManager, DatasetManifest
from core_ai.dataset.annotator import AssistedAnnotator, AnnotationStatus
from core_ai.dataset.review_queue import ReviewQueue, CandidateSample

__all__ = [
    "VideoIngestPipeline",
    "VideoMetadata",
    "FrameExtractor",
    "ExtractionConfig",
    "ExtractedFrame",
    "DatasetManager",
    "DatasetManifest",
    "AssistedAnnotator",
    "AnnotationStatus",
    "ReviewQueue",
    "CandidateSample",
]
