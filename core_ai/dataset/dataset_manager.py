"""
ORBITA Dataset Manager
======================
Manages filesystem directory layout, split assignment by video/trial,
manifest generation, and YOLO data.yaml configuration.
Ensures zero data leakage between train/val/test splits.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ORBITA_CLASSES = [
    "PERSON",
    "MAIN_BOX",
    "RED_BOX",
    "YELLOW_BOX",
    "SAMPLE",
    "TOOL",
]


@dataclass
class DatasetManifest:
    """Metadata tracking dataset statistics, versions, and split sizes."""
    dataset_version: str = "v1"
    source: str = "vdata"
    created_at: str = ""
    classes: list[str] = field(default_factory=lambda: list(ORBITA_CLASSES))
    total_videos: int = 0
    total_extracted_frames: int = 0
    annotated_frames: int = 0
    verified_frames: int = 0
    approved_samples: int = 0
    pending_review: int = 0
    rejected_samples: int = 0
    split_counts: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "val": 0, "test": 0}
    )
    video_split_map: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetManager:
    """
    Coordinates directory creation, video split assignment, and YOLO dataset preparation.
    """

    def __init__(self, base_dir: str | Path = "datasets/orbita"):
        self.base_dir = Path(base_dir).resolve()
        self.images_dir = self.base_dir / "images"
        self.labels_dir = self.base_dir / "labels"
        self.metadata_dir = self.base_dir / "metadata"
        self.candidates_dir = self.base_dir / "candidates"
        self.data_yaml_path = self.base_dir / "data.yaml"
        self.manifest_path = self.metadata_dir / "manifest.json"

        self.splits = ["train", "val", "test"]

    def initialize_directories(self) -> None:
        """Create clean dataset directory hierarchy."""
        for split in self.splits:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.generate_data_yaml()

    def generate_data_yaml(self) -> Path:
        """Generate standard YOLOv8 data.yaml."""
        yaml_lines = [
            f"# ORBITA Dataset Configuration",
            f"path: {self.base_dir.as_posix()}",
            f"train: images/train",
            f"val: images/val",
            f"test: images/test",
            f"",
            f"# Class mapping",
            f"names:",
        ]
        for idx, name in enumerate(ORBITA_CLASSES):
            yaml_lines.append(f"  {idx}: {name}")

        self.data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yaml_lines) + "\n")

        return self.data_yaml_path

    def assign_video_splits(
        self,
        video_ids: list[str],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> dict[str, str]:
        """
        Assign each video/trial to a specific split.
        Splitting by video/trial strictly avoids data leakage between adjacent frames.
        """
        video_ids = sorted(list(set(video_ids)))
        n = len(video_ids)
        mapping: dict[str, str] = {}

        if n == 0:
            return mapping
        elif n == 1:
            mapping[video_ids[0]] = "train"
        elif n == 2:
            mapping[video_ids[0]] = "train"
            mapping[video_ids[1]] = "val"
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            for i, vid in enumerate(video_ids):
                if i < n_train:
                    mapping[vid] = "train"
                elif i < n_train + n_val:
                    mapping[vid] = "val"
                else:
                    mapping[vid] = "test"

        return mapping

    def update_manifest(
        self,
        total_videos: int,
        video_split_map: dict[str, str],
        dataset_version: str = "v1",
    ) -> DatasetManifest:
        """Scan directories and calculate accurate manifest metrics."""
        import datetime

        split_counts = {"train": 0, "val": 0, "test": 0}
        total_extracted = 0
        annotated = 0

        for split in self.splits:
            img_dir = self.images_dir / split
            lbl_dir = self.labels_dir / split
            if img_dir.exists():
                imgs = [p for p in img_dir.glob("*.jpg")]
                split_counts[split] = len(imgs)
                total_extracted += len(imgs)

            if lbl_dir.exists():
                lbls = [p for p in lbl_dir.glob("*.txt")]
                annotated += len(lbls)

        # Count candidate review queue
        approved = len(list(self.candidates_dir.glob("approved_*"))) if self.candidates_dir.exists() else 0
        pending = len(list(self.candidates_dir.glob("pending_*"))) if self.candidates_dir.exists() else 0
        rejected = len(list(self.candidates_dir.glob("rejected_*"))) if self.candidates_dir.exists() else 0

        manifest = DatasetManifest(
            dataset_version=dataset_version,
            source="vdata",
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            classes=list(ORBITA_CLASSES),
            total_videos=total_videos,
            total_extracted_frames=total_extracted,
            annotated_frames=annotated,
            approved_samples=approved,
            pending_review=pending,
            rejected_samples=rejected,
            split_counts=split_counts,
            video_split_map=video_split_map,
        )

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        return manifest

    def get_manifest(self) -> Optional[DatasetManifest]:
        """Load manifest if present."""
        if not self.manifest_path.exists():
            return None
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid_keys = {field_obj.name for field_obj in fields(DatasetManifest)}
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return DatasetManifest(**filtered_data)
        except Exception as exc:
            logger.warning("Failed to load dataset manifest: %s", exc)
            return None
