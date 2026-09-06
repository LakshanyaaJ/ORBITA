"""
ORBITA Human Review Queue
=========================
Governs candidate dataset samples recorded from real experiments.
Provides human-in-the-loop approval, rejection, and label editing
before candidate samples are admitted into the active training set.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CandidateSample:
    """Summary of a staged candidate run in the review queue."""
    run_id: str
    experiment_id: str
    status: str                         # "pending" | "approved" | "rejected"
    recorded_at: str
    duration_seconds: float
    samples_count: int
    quality_score: float
    is_acceptable: bool
    rejection_reasons: list[str] = field(default_factory=list)
    folder_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReviewQueue:
    """
    Manages the staging and human validation workflow for recorded candidate samples.
    """

    def __init__(
        self,
        candidates_dir: str | Path = "datasets/orbita/candidates",
        dataset_base_dir: str | Path = "datasets/orbita",
    ):
        self.candidates_dir = Path(candidates_dir).resolve()
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_base_dir = Path(dataset_base_dir).resolve()

    def list_candidates(self, status_filter: Optional[str] = None) -> list[CandidateSample]:
        """List all candidate runs matching filter."""
        results: list[CandidateSample] = []
        if not self.candidates_dir.exists():
            return results

        for folder in sorted(self.candidates_dir.iterdir()):
            if not folder.is_dir():
                continue

            meta_file = folder / "run_metadata.json"
            if not meta_file.exists():
                continue

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                status = "pending"
                if folder.name.startswith("approved_"):
                    status = "approved"
                elif folder.name.startswith("rejected_"):
                    status = "rejected"
                elif folder.name.startswith("pending_"):
                    status = "pending"

                if status_filter and status != status_filter:
                    continue

                quality = meta.get("quality", {})

                results.append(CandidateSample(
                    run_id=meta.get("run_id", folder.name),
                    experiment_id=meta.get("experiment_id", "UNKNOWN"),
                    status=status,
                    recorded_at=meta.get("recorded_at", ""),
                    duration_seconds=meta.get("duration_seconds", 0.0),
                    samples_count=meta.get("saved_samples_count", 0),
                    quality_score=quality.get("overall_score", 0.0),
                    is_acceptable=quality.get("is_acceptable", False),
                    rejection_reasons=quality.get("rejection_reasons", []),
                    folder_name=folder.name,
                ))
            except Exception as exc:
                logger.warning("Error reading candidate metadata in %s: %s", folder.name, exc)

        return results

    def get_candidate_details(self, run_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full details, image paths, and telemetry for a specific run."""
        folder = self._find_folder_by_run_id(run_id)
        if not folder:
            return None

        meta_file = folder / "run_metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        images_dir = folder / "images"
        image_files = []
        if images_dir.exists():
            image_files = [p.name for p in sorted(images_dir.glob("*.jpg"))]

        meta["images"] = image_files
        meta["folder_name"] = folder.name
        return meta

    def approve_candidate(
        self,
        run_id: str,
        reviewer_name: str = "human_reviewer",
        target_split: str = "train",
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Approve candidate run: copy images and labels into datasets/orbita/{images,labels}/{split}.
        """
        folder = self._find_folder_by_run_id(run_id)
        if not folder:
            raise FileNotFoundError(f"Candidate run '{run_id}' not found.")

        meta_file = folder / "run_metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Destination directories
        dest_img_dir = self.dataset_base_dir / "images" / target_split
        dest_lbl_dir = self.dataset_base_dir / "labels" / target_split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Storage for approved runs (Section 28)
        approved_storage_dir = self.dataset_base_dir / "approved"
        approved_storage_dir.mkdir(parents=True, exist_ok=True)

        images_dir = folder / "images"
        copied_count = 0
        import datetime

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        clean_run_id = meta.get("run_id", run_id)
        experiment_id = meta.get("experiment_id", "UNKNOWN")
        model_version = meta.get("model_version", "unknown")

        provenance_entries = []

        if images_dir.exists():
            for img_file in images_dir.glob("*.jpg"):
                dest_img = dest_img_dir / img_file.name
                shutil.copy2(img_file, dest_img)

                # Look for corresponding label
                lbl_file = images_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    shutil.copy2(lbl_file, dest_lbl_dir / lbl_file.name)

                # Write verified sidecar .meta.json for supervised YOLO training safety gate
                meta_sidecar = dest_lbl_dir / f"{img_file.stem}.meta.json"
                sidecar_data = {
                    "image_name": img_file.name,
                    "image_path": str(dest_img.resolve()),
                    "label_path": str((dest_lbl_dir / f"{img_file.stem}.txt").resolve()),
                    "meta_path": str(meta_sidecar.resolve()),
                    "status": "verified",
                    "annotated_by": "live_run_candidate",
                    "verified_by": reviewer_name,
                    "verification_notes": notes or "Approved from human review queue",
                    "provenance": {
                        "source": "live_run",
                        "source_video": meta.get("video_path") or clean_run_id,
                        "experiment": experiment_id,
                        "run_id": clean_run_id,
                        "model_version": model_version,
                        "approval_status": "approved",
                        "approved_by": reviewer_name,
                        "timestamp": now_iso,
                        "target_split": target_split,
                    }
                }
                with open(meta_sidecar, "w", encoding="utf-8") as sf:
                    json.dump(sidecar_data, sf, indent=2)

                provenance_entries.append({
                    "sample_id": img_file.name,
                    "source": "live_run",
                    "source_video": meta.get("video_path") or clean_run_id,
                    "experiment": experiment_id,
                    "run_id": clean_run_id,
                    "model_version": model_version,
                    "approval_status": "approved",
                    "approved_by": reviewer_name,
                    "timestamp": now_iso,
                    "target_split": target_split,
                })

                copied_count += 1

        # Update metadata
        meta["approval"] = {
            "approved_by": reviewer_name,
            "target_split": target_split,
            "notes": notes,
            "approved_samples_count": copied_count,
            "approved_at": now_iso,
        }
        meta["review_status"] = "approved"

        # Rename candidate folder to approved_ in candidates dir
        new_folder = self.candidates_dir / f"approved_{clean_run_id}"
        if folder != new_folder:
            if new_folder.exists():
                shutil.rmtree(new_folder)
            folder.rename(new_folder)
            folder = new_folder

        with open(folder / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Store a copy in datasets/orbita/approved/ (Section 28)
        run_approved_copy = approved_storage_dir / clean_run_id
        if run_approved_copy.exists():
            shutil.rmtree(run_approved_copy)
        shutil.copytree(folder, run_approved_copy)

        # Append to sample_provenance.json (Section 17)
        prov_file = self.dataset_base_dir / "metadata" / "sample_provenance.json"
        prov_file.parent.mkdir(parents=True, exist_ok=True)
        all_provenance = []
        if prov_file.exists():
            try:
                with open(prov_file, "r", encoding="utf-8") as pf:
                    all_provenance = json.load(pf)
                    if not isinstance(all_provenance, list):
                        all_provenance = []
            except Exception:
                all_provenance = []
        all_provenance.extend(provenance_entries)
        with open(prov_file, "w", encoding="utf-8") as pf:
            json.dump(all_provenance, pf, indent=2)

        logger.info("Candidate run %s APPROVED by %s into split '%s' (%d samples, provenance logged)",
                    run_id, reviewer_name, target_split, copied_count)

        return {
            "status": "approved",
            "run_id": clean_run_id,
            "samples_added": copied_count,
            "target_split": target_split,
            "approved_path": str(run_approved_copy.resolve()),
        }

    def reject_candidate(
        self,
        run_id: str,
        reviewer_name: str = "human_reviewer",
        reason: str = "Quality gate rejection",
    ) -> dict[str, Any]:
        """
        Reject candidate run with reason.
        """
        folder = self._find_folder_by_run_id(run_id)
        if not folder:
            raise FileNotFoundError(f"Candidate run '{run_id}' not found.")

        meta_file = folder / "run_metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta["rejection"] = {
            "rejected_by": reviewer_name,
            "reason": reason,
        }
        meta["review_status"] = "rejected"

        clean_run_id = meta.get("run_id", run_id)
        new_folder = self.candidates_dir / f"rejected_{clean_run_id}"
        if folder != new_folder:
            if new_folder.exists():
                shutil.rmtree(new_folder)
            folder.rename(new_folder)
            folder = new_folder

        with open(folder / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Candidate run %s REJECTED by %s: %s", run_id, reviewer_name, reason)

        return {
            "status": "rejected",
            "run_id": clean_run_id,
            "reason": reason,
        }

    def _find_folder_by_run_id(self, run_id: str) -> Optional[Path]:
        """Find folder regardless of prefix (pending_, approved_, rejected_)."""
        for folder in self.candidates_dir.iterdir():
            if not folder.is_dir():
                continue
            if run_id in folder.name:
                return folder
        return None
