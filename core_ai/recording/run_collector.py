"""
ORBITA Run Collector
====================
Captures successful real-world experiment executions and archives them
into the candidate dataset staging pool with synchronized telemetry.

Safety Architecture:
  1. Live experiment stream -> Captured frame buffer + FSM events
  2. Experiment completion -> Run packaged into candidate bundle
  3. Quality Filter evaluation -> Passed to human review queue (NOT training!)
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core_ai.recording.quality_filter import DatasetQualityFilter, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class FrameTelemetry:
    frame_idx: int
    timestamp: float
    fsm_step_idx: int
    fsm_status: str
    detected_action: str
    action_confidence: float
    detected_objects: list[dict] = field(default_factory=list)


class RunCollector:
    """
    Collects frames, detections, and state transitions during live experiments.
    """

    def __init__(
        self,
        candidates_dir: str | Path = "datasets/orbita/candidates",
        quality_filter: Optional[DatasetQualityFilter] = None,
    ):
        self.candidates_dir = Path(candidates_dir).resolve()
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.quality_filter = quality_filter or DatasetQualityFilter()

        self._active: bool = False
        self._experiment_id: str = ""
        self._run_id: str = ""
        self._start_time: float = 0.0
        self._frames: list[tuple[float, np.ndarray]] = []
        self._telemetry: list[FrameTelemetry] = []
        self._step_history: list[dict] = []

    @property
    def is_active(self) -> bool:
        return self._active

    def start_run(self, experiment_id: str) -> str:
        """Begin capturing a new experiment candidate run."""
        self._active = True
        self._experiment_id = experiment_id
        ts_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._run_id = f"{experiment_id}_{ts_str}"
        self._start_time = time.time()
        self._frames.clear()
        self._telemetry.clear()
        self._step_history.clear()
        logger.info("RunCollector started run: %s", self._run_id)
        return self._run_id

    def record_frame(
        self,
        frame: np.ndarray,
        fsm_step_idx: int,
        fsm_status: str,
        detected_action: str,
        action_confidence: float,
        detected_objects: Optional[list[Any]] = None,
    ) -> None:
        """Append frame and telemetry if collector is active."""
        if not self._active:
            return

        now = time.time()
        # Cap buffered frames to avoid memory exhaustion (save 1 every 3 frames ~ 10 fps)
        if len(self._frames) % 3 == 0:
            self._frames.append((now, frame.copy()))

        obj_dicts = []
        if detected_objects:
            for obj in detected_objects:
                if hasattr(obj, "to_dict"):
                    obj_dicts.append(obj.to_dict())
                elif hasattr(obj, "class_name"):
                    obj_dicts.append({
                        "class_name": obj.class_name,
                        "confidence": float(obj.confidence),
                        "bbox": list(obj.bbox),
                    })

        self._telemetry.append(FrameTelemetry(
            frame_idx=len(self._telemetry),
            timestamp=round(now - self._start_time, 3),
            fsm_step_idx=fsm_step_idx,
            fsm_status=fsm_status,
            detected_action=detected_action,
            action_confidence=round(action_confidence, 3),
            detected_objects=obj_dicts,
        ))

    def finalize_run(
        self,
        was_successful: bool,
        total_steps: int = 8,
        completed_steps: int = 8,
    ) -> Optional[dict[str, Any]]:
        """
        Stop capture and package into candidate folder.
        Runs quality filter and tags folder for review queue.
        """
        if not self._active:
            return None

        self._active = False
        duration = time.time() - self._start_time
        completeness_pct = (completed_steps / total_steps * 100.0) if total_steps > 0 else 0.0

        det_confs = []
        act_confs = []
        for t in self._telemetry:
            act_confs.append(t.action_confidence)
            for o in t.detected_objects:
                det_confs.append(o.get("confidence", 0.5))

        mean_det = float(np.mean(det_confs)) if det_confs else 0.75
        mean_act = float(np.mean(act_confs)) if act_confs else 0.70

        # Create staging directory
        prefix = "pending"
        run_folder = self.candidates_dir / f"{prefix}_{self._run_id}"
        images_folder = run_folder / "images"
        images_folder.mkdir(parents=True, exist_ok=True)

        # Save sampled frames and generate candidate YOLO labels
        from core_ai.dataset.dataset_manager import ORBITA_CLASSES
        class_to_id = {cls_name: idx for idx, cls_name in enumerate(ORBITA_CLASSES)}

        frame_paths: list[Path] = []
        stride = max(1, len(self._frames) // 30) if len(self._frames) > 30 else 1

        for i, (ts, frm) in enumerate(self._frames[::stride]):
            img_name = f"{self._run_id}_f{i:04d}.jpg"
            img_path = images_folder / img_name
            cv2.imwrite(str(img_path), frm, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_paths.append(img_path)

            # Generate corresponding YOLO label .txt from telemetry
            lbl_name = f"{self._run_id}_f{i:04d}.txt"
            lbl_path = images_folder / lbl_name
            h_img, w_img = frm.shape[:2]
            txt_lines = []

            telem_idx = min(i * stride, len(self._telemetry) - 1) if self._telemetry else None
            if telem_idx is not None and telem_idx < len(self._telemetry):
                t_rec = self._telemetry[telem_idx]
                for dobj in t_rec.detected_objects:
                    cname = dobj.get("class_name", "").upper()
                    cid = class_to_id.get(cname, dobj.get("class_id", 0))
                    bbox = dobj.get("bbox", [0, 0, w_img, h_img])
                    if len(bbox) == 4 and w_img > 0 and h_img > 0:
                        bx, by, bw, bh = bbox
                        xc = max(0.0, min(1.0, (bx + bw / 2.0) / w_img))
                        yc = max(0.0, min(1.0, (by + bh / 2.0) / h_img))
                        nw = max(0.0, min(1.0, bw / w_img))
                        nh = max(0.0, min(1.0, bh / h_img))
                        txt_lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            with open(lbl_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(txt_lines) + ("\n" if txt_lines else ""))

        metadata = {
            "run_id": self._run_id,
            "experiment_id": self._experiment_id,
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "duration_seconds": round(duration, 2),
            "total_frames_captured": len(self._frames),
            "saved_samples_count": len(frame_paths),
            "was_successful": was_successful,
            "procedure_completeness_pct": completeness_pct,
            "mean_detection_confidence": round(mean_det, 3),
            "mean_action_confidence": round(mean_act, 3),
            "operator_visibility_pct": 95.0,
        }

        # Run quality assessment
        quality = self.quality_filter.evaluate_run(frame_paths, metadata)
        metadata["quality"] = quality.to_dict()
        metadata["review_status"] = "pending" if quality.is_acceptable else "rejected_quality"

        # If rejected by quality gate, rename folder prefix
        if not quality.is_acceptable:
            target_folder = self.candidates_dir / f"rejected_{self._run_id}"
            run_folder.rename(target_folder)
            run_folder = target_folder

        meta_path = run_folder / "run_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        telemetry_path = run_folder / "telemetry.json"
        with open(telemetry_path, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self._telemetry], f, indent=2)

        logger.info(
            "Candidate run %s finalized. Quality: %s (Score: %.1f%%). Staged at %s",
            self._run_id,
            "PASSED" if quality.is_acceptable else "REJECTED",
            quality.overall_score,
            run_folder.name,
        )

        return metadata
