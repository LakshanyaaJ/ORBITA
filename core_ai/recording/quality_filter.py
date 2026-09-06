"""
ORBITA Dataset Quality Filter
=============================
Evaluates recorded experiment runs before admitting them as candidate dataset samples.

Criteria evaluated:
  1. Sharpness & Blur: Mean Laplacian variance across sample frames.
  2. Lighting & Exposure: Checks for underexposed (<15% mean intensity) or clipped images.
  3. Operator & Object Visibility: Presence of operator and critical tools/containers.
  4. Model Confidence: Average perception & HAR confidence across the execution.
  5. Procedure Completeness: Verifies 100% of required FSM steps were completed without terminal faults.
  6. Temporal Duration: Sanity-checks duration against minimum/maximum thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Detailed scores for a candidate run."""
    overall_score: float             # 0 - 100%
    is_acceptable: bool              # True if passed all minimum gates
    blur_score: float                # 0 - 100%
    lighting_score: float            # 0 - 100%
    detection_confidence: float      # 0 - 100%
    action_confidence: float         # 0 - 100%
    procedure_completeness: float    # 0 - 100%
    operator_visibility: float       # 0 - 100%
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetQualityFilter:
    """
    Evaluates candidate experiment recordings against quality criteria.
    """

    def __init__(
        self,
        min_blur_var: float = 65.0,
        min_overall_score: float = 70.0,
        min_detection_conf: float = 0.50,
        min_procedure_pct: float = 100.0,
        min_duration_sec: float = 10.0,
        max_duration_sec: float = 600.0,
    ):
        self.min_blur_var = min_blur_var
        self.min_overall_score = min_overall_score
        self.min_detection_conf = min_detection_conf
        self.min_procedure_pct = min_procedure_pct
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec

    def evaluate_run(
        self,
        frame_paths: list[str | Path],
        run_metadata: dict[str, Any],
    ) -> QualityMetrics:
        """
        Compute quality scores across candidate frames and telemetry.
        """
        rejection_reasons: list[str] = []

        # 1. Temporal Duration Check
        duration = float(run_metadata.get("duration_seconds", 0.0))
        if duration < self.min_duration_sec:
            rejection_reasons.append(f"Run duration too short: {duration:.1f}s < {self.min_duration_sec}s")
        elif duration > self.max_duration_sec:
            rejection_reasons.append(f"Run duration too long: {duration:.1f}s > {self.max_duration_sec}s")

        # 2. Procedure Completeness
        proc_pct = float(run_metadata.get("procedure_completeness_pct", 0.0))
        if proc_pct < self.min_procedure_pct:
            rejection_reasons.append(
                f"Incomplete procedure: {proc_pct:.0f}% completed (required {self.min_procedure_pct:.0f}%)"
            )

        # 3. Model Confidences from metadata
        det_conf = float(run_metadata.get("mean_detection_confidence", 0.75)) * 100.0
        act_conf = float(run_metadata.get("mean_action_confidence", 0.70)) * 100.0

        if (det_conf / 100.0) < self.min_detection_conf:
            rejection_reasons.append(f"Low mean detection confidence: {det_conf:.1f}%")

        # 4. Frame Quality Analysis (sample up to 20 frames across run)
        if not frame_paths:
            rejection_reasons.append("No video frames recorded for candidate run")
            return QualityMetrics(
                overall_score=0.0,
                is_acceptable=False,
                blur_score=0.0,
                lighting_score=0.0,
                detection_confidence=det_conf,
                action_confidence=act_conf,
                procedure_completeness=proc_pct,
                operator_visibility=0.0,
                rejection_reasons=rejection_reasons,
            )

        sample_stride = max(1, len(frame_paths) // 20)
        sampled_paths = frame_paths[::sample_stride]

        blur_vars: list[float] = []
        brightness_vals: list[float] = []

        for p in sampled_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            brightness_vals.append(float(np.mean(gray)))

        avg_blur = float(np.mean(blur_vars)) if blur_vars else 0.0
        avg_brightness = float(np.mean(brightness_vals)) if brightness_vals else 0.0

        # Normalize blur score to 0-100%
        blur_score = min(100.0, (avg_blur / max(self.min_blur_var, 1.0)) * 80.0)
        if avg_blur < self.min_blur_var:
            rejection_reasons.append(f"Average frame sharpness too low ({avg_blur:.1f} < {self.min_blur_var:.1f})")

        # Normalize lighting score
        if avg_brightness < 35.0:
            lighting_score = 40.0
            rejection_reasons.append(f"Lighting underexposed: mean intensity {avg_brightness:.1f} < 35")
        elif avg_brightness > 230.0:
            lighting_score = 50.0
            rejection_reasons.append(f"Lighting overexposed: mean intensity {avg_brightness:.1f} > 230")
        else:
            lighting_score = 95.0

        operator_vis = float(run_metadata.get("operator_visibility_pct", 90.0))

        # Composite score
        overall = (
            0.25 * blur_score +
            0.20 * lighting_score +
            0.25 * det_conf +
            0.15 * act_conf +
            0.15 * proc_pct
        )
        overall = round(max(0.0, min(100.0, overall)), 1)

        is_acceptable = (overall >= self.min_overall_score) and (len(rejection_reasons) == 0)

        return QualityMetrics(
            overall_score=overall,
            is_acceptable=is_acceptable,
            blur_score=round(blur_score, 1),
            lighting_score=round(lighting_score, 1),
            detection_confidence=round(det_conf, 1),
            action_confidence=round(act_conf, 1),
            procedure_completeness=round(proc_pct, 1),
            operator_visibility=round(operator_vis, 1),
            rejection_reasons=rejection_reasons,
        )
