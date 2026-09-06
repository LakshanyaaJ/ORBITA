"""
ORBITA Model Registry & Promotion Engine
========================================
Maintains versioned checkpoints, training histories, and validation metrics.
Implements the Promotion Policy: A candidate model is promoted to production
ONLY if it strictly outperforms the current production baseline on the
held-out validation/test split.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelVersionRecord:
    """Metadata for a versioned model artifact."""
    version: str                            # e.g. "orbita_v1"
    model_type: str                         # "yolo_detector" | "har_gru" | "fused"
    created_at: str
    weights_path: str
    dataset_version: str
    training_samples_count: int
    classes: list[str]
    is_production: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    promotion_notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """
    Manages model versioning, validation metric history, and promotion gates.
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        registry_file: str | Path = "models/model_registry.json",
    ):
        self.models_dir = Path(models_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir = self.models_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = Path(registry_file).resolve()
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        self._records: dict[str, ModelVersionRecord] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for ver, item in data.items():
                    self._records[ver] = ModelVersionRecord(**item)
            except Exception as exc:
                logger.warning("Failed to parse model registry: %s", exc)

    def _save_registry(self) -> None:
        data = {ver: rec.to_dict() for ver, rec in self._records.items()}
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def list_models(self) -> list[ModelVersionRecord]:
        return list(self._records.values())

    def get_production_model(self, model_type: str = "yolo_detector") -> Optional[ModelVersionRecord]:
        for rec in self._records.values():
            if rec.model_type == model_type and rec.is_production:
                return rec
        return None

    def register_candidate_version(
        self,
        weights_file: str | Path,
        model_type: str,
        dataset_version: str,
        training_samples: int,
        classes: list[str],
        metrics: dict[str, float],
    ) -> ModelVersionRecord:
        """
        Store candidate model weights into versions/ directory and log metrics.
        Does NOT automatically make it production!
        """
        src_weights = Path(weights_file)
        if not src_weights.exists():
            raise FileNotFoundError(f"Source weights '{weights_file}' do not exist.")

        next_ver_num = len([r for r in self._records.values() if r.model_type == model_type]) + 1
        version_name = f"orbita_{model_type}_v{next_ver_num}"

        dest_folder = self.versions_dir / version_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_weights = dest_folder / src_weights.name
        shutil.copy2(src_weights, dest_weights)

        # Check if any production model currently exists for this type
        is_prod = (self.get_production_model(model_type) is None)

        record = ModelVersionRecord(
            version=version_name,
            model_type=model_type,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            weights_path=str(dest_weights.resolve()),
            dataset_version=dataset_version,
            training_samples_count=training_samples,
            classes=classes,
            is_production=is_prod,
            metrics=metrics,
            promotion_notes="Initial baseline model" if is_prod else "Pending promotion validation",
        )

        self._records[version_name] = record
        self._save_registry()

        logger.info(
            "Registered candidate model %s (Production: %s, Metrics: %s)",
            version_name, is_prod, metrics
        )
        return record

    def evaluate_and_promote(
        self,
        candidate_version: str,
        primary_metric_key: str = "mAP50",
        min_improvement: float = 0.005,
    ) -> tuple[bool, str]:
        """
        Compare candidate model against production baseline.
        If candidate metric >= prod_metric + min_improvement, promotes candidate.
        """
        candidate = self._records.get(candidate_version)
        if not candidate:
            return False, f"Candidate version '{candidate_version}' not found."

        current_prod = self.get_production_model(candidate.model_type)
        cand_score = candidate.metrics.get(primary_metric_key, 0.0)

        if current_prod is None:
            candidate.is_production = True
            candidate.promotion_notes = f"Promoted as first production baseline ({primary_metric_key}={cand_score:.4f})"
            self._save_registry()
            return True, f"Promoted {candidate_version} as initial production model."

        prod_score = current_prod.metrics.get(primary_metric_key, 0.0)
        diff = cand_score - prod_score

        if diff >= min_improvement:
            # Demote old production
            current_prod.is_production = False
            current_prod.promotion_notes = f"Demoted by {candidate_version}"

            # Promote candidate
            candidate.is_production = True
            candidate.promotion_notes = (
                f"Promoted: {primary_metric_key} improved from {prod_score:.4f} to {cand_score:.4f} (+{diff:.4f})"
            )
            self._save_registry()
            logger.info("PROMOTED %s over %s (%s: %.4f vs %.4f)",
                        candidate_version, current_prod.version, primary_metric_key, cand_score, prod_score)
            return True, f"Successfully promoted {candidate_version} (+{diff:.4f} {primary_metric_key})."
        else:
            candidate.promotion_notes = (
                f"Promotion rejected: {primary_metric_key} ({cand_score:.4f}) did not beat production ({prod_score:.4f})"
            )
            self._save_registry()
            logger.info("REJECTED promotion of %s: candidate %.4f <= prod %.4f",
                        candidate_version, cand_score, prod_score)
            return False, f"Promotion rejected: {primary_metric_key} diff ({diff:.4f}) < {min_improvement:.4f}."
