"""
ORBITA Experiment Logger
========================
Real-time event logging to JSON + SQLite.

Generates structured logs with:
  - ISO timestamps
  - Step number, action, status
  - Error type and recovery message
  - Confidence values
  - End-to-end latency

Exports:
  - experiment_log.json  (portable, human-readable)
  - experiment_log.csv   (for data analysis)
  - SQLite records       (persistent, queryable)
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from core_ai.database.sqlite_db import OrbitaDB
from core_ai.reasoning.state_manager import ValidationResult
from core_ai.reasoning.fsm import FSMStatus

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    timestamp: str
    elapsed_seconds: float
    experiment_id: str
    step_number: int
    action: str
    label: str
    status: str
    detected_action: str
    detected_object: str
    confidence: float
    error_type: Optional[str]
    recovery_message: Optional[str]
    latency_ms: float
    fps: float


class ExperimentLogger:
    """
    Logs experiment events to JSON, CSV, and SQLite.

    Usage:
        logger = ExperimentLogger(experiment_id="EXP001", output_dir="experiments")
        logger.log(validation_result)
        logger.export()
    """

    def __init__(
        self,
        experiment_id: str,
        experiment_name: str,
        output_dir: str | Path,
        total_steps: int,
    ):
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps
        self.start_time = time.time()
        self._entries: list[LogEntry] = []
        self._db = OrbitaDB()
        self._db.create_experiment(experiment_id, experiment_name, total_steps)

    def log(self, result: ValidationResult) -> None:
        """Log a validation result."""
        state = result.fsm_state
        step = state.current_step
        now = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        elapsed = time.time() - self.start_time

        entry = LogEntry(
            timestamp=now,
            elapsed_seconds=round(elapsed, 2),
            experiment_id=self.experiment_id,
            step_number=step.id if step else 0,
            action=step.action if step else "NONE",
            label=step.label if step else "NONE",
            status=state.status.name,
            detected_action=state.detected_action,
            detected_object=state.detected_object,
            confidence=round(result.action_prediction.confidence if result.action_prediction else 0.0, 3),
            error_type=state.error_type,
            recovery_message=state.recovery_message,
            latency_ms=round(result.latency_ms, 2),
            fps=round(result.fps, 1),
        )
        self._entries.append(entry)

        # Log to SQLite
        try:
            self._db.log_step(
                experiment_id=self.experiment_id,
                step_number=entry.step_number,
                action=entry.action,
                label=entry.label,
                status=entry.status,
                error_type=entry.error_type,
                detected_action=entry.detected_action,
                detected_object=entry.detected_object,
                confidence=entry.confidence,
                latency_ms=entry.latency_ms,
            )
        except Exception as exc:
            logger.warning("DB log error: %s", exc)

    def log_event(self, event_type: str, data: dict) -> None:
        """Log a freeform event to SQLite."""
        try:
            self._db.log_event(self.experiment_id, event_type, data)
        except Exception as exc:
            logger.warning("DB event log error: %s", exc)

    def export(self, final_status: str = "SUCCESS") -> dict[str, str]:
        """
        Export logs to JSON and CSV.

        Returns:
            Dict of {format: filepath}
        """
        # Build summary
        error_count = sum(1 for e in self._entries if e.error_type is not None)
        completed_steps = list(dict.fromkeys(
            e.step_number for e in self._entries if e.status == FSMStatus.CORRECT.name
        ))

        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "start_time": datetime.utcfromtimestamp(self.start_time).isoformat() + "Z",
            "end_time": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "duration_seconds": round(time.time() - self.start_time, 1),
            "total_steps": self.total_steps,
            "completed_steps": completed_steps,
            "error_count": error_count,
            "overall_status": final_status,
            "entries": [asdict(e) for e in self._entries],
        }

        # JSON export
        json_path = self.output_dir / "experiment_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # CSV export
        csv_path = self.output_dir / "experiment_log.csv"
        if self._entries:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self._entries[0]).keys())
                writer.writeheader()
                for e in self._entries:
                    writer.writerow(asdict(e))

        # SQLite completion
        try:
            self._db.complete_experiment(self.experiment_id, final_status, summary)
        except Exception as exc:
            logger.warning("DB complete error: %s", exc)

        logger.info("Experiment log exported: %s", json_path)
        return {
            "json": str(json_path),
            "csv": str(csv_path) if self._entries else "",
        }

    def get_summary(self) -> dict:
        """Get current in-progress summary."""
        return {
            "experiment_id": self.experiment_id,
            "total_entries": len(self._entries),
            "error_count": sum(1 for e in self._entries if e.error_type),
            "elapsed_seconds": round(time.time() - self.start_time, 1),
        }
