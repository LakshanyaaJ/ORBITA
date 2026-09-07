"""
ORBITA Experiment Logger
========================
Real-time write-through event logging to JSONL, CSV, and SQLite.

Generates structured logs with:
  - ISO timestamps (timezone-aware UTC)
  - Monotonic elapsed seconds
  - Step number, action, status, outcome
  - Error type and recovery message
  - Confidence values
  - End-to-end latency and FPS

Persistence:
  - experiment_log.jsonl (append-only write-through for crash resilience)
  - experiment_log.csv   (real-time appended CSV for data analysis)
  - experiment_log.json  (summary JSON exported on demand / shutdown)
  - SQLite records       (persistent, queryable ACID store)
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
    step_name: str
    expected_action: str
    detected_action: str
    detected_object: str
    confidence: float
    validation_status: str
    outcome: str
    error_type: Optional[str]
    recovery_message: Optional[str]
    latency_ms: float
    fps: float
    # Backwards compatibility aliases
    action: str = ""
    label: str = ""
    status: str = ""

    def __post_init__(self):
        if not self.action:
            self.action = self.expected_action
        if not self.label:
            self.label = self.step_name
        if not self.status:
            self.status = self.validation_status


class ExperimentLogger:
    """
    Logs experiment events to JSONL, CSV, and SQLite with immediate write-through.

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
        self._lock = threading.Lock()
        self._db = OrbitaDB()
        self._db.create_experiment(experiment_id, experiment_name, total_steps)

        # File paths
        self.jsonl_path = self.output_dir / "experiment_log.jsonl"
        self.csv_path = self.output_dir / "experiment_log.csv"
        self.json_path = self.output_dir / "experiment_log.json"

        # Initialize CSV header if file doesn't exist
        self._csv_initialized = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._last_event_timestamp: str = ""

    def log(self, result: ValidationResult) -> None:
        """Log a validation result immediately to SQLite, JSONL, and CSV."""
        state = result.fsm_state
        step = state.current_step
        now_utc = datetime.now(timezone.utc)
        now = now_utc.isoformat(timespec="milliseconds")
        elapsed = time.time() - self.start_time

        step_num = step.id if step else 0
        step_lbl = step.label if step else "NONE"
        step_act = step.action if step else "NONE"
        status_name = state.status.name

        if state.status == FSMStatus.CORRECT:
            outcome = "SUCCESS"
        elif state.status in (
            FSMStatus.WRONG_OBJECT,
            FSMStatus.WRONG_ACTION,
            FSMStatus.WRONG_SEQUENCE,
            FSMStatus.STEP_SKIPPED,
            FSMStatus.OUT_OF_SEQUENCE,
        ):
            outcome = "ERROR"
        elif state.status == FSMStatus.UNCERTAIN:
            outcome = "UNCERTAIN"
        else:
            outcome = "WAITING"

        conf = round(result.action_prediction.confidence if result.action_prediction else 0.0, 3)

        entry = LogEntry(
            timestamp=now,
            elapsed_seconds=round(elapsed, 3),
            experiment_id=self.experiment_id,
            step_number=step_num,
            step_name=step_lbl,
            expected_action=step_act,
            detected_action=state.detected_action,
            detected_object=state.detected_object,
            confidence=conf,
            validation_status=status_name,
            outcome=outcome,
            error_type=state.error_type,
            recovery_message=state.recovery_message,
            latency_ms=round(result.latency_ms, 2),
            fps=round(result.fps, 1),
            action=step_act,
            label=step_lbl,
            status=status_name,
        )

        with self._lock:
            self._entries.append(entry)
            self._last_event_timestamp = now

            # 1. SQLite write-through
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

            # 2. JSONL write-through (append-only)
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(asdict(entry)) + "\n")
                    jf.flush()
            except Exception as exc:
                logger.error("JSONL write error: %s", exc)

            # 3. CSV write-through (append-only with header management)
            try:
                fieldnames = list(asdict(entry).keys())
                with open(self.csv_path, "a", newline="", encoding="utf-8") as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    if not self._csv_initialized:
                        writer.writeheader()
                        self._csv_initialized = True
                    writer.writerow(asdict(entry))
                    cf.flush()
            except Exception as exc:
                logger.error("CSV write error: %s", exc)

    def log_event(self, event_type: str, data: dict) -> None:
        """Log a freeform event to SQLite."""
        try:
            self._db.log_event(self.experiment_id, event_type, data)
        except Exception as exc:
            logger.warning("DB event log error: %s", exc)

    def export(self, final_status: str = "SUCCESS") -> dict[str, str]:
        """
        Export complete summary log to JSON and finalize SQLite.

        Returns:
            Dict of {format: filepath}
        """
        with self._lock:
            error_count = sum(1 for e in self._entries if e.error_type is not None)
            completed_steps = list(dict.fromkeys(
                e.step_number for e in self._entries if e.status == FSMStatus.CORRECT.name
            ))

            summary = {
                "experiment_id": self.experiment_id,
                "experiment_name": self.experiment_name,
                "start_time": datetime.fromtimestamp(self.start_time, timezone.utc).isoformat(timespec="milliseconds"),
                "end_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "duration_seconds": round(time.time() - self.start_time, 1),
                "total_steps": self.total_steps,
                "completed_steps": completed_steps,
                "error_count": error_count,
                "overall_status": final_status,
                "entries": [asdict(e) for e in self._entries],
            }

            # JSON summary export
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            # SQLite completion
            try:
                self._db.complete_experiment(self.experiment_id, final_status, summary)
            except Exception as exc:
                logger.warning("DB complete error: %s", exc)

            logger.info("Experiment log exported: %s", self.json_path)
            return {
                "json": str(self.json_path),
                "csv": str(self.csv_path) if self.csv_path.exists() else "",
                "jsonl": str(self.jsonl_path) if self.jsonl_path.exists() else "",
            }

    def get_summary(self) -> dict:
        """Get current in-progress summary."""
        with self._lock:
            return {
                "experiment_id": self.experiment_id,
                "total_entries": len(self._entries),
                "error_count": sum(1 for e in self._entries if e.error_type),
                "elapsed_seconds": round(time.time() - self.start_time, 1),
                "jsonl_exists": self.jsonl_path.exists(),
                "csv_exists": self.csv_path.exists(),
            }

    def get_telemetry(self) -> dict:
        """Structured telemetry for API and GUI monitoring."""
        with self._lock:
            return {
                "status": "ACTIVE",
                "events_written": len(self._entries),
                "last_event_timestamp": self._last_event_timestamp,
                "sqlite": "ACTIVE",
                "jsonl": "ACTIVE" if self.jsonl_path.exists() else "READY",
                "csv": "ACTIVE" if self.csv_path.exists() else "READY",
            }
