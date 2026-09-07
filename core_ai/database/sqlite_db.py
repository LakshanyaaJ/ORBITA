"""
ORBITA SQLite Database
=======================
Persistent storage for experiment records.
Uses Python's built-in sqlite3 — no external database server.

Schema:
  experiments  — experiment metadata
  steps        — per-step records
  events       — all action events (correct, errors, recoveries)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "experiments" / "orbita.db"


class OrbitaDB:
    """SQLite-backed experiment storage."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialize database schema."""
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id          TEXT PRIMARY KEY,
                    name        TEXT,
                    start_time  REAL,
                    end_time    REAL,
                    status      TEXT DEFAULT 'IN_PROGRESS',
                    total_steps INTEGER,
                    summary_json TEXT
                );

                CREATE TABLE IF NOT EXISTS step_records (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT REFERENCES experiments(id),
                    step_number     INTEGER,
                    action          TEXT,
                    label           TEXT,
                    timestamp       REAL,
                    status          TEXT,
                    error_type      TEXT,
                    detected_action TEXT,
                    detected_object TEXT,
                    confidence      REAL,
                    latency_ms      REAL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT REFERENCES experiments(id),
                    timestamp       REAL,
                    event_type      TEXT,
                    data_json       TEXT
                );
            """)
        logger.info("Database initialized at %s", self.db_path)

    def create_experiment(self, experiment_id: str, name: str, total_steps: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiments (id, name, start_time, total_steps) VALUES (?,?,?,?)",
                (experiment_id, name, time.time(), total_steps),
            )

    def log_step(
        self,
        experiment_id: str,
        step_number: int,
        action: str,
        label: str,
        status: str,
        error_type: Optional[str] = None,
        detected_action: str = "",
        detected_object: str = "",
        confidence: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO step_records
                   (experiment_id, step_number, action, label, timestamp, status,
                    error_type, detected_action, detected_object, confidence, latency_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (experiment_id, step_number, action, label, time.time(), status,
                 error_type, detected_action, detected_object, confidence, latency_ms),
            )

    def log_event(self, experiment_id: str, event_type: str, data: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (experiment_id, timestamp, event_type, data_json) VALUES (?,?,?,?)",
                (experiment_id, time.time(), event_type, json.dumps(data)),
            )

    def complete_experiment(self, experiment_id: str, status: str, summary: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET end_time=?, status=?, summary_json=? WHERE id=?",
                (time.time(), status, json.dumps(summary), experiment_id),
            )

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_steps(self, experiment_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM step_records WHERE experiment_id=? ORDER BY timestamp",
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_experiments(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY start_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
