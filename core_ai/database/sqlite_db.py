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
from typing import Any, Optional, Tuple

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
                    objective   TEXT DEFAULT '',
                    start_time  REAL,
                    end_time    REAL,
                    status      TEXT DEFAULT 'IN_PROGRESS',
                    total_steps INTEGER DEFAULT 0,
                    configuration TEXT DEFAULT '{}',
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
                    type            TEXT,
                    confidence      REAL DEFAULT 1.0,
                    metadata        TEXT,
                    data_json       TEXT
                );

                CREATE TABLE IF NOT EXISTS observations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT,
                    parameter       TEXT,
                    value           REAL,
                    unit            TEXT,
                    timestamp       REAL
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT,
                    person_id       INTEGER DEFAULT 1,
                    confidence      REAL,
                    timestamp       REAL,
                    metadata        TEXT
                );

                CREATE TABLE IF NOT EXISTS activities (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT,
                    person_id       INTEGER DEFAULT 1,
                    activity        TEXT,
                    confidence      REAL,
                    start_time      REAL,
                    end_time        REAL
                );

                CREATE TABLE IF NOT EXISTS rules (
                    id              TEXT PRIMARY KEY,
                    experiment_id   TEXT,
                    type            TEXT,
                    configuration   TEXT
                );

                CREATE TABLE IF NOT EXISTS results (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT,
                    payload         TEXT,
                    checksum        TEXT,
                    sync_status     TEXT DEFAULT 'queued',
                    created_at      REAL,
                    synced_at       REAL
                );

                CREATE TABLE IF NOT EXISTS sync_queue (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT,
                    result_id       INTEGER REFERENCES results(id),
                    payload         TEXT,
                    checksum        TEXT,
                    attempts        INTEGER DEFAULT 0,
                    status          TEXT DEFAULT 'QUEUED',
                    last_attempt    REAL DEFAULT 0.0,
                    error           TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS ground_received_results (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT,
                    checksum        TEXT UNIQUE,
                    payload         TEXT,
                    received_at     REAL,
                    status          TEXT DEFAULT 'ACKNOWLEDGED'
                );
            """)
            # Migration check for existing SQLite files
            for col, d_val in [("objective", "''"), ("configuration", "'{}'"), ("total_steps", "0")]:
                try:
                    conn.execute(f"ALTER TABLE experiments ADD COLUMN {col} TEXT DEFAULT {d_val}")
                except Exception:
                    pass
        logger.info("Database initialized at %s with extended schema", self.db_path)

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        total_steps: int = 0,
        objective: str = "",
        configuration: Optional[dict] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiments 
                   (id, name, objective, start_time, total_steps, configuration, status) 
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    experiment_id,
                    name,
                    objective,
                    time.time(),
                    total_steps,
                    json.dumps(configuration or {}),
                    "READY",
                ),
            )

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE experiments SET status=? WHERE id=?", (status, experiment_id))

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

    def log_event(self, experiment_id: str, event_type: str, data: dict, confidence: float = 1.0) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO events 
                   (experiment_id, timestamp, event_type, type, confidence, metadata, data_json) 
                   VALUES (?,?,?,?,?,?,?)""",
                (experiment_id, time.time(), event_type, event_type, confidence, json.dumps(data), json.dumps(data)),
            )

    def log_observation(self, experiment_id: str, parameter: str, value: float, unit: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO observations (experiment_id, parameter, value, unit, timestamp) VALUES (?,?,?,?,?)",
                (experiment_id, parameter, value, unit, time.time()),
            )

    def log_detection(self, experiment_id: str, confidence: float, metadata: dict, person_id: int = 1) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO detections (experiment_id, person_id, confidence, timestamp, metadata) VALUES (?,?,?,?,?)",
                (experiment_id, person_id, confidence, time.time(), json.dumps(metadata)),
            )

    def log_activity(
        self, experiment_id: str, activity: str, confidence: float, start_time: float, end_time: float, person_id: int = 1
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO activities 
                   (experiment_id, person_id, activity, confidence, start_time, end_time) 
                   VALUES (?,?,?,?,?,?)""",
                (experiment_id, person_id, activity, confidence, start_time, end_time),
            )

    def log_rule(self, experiment_id: str, rule_type: str, configuration: dict, rule_id: str = None) -> None:
        r_id = rule_id or f"RULE-{int(time.time() * 1000)}"
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO rules (id, experiment_id, type, configuration)
                   VALUES (?,?,?,?)""",
                (r_id, experiment_id, rule_type, json.dumps(configuration)),
            )

    def get_rules(self, experiment_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM rules WHERE experiment_id=?", (experiment_id,)).fetchall()
            items = []
            for r in rows:
                d = dict(r)
                if isinstance(d.get("configuration"), str):
                    try:
                        d["configuration"] = json.loads(d["configuration"])
                    except Exception:
                        pass
                items.append(d)
            return items

    def complete_experiment(self, experiment_id: str, status: str, summary: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET end_time=?, status=?, summary_json=? WHERE id=?",
                (time.time(), status, json.dumps(summary), experiment_id),
            )

    def store_result(self, experiment_id: str, payload: dict, checksum: str, sync_status: str = "queued") -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO results (experiment_id, payload, checksum, sync_status, created_at)
                   VALUES (?,?,?,?,?)""",
                (experiment_id, json.dumps(payload), checksum, sync_status, time.time()),
            )
            result_id = cursor.lastrowid
            # Also enqueue automatically into sync_queue
            conn.execute(
                """INSERT INTO sync_queue (experiment_id, result_id, payload, checksum, status)
                   VALUES (?,?,?,?, 'QUEUED')""",
                (experiment_id, result_id, json.dumps(payload), checksum),
            )
            return result_id

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

    def get_events(self, experiment_id: str, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE experiment_id=? ORDER BY timestamp DESC LIMIT ?",
                (experiment_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_activities(self, experiment_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM activities WHERE experiment_id=? ORDER BY start_time DESC",
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_detections(self, experiment_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM detections WHERE experiment_id=? ORDER BY timestamp DESC",
                (experiment_id,),
            ).fetchall()
            items = []
            for r in rows:
                d = dict(r)
                if isinstance(d.get("metadata"), str):
                    try:
                        d["metadata"] = json.loads(d["metadata"])
                    except Exception:
                        pass
                items.append(d)
            return items

    def get_observations(self, experiment_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM observations WHERE experiment_id=? ORDER BY timestamp DESC",
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_result(self, result_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM results WHERE id=?", (result_id,)).fetchone()
            if not row:
                return None
            res = dict(row)
            if isinstance(res.get("payload"), str):
                try:
                    res["payload"] = json.loads(res["payload"])
                except Exception:
                    pass
            return res

    def get_latest_result(self, experiment_id: Optional[str] = None) -> Optional[dict]:
        with self._connect() as conn:
            if experiment_id:
                row = conn.execute(
                    "SELECT * FROM results WHERE experiment_id=? ORDER BY created_at DESC LIMIT 1",
                    (experiment_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM results ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            if not row:
                return None
            res = dict(row)
            if isinstance(res.get("payload"), str):
                try:
                    res["payload"] = json.loads(res["payload"])
                except Exception:
                    pass
            return res

    def list_results(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM results ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                if isinstance(d.get("payload"), str):
                    try:
                        d["payload"] = json.loads(d["payload"])
                    except Exception:
                        pass
                results.append(d)
            return results

    def list_experiments(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY start_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_pending_sync_items(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sync_queue WHERE status IN ('QUEUED', 'RETRY') ORDER BY id ASC"
            ).fetchall()
            items = []
            for r in rows:
                d = dict(r)
                if isinstance(d.get("payload"), str):
                    try:
                        d["payload"] = json.loads(d["payload"])
                    except Exception:
                        pass
                items.append(d)
            return items

    def update_sync_status(self, queue_id: int, result_id: int, status: str, error: str = "") -> None:
        with self._connect() as conn:
            now = time.time()
            conn.execute(
                """UPDATE sync_queue 
                   SET status=?, attempts=attempts+1, last_attempt=?, error=? 
                   WHERE id=?""",
                (status, now, error, queue_id),
            )
            if status == "SYNCED":
                conn.execute(
                    "UPDATE results SET sync_status='synced', synced_at=? WHERE id=?",
                    (now, result_id),
                )
            elif status == "FAILED":
                conn.execute(
                    "UPDATE results SET sync_status='failed' WHERE id=?",
                    (result_id,),
                )

    def get_sync_queue_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM sync_queue WHERE status IN ('QUEUED', 'RETRY')"
            ).fetchone()
            return int(row["count"]) if row else 0

    def record_ground_receipt(self, experiment_id: str, checksum: str, payload: dict) -> Tuple[bool, str]:
        """Ground receiver persistence. Enforces duplicate prevention via checksum."""
        with self._connect() as conn:
            # Check for duplicate
            existing = conn.execute(
                "SELECT id FROM ground_received_results WHERE checksum=?", (checksum,)
            ).fetchone()
            if existing:
                return False, f"Duplicate payload rejected. Checksum {checksum} already recorded."

            conn.execute(
                """INSERT INTO ground_received_results (experiment_id, checksum, payload, received_at)
                   VALUES (?,?,?,?)""",
                (experiment_id, checksum, json.dumps(payload), time.time()),
            )
            return True, "Payload verified and accepted by Ground Station."

    def get_ground_results(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ground_received_results ORDER BY received_at DESC LIMIT ?", (limit,)
            ).fetchall()
            items = []
            for r in rows:
                d = dict(r)
                if isinstance(d.get("payload"), str):
                    try:
                        d["payload"] = json.loads(d["payload"])
                    except Exception:
                        pass
                items.append(d)
            return items

    def save_observation(self, experiment_id: str, parameter: str, value: float, unit: str) -> None:
        self.log_observation(experiment_id, parameter, value, unit)

    def save_detection(self, experiment_id: str, person_id: int, confidence: float, metadata: dict) -> None:
        self.log_detection(experiment_id, confidence, metadata, person_id=person_id)

    def save_activity(self, experiment_id: str, person_id: int, activity: str, confidence: float, start_time: float, end_time: float) -> None:
        self.log_activity(experiment_id, activity, confidence, start_time, end_time, person_id=person_id)

    def save_rule(self, experiment_id: str, rule_type: str, configuration: dict) -> None:
        self.log_rule(experiment_id, rule_type, configuration)

    def save_result(self, result_id: Any, experiment_id: str, payload: dict, checksum: str, sync_status: str = "queued") -> int:
        return self.store_result(experiment_id, payload, checksum, sync_status)

    def enqueue_sync(self, result_id: Any, experiment_id: str, payload: dict, checksum: str, status: str = "QUEUED") -> int:
        with self._connect() as conn:
            r_id = 1 if isinstance(result_id, str) else int(result_id)
            cursor = conn.execute(
                """INSERT INTO sync_queue (experiment_id, result_id, payload, checksum, status)
                   VALUES (?,?,?,?,?)""",
                (experiment_id, r_id, json.dumps(payload), checksum, status),
            )
            return cursor.lastrowid

    def get_pending_syncs(self) -> list[dict]:
        return self.get_pending_sync_items()

    def close(self) -> None:
        """Explicitly close persistent connection if open."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# Alias for architectural conformance
SQLiteStorage = OrbitaDB
