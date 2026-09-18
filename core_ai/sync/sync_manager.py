"""
ORBITA Synchronization Queue Manager
====================================
Implements edge-to-ground offline-tolerant synchronization.

Requirements (Sections 15, 16, 18):
- Manages local sync queue
- Supports Ground Link simulation toggle (Online / Offline)
- Continues local edge processing during loss of signal
- Queues results locally in SQLite
- Uploads queued items upon ground link restoration with SHA-256 verification
- Handles retries and error reporting
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core_ai.database.sqlite_db import OrbitaDB
from core_ai.sync.ground_system import GroundStationReceiver

logger = logging.getLogger(__name__)


class SyncQueueManager:
    """
    Manages offline queueing, ground link state, and sync execution.
    """

    def __init__(self, db: Optional[OrbitaDB] = None, ground_station: Optional[GroundStationReceiver] = None):
        self.db = db or OrbitaDB()
        self.ground_station = ground_station or GroundStationReceiver(self.db)
        self.ground_link_online: bool = True
        self._last_sync_time: float = time.time()

    def set_ground_link(self, online: bool, auto_process: bool = True) -> bool:
        """Toggle simulated ground communication link."""
        self.ground_link_online = bool(online)
        logger.info(
            "Ground Link state changed to: %s",
            "ONLINE" if self.ground_link_online else "OFFLINE",
        )
        # If link restored to ONLINE and auto_process enabled, trigger automated queue processing
        if self.ground_link_online and auto_process:
            self.last_sync_result = self.process_queue()
        return self.ground_link_online

    def is_online(self) -> bool:
        return self.ground_link_online

    def is_ground_link_online(self) -> bool:
        return self.ground_link_online

    def get_pending_count(self) -> int:
        return self.db.get_sync_queue_count()

    def sync_all_pending(self) -> Dict[str, Any]:
        return self.process_queue()

    def queue_result(self, result_id: Any, experiment_id: str, payload: dict, checksum: str) -> Dict[str, Any]:
        self.db.enqueue_sync(result_id, experiment_id, payload, checksum)
        if self.ground_link_online:
            self.process_queue()
            return {"status": "synced", "message": "Uploaded immediately to Ground Station."}
        return {"status": "queued", "message": "Safely held in offline sync queue."}

    def process_queue(self) -> Dict[str, Any]:
        """
        Processes pending items in the synchronization queue.
        If ground link is offline, items remain safely held in the local queue.
        """
        if not self.ground_link_online:
            pending = self.get_pending_count()
            return {
                "status": "OFFLINE",
                "ground_link": "OFFLINE",
                "synced": 0,
                "pending": pending,
                "message": f"Ground link is offline. {pending} payloads securely queued locally.",
            }

        items = self.db.get_pending_sync_items()
        if not items:
            return {
                "status": "IDLE",
                "ground_link": "ONLINE",
                "synced": 0,
                "pending": 0,
                "message": "Sync queue is empty.",
            }

        synced_count = 0
        failed_count = 0
        errors: List[str] = []

        for item in items:
            queue_id = item["id"]
            result_id = item.get("result_id", 0)
            exp_id = item.get("experiment_id", "")
            checksum = item.get("checksum", "")
            payload = item.get("payload")

            envelope = {
                "experiment_id": exp_id,
                "payload": payload,
                "checksum": checksum,
                "client_timestamp": time.time(),
            }

            # Transmit to Ground Station receiver
            success, code, resp = self.ground_station.receive_payload(envelope)
            if success or code == 409:  # 409 duplicate is also considered successfully synced
                self.db.update_sync_status(queue_id, result_id, "SYNCED")
                synced_count += 1
                self._last_sync_time = time.time()
            else:
                err_msg = resp.get("error", "Ground upload rejected")
                self.db.update_sync_status(queue_id, result_id, "RETRY", error=err_msg)
                failed_count += 1
                errors.append(f"Result {result_id}: {err_msg}")

        pending_remaining = self.get_pending_count()
        return {
            "status": "SUCCESS" if failed_count == 0 else "PARTIAL",
            "ground_link": "ONLINE",
            "synced": synced_count,
            "failed": failed_count,
            "pending": pending_remaining,
            "errors": errors,
            "last_sync_time": self._last_sync_time,
            "message": f"Synchronized {synced_count} payloads with Ground Station.",
        }

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive synchronization telemetry."""
        return {
            "ground_link": "ONLINE" if self.ground_link_online else "OFFLINE",
            "is_online": self.ground_link_online,
            "pending_count": self.get_pending_count(),
            "last_sync_time": self._last_sync_time,
            "last_sync_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_sync_time)),
            "ground_records_count": len(self.ground_station.list_received_results(limit=100)),
        }
