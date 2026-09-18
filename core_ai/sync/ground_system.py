"""
ORBITA Ground System Receiver
=============================
Simulates the remote ground station interface that receives, validates,
and stores telemetry and structured experiment results transmitted from the edge.

Requirements (Sections 19, 21):
- Receive payload
- Validate payload structure
- Validate SHA-256 checksum
- Reject duplicate payloads (duplicate prevention)
- Acknowledge valid payload with status
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from core_ai.database.integrity import calculate_checksum, verify_payload_checksum
from core_ai.database.sqlite_db import OrbitaDB

logger = logging.getLogger(__name__)


class GroundStationReceiver:
    """
    Ground Station API verification engine.
    Persists accepted results into ground_received_results table.
    """

    def __init__(self, db: Optional[OrbitaDB] = None):
        self.db = db or OrbitaDB()

    def receive_payload(self, envelope: Dict[str, Any]) -> Tuple[bool, int, Dict[str, Any]]:
        """
        Processes an incoming synchronization transmission.

        Args:
            envelope: {
                "experiment_id": str,
                "payload": dict,
                "checksum": str,
                "client_timestamp": float
            }

        Returns:
            (success: bool, http_code: int, response_dict: dict)
        """
        experiment_id = envelope.get("experiment_id", "")
        payload = envelope.get("payload")
        provided_checksum = envelope.get("checksum", "")

        if not experiment_id or not payload or not provided_checksum:
            return False, 400, {
                "status": "REJECTED",
                "error": "Malformed payload envelope. Required: experiment_id, payload, checksum.",
            }

        # 1. SHA-256 Data Integrity Verification
        is_valid = verify_payload_checksum(payload, provided_checksum)
        if not is_valid:
            calc = calculate_checksum(payload)
            logger.warning(
                "Ground Station rejected payload for %s: Checksum mismatch (calc: %s, received: %s)",
                experiment_id,
                calc,
                provided_checksum,
            )
            return False, 400, {
                "status": "REJECTED",
                "error": "Data integrity violation: SHA-256 checksum mismatch.",
                "calculated": calc,
                "expected": provided_checksum,
            }

        # 2. Duplicate Prevention & Persistence
        success, message = self.db.record_ground_receipt(
            experiment_id=experiment_id,
            checksum=provided_checksum,
            payload=payload,
        )

        if not success:
            logger.info("Ground Station detected duplicate payload for %s (%s)", experiment_id, provided_checksum[:8])
            return False, 409, {
                "status": "DUPLICATE_REJECTED",
                "message": message,
                "experiment_id": experiment_id,
                "checksum": provided_checksum,
            }

        logger.info(
            "Ground Station accepted payload for %s (Checksum: %s...)",
            experiment_id,
            provided_checksum[:12],
        )

        return True, 200, {
            "status": "ACKNOWLEDGED",
            "experiment_id": experiment_id,
            "checksum": provided_checksum,
            "received_at": time.time(),
            "message": "Payload verified and stored by Ground Station.",
        }

    def list_received_results(self, limit: int = 50) -> list[dict]:
        return self.db.get_ground_results(limit=limit)
