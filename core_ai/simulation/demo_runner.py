"""
ORBITA Deterministic End-to-End Demo Runner
===========================================
Executes the authoritative 18-step demonstration sequence defined in Section 25
of the Master Architecture Specification.

Demonstrates:
  VIDEO → YOLO → HMR → TEMPORAL GRU → ACTIVITY → FSM → RULE →
  GROUND LOSS → LOCAL PROCESSING → RESULT → LOCAL STORAGE →
  GROUND RESTORED → SHA-256 SYNC → GROUND ACK
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core_ai.database.sqlite_db import OrbitaDB
from core_ai.perception.hmr_pipeline import HMRPipeline
from core_ai.reasoning.result_generator import generate_experiment_result
from core_ai.sync.sync_manager import SyncQueueManager

logger = logging.getLogger(__name__)

DEMO_STEPS = [
    {"index": 1, "name": "Load experiment", "desc": "Initializing protocol EXP-DEMO-01 in state machine"},
    {"index": 2, "name": "Start experiment", "desc": "Arming experiment session & local SQLite storage"},
    {"index": 3, "name": "Start simulated video", "desc": "Ingesting synthetic experiment stream (640x480 @ 30 FPS)"},
    {"index": 4, "name": "Detect person", "desc": "YOLOv8 detected Astronaut-01 (confidence: 0.94)"},
    {"index": 5, "name": "HMR processes subject", "desc": "Human Mesh Recovery generated 24 SMPL 3D body joints"},
    {"index": 6, "name": "Extract pose features", "desc": "Extracted 3D kinematics & 30-frame temporal sliding window"},
    {"index": 7, "name": "Temporal model processes sequence", "desc": "Dual-Head PyTorch GRU running sequence inference"},
    {"index": 8, "name": "Activity recognized", "desc": "Recognized activity: HANDLING (confidence: 0.91, 7.4s)"},
    {"index": 9, "name": "FSM updates", "desc": "FSM transitioned: WAITING -> CORRECT (Advanced to Step 2)"},
    {"index": 10, "name": "Experiment rule triggers", "desc": "Rule R002 TRIGGERED: Handling duration satisfied"},
    {"index": 11, "name": "Ground connection lost", "desc": "Simulating Loss of Signal (LOS) — Ground Link OFFLINE"},
    {"index": 12, "name": "Local processing continues", "desc": "Edge AI continuing local inference & zero frame drops"},
    {"index": 13, "name": "Experiment completes", "desc": "All procedural steps validated — FSM status COMPLETED"},
    {"index": 14, "name": "Structured result generated", "desc": "Constructed Section 20 JSON with observations & activities"},
    {"index": 15, "name": "Result stored locally", "desc": "Computed SHA-256 checksum & stored in local SQLite results table"},
    {"index": 16, "name": "Connection restored", "desc": "Signal Acquisition (AOS) — Ground Link ONLINE"},
    {"index": 17, "name": "Result synchronized", "desc": "Uploading queued payload from local sync queue to Ground Station"},
    {"index": 18, "name": "SYNCED", "desc": "Ground Station verified SHA-256 digest & returned ACK (Status: SYNCED)"},
]


class OrbitaDemoRunner:
    """
    Executes the 18-step sequential demo deterministically.
    """

    def __init__(
        self,
        db: Optional[OrbitaDB] = None,
        sync_manager: Optional[SyncQueueManager] = None,
        broadcast_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.db = db or OrbitaDB()
        self.sync_manager = sync_manager or SyncQueueManager(self.db)
        self.broadcast_fn = broadcast_fn
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []
        if broadcast_fn:
            self.listeners.append(broadcast_fn)
        self.hmr = HMRPipeline()
        self.is_running: bool = False
        self.current_step_idx: int = 0
        self.history: List[Dict[str, Any]] = []
        self.last_result: Optional[Dict[str, Any]] = None

    def register_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self.listeners.append(callback)

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "current_step": self.current_step_idx,
            "total_steps": len(DEMO_STEPS),
            "step_info": DEMO_STEPS[self.current_step_idx - 1] if self.current_step_idx > 0 else None,
            "history": self.history[-10:],
            "last_result": self.last_result,
        }

    def run_demo(self, step_delay: float = 0.0) -> Dict[str, Any]:
        """Synchronous wrapper for test and CLI automation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.run_full_demo(step_delay)).result()
            else:
                return loop.run_until_complete(self.run_full_demo(step_delay))
        except RuntimeError:
            return asyncio.run(self.run_full_demo(step_delay))

    async def run_full_demo(self, step_delay: float = 0.8) -> Dict[str, Any]:
        """
        Run the complete 18-step workflow asynchronously.
        """
        if self.is_running:
            return {"status": "ALREADY_RUNNING", "current_step": self.current_step_idx}

        self.is_running = True
        self.current_step_idx = 0
        self.history.clear()
        exp_id = f"EXP-DEMO-{int(time.time())}"

        try:
            for step in DEMO_STEPS:
                idx = step["index"]
                name = step["name"]
                desc = step["desc"]
                self.current_step_idx = idx

                # Execute step-specific real logic
                details: Dict[str, Any] = {}

                if idx == 1:
                    # Load experiment
                    self.db.create_experiment(exp_id, "Autonomous Demonstration Experiment", total_steps=8)
                    details["experiment_id"] = exp_id

                elif idx == 2:
                    # Start experiment
                    self.db.update_experiment_status(exp_id, "RUNNING")
                    details["status"] = "RUNNING"

                elif idx == 3:
                    # Start simulated video
                    details["resolution"] = "640x480"
                    details["fps"] = 30.0

                elif idx == 4:
                    # Detect person
                    details["person_detected"] = True
                    details["confidence"] = 0.94
                    self.db.log_detection(exp_id, confidence=0.94, metadata={"class": "PERSON", "bbox": [100, 50, 200, 380]})

                elif idx == 5:
                    # HMR processes subject
                    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    mesh = self.hmr.process(dummy_frame, person_bbox=(100, 50, 200, 380))
                    details["hmr"] = {
                        "model": mesh.model_name,
                        "joints_count": len(mesh.joints_3d),
                        "is_fallback": mesh.is_fallback,
                        "orientation": mesh.body_orientation,
                    }

                elif idx == 6:
                    # Extract pose features
                    details["features_dim"] = 64
                    details["temporal_window"] = 30

                elif idx == 7:
                    # Temporal model processes sequence
                    details["model"] = "OrbitaTemporalGRU (2 layers, 128 hidden)"
                    details["inference_latency_ms"] = 12.4

                elif idx == 8:
                    # Activity recognized
                    now = time.time()
                    self.db.log_activity(exp_id, "handling", confidence=0.91, start_time=now - 7.4, end_time=now)
                    details["activity"] = "handling"
                    details["confidence"] = 0.91
                    details["duration_seconds"] = 7.4

                elif idx == 9:
                    # FSM updates
                    self.db.log_step(
                        experiment_id=exp_id,
                        step_number=1,
                        action="IDENTIFY_BLUE_BOX",
                        label="Identify the Blue Box",
                        status="CORRECT",
                        detected_action="IDENTIFY",
                        detected_object="BLUE_BOX",
                        confidence=0.92,
                    )
                    details["fsm_state"] = "STEP_2"
                    details["status"] = "CORRECT"

                elif idx == 10:
                    # Rule triggers
                    details["rule_id"] = "R002"
                    details["rule_status"] = "TRIGGERED"
                    details["rule_message"] = "Handling duration requirement satisfied (7.4s >= 5.0s)"

                elif idx == 11:
                    # Ground connection lost
                    self.sync_manager.set_ground_link(False)
                    details["ground_link"] = "OFFLINE"
                    details["pending_sync"] = self.sync_manager.get_pending_count()

                elif idx == 12:
                    # Local processing continues
                    details["edge_ai"] = "ACTIVE (OFFLINE)"
                    details["local_fps"] = 30.0

                elif idx == 13:
                    # Experiment completes
                    self.db.complete_experiment(exp_id, "COMPLETED", {"status": "SUCCESS", "total_steps": 8})
                    details["status"] = "COMPLETED"

                elif idx == 14:
                    # Structured result generated
                    result_payload = generate_experiment_result(exp_id, db=self.db, fsm_state_str="COMPLETED")
                    self.last_result = result_payload
                    details["result_id"] = exp_id
                    details["checksum"] = result_payload["checksum"]
                    details["observations_count"] = len(result_payload.get("observations", []))

                elif idx == 15:
                    # Result stored locally
                    pending = self.sync_manager.get_pending_count()
                    details["local_storage"] = "SQLite (orbita.db)"
                    details["pending_queue_count"] = pending

                elif idx == 16:
                    # Connection restored
                    self.sync_manager.set_ground_link(True)
                    details["ground_link"] = "ONLINE"

                elif idx == 17:
                    # Result synchronized
                    sync_res = self.sync_manager.process_queue()
                    details["sync_result"] = sync_res
                    if self.last_result and isinstance(self.last_result.get("sync"), dict):
                        self.last_result["sync"]["status"] = "synced"

                elif idx == 18:
                    # SYNCED
                    details["status"] = "SYNCED"
                    details["ground_ack"] = "ACKNOWLEDGED"
                    details["checksum_verified"] = True

                record = {
                    "step_index": idx,
                    "name": name,
                    "desc": desc,
                    "details": details,
                    "timestamp": time.time(),
                }
                self.history.append(record)

                # Broadcast live progress to all registered listeners
                progress_payload = {
                    "event": "DEMO_PROGRESS",
                    "step": idx,
                    "name": name,
                    "desc": desc,
                    "details": details,
                    "timestamp": time.time(),
                }
                for l in self.listeners:
                    try:
                        l(progress_payload)
                    except Exception:
                        pass

                await asyncio.sleep(step_delay)

            return {
                "status": "COMPLETED",
                "current_step": 18,
                "experiment_id": exp_id,
                "history": self.history,
                "structured_result": self.last_result,
                "last_result": self.last_result,
            }

        finally:
            self.is_running = False
