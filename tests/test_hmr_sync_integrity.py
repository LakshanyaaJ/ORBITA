"""
ORBITA Integration Test Suite
==============================
Tests for:
1. HMR 3D Mesh Recovery & Kinematic Fallback Pipeline
2. SHA-256 Canonical Data Integrity Verification
3. SQLite Persistence Layer (Observations, Detections, Activities, Rules, Results, Sync Queue)
4. Offline Sync Queue and Ground Station Receiver with Duplicate Prevention
5. OrbitaDemoRunner 18-Step Deterministic Sequence
"""

import os
import tempfile
import numpy as np
import pytest

from core_ai.perception.hmr_pipeline import HMRPipeline, HMRJoint3D, HMRMeshResult
from core_ai.database.integrity import calculate_checksum, verify_payload_checksum, canonical_json
from core_ai.database.sqlite_db import SQLiteStorage
from core_ai.sync.ground_system import GroundStationReceiver
from core_ai.sync.sync_manager import SyncQueueManager
from core_ai.reasoning.result_generator import ResultGenerator
from core_ai.simulation.demo_runner import OrbitaDemoRunner


def test_hmr_pipeline_topology_and_fallback():
    """HMR produces 24 SMPL joints, 3D translation/orientation, and labels fallback mode."""
    pipeline = HMRPipeline()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run inference with bbox
    result = pipeline.recover_mesh(frame, bbox=(100, 50, 200, 350))
    
    assert isinstance(result, HMRMeshResult)
    assert len(result.joints_3d) == 24
    assert result.is_fallback is True
    assert "PROTOTYPE_FALLBACK" in result.model_name
    
    # Check joint schema
    pelvis = result.joints_3d[0]
    assert pelvis.name == "Pelvis"
    assert len([pelvis.x, pelvis.y, pelvis.z]) == 3
    assert 0.0 <= pelvis.confidence <= 1.0
    
    # Check vertices and orientation
    assert "num_vertices" in result.vertices_summary
    assert result.vertices_summary["num_vertices"] == 6890
    assert "yaw" in result.body_orientation
    assert len(result.camera_translation) == 3
    
    # Dictionary serialization
    res_dict = result.to_dict()
    assert res_dict["is_fallback"] is True
    assert len(res_dict["joints_3d"]) == 24


def test_sha256_canonical_data_integrity():
    """SHA-256 canonical hashing ensures immutable payload verification."""
    payload = {
        "experiment_id": "EXP-INTEGRITY-01",
        "status": "completed",
        "metrics": {"fps": 30.0, "latency_ms": 42.5},
        "items": [1, 2, 3]
    }
    
    # Checksum computation
    checksum = calculate_checksum(payload)
    assert len(checksum) == 64
    assert all(c in "0123456789abcdef" for c in checksum)
    
    # Verification succeeds with untouched payload
    assert verify_payload_checksum(payload, checksum) is True
    
    # Key reordering does NOT change canonical checksum
    reordered_payload = {
        "status": "completed",
        "items": [1, 2, 3],
        "experiment_id": "EXP-INTEGRITY-01",
        "metrics": {"latency_ms": 42.5, "fps": 30.0}
    }
    assert verify_payload_checksum(reordered_payload, checksum) is True
    
    # Tampering payload invalidates checksum
    tampered = dict(payload)
    tampered["status"] = "tampered"
    assert verify_payload_checksum(tampered, checksum) is False


def test_sqlite_persistence_and_sync_queue():
    """SQLite correctly persists all entities and tracks sync queue state."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        storage = SQLiteStorage(db_path)
        
        # 1. Store experiment
        storage.create_experiment("EXP-TEST-DB", "Test Exp", total_steps=13)
        exp = storage.get_experiment("EXP-TEST-DB")
        assert exp is not None
        assert exp["name"] == "Test Exp"
        
        # 2. Store observations and detections
        storage.save_observation("EXP-TEST-DB", "temperature", 24.5, "C")
        storage.save_detection("EXP-TEST-DB", 1, 0.95, {"class": "BLUE_BOX"})
        storage.save_activity("EXP-TEST-DB", 1, "handling", 0.91, 100.0, 105.0)
        storage.save_rule("EXP-TEST-DB", "person_detected", {"required": True})
        
        obs = storage.get_observations("EXP-TEST-DB")
        assert len(obs) == 1
        assert obs[0]["parameter"] == "temperature"
        
        dets = storage.get_detections("EXP-TEST-DB")
        assert len(dets) == 1
        
        acts = storage.get_activities("EXP-TEST-DB")
        assert len(acts) == 1
        assert acts[0]["activity"] == "handling"
        
        # 3. Store result
        res_payload = {"experiment_id": "EXP-TEST-DB", "summary": "Sample completed"}
        csum = calculate_checksum(res_payload)
        res_id = storage.save_result("RES-001", "EXP-TEST-DB", res_payload, csum, "queued")
        
        res = storage.get_result(res_id)
        assert res is not None
        assert res["checksum"] == csum
        assert res["sync_status"] == "queued"
        
        # 4. Sync queue operations
        pending = storage.get_pending_syncs()
        assert len(pending) >= 1
        
        storage.update_sync_status(pending[0]["id"], res_id, "SYNCED")
        pending_after = storage.get_pending_syncs()
        assert len(pending_after) == 0
    finally:
        storage.close()
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except Exception:
            pass


def test_offline_sync_and_ground_duplicate_rejection():
    """SyncQueueManager queues offline and GroundStation rejects duplicate payloads."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        storage = SQLiteStorage(db_path)
        ground = GroundStationReceiver(storage)
        sync_mgr = SyncQueueManager(storage, ground)
        
        payload = {"experiment_id": "EXP-OFFLINE-01", "data": [10, 20, 30]}
        csum = calculate_checksum(payload)
        storage.save_result("RES-OFF-1", "EXP-OFFLINE-01", payload, csum, "queued")
        
        # 1. Ground link offline: item gets queued
        sync_mgr.set_ground_link(False)
        assert sync_mgr.is_ground_link_online() is False
        
        queued_res = sync_mgr.queue_result("RES-OFF-1", "EXP-OFFLINE-01", payload, csum)
        assert queued_res["status"] == "queued"
        assert sync_mgr.get_pending_count() >= 1
        
        # 2. Restore ground link: flush queue
        sync_mgr.set_ground_link(True, auto_process=False)
        assert sync_mgr.is_ground_link_online() is True
        
        sync_report = sync_mgr.sync_all_pending()
        assert sync_report["synced"] >= 1
        assert sync_mgr.get_pending_count() == 0
        
        # 3. Ground duplicate rejection
        dup_envelope = {"experiment_id": "EXP-OFFLINE-01", "payload": payload, "checksum": csum}
        success, code, dup_resp = ground.receive_payload(dup_envelope)
        assert success is False
        assert code == 409
        assert dup_resp["status"] == "DUPLICATE_REJECTED"
        
        # 4. Corrupted checksum rejection
        bad_envelope = {"experiment_id": "EXP-OFFLINE-01", "payload": payload, "checksum": "0000000000000000000000000000000000000000000000000000000000000000"}
        bad_succ, bad_code, bad_resp = ground.receive_payload(bad_envelope)
        assert bad_succ is False
        assert bad_code == 400
        assert bad_resp["status"] == "REJECTED"
    finally:
        storage.close()
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except Exception:
            pass


def test_structured_result_generation():
    """ResultGenerator generates Section 20 compliant JSON with SHA-256 checksum."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        storage = SQLiteStorage(db_path)
        ground = GroundStationReceiver(storage)
        sync_mgr = SyncQueueManager(storage, ground)
        generator = ResultGenerator(storage, sync_mgr)
        
        result = generator.generate_result(
            experiment_id="EXP-GEN-01",
            status="completed",
            observations=[{"parameter": "lux", "value": 450, "unit": "lx", "timestamp": 1.0}],
            human_activity=[{"activity": "handling", "confidence": 0.91, "duration_seconds": 7.4}],
            events=[{"type": "rule_triggered", "name": "R002", "timestamp": 1.0}],
            rules_triggered=[{"id": "R002", "status": "TRIGGERED", "event": "handling", "confidence": 0.91}],
            fsm_state="COMPLETED",
            telemetry={"fps": 28.5, "latency_ms": 32.0}
        )
        
        # Assert Section 20 JSON structure
        assert result["experiment_id"] == "EXP-GEN-01"
        assert result["schema_version"] == "1.0"
        assert result["status"] == "completed"
        assert result["experiment_state"] == "COMPLETED"
        assert len(result["human_activity"]) >= 1
        assert "checksum" in result
        assert "sync" in result
        
        # Verify checksum on output
        assert verify_payload_checksum(result, result["checksum"]) is True
    finally:
        storage.close()
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except Exception:
            pass


def test_orbita_demo_runner_18_steps():
    """OrbitaDemoRunner executes all 18 deterministic steps to completion."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        storage = SQLiteStorage(db_path)
        ground = GroundStationReceiver(storage)
        sync_mgr = SyncQueueManager(storage, ground)
        runner = OrbitaDemoRunner(storage, sync_mgr)
        
        progress_events = []
        def on_step(evt):
            progress_events.append(evt)
            
        runner.register_listener(on_step)
        
        # Run demo synchronously
        status = runner.run_demo()
        
        assert status["status"] == "COMPLETED"
        assert status["current_step"] == 18
        assert len(status["history"]) == 18
        assert status["structured_result"] is not None
        assert status["structured_result"]["experiment_state"] == "COMPLETED"
        assert status["structured_result"]["sync"]["status"] == "synced"
        
        # Verify 18 steps received by listener
        assert len(progress_events) == 18
        assert progress_events[0]["step"] == 1
        assert progress_events[-1]["step"] == 18
    finally:
        storage.close()
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except Exception:
            pass
