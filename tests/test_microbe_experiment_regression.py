"""
ORBITA EXP-MICROBE Integration & Regression Test Suite
=====================================================
Verifies that EXP-MICROBE (Microbial Experiment in Microgravity) is correctly
added and completely isolated from existing experiments (EXP-01, EXP-VDATA, EXP-02).
"""

import pytest
from core_ai.database.sqlite_db import OrbitaDB
import tempfile
from pathlib import Path


def test_exp_microbe_db_isolation_and_logging():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_microbe.db"
        db = OrbitaDB(db_path=db_path)

        # 1. Create EXP-MICROBE record
        db.create_experiment(
            experiment_id="EXP-MICROBE",
            name="Microbial Experiment in Microgravity",
            objective="Observe microbial experiment behavior and demonstrate automated experiment monitoring.",
            total_steps=7
        )

        # 2. Log step for EXP-MICROBE
        db.log_step(
            experiment_id="EXP-MICROBE",
            step_number=1,
            action="SETUP",
            label="Prepare Experiment Setup",
            status="CORRECT",
            detected_action="SETUP",
            detected_object="WORK_SURFACE",
            confidence=0.96
        )

        # 3. Log event for EXP-MICROBE
        db.log_event(
            experiment_id="EXP-MICROBE",
            event_type="STEP_VALIDATED",
            data={"step": 1, "label": "Prepare Experiment Setup"},
            confidence=0.96
        )

        # 4. Create EXP-01 record to verify non-interference
        db.create_experiment(
            experiment_id="EXP-01",
            name="Yellow and Blue Box",
            total_steps=13
        )

        # 5. Verify EXP-MICROBE logs are distinct from EXP-01
        microbe_events = db.get_events("EXP-MICROBE")
        exp01_events = db.get_events("EXP-01")

        assert len(microbe_events) == 1, "EXP-MICROBE should have 1 logged event"
        assert len(exp01_events) == 0, "EXP-01 should have 0 logged events"
        assert microbe_events[0]["experiment_id"] == "EXP-MICROBE"

        db.close()
        import gc
        gc.collect()
