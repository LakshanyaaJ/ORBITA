# ORBITA — Architecture Decision Records (ADRs)

This document records the critical design and architectural decisions made during the integration and enhancement of the ORBITA platform.

---

## ADR-001: Modular Human Mesh Recovery (HMR) with Deterministic Kinematic Fallback

### Context
Section 6 requires 3D Human Mesh Recovery (HMR) / 4D-Humans. However, full 4D-Humans/SMPL neural network model weights exceed several gigabytes, require specialized CUDA environments, and were not pre-bundled in the original repository.

### Decision
1. Designed a clean, decoupled abstraction: `HMRPipeline` in `core_ai/perception/hmr_pipeline.py`.
2. Defined the 24 standard SMPL joint topology (`HMRMeshResult` with 24 3D coordinates, orientation yaw/pitch/roll, and translation vector).
3. Built a deterministic 3D kinematic joint lifter as the development and offline fallback engine.
4. **Transparency Commitment**: Set `is_fallback=True` explicitly in telemetry payloads and displayed the prominent badge `HMR_PROTOTYPE_FALLBACK (Deterministic 3D Kinematic Lifter)` in the UI. No fake neural output is disguised.

### Consequences
- Pluggable interface allows drop-in neural 4D-Humans weights without altering the rest of the application.
- 100% deterministic testability in CPU-only and lightweight edge environments.

---

## ADR-002: Canonical SHA-256 Hashing for Data Integrity

### Context
Section 21 requires SHA-256 payload integrity to prevent tampering, corrupt packet detection, and ground-side validation. JSON serialization in Python can yield varying key orders and whitespace, causing false hash mismatches.

### Decision
1. Implemented `canonical_json_bytes()` in `core_ai/database/integrity.py` using `sort_keys=True` and `separators=(',', ':')` encoded as UTF-8.
2. Stripped self-referential `"checksum"` keys prior to hashing.
3. Ground station recalculates hash on received payload and strictly enforces equality before issuing ACK 200.

### Consequences
- Idempotent and deterministic integrity validation across heterogeneous operating systems and network boundaries.

---

## ADR-003: Offline-First SQLite Persistence with Automated Schema Migration

### Context
Section 15 and 33 require full local persistence of experiments, observations, detections, activities, rules, results, and sync queues so the platform operates autonomously without internet or server connectivity. Existing databases in `experiments/orbita.db` had legacy table definitions.

### Decision
1. Implemented complete relational tables in `core_ai/database/sqlite_db.py`.
2. Added automated `ALTER TABLE ... ADD COLUMN` migration guards during initialization to ensure legacy database files seamlessly acquire required columns (`objective`, `configuration`, `total_steps`).

### Consequences
- Zero data loss when upgrading existing installations.
- All experiment steps, detections, and rule triggers persist across browser refreshes and system reboots.

---

## ADR-004: Idempotent Ground Synchronization with HTTP 409 Duplicate Prevention

### Context
Section 18 and 19 require offline buffering and reliable ground synchronization without creating duplicate experiment records if packets are retried.

### Decision
1. Implemented `GroundStationReceiver` with an append-only `ground_receipts` table keyed by `checksum` and `experiment_id`.
2. When a payload with an already-processed checksum is submitted, the receiver responds with HTTP 409 and existing receipt metadata, ensuring the local queue marks it as synced without throwing an unrecoverable failure.

### Consequences
- Safe network retries under unstable radio communications blackouts.

---

## ADR-005: Deterministic 18-Step Automated Demo Sequence

### Context
Section 25 and 26 mandate an automated, deterministic "RUN ORBITA DEMO" workflow that technical evaluators and judges can trigger with one click, without requiring a live physical webcam or lab environment.

### Decision
1. Implemented `OrbitaDemoRunner` in `core_ai/simulation/demo_runner.py` executing the exact 18-step progression:
   Experiment Load → Start → Stream Mount → YOLO Detection → HMR 3D Mesh → Pose Feature Extraction → Temporal Window → GRU Classification → FSM Transition → Rule Trigger → Ground Link Drop → Local Edge Processing → Completion → Section 20 JSON Generation → SQLite Persistence → Link Reconnect → Queue Upload → Ground Station ACK 200.
2. Emits real-time progress events over WebSocket to drive interactive UI modals.

### Consequences
- Evaluators can verify every arrow in the system workflow deterministically and reliably.
