# ORBITA — System Limitations & Assumptions

In accordance with the core ORBITA architectural principle of **Absolute Transparency**, this document candidly records all existing constraints, assumptions, and hardware fallbacks.

---

## 1. Human Mesh Recovery (HMR) Model Weights
- **Current State**: The full 4D-Humans / SMPL-X neural network weights (several gigabytes in size) and associated CUDA mesh decoders are not pre-packaged in this prototype repository.
- **Fallback Implementation**: A deterministic 3D kinematic joint lifter (`core_ai/perception/hmr_pipeline.py`) calculates 3D coordinates for all 24 SMPL joints, global orientation angles, and camera translation from 2D keypoints and depth heuristics.
- **UI Labeling**: All fallback outputs are explicitly marked with `is_fallback: true` in the API and displayed with the badge `HMR_PROTOTYPE_FALLBACK (Deterministic 3D Kinematic Lifter)`.
- **Pluggability**: The `HMRPipeline` interface is designed as an architectural drop-in, allowing real neural 4D-Humans weights to be mounted without any modifications to downstream HAR or FSM modules.

---

## 2. Edge Hardware Execution Environment
- **Target Deployment**: NVIDIA Jetson Orin Nano (6-core ARM Cortex-A78AE, Ampere GPU, 40 TOPS INT8).
- **Evaluation Environment**: Development workstation (x86_64 CPU / local GPU).
- **System Telemetry**: The application dynamically reports the actual execution compute device (`Development CPU (Intel/AMD x86_64)` or `NVIDIA GPU`), while explicitly documenting the target deployment target (`NVIDIA Jetson Orin Nano`). The software never simulates or disguises CPU execution as Jetson hardware.

---

## 3. Ground Station Uplink
- **Operational Reality**: In deep space or extreme analog deployments, ground station communication is intermittent and subject to orbital delays.
- **Prototype Implementation**: The Ground Station receiver (`GroundStationReceiver`) runs locally on the edge node or loopback network, listening at `POST /api/sync`.
- **Integrity Validation**: While the transmission channel is local/loopback in the demo environment, payload validation is genuine: canonical SHA-256 hashes are recalculated, verified against the payload, and logged in SQLite. Duplicate payloads are strictly rejected with HTTP 409.

---

## 4. Activity Recognition Ontology
- **Trained Model**: PyTorch `OrbitaTemporalGRU` checkpoint trained over procedural laboratory interactions (`TAKE`, `PLACE`, `OPEN`, `CLOSE`, `ACTIVATE`, `PERFORM`, `STORE`).
- **Target Human Activities**: Mapped dynamically to canonical human activity classifications (`standing`, `walking`, `reaching`, `handling`, `inspection`, `interaction`, `unknown`).
- **Prototype Status**: Labeled as `Prototype Activity Recognition` in the interface.

---

## 5. Security & Authentication
- **Current State**: Prototype-grade security suitable for closed research networks:
  - Input validation via Pydantic models.
  - Safe parameterized SQLite queries.
  - SHA-256 cryptographic payload integrity validation.
  - Idempotent duplicate sync protection.
- **Production Gaps**: Role-based access control (RBAC), end-to-end TLS client certificates, and encrypted database-at-rest are not enabled in this prototype build.
