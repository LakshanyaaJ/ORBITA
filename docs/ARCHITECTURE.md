# ORBITA — System Architecture Documentation

## 1. System Overview

**ORBITA** is an edge-first AI experiment intelligence platform designed for extreme operational environments (space exploration, remote scientific outposts, analog research facilities) where continuous ground uplink cannot be assumed.

The platform executes real-time computer vision, 3D human mesh recovery, temporal activity recognition, and procedural verification locally on edge hardware (NVIDIA Jetson Orin Nano / local workstation), persists all telemetry to an append-only local SQLite database, and synchronizes payloads to mission control via an autonomous offline sync queue with cryptographic SHA-256 integrity validation.

```
                      [ OPERATIONAL WORKFLOW ]
                                 │
                        [ VIDEO / SENSOR INPUT ]
                                 │
                       [ LOCAL EDGE PROCESSING ]
                                 │
                         [ COMPUTER VISION ]
                     (YOLOv8 Multi-Object Detect)
                                 │
                     [ HUMAN MESH RECOVERY (HMR) ]
                   (3D Pose & 24 SMPL Joint Kinematics)
                                 │
                       [ TEMPORAL REASONING ]
                    (30-Frame Sliding Feature Window)
                                 │
                     [ ACTIVITY RECOGNITION (HAR) ]
                         (Dual-Head PyTorch GRU)
                                 │
                     [ EXPERIMENT INTELLIGENCE ]
                   (Deterministic 13-Step FSM & Rules)
                                 │
                    [ SECTION 20 STRUCTURED RESULT ]
                      (Canonical JSON + SHA-256)
                                 │
                       [ LOCAL OFFLINE STORAGE ]
                          (SQLite: orbita.db)
                                 │
                     [ AUTONOMOUS SYNC QUEUE ]
                    (Retry, Dupe-Guard, ACK 200)
                                 │
                       [ GROUND CONTROL SYSTEM ]
```

---

## 2. Component Architecture

### A. Presentation Layer (Frontend)
- **Framework**: React 19 + TypeScript 5.8 + Vite 8
- **Styling**: Tailwind CSS 4 + Space / Mission Control Design System
- **Key Modules**:
  - `LiveExperiment.tsx`: Unified cockpit displaying low-latency MJPEG video overlay (<100ms), step progress, astronaut guidance, 3D HMR Mesh Recovery Card, and procedural checklist.
  - `MissionHome.tsx`: Executive dashboard showing real-time compute device status (CPU vs CUDA), Ground Link online/offline state, pending sync queue count, and SHA-256 integrity badge.
  - `OrbitaDemoModal.tsx`: Deterministic 18-step sequential workflow visualization executing end-to-end trace with live status badges and telemetry inspection.
  - `StructuredResultModal.tsx`: Section 20 compliant JSON result inspector with SHA-256 checksum verification, copy-to-clipboard, and JSON file export.
  - `HmrMeshViewer.tsx`: 3D SMPL kinematic tree viewport with 24 joints, body orientation (yaw/pitch/roll), camera translation, and fallback state badge.

### B. Application & Routing Layer (Backend)
- **Framework**: FastAPI (asyncio) + Pydantic + Uvicorn
- **Concurrency**:
  - *Capture Loop*: Dedicated thread continuously filling `LatestFrameBuffer(size=1)` at 25–30 FPS.
  - *AI Inference Loop*: Thread-safe inference worker running YOLO + Pose + HMR + Hands + HAR + FSM at 10–15 FPS.
  - *MJPEG Streamer*: Blits cached AI bounding boxes and annotations in <1ms to serve browser streams without blocking API calls.
  - *WebSocket Telemetry*: Real-time state broadcasting over `/ws/telemetry` and `/ws/experiments/{id}`.

### C. Perception & Computer Vision Layer
- **Object Detection**: `ObjectDetector` (`core_ai/perception/object_detector.py`) running YOLOv8 nano with custom fine-tuned weights (`orbita_yolo_detector_v4.pt`) and chroma color mapping. Detects lab objects: `BLUE_BOX`, `YELLOW_BOX`, `PEN`, `WATCH`, `MAIN_BOX`, `PERSON`.
- **2D Pose Estimation**: `PoseEstimator` (`core_ai/perception/pose_estimator.py`) utilizing `yolov8n-pose.pt` extracting 17 keypoints with confidence scores.
- **Hand Landmark Tracking**: `HandTracker` (`core_ai/perception/hand_tracker.py`) powered by MediaPipe hand landmarker, detecting wrists, fingertips, and interaction grasp centers.
- **Hand-Object Interaction**: `HandObjectInteractionTracker` calculating Euclidean distance, bounding-box overlap ratio, displacement velocity, and contact duration.

### D. Human Mesh Recovery (HMR) Subsystem
- **Module**: `HMRPipeline` (`core_ai/perception/hmr_pipeline.py`)
- **Topology**: 24 standard SMPL body joints (Pelvis, Spine, Neck, Head, Shoulders, Elbows, Wrists, Hips, Knees, Ankles, Feet, Hands).
- **Interface**: `recover_mesh(frame, person_bbox, keypoints_2d) -> HMRMeshResult`
- **Telemetry**:
  - 3D spatial joint coordinates $(X, Y, Z)$ in meters.
  - 3D global body orientation (Yaw, Pitch, Roll in radians and degrees).
  - Estimated camera translation $[T_x, T_y, T_z]$.
  - Fallback indicator: Explicit `is_fallback=True` flag and badge `HMR_PROTOTYPE_FALLBACK (Deterministic 3D Kinematic Lifter)` ensuring complete scientific honesty.

### E. Temporal Activity Recognition (HAR) Subsystem
- **Module**: `core_ai/har/temporal_model.py` & `core_ai/har/feature_fusion.py`
- **Sliding Window**: 30 consecutive frames at 15 FPS ($\approx 2.0$ seconds temporal context).
- **Feature Vector**: 64 dimensions combining normalized torso-relative joint positions, hand-object distances, velocity derivatives, and object presence flags.
- **Neural Model**: PyTorch `OrbitaTemporalGRU` (2 layers, 128 hidden units, dual head).
- **Activity Ontology**: Maps low-level actions to canonical human activities:
  `standing`, `walking`, `reaching`, `handling`, `inspection`, `interaction`, `unknown`.

### F. Procedural Reasoning & Rule Engine
- **Procedural FSM**: `ActionGate` & `StateManager` enforcing the official 13-step lab protocol:
  `IDENTIFY_BLUE_BOX` → `PICKUP_BLUE_BOX` → `PLACE_BLUE_BOX_A` → `IDENTIFY_YELLOW_BOX` → `PICKUP_YELLOW_BOX` → `PLACE_YELLOW_BOX_B` → `PICKUP_PEN` → `PLACE_PEN_BLUE_BOX` → `PICKUP_WATCH` → `PLACE_WATCH_YELLOW_BOX` → `MOVE_BLUE_BOX_A_TO_B` → `MOVE_YELLOW_BOX_B_TO_A` → `EXPERIMENT_COMPLETE`.
- **Configurable Rule Engine**: Evaluates safety and procedural rules (R001–R007) dynamically from JSON configurations, recording rule triggers with timestamps, duration, and confidence.

### G. Offline Storage & Cryptographic Integrity
- **Database**: `OrbitaDB` (`core_ai/database/sqlite_db.py`) managing SQLite storage with tables:
  `experiments`, `step_records`, `observations`, `detections`, `activities`, `rules`, `events`, `results`, `sync_queue`, `ground_receipts`.
- **Integrity**: `core_ai/database/integrity.py` computes canonical SHA-256 hashes of JSON payloads (sorted keys, compact separators). Any modification to the payload invalidates the checksum.

### H. Synchronization & Ground Station
- **Queue Manager**: `SyncQueueManager` (`core_ai/sync/sync_manager.py`) queues offline experiment results and dispatches batches upon ground link re-establishment.
- **Ground Station Receiver**: `GroundStationReceiver` (`core_ai/sync/ground_system.py`) validates SHA-256 hashes, rejects duplicate payloads (HTTP 409), and issues cryptographic receipts (ACK 200).
- **Ground Link Simulator**: Live toggle control (`POST /api/sync/ground_link`) allowing judges and operators to simulate communications blackouts without stopping local experiment execution.
