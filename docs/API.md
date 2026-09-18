# ORBITA — REST API & WebSocket Reference

This document describes the complete API surface of the ORBITA backend server.

---

## 1. Experiment Lifecycle Management

### `GET /api/experiments`
Returns a list of all configured and past experiments.
- **Response**: Array of experiment summaries (ID, name, status, total steps, created timestamp).

### `GET /api/experiments/{id}`
Returns the detailed configuration and step protocol for a specific experiment.

### `POST /api/experiments/{id}/start`
Starts or arms the experiment session.
- **Payload**: `{ "source": "video_file" | "webcam" | "sim", "path": "vdata/..." }`

### `POST /api/experiments/{id}/pause`
Pauses the current procedural FSM and halts step timers.

### `POST /api/experiments/{id}/resume`
Resumes execution of a paused experiment session.

### `POST /api/experiments/{id}/stop`
Terminates the experiment, stops video recording, and finalizes structured logging.

---

## 2. Perception & Telemetry Endpoints

### `GET /api/detection`
Returns the latest YOLO object detections, class names, bounding boxes, and confidence scores.

### `GET /api/hmr`
Returns the latest 3D Human Mesh Recovery (HMR) SMPL data:
- `is_fallback`: Boolean indicating prototype kinematic lifter state.
- `joints_3d`: 24 SMPL joints with $(X, Y, Z)$ coordinates and confidence.
- `global_orientation`: Yaw, pitch, roll angles.
- `camera_translation`: $[T_x, T_y, T_z]$.

### `GET /api/activity`
Returns the active temporal human activity recognition prediction from the dual-head GRU:
- `activity`: e.g. `"handling"`, `"reaching"`, `"standing"`.
- `confidence`: Float between 0.0 and 1.0.
- `duration_seconds`: Float duration in seconds.

### `GET /api/events`
Returns recent procedural and safety rule events.

---

## 3. Results & Cryptographic Integrity

### `GET /api/results`
Returns all completed experiment results stored in the local SQLite database.

### `GET /api/results/{id}`
Returns the Section 20 structured experiment JSON for a given experiment ID, including canonical SHA-256 checksum and sync status.

---

## 4. Offline Synchronization & Ground System

### `POST /api/sync`
Ground Station receiver endpoint.
- **Payload**:
  ```json
  {
    "payload": { ... },
    "checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  }
  ```
- **Responses**:
  - `200 OK`: Payload verified and stored.
  - `400 Bad Request`: Checksum mismatch or invalid payload format.
  - `409 Conflict`: Duplicate payload already acknowledged.

### `GET /api/sync/status`
Returns synchronization queue status:
- `ground_link_online`: Boolean.
- `pending_count`: Number of buffered payloads awaiting upload.
- `synced_count`: Number of successfully uploaded payloads.
- `failed_count`: Number of failed attempts.

### `POST /api/sync/ground_link`
Toggles simulated ground link connectivity.
- **Payload**: `{ "online": true | false }`

---

## 5. System Status & Demo Controls

### `GET /api/system/status`
Returns Edge AI hardware compute status:
- `target_hardware`: `"NVIDIA Jetson Orin Nano"`
- `active_device`: `"Development CPU"` or `"CUDA GPU"`
- `cuda_available`: Boolean
- `fps`: Float streaming rate
- `ai_fps`: Float inference rate
- `latency_ms`: Float inference latency
- `ground_link`: `"ONLINE"` or `"OFFLINE"`
- `pending_sync`: Integer queue size

### `POST /api/demo/run`
Triggers the deterministic 18-step end-to-end ORBITA demonstration sequence in the background.

### `GET /api/demo/status`
Returns current step, running state, and latest payload of the active demo run.

---

## 6. Realtime WebSockets

### `WS /ws/telemetry`
High-speed broadcast channel streaming live procedural FSM state, YOLO detections, hand interactions, 3D HMR mesh, and telemetry at 10–15 Hz.

### `WS /ws/experiments/{experiment_id}`
Experiment-scoped WebSocket stream delivering live telemetry and procedural event updates for a specific experiment ID.
