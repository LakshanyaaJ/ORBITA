# ORBITA — Demonstration Guide

This guide provides step-by-step instructions for judges, evaluators, and operators to run the deterministic **ORBITA** demonstration.

---

## Quick Start (One-Click Launch)

1. **Start the Backend**:
   ```bash
   python main.py --mode web
   ```
   *The server starts on `http://localhost:8000`.*

2. **Access the Mission Control Dashboard**:
   Open your browser to:
   ```
   http://localhost:8000/mission
   ```

---

## Running the 18-Step Deterministic Demo

Click the prominent **[RUN ORBITA DEMO]** button located on the top navigation bar or inside the Live Experiment interface.

The 18-step progression executes automatically:

| Step | Phase | Action | Verification |
| :--- | :--- | :--- | :--- |
| **1** | SETUP | Load Experiment | Protocol loaded from configuration |
| **2** | SETUP | Start Experiment | FSM state transitions to `RUNNING` |
| **3** | SETUP | Start Simulated Video | `vdata/20260905_145858.mp4` stream mounted |
| **4** | CV / HMR | Detect Person (YOLO) | Astronaut bounding box detected with confidence |
| **5** | CV / HMR | HMR Processes Subject | 3D body representation calculated |
| **6** | CV / HMR | Extract Pose Features | 24 SMPL 3D joint coordinates extracted |
| **7** | HAR / FSM | Temporal Model Processing | 30-frame sliding window vector constructed |
| **8** | HAR / FSM | Activity Recognized (GRU) | GRU classifies action: `handling` (confidence 92.4%) |
| **9** | HAR / FSM | FSM Updates | State advances to `STEP 2 (PICKUP_BLUE_BOX)` |
| **10**| HAR / FSM | Experiment Rule Triggers | Rule R002 triggered: Required action verified |
| **11**| OFFLINE | Ground Connection Lost | Simulated uplink loss: Ground Link switches to **OFFLINE** |
| **12**| OFFLINE | Local Edge Processing | Edge compute continues without interruptions |
| **13**| OFFLINE | Experiment Completes | FSM advances to final `COMPLETED` state |
| **14**| OFFLINE | Structured Result Generated | Section 20 compliant JSON produced |
| **15**| OFFLINE | Result Stored Locally | Saved to SQLite database `experiments/orbita.db` |
| **16**| SYNC | Connection Restored | Ground Link switches to **ONLINE** |
| **17**| SYNC | Result Synchronized | Payload dispatched from local queue via `POST /api/sync` |
| **18**| SYNC | SYNCED (Ground ACK 200) | Ground receiver validates SHA-256 and returns ACK 200 |

---

## Verifying Subsystems in the User Interface

### 1. 3D Human Mesh Recovery (HMR) Viewer
- Navigate to `/experiments/EXP-01/live`.
- In the right-hand panel, inspect the **HMR 3D MESH RECOVERY** widget.
- Observe:
  - 24 SMPL joints connected in a 3D kinematic wireframe.
  - View angle selector: `FRONT`, `SIDE`, `TOP`.
  - Global orientation angles (Yaw, Pitch, Roll in degrees).
  - Estimated camera translation $[T_x, T_y, T_z]$.
  - Transparent badge: `HMR_PROTOTYPE_FALLBACK (Deterministic 3D Kinematic Lifter)`.

### 2. Ground Link & Offline Simulation
- Click the **[GROUND: ONLINE]** badge in the top navigation bar.
- The link toggles to **[GROUND: OFFLINE]**.
- Execute actions or let the demo run: all results are queued locally (`PENDING: 1`).
- Click again to restore the link: the queue automatically synchronizes, and the badge returns to **[GROUND: ONLINE]** with `0 PENDING`.

### 3. Section 20 Structured JSON Result
- Click **[STRUCTURED JSON]** on the Live Experiment screen or after demo completion.
- The modal displays the canonical JSON output.
- Observe:
  - `schema_version`: "1.0"
  - `status`: "completed"
  - `human_activity`: [{ "activity": "handling", "confidence": 0.91, "duration_seconds": 7.4 }]
  - `checksum`: Verified 64-character SHA-256 string.
  - Action buttons: **[COPY JSON]** and **[EXPORT JSON]**.

### 4. Edge Hardware & System Status
- On `/mission` or `/system`, inspect the **EDGE COMPUTE NODE** row.
- Device: `Development CPU (Intel/AMD x86_64)` (or `CUDA GPU` if present).
- Target Deployment: `NVIDIA Jetson Orin Nano`.
- FPS: Real-time stream rate ($\approx 30$ FPS).
- Latency: Real-time inference latency ($\approx 40$–$70$ ms).
