# ORBITA — Edge-First Autonomous Experiment Intelligence Platform

**Offline AI Human Activity Recognition (HAR), 3D Human Mesh Recovery (HMR) & Procedural Experiment Intelligence**  
*Target Platform: NVIDIA Jetson Orin Nano / Local Workstation*

---

## 1. Overview

**ORBITA** is an offline-capable experiment copilot platform designed for extreme environments (spaceflight, analog habitats, polar outposts) where continuous ground communication cannot be assumed.

ORBITA processes multi-modal video and sensor inputs locally at the edge, extracts 3D human pose representations (SMPL-24), recognizes temporal activity patterns (PyTorch GRU), verifies manual procedural execution using a safety-critical Finite State Machine (FSM), generates canonical Section 20 JSON results with SHA-256 integrity validation, buffers records locally in SQLite, and synchronizes with Mission Control whenever the ground link becomes available.

---

## 2. Core Workflow Architecture

```text
EXPERIMENT INPUT (Video / Stream / Sim)
     │
     ▼
LOCAL EDGE PROCESSING (YOLOv11 Object Detection)
     │
     ▼
HUMAN MESH RECOVERY (HMR / 24 SMPL 3D Joints & Orientation)
     │
     ▼
TEMPORAL REASONING (30-Frame Sliding Window Feature Fusion)
     │
     ▼
ACTIVITY RECOGNITION (Dual-Head PyTorch GRU)
     │
     ▼
EXPERIMENT INTELLIGENCE (Procedural FSM & Rule Engine R01–R07)
     │
     ▼
SECTION 20 STRUCTURED RESULT (Canonical JSON + SHA-256 Checksum)
     │
     ▼
LOCAL PERSISTENCE (SQLite Database: experiments/orbita.db)
     │
     ▼
OFFLINE SYNCHRONIZATION QUEUE (Retry & Idempotent Duplicate Protection)
     │
     ▼
GROUND CONTROL SYSTEM (HTTP 200 ACK / Hash Verification)
```

---

## 3. Technology Stack

- **Frontend**: React 19, TypeScript 5.8, Tailwind CSS 4, Vite 8, Recharts
- **Backend**: Python 3.13, FastAPI, Pydantic, Uvicorn
- **Computer Vision**: OpenCV, YOLOv11 (`yolo11n.pt`, custom fine-tuned weights), MediaPipe
- **Human Mesh Recovery**: 24-Joint SMPL 3D Kinematic Lifter (`HMRPipeline`) with explicit fallback transparency
- **Temporal HAR**: PyTorch GRU (30-frame temporal window, 64-dim feature vector)
- **Procedural Engine**: 13-step deterministic FSM + Dynamic JSON Rule Engine
- **Data Integrity**: Cryptographic canonical JSON serialization + SHA-256 checksums
- **Database & Sync**: SQLite3 (`orbita.db`) + `SyncQueueManager` + `GroundStationReceiver`
- **Edge AI Compute**: NVIDIA Jetson Orin Nano (Target) / Development CPU & CUDA (Active)

---

## 4. Quick Start

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Node.js 18+ and npm

### Installation
1. Clone the repository and install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Build the React frontend:
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

### Launching ORBITA
Start the integrated server:
```bash
python main.py --mode web
```
Navigate in your browser to:
```
http://localhost:8000/mission
```

---

## 5. Running the 18-Step Deterministic Demo

To run the end-to-end verification demo:
1. Open the UI at `http://localhost:8000/mission` or `/experiments/EXP-01/live`.
2. Click the **[RUN ORBITA DEMO]** button on the top navigation bar.
3. Observe the live 18-step trace from video input, YOLO, 3D HMR, activity recognition, FSM transition, ground connection loss, local offline storage, link restoration, and final Ground ACK with verified SHA-256 checksum.

---

## 6. Running Automated Tests

Run the complete 182-test automated suite:
```bash
pytest tests/ -v
```
All 182 tests pass 100% with zero regressions.

---

## 7. Documentation Index

- [`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md): Comprehensive initial audit and post-implementation verification matrix.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Complete subsystem architecture document.
- [`docs/ARCHITECTURE_DECISIONS.md`](docs/ARCHITECTURE_DECISIONS.md): Architecture Decision Records (ADRs).
- [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md): Evaluator step-by-step demonstration walkthrough.
- [`docs/API.md`](docs/API.md): Full REST API & WebSocket reference.
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): Transparent documentation of constraints and prototype assumptions.
