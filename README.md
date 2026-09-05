# ORBITA

**Offline AI Human Activity Recognition (HAR) & Autonomous Experiment Copilot**  
*Optimized for Edge Deployment on NVIDIA Jetson Orin Nano*

---

## Architecture Overview

ORBITA is an edge-first AI system designed to monitor, guide, and validate manual procedures performed by astronauts and researchers during scientific spaceflight experiments.

```text
Phone IP Camera / Jetson Cam / Simulation
                    │
                    ▼
          Camera Manager (OpenCV)
          (Low-latency drop-stale buffer)
                    │
                    ▼
          Real-Time Perception Pipeline
          ├── Object Detector (YOLOv8 + Ambient Chroma)
          ├── Pose Estimator (YOLOv8-Pose, 17 Keypoints)
          └── Hand Tracker (L/R Wrists & Fingertip Trajectory)
                    │
                    ▼
          Feature Fusion
          (30-frame temporal sliding window, 64-dim vector)
                    │
                    ▼
          Temporal HAR (Action Classifier)
                    │
                    ▼
          Procedural Reasoning
          ├── Finite State Machine (Step Verification)
          ├── Rule Engine (Out-of-sequence, wrong object, skipped steps)
          └── Audio & HUD Feedback Generator (TTS)
                    │
                    ▼
          FastAPI & React Dashboard (frontend)
          (Real-time video feed, overlays, telemetry WebSocket, mission control)
```

---

## Directory Structure

```text
ORBITA/
├── config/                               # Central configuration & experiment steps
│   ├── experiment_config.json            # Step definitions, expected objects & timeouts
│   └── orbita_config.json                # Hardware, model, voice & logging settings
│
├── data/                                 # Runtime data storage
│   └── .gitkeep
│
├── experiments/                          # Database & runtime experiment sessions
│   ├── .gitkeep
│   └── orbita.db                         # SQLite session database
│
├── models/                               # AI Neural Network Weights
│   ├── yolov8n.pt                        # YOLOv8 Object Detection (COCO classes)
│   └── yolov8n-pose.pt                   # YOLOv8 Pose Estimation (17 keypoints)
│
├── server/                               # Top-level API service facade (uvicorn server.server:app)
│   ├── __init__.py
│   └── server.py
│
├── core_ai/                              # Core Python Edge-AI Package
│   ├── app/                              # Configuration loaders
│   ├── backend/                          # FastAPI REST API & WebSocket Server
│   ├── database/                         # SQLite session persistence
│   ├── har/                              # Human Activity Recognition & Feature Fusion
│   ├── interaction/                      # Hand-object spatial interaction tracking
│   ├── logging/                          # Experiment session event logger
│   ├── perception/                       # Computer Vision (YOLOv8, Pose, Hands)
│   ├── reasoning/                        # Finite State Machine & Rule Engine
│   ├── simulation/                       # Offline synthetic simulator (Scenarios A/B/C)
│   ├── video/                            # Unified Camera Manager & Low-Latency IP Camera
│   └── voice/                            # Pyttsx3 offline astronaut voice assistant
│
├── frontend/                             # Unified React + Vite Frontend
│   ├── src/
│   │   ├── api/                          # Camera & telemetry client hooks
│   │   ├── components/                   # CameraControlPanel, status, metrics
│   │   ├── mission-control/              # Integrated Mission Control suite
│   │   └── pages/                        # LiveExperiment, Diagnostics, Timeline
│   ├── package.json
│   └── vite.config.ts
│
├── archive/                              # Consolidated legacy implementations
│   ├── README.md                         # Documentation of archived modules
│   ├── frontend-prototype/               # Initial prototype React application
│   └── orbita-mission-control/           # Standalone mission control template
│
├── tests/                                # Automated Pytest Suite (61 tests)
│   ├── test_camera.py                    # IP webcam validation & buffer tests
│   ├── test_end_to_end.py                # End-to-end scenario validation (A/B/C)
│   ├── test_fsm.py                       # FSM state transition & rule engine tests
│   ├── test_har.py                       # HAR feature window & prediction tests
│   └── test_perception.py                # Object detection, pose & hand tracking tests
│
├── scripts/                              # Deployment & utility automation
│   ├── download_models.py                # Model verification/download utility
│   └── run_jetson.sh                     # Jetson Orin Nano boot & launch script
│
├── docs/                                 # Architectural & deployment specifications
│   ├── ARCHITECTURE.md                   # Full end-to-end pipeline architecture
│   └── JETSON_DEPLOYMENT.md              # Hardware setup & performance guide
│
├── main.py                               # Top-level CLI launcher (web, desktop, headless)
├── requirements.txt                      # Python dependencies
├── HOW_TO_RUN.md                         # Quick-start execution guide
└── .gitignore                            # Clean Git repository ignore rules
```

---

## Quick Start

### 1. Start Backend
```bash
python main.py --mode web --scenario A
```

### 2. Start Frontend
```bash
cd frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser.

For complete execution instructions, see [HOW_TO_RUN.md](HOW_TO_RUN.md).
