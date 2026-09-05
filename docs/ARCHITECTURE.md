# ORBITA System Architecture

ORBITA is an edge-native AI Activity Recognition and Mission Oversight System designed for deployment on edge compute platforms such as the **NVIDIA Jetson Orin Nano** and workstation environments.

---

## 1. End-to-End Pipeline

```text
       [ Camera / Input Stream ]
 (Jetson CSI / USB V4L2 / Phone IP / Simulation)
                   │
                   ▼
       [ Video Capture Engine ]
     (OpenCV VideoCapture / GStreamer)
                   │
                   ▼
     [ Multi-Task Perception Engine ]
     ├── YOLOv8 Object Detection (Boxes & Labels)
     ├── YOLOv8-Pose / Fallback Keypoint Estimator
     └── Color-Centroid Fallback Chroma Tracker
                   │
                   ▼
    [ Hand & Interaction Tracking ]
     ├── Wrist Keypoint Projection
     ├── Hand Bounding Box Extraction
     └── Distance & Grasp Heuristics (IoU / Proximity)
                   │
                   ▼
     [ Feature Fusion & Windowing ]
     ├── Pose Vector (Normalized Coordinates)
     ├── Object Proximities & One-Hot Flags
     └── 15-Frame Rolling Temporal Buffer
                   │
                   ▼
        [ Temporal HAR Model ]
     ├── Heuristic Classifier (Fast, Rule-Based)
     └── Optional GRU/LSTM Temporal Classifier
                   │
                   ▼
       [ Reasoning & FSM Engine ]
     ├── State Manager & Transition Graph
     ├── Rule Engine (R01 Wrong Object, R02 Skipped Step, etc.)
     └── Anomaly Detection & State Validation
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 [ Voice Feedback ]   [ Persistence & Telemetry ]
  (pyttsx3 / TTS)      (SQLite experiments/orbita.db)
                              │
                              ▼
                [ FastAPI & WebSocket Server ]
                   (Telemetry / Video Stream)
                              │
                              ▼
                  [ React Dashboard UI ]
          (Live Feed, Diagnostics, Mission Control)
```

---

## 2. Core Python Components (`orbita/`)

| Module | Purpose |
|---|---|
| `orbita.perception` | YOLOv8 object detector and pose keypoint estimator |
| `orbita.interaction` | Hand-object contact analysis and grasp detection |
| `orbita.har` | Feature vector fusion and temporal action recognition |
| `orbita.reasoning` | Finite State Machine (FSM) procedure tracking and rule validation |
| `orbita.video` | Multi-source camera manager (CSI, USB, Phone IP Webcam, Simulation) |
| `orbita.database` | SQLite telemetry logger and step event tracking |
| `orbita.voice` | Real-time text-to-speech feedback and procedural guidance |
| `orbita.simulation` | Synthetic frame generation for testing scenarios A, B, and C |
| `orbita.backend` | FastAPI REST endpoints and WebSockets for telemetry & video |
| `orbita.app` | Central dataclass configurations and JSON loaders |

---

## 3. Frontend Architecture (`frontend/`)

- **Framework**: React 19 + TypeScript + Vite 8
- **Styling**: Tailwind CSS v4 + Vanilla CSS animations
- **Key Modules**:
  - `pages/LiveExperiment.tsx`: Real-time annotated video stream, live step checklist, action status, and voice logs.
  - `pages/ProcedureTimeline.tsx`: Sequential timeline with durations, pass/fail status, and transition logs.
  - `pages/SystemDiagnostics.tsx`: Live FPS, camera health, latency, database size, and model states.
  - `pages/PastExperiments.tsx`: Historical runs loaded directly from `experiments/orbita.db`.
  - `mission-control/`: Integrated orbital mission control view with interactive 3D globe and telemetry HUD.
  - `components/camera/`: Live camera switching (Jetson CSI/USB, Phone IP Camera, Simulation).
