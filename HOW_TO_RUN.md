# ORBITA — How to Run Guide
**SIH Problem ID: 26174 — AI Human Activity Recognition for On-Board BAS Experiments**
**Project:** ORBITA — Offline AI Experiment Copilot for Astronauts

---

## 🚀 Quick Start (Fastest Way to Run)

The easiest and recommended way to run ORBITA is in **Web Mission Control Mode**:

```powershell
# 1. Navigate to the project root
cd c:\project\ORB

# 2. Run the main application
python main.py --mode web --scenario A --port 8000
```

Once started, open your web browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🛠️ Prerequisites & Installation

### 1. Python Environment (3.10 – 3.13)
Install Python dependencies:
```powershell
pip install -r requirements.txt
```
*(Or install core dependencies directly)*:
```powershell
pip install fastapi "uvicorn[standard]" websockets pyttsx3 opencv-python numpy psutil rich python-multipart pydantic pydantic-settings Pillow pytest
```

### 2. Frontend Dashboard (React + TypeScript)
The production bundle is already pre-compiled inside `frontend/dist/` and served automatically by FastAPI. 

If you want to edit or develop the frontend live with hot reload:
```powershell
cd frontend
npm install
npm run dev
```
The dev server will run on **`http://localhost:5173`** (proxied to the backend at port 8000).

To build a fresh production bundle:
```powershell
cd frontend
npm run build
```

---

## 🕹️ Execution Modes

ORBITA supports three primary run modes:

### Mode 1: Web Mission Control (Recommended)
Combines the AI perception pipeline, deterministic FSM, offline TTS voice assistant, MJPEG video streaming, and the React dashboard in a single command:

```powershell
# Run with Scenario A (Nominal execution)
python main.py --mode web --scenario A --port 8000

# Run with Scenario B (Simulate wrong object procedural deviation)
python main.py --mode web --scenario B --port 8000

# Run with Scenario C (Simulate skipped step procedural deviation)
python main.py --mode web --scenario C --port 8000
```

### Mode 2: Live Physical Webcam
To use your laptop or workstation webcam instead of the visual simulator:

```powershell
# Use webcam 0
python main.py --mode web --camera 0

# Use an external USB or payload camera (e.g. device 1)
python main.py --mode web --camera 1

# Process a pre-recorded test video file
python main.py --mode web --camera "path/to/experiment_video.mp4"
```

### Mode 3: Desktop Mode (OpenCV Window)
Direct local desktop window (requires GUI-enabled OpenCV build):

```powershell
python main.py --mode desktop --scenario A
```
*Keyboard controls in desktop mode:*
- `a`: Switch to Scenario A (Nominal)
- `b`: Switch to Scenario B (Wrong Object)
- `c`: Switch to Scenario C (Skipped Step)
- `r`: Reset experiment to Step 1
- `q`: Quit

*(Note: If your environment uses a headless OpenCV build, it will automatically fall back to Web mode).*

---

## 🖥️ Using the Mission Control Dashboard (`http://localhost:8000`)

When the dashboard opens, you have access to:

1. **Live Video Feed (Top-Left)**:
   - Real-time video with AI perception overlays, bounding boxes, action status, and FPS counter.
2. **Alert & Recovery Panel (Bottom-Left)**:
   - Displays real-time procedural alerts.
   - Compares **Expected Action** vs. **Detected Action**.
   - Gives clear spoken and visual voice recovery guidance for astronauts.
3. **Current & Next Step Cards (Top-Right)**:
   - Shows active step number, procedure label, action confidence bar, and upcoming step preview.
4. **Procedure Timeline (Middle-Right)**:
   - Live progress indicator with status markers:
     - `✓ DONE` (Step passed successfully)
     - `→ IN PROGRESS` (Currently awaiting or executing)
     - `⚠ ERROR / SKIPPED` (Procedural deviation flagged)
5. **System Health Status (Bottom-Right)**:
   - Live heartbeats for AI Engine, Camera, TTS Voice, Recording, MJPEG Stream, CPU %, and RAM %.
6. **Top Control Bar**:
   - **Scenario Buttons (`A`, `B`, `C`)**: Instantly switch simulation scenarios live without restarting the server.
   - **RESET**: Reset the experiment FSM back to Step 1.
   - **RECORD / STOP REC**: Save an annotated `.mp4` video recording into `experiments/`.
   - **EXPORT LOG**: Export audit logs to JSON, CSV, and SQLite.

---

## 🧪 Running the Automated Test Suite

ORBITA includes 49 automated tests covering Perception, Temporal HAR, Finite State Machine, Rule Engine, and full End-to-End simulation:

```powershell
# Run all tests
python -m pytest tests/ -v

# Run with concise summary
python -m pytest tests/ -v --tb=short
```

Expected result:
```
============================= 49 passed in ~35s =============================
```

---

## 📡 API Endpoints & Ports

When the backend runs on port 8000:
- **Dashboard UI**: `http://localhost:8000/`
- **MJPEG Live Video Stream**: `http://localhost:8000/video_feed`
- **WebSocket Telemetry**: `ws://localhost:8000/ws/telemetry`
- **System Health Status**: `http://localhost:8000/api/status`
- **Past Experiment History**: `http://localhost:8000/api/experiments`
- **Experiment Control API**: `POST http://localhost:8000/api/control`

---

## ❓ Troubleshooting

### Error: `[Errno 10048] Only one usage of each socket address is normally permitted`
- **Cause**: An instance of ORBITA is already running on port 8000.
- **Fix**: Either use the already-running instance in your browser at `http://localhost:8000`, or terminate the existing process:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess | Stop-Process -Force
  ```
  Or run ORBITA on a different port:
  ```powershell
  python main.py --mode web --scenario A --port 8080
  ```

### Voice output is silent
- Voice is powered by `pyttsx3` (Windows SAPI5 offline speech engine).
- If your system has no audio output device connected or SAPI5 is disabled, ORBITA continues running smoothly in silent mode without throwing errors. All voice messages are still transcribed in real-time on the dashboard HUD.
