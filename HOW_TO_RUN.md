# ORBITA — Execution & Deployment Guide

This guide describes how to run and deploy **ORBITA** (Offline AI Human Activity Recognition & Experiment Copilot) on local workstations and edge devices like the **NVIDIA Jetson Orin Nano**.

---

## 1. System Requirements

### Workstation / Jetson Orin Nano
- **Operating System**: Linux (Ubuntu 20.04 / 22.04 / JetPack 5.x–6.x) or Windows 10/11
- **Python**: 3.9+ (Python 3.10–3.13 supported)
- **Node.js**: v18+ or v20+ with npm
- **Hardware Acceleration**: NVIDIA Jetson Orin Nano (CUDA / TensorRT supported via PyTorch)

---

## 2. Environment Setup

### A. Python Backend Setup
```bash
# 1. Clone or navigate to the repository
cd ORBITA

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv

# On Linux / Jetson:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### B. Frontend Setup
```bash
cd frontend
npm install
```

---

## 3. Launching ORBITA

### Option 0: One-Click Windows Launcher (Fastest)
Double-click or execute from the terminal:
```cmd
start.bat
```
This automatically:
1. Detects your Python environment (`.venv`, `venv`, or system Python).
2. Launches the **FastAPI Backend** on port `8000` in a titled console window.
3. Launches the **React Frontend** on port `5173` in a titled console window.
4. Opens `http://localhost:5173` in your default browser.

To terminate both services at any time, close their windows or run:
```cmd
stop.bat
```

---

### Option 1: Web Mode (Manual Full Stack)

**Terminal 1 — Start the FastAPI Backend**:
```bash
python main.py --mode web --scenario A
```
* Backend API serves at: `http://localhost:8000`
* MJPEG Video Stream at: `http://localhost:8000/video_feed` (and `/api/camera/stream`)
* Telemetry WebSocket at: `ws://localhost:8000/ws/telemetry`

**Terminal 2 — Start the React Dashboard**:
```bash
cd frontend
npm run dev
```
* Dashboard will be live at: `http://localhost:5173`

---

### Option 2: Desktop Mode (OpenCV Window — Offline)
Run offline without starting a web browser:
```bash
python main.py --mode desktop --scenario A
```
Press `q` or `ESC` in the OpenCV window to exit.

---

### Option 3: Headless Mode (CLI / Benchmarking)
```bash
python main.py --mode headless --scenario A
```

---

## 4. Camera Sources & Streaming

ORBITA supports seamless switching between camera sources directly from the dashboard:

```text
Camera Source
[ Jetson Camera ]   [ Phone IP Camera ]   [ Simulation ]
```

### 1. Jetson CSI / USB Camera
- Select **Jetson Camera** in the dashboard or launch with `--camera 0`:
  ```bash
  python main.py --mode web --camera 0
  ```

### 2. Phone IP Camera (Wi-Fi Streaming)
Connect any smartphone as a high-definition wireless sensor:
1. Install an IP webcam app on your phone:
   - **Android**: *IP Webcam* (by Pavel Khlebovich) or *DroidCam*.
   - **iOS**: Any RTSP/MJPEG IP streaming app.
2. Ensure phone and Jetson/PC are connected to the **same Wi-Fi network** (or connect your device to your phone's Wi-Fi hotspot).
3. Tap **Start server** in the phone app.
4. In the ORBITA dashboard under **Live Experiment**:
   - Select **Phone IP Camera**.
   - Enter your phone's IP address (e.g. `192.168.1.105`), Port `8080`, and Path `/video`.
   - Click **Connect Camera**.

### 3. Synthetic Simulation
- Select **Simulation** in the dashboard or run with `--scenario A|B|C`.
  - **Scenario A**: Nominal execution (all steps pass sequentially).
  - **Scenario B**: Deviation error (wrong object used).
  - **Scenario C**: Protocol violation (step skipped).

---

## 5. Verification & Testing

Run all automated unit and integration tests:
```bash
pytest tests/ -v
```

Build the production frontend bundle:
```bash
cd frontend
npm run build
```
