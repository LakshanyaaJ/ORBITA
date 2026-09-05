# NVIDIA Jetson Orin Nano Deployment Guide

This guide details the procedure for deploying and running ORBITA on an **NVIDIA Jetson Orin Nano Developer Kit** (or Orin NX / AGX Orin).

---

## 1. Prerequisites

- **Hardware**: NVIDIA Jetson Orin Nano (8GB or 4GB)
- **OS**: JetPack 5.1.x or JetPack 6.x (Ubuntu 20.04 / 22.04 LTS)
- **Python**: Python 3.8+ with PyTorch built for Jetson (`torch-*.whl` from NVIDIA)
- **Node.js**: Node 18+ and npm (for building the React frontend)
- **Camera**:
  - Raspberry Pi Camera Module v2 / HQ Camera (via MIPI-CSI)
  - USB UVC Webcam (e.g. Logitech C920)
  - Smartphone on same Wi-Fi network running IP Webcam app

---

## 2. Environment Setup

### 2.1 Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libopenblas-base libopenmpi-dev \
    libjpeg-dev zlib1g-dev libgl1-mesa-glx gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad espeak ffmpeg
```

### 2.2 Install Python Requirements

```bash
pip3 install -r requirements.txt
```

### 2.3 Download or Verify Models

Run the built-in model verification script:

```bash
python3 scripts/download_models.py
```

---

## 3. Camera Configuration on Jetson

### 3.1 MIPI-CSI Camera (GStreamer Pipeline)

ORBITA supports GStreamer pipelines natively. In `config/orbita_config.json` or via CLI:

```bash
python3 main.py --mode web --camera "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

### 3.2 USB V4L2 Webcam

Connect your USB camera to a USB 3.0 port:

```bash
python3 main.py --mode web --camera 0
```

### 3.3 Phone IP Camera

Start the IP Webcam app on your smartphone (connected to the same Wi-Fi as Jetson):
1. In the ORBITA web dashboard (`http://<jetson-ip>:8000`), open the **Camera Control Panel**.
2. Select **Phone IP Stream**.
3. Enter the phone IP (e.g., `192.168.1.50`) and port (`8080`).
4. Click **Connect**.

---

## 4. Hardware Optimization

To maximize FPS on Jetson Orin Nano:

```bash
# Set maximum performance power mode
sudo nvpmodel -m 0

# Lock clock rates at maximum frequency
sudo jetson_clocks
```

Alternatively, use the convenience script:

```bash
bash scripts/run_jetson.sh
```

---

## 5. Running as a Systemd Service

To automatically launch ORBITA upon Jetson boot:

Create `/etc/systemd/system/orbita.service`:

```ini
[Unit]
Description=ORBITA Edge AI Mission Assistant
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/ORBITA
ExecStart=/usr/bin/python3 main.py --mode web --camera 0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable orbita
sudo systemctl start orbita
```
