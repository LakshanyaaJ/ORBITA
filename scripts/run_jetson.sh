#!/usr/bin/env bash
set -e

echo "=== Starting ORBITA on NVIDIA Jetson Orin Nano ==="

# Set Jetson clocks to maximum performance mode if privileged
if command -v nvpmodel &> /dev/null; then
    echo "[*] Setting MAXN power mode..."
    sudo nvpmodel -m 0 || true
fi

if command -v jetson_clocks &> /dev/null; then
    echo "[*] Locking clock rates..."
    sudo jetson_clocks || true
fi

# Ensure models directory has weights
python3 scripts/download_models.py

# Build frontend if dist does not exist
if [ ! -d "frontend/dist" ]; then
    echo "[*] Building frontend dashboard..."
    cd frontend && npm install && npm run build && cd ..
fi

# Start FastAPI backend server with Jetson CSI/USB camera support
echo "[*] Launching ORBITA Server..."
python3 main.py --mode web --camera 0
