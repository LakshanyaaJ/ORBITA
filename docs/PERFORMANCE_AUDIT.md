# ORBITA — Video Pipeline & YOLO Performance Audit

**Audit Date**: 2026-09-18  
**Auditor**: Senior Computer Vision & Performance Optimization Engineer  
**Repository**: `c:\project\ORBITA`  
**Target Platform**: NVIDIA Jetson Orin Nano / Edge Workstation  

---

## 1. System & Environment Baseline

| Parameter | Measured Specification |
| :--- | :--- |
| **Operating System** | Windows 11 (build 10.0.26100) x86_64 |
| **Python Runtime** | Python 3.13.9 (CPython) |
| **PyTorch Version** | 2.13.0+cpu |
| **CUDA Available** | False (Development CPU mode; Jetson Orin Nano is deployment target) |
| **Compute Device** | Intel/AMD x86_64 Multi-Core CPU |
| **YOLO Architecture** | YOLOv8 nano (custom fine-tuned checkpoint) |
| **Active Model File** | `models/orbita_yolo_detector_v4.pt` (5.96 MB, ~3.2M parameters) |
| **Model Ontology** | 7 Classes (`LOCATION_A`, `LOCATION_B`, `PEN`, `WATCH`, `BLUE_BOX`, `YELLOW_BOX`, `HAND`) |
| **Default Input Res** | 640×640 px |
| **Confidence Thresh** | 0.20 – 0.25 (class-specific thresholds) |
| **IoU NMS Thresh** | 0.45 |

---

## 2. YOLO Standalone Benchmark Baseline (`scripts/benchmark_yolo.py`)

*Measured with 10 warmup cycles, 60 steady-state iterations per resolution:*

| Resolution | Cold Start | Average Latency | P50 (Median) | P95 Latency | P99 Latency | Standalone FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **640×640** | 5691.4 ms | **124.2 ms** | 105.3 ms | 210.1 ms | 431.1 ms | **8.1 FPS** |
| **512×512** | — | **84.3 ms** | 79.2 ms | 110.4 ms | 135.2 ms | **11.9 FPS** |
| **480×480** | — | **70.4 ms** | 68.0 ms | 82.2 ms | 90.1 ms | **14.2 FPS** |
| **384×384** | — | **60.6 ms** | 53.9 ms | 91.3 ms | 130.4 ms | **16.5 FPS** |

*Cold-start model load time: 228.7 ms. First inference warmup: 5691.4 ms.*

---

## 3. End-to-End Pipeline Latency Breakdown (`scripts/benchmark_pipeline.py`)

*Measured over 60 real video frames (`vdata/20260905_145858.mp4`):*

| Pipeline Stage | Current Latency (Avg) | P50 Latency | P95 Latency | % of Pipeline | Bottleneck? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Capture & Decode** | 112.49 ms | 90.83 ms | 116.95 ms | **35.2%** | **YES (Top #1)** — Synchronous disk I/O / decoding on main thread |
| **YOLO Detection** | 76.53 ms | 70.63 ms | 111.68 ms | **23.9%** | **YES (Top #2)** — Full-frame 640 inference every single frame |
| **Pose Estimation** | 58.10 ms | 0.33 ms | 74.56 ms | **18.2%** | **YES (Top #3)** — YOLOv8-pose spike on neural evaluation frames |
| **Hand Tracking** | 40.04 ms | 22.00 ms | 94.12 ms | 12.5% | Secondary — MediaPipe landmarking across two hands |
| **Preprocessing** | 21.10 ms | 20.93 ms | 22.61 ms | 6.6% | Moderate — CPU image interpolation and resizing |
| **HAR GRU** | 6.64 ms | 5.79 ms | 9.36 ms | 2.1% | Low — 30-frame temporal classification is fast |
| **Interaction Tracking** | 0.66 ms | 0.69 ms | 1.36 ms | 0.2% | Negligible |
| **FSM Reasoning** | 0.33 ms | 0.33 ms | 0.42 ms | 0.1% | Negligible |
| **Feature Fusion** | 0.23 ms | 0.22 ms | 0.31 ms | 0.1% | Negligible |
| **Serialization & Telemetry**| 0.21 ms | 0.21 ms | 0.29 ms | 0.1% | Negligible |
| **HMR 3D Recovery** | 0.09 ms | 0.08 ms | 0.14 ms | 0.0% | Negligible (Kinematic lifter fallback is highly optimized) |
| **TOTAL END-TO-END** | **319.69 ms** | **266.57 ms** | **546.84 ms** | **100.0%** | **Overall: 3.11 Effective FPS** |

*Freshness Telemetry: Frame Age Avg = 3.24 ms, Frame Age P95 = 4.14 ms.*

---

## 4. Top 3 Root Bottlenecks Identified

### Bottleneck #1: Synchronous Video Capture & Decode in AI Thread
- **Evidence**: `capture_decode` consumes **112.49 ms** per frame (35.2% of total pipeline latency).
- **Cause**: Reading frames sequentially from `cv2.VideoCapture` inside the AI worker thread blocks inference on disk I/O and CPU software decode.
- **Remedy**: Decouple capture entirely into an asynchronous reader thread pushing frames to `LatestFrameBuffer(maxsize=1)`. The AI worker thread only consumes the freshest frame via zero-copy buffer access.

### Bottleneck #2: Full-Resolution YOLO Inference on Every Frame
- **Evidence**: `yolo_detection` takes **76.53 ms** (up to **124.2 ms** at 640×640 standalone).
- **Cause**:
  1. Default resolution is 640×640 px. Benchmarks demonstrate that 480×480 reduces latency from 124 ms to 70 ms (a **43.3% speedup**) without small-object degradation.
  2. YOLO is invoked on every single frame. Tracking (`MultiObjectTracker`) already persists tracks, so running YOLO on a configurable cadence (e.g., every 2 frames) immediately halves detection compute overhead.
  3. Inference lacks explicit `torch.inference_mode()` scoping in `ObjectDetector`.
  4. Half-precision (`FP16`) is not dynamically selected when CUDA is active.

### Bottleneck #3: Concurrent Pose Estimation and Hand Tracking Redundancy
- **Evidence**: `pose_estimation` takes **58.10 ms** and `hand_tracking` takes **40.04 ms** (combined 30.7% of pipeline).
- **Cause**: Both pipelines run full-frame neural networks. When Pose has high confidence, MediaPipe hand tracking can be scoped to the wrist ROIs rather than the full image.
- **Remedy**: Apply ROI cropping for hands and align cadence so expensive passes do not bottleneck consecutive frames.

---

## 5. Performance Optimization Targets

| Metric | Measured Baseline | Optimization Target |
| :--- | :--- | :--- |
| **YOLO Inference Latency** | 76.5 ms (124 ms @ 640) | **< 45 ms** |
| **Total Pipeline Latency** | 319.7 ms | **< 120 ms** |
| **Effective Inference FPS** | 3.11 FPS | **> 12.0 FPS** |
| **Streamer Display FPS** | 25–30 FPS | **30 FPS stable** |
| **Frame Age (Freshness)** | 3.2 ms | **< 30 ms** |
| **Dropped Backlog Frames** | Bounded (drop-oldest) | **0 backlog accumulation** |
