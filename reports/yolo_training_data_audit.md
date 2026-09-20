# ORBITA YOLO Training Data & Domain Shift Audit Report

## 1. Training Pipeline Overview

* **Training Script**: [scripts/train_yolo_7class.py](file:///e:/VS_CODE/Orbita/ORBITA/scripts/train_yolo_7class.py)
* **Dataset Generator**: [scripts/build_dataset_7class.py](file:///e:/VS_CODE/Orbita/ORBITA/scripts/build_dataset_7class.py)
* **Base Pretrained Weights**: `models/yolov8n.pt` (Ultralytics YOLOv8 Nano)
* **Output Production Weights**: [models/orbita_yolo_detector_v4.pt](file:///e:/VS_CODE/Orbita/ORBITA/models/orbita_yolo_detector_v4.pt)

---

## 2. Dataset Generation Analysis

Inspection of `build_dataset_7class.py` reveals the exact origin of the dataset used to train `orbita_yolo_detector_v4.pt`:

```python
# Video source used to build datasets/orbita/v4/
cap = cv2.VideoCapture("vdata/20260908_135006.mp4")
sampled_frame_indices = list(range(10, total_frames - 10, 15)) # ~120 frames
```

### Critical Findings:
1. **Single Video Source**: `orbita_yolo_detector_v4.pt` was trained on keyframes extracted **exclusively from one video file**: `vdata/20260908_135006.mp4`.
2. **Dataset Size**: ~120 frames total.
3. **Validation Split Leakage**: `build_dataset_7class.py` split frames from the SAME single video randomly into train/val splits (`train`: 80%, `val`: 20%). This caused the training evaluation to report misleadingly high validation metrics ($\text{mAP50} \approx 0.95$) on frames from that single video.
4. **Domain Shift on Test Video**: When `orbita_yolo_detector_v4.pt` is evaluated on `vdata/20260905_145858.mp4` (a different video recorded 3 days prior with different camera framing, lighting, and object placement), the detector fails to generalize, yielding **0% detection coverage for experiment items** while detecting hands and pose keypoints via MediaPipe/YOLO-pose.

---

## 3. Training Augmentation & Configuration

```python
train_args = dict(
    data="datasets/orbita/v4/data.yaml",
    epochs=10,
    batch=8,
    imgsz=640,
    device="cpu",
    degrees=5.0,
    translate=0.08,
    scale=0.10,
    fliplr=0.0,    # Don't flip horizontally to keep Location A/B semantics aligned
    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.3,
)
```

- **Epoch Count**: 10 epochs (extremely low for zero-shot generalization across scenes).
- **Class Balance**: High presence of `HAND` and `LOCATION_A`/`LOCATION_B` labels; sparse annotations for small objects like `PEN` and `WATCH`.

---

## 4. Recommendations for Production Fine-Tuning

1. **Multi-Video Sampling**: Sample keyframes across **all 4 available video recordings** in `vdata/`:
   - `20260905_145858.mp4`
   - `20260905_149948.mp4`
   - `20260905_150132.mp4`
   - `20260908_135006.mp4`
2. **Video-Separated Train/Val Split**: Put `20260905_145858.mp4` and `20260908_135006.mp4` into `train`, and `20260905_149948.mp4` into `val` to guarantee strict out-of-distribution validation without leakage.
3. **Epoch Increase**: Increase fine-tuning to 50–100 epochs with cosine learning rate schedule.
