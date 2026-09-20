import time
import os
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def evaluate_model(model_path: str, data_yaml: str = "datasets/orbita/v3/data.yaml"):
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(data=data_yaml, split="val", verbose=False)
    
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    
    # Measure inference latency and FPS over 50 dummy passes
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Warmup
    for _ in range(5):
        model(dummy_img, verbose=False)
        
    start_t = time.perf_counter()
    n_frames = 50
    for _ in range(n_frames):
        model(dummy_img, verbose=False)
    total_time = time.perf_counter() - start_t
    
    avg_latency_ms = (total_time / n_frames) * 1000.0
    fps = n_frames / total_time
    
    per_class = {}
    if hasattr(metrics.box, 'ap_class_index') and len(metrics.box.ap_class_index) > 0:
        names = model.names
        for idx, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = names.get(cls_idx, f"class_{cls_idx}")
            cls_ap50 = float(metrics.box.ap50[idx]) if idx < len(metrics.box.ap50) else 0.0
            per_class[cls_name] = cls_ap50

    return {
        "model_path": model_path,
        "mAP50": round(map50, 4),
        "mAP50-95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fps": round(fps, 1),
        "latency_ms": round(avg_latency_ms, 2),
        "model_size_mb": round(model_size_mb, 2),
        "per_class_ap50": per_class
    }

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "models/orbita_yolov8_backup.pt"
    res = evaluate_model(path)
    print(json.dumps(res, indent=2))
