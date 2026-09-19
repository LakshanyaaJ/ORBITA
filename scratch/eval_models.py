from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

for mpath in ["models/orbita_yolo_detector_v4.pt", "models/orbita_yolo_detector_v3.pt"]:
    p = Path(mpath)
    if not p.exists():
        continue
    print(f"==================================================")
    print(f"EVALUATING MODEL: {mpath}")
    print(f"==================================================")
    model = YOLO(str(p))
    print(f"Classes: {model.names}")
    
    # Run validation on val split
    try:
        metrics = model.val(data="datasets/orbita/data.yaml", imgsz=640, conf=0.10, verbose=False)
        print("\nValidation Metrics (conf=0.10, imgsz=640):")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50:    {metrics.box.map50:.4f}")
        for i, cname in model.names.items():
            if i < len(metrics.box.maps):
                p_c = metrics.box.p[i] if hasattr(metrics.box, 'p') and i < len(metrics.box.p) else 0.0
                r_c = metrics.box.r[i] if hasattr(metrics.box, 'r') and i < len(metrics.box.r) else 0.0
                map50_c = metrics.box.maps[i]
                print(f"  {cname:<12}: Precision={p_c:.3f} | Recall={r_c:.3f} | mAP50={map50_c:.3f}")
    except Exception as e:
        print(f"Validation error: {e}")
