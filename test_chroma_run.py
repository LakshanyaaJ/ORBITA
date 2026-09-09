import cv2
import numpy as np
from core_ai.app.config import load_config
from core_ai.perception.object_detector import ObjectDetector

cfg = load_config()
cfg.detection.allow_chroma_fallback = True
detector = ObjectDetector(cfg.detection)

cap = cv2.VideoCapture("vdata/20260908_135006.mp4")
for f_idx in [100, 300, 600, 900, 1200, 1500, 1814]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: continue
    objs = detector.detect(frame, f_idx / 30.0)
    grouped = detector.get_grouped_detections()
    summary = {k: len(v) for k, v in grouped.items()}
    details = {k: [(d['confidence'], d.get('bbox', [d.get('x1'), d.get('y1'), d.get('x2')-d.get('x1'), d.get('y2')-d.get('y1')])) for d in v] for k, v in grouped.items() if v}
    print(f"F{f_idx}: summary={summary} details={details}")
cap.release()
