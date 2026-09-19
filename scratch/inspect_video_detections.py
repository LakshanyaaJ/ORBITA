import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("models/orbita_yolo_detector_v4.pt")
cap = cv2.VideoCapture("vdata/20260905_145858.mp4")
total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total video frames: {total_f}")

for f_idx in [50, 150, 250, 350, 450, 550, 650]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    h, w = frame.shape[:2]
    # Predict at 640 and at 480
    res = model.predict(frame, conf=0.15, imgsz=640, verbose=False)
    dets = []
    for r in res:
        for b in r.boxes:
            cid = int(b.cls[0])
            cname = r.names[cid]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            dets.append(f"{cname}:{conf:.2f}({x2-x1}x{y2-y1})")
    print(f"Frame {f_idx:04d}: {dets}")

cap.release()
