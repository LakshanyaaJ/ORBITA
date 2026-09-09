import cv2
import numpy as np
from ultralytics import YOLO

# Let's test the YOLO models on frame 1814 of 20260908_135006.mp4
cap = cv2.VideoCapture("vdata/20260908_135006.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 1814)
ret, frame = cap.read()

# Let's crop around the white paper with the pen:
# In 1080x1920, the paper is around x: 200..700, y: 600..1100
paper_crop = frame[600:1100, 200:700]
print("Paper crop shape:", paper_crop.shape)

for m_path in ["models/orbita_yolo_detector_v3.pt", "models/versions/orbita_yolo_detector_v1/best.pt", "models/versions/orbita_yolo_detector_v2/best.pt"]:
    m = YOLO(m_path)
    res = m.predict(paper_crop, conf=0.05, verbose=False)
    for r in res:
        for b in r.boxes:
            print(f"{m_path} -> {m.names[int(b.cls[0])]}: {float(b.conf[0]):.2f}")

cap.release()
