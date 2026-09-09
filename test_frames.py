import cv2
from ultralytics import YOLO

model = YOLO("models/orbita_yolo_detector_v3.pt")
cap = cv2.VideoCapture("vdata/20260908_135006.mp4")

# Test frames around 1700-1830 in steps of 20
for f_idx in range(1600, 1830, 20):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    res = model.predict(frame, conf=0.05, verbose=False)
    dets = []
    for r in res:
        for b in r.boxes:
            cid = int(b.cls[0])
            cname = r.names[cid]
            conf = float(b.conf[0])
            dets.append(f"{cname}:{conf:.2f}")
    if dets:
        print(f"Frame {f_idx}: {dets}")

cap.release()
