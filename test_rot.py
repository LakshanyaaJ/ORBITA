import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("models/orbita_yolo_detector_v3.pt")
cap = cv2.VideoCapture("vdata/20260908_135006.mp4")

# Check frame 100 with different rotations and imgsz
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, raw_frame = cap.read()
print("Raw frame shape:", raw_frame.shape)

for rot_deg in [0, 90, 180, 270]:
    if rot_deg == 0:
        f = raw_frame.copy()
    elif rot_deg == 90:
        f = cv2.rotate(raw_frame, cv2.ROTATE_90_CLOCKWISE)
    elif rot_deg == 180:
        f = cv2.rotate(raw_frame, cv2.ROTATE_180)
    elif rot_deg == 270:
        f = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # test with resize target_w = 960 (like in Camera._preprocess_frame)
    fh, fw = f.shape[:2]
    if fw > 960:
        th = int(fh * (960 / fw))
        f_resized = cv2.resize(f, (960, th))
    else:
        f_resized = f

    res = model.predict(f_resized, conf=0.10, verbose=False)
    dets = []
    for r in res:
        for b in r.boxes:
            cid = int(b.cls[0])
            cname = r.names[cid]
            conf = float(b.conf[0])
            dets.append(f"{cname}:{conf:.2f}")
    print(f"Rot {rot_deg}°: shape={f_resized.shape} -> {dets}")

cap.release()
