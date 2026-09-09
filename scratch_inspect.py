import cv2
from ultralytics import YOLO

model = YOLO("models/orbita_yolo_detector_v3.pt")
print("MODEL NAMES:", model.names)

cap = cv2.VideoCapture("vdata/20260908_135006.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", total_frames)

for f_idx in [0, 10, 30, 100, 500, 1000, 1500, 1814]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {f_idx}: Read failed")
        continue
    results = model.predict(frame, conf=0.15, verbose=False)
    boxes_info = []
    for r in results:
        for b in r.boxes:
            cls_id = int(b.cls[0])
            name = r.names.get(cls_id, str(cls_id))
            conf = float(b.conf[0])
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            boxes_info.append(f"{name} ({conf:.2f}) {xyxy}")
    print(f"Frame {f_idx}: {len(boxes_info)} detections -> {boxes_info}")

cap.release()
