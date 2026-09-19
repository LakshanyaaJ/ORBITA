import cv2
from pathlib import Path

for vp in Path("vdata").glob("*.mp4"):
    cap = cv2.VideoCapture(str(vp))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{vp.name:<25}: {w}x{h} | {fps:.1f} fps | {count} frames | Aspect: {w/h:.3f}")
    cap.release()
