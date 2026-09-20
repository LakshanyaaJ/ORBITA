import cv2
import sys
import glob

def analyze_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"=== {path} ===")
    print(f"FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration:.2f}s")
    cap.release()

for f in sorted(glob.glob("vdata/*.mp4")):
    analyze_video(f)
