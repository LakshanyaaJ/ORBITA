import cv2
import numpy as np

def inspect_video_events(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"\n==========================================")
    print(f"VIDEO: {path}")
    print(f"Total Frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
    print(f"==========================================")
    
    # Sample every 1 sec (30 frames)
    for sec in range(0, int(duration) + 1):
        f_idx = int(sec * fps)
        if f_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        # Print frame metric
        mean_val = np.mean(frame)
        print(f"Time {sec:02d}s (Frame {f_idx:04d}): Frame loaded OK (mean brightness: {mean_val:.1f})")
        
    cap.release()

inspect_video_events("vdata/20260905_145858.mp4")
inspect_video_events("vdata/20260908_135006.mp4")
