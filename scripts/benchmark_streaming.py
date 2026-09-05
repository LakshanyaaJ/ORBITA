"""
ORBITA Streaming Pipeline Benchmark
====================================
Measures latency, stream FPS, and dropped frames under simulated AI workloads.
Validates that AI inference latency does NOT degrade streaming throughput.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import threading
import numpy as np

from core_ai.video.frame_buffer import LatestFrameBuffer
from core_ai.video.streamer import MJPEGStreamer


def run_benchmark(duration_sec: float = 3.0, ai_delay_sec: float = 0.12):
    """
    Simulates:
      - Camera capture thread pushing 720p frames at 30 FPS.
      - Decoupled Streamer producing MJPEG at camera frame rate.
      - Independent AI worker reading latest frame and processing with simulated delay.
    """
    buffer = LatestFrameBuffer("bench-buffer")
    streamer = MJPEGStreamer(jpeg_quality=70)
    stop_event = threading.Event()

    camera_pushed = 0
    stream_frames = 0
    ai_frames = 0
    latencies = []

    # 1. Camera capture worker (30 FPS)
    def _camera_worker():
        nonlocal camera_pushed
        target_interval = 1.0 / 30.0
        while not stop_event.is_set():
            t0 = time.time()
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            buffer.push(frame, t0)
            camera_pushed += 1
            elapsed = time.time() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # 2. Decoupled Streamer worker (30 FPS target)
    def _stream_worker():
        nonlocal stream_frames
        target_interval = 1.0 / 30.0
        while not stop_event.is_set():
            t0 = time.time()
            frame, cap_ts, _ = buffer.get_latest()
            if frame is not None:
                streamer.update(frame, timestamp=cap_ts)
                stream_frames += 1
                latencies.append((time.time() - cap_ts) * 1000.0)

            elapsed = time.time() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # 3. Independent AI Worker (simulating 120ms heavy inference)
    def _ai_worker():
        nonlocal ai_frames
        while not stop_event.is_set():
            frame, cap_ts, _ = buffer.get_latest()
            if frame is not None:
                # Simulate heavy YOLO + Pose + Hands inference
                time.sleep(ai_delay_sec)
                ai_frames += 1
            else:
                time.sleep(0.01)

    t_cam = threading.Thread(target=_camera_worker, daemon=True)
    t_str = threading.Thread(target=_stream_worker, daemon=True)
    t_ai = threading.Thread(target=_ai_worker, daemon=True)

    start_time = time.time()
    t_cam.start()
    t_str.start()
    t_ai.start()

    time.sleep(duration_sec)
    stop_event.set()

    t_cam.join(timeout=1.0)
    t_str.join(timeout=1.0)
    t_ai.join(timeout=1.0)

    total_time = time.time() - start_time
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    cam_fps = camera_pushed / total_time
    str_fps = stream_frames / total_time
    ai_fps = ai_frames / total_time
    dropped = buffer.stats["dropped_pct"]

    return {
        "duration": round(total_time, 2),
        "camera_fps": round(cam_fps, 1),
        "stream_fps": round(str_fps, 1),
        "ai_fps": round(ai_fps, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "dropped_pct": dropped,
        "encode_latency_ms": streamer.encode_latency_ms,
    }


def run_old_pipeline_simulation(duration_sec: float = 3.0, ai_delay_sec: float = 0.12):
    """
    Simulates the OLD coupled architecture where camera frame is read,
    then AI inference runs synchronously, and only then is the frame encoded and streamed.
    """
    stop_event = threading.Event()
    frames_processed = 0
    latencies = []

    start_time = time.time()
    while time.time() - start_time < duration_sec:
        t_capture = time.time()
        # Camera capture + AI inference synchronously
        time.sleep(ai_delay_sec)
        # Fast JPEG encode
        encode_start = time.time()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames_processed += 1
        latencies.append((time.time() - t_capture) * 1000.0)

    total_time = time.time() - start_time
    return {
        "stream_fps": round(frames_processed / total_time, 1),
        "avg_latency_ms": round(float(np.mean(latencies)), 1) if latencies else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0,
    }


if __name__ == "__main__":
    import cv2
    print("=" * 60)
    print(" ORBITA Streaming Pipeline Latency Benchmark (Old vs New)")
    print("=" * 60)
    print("\n--- 1. OLD PIPELINE (Synchronous AI -> Stream) ---")
    old_res = run_old_pipeline_simulation(duration_sec=2.5, ai_delay_sec=0.12)
    print(f"OLD STREAM FPS:     {old_res['stream_fps']} FPS")
    print(f"OLD AVG LATENCY:    {old_res['avg_latency_ms']} ms")
    print(f"OLD P95 LATENCY:    {old_res['p95_latency_ms']} ms")

    print("\n--- 2. NEW DECOUPLED PIPELINE (LatestFrameBuffer + Independent AI) ---")
    new_res = run_benchmark(duration_sec=2.5, ai_delay_sec=0.12)
    print(f"NEW CAMERA FPS:     {new_res['camera_fps']} FPS")
    print(f"NEW STREAM FPS:     {new_res['stream_fps']} FPS")
    print(f"NEW AI FPS:         {new_res['ai_fps']} FPS")
    print(f"NEW AVG LATENCY:    {new_res['avg_latency_ms']} ms")
    print(f"NEW P95 LATENCY:    {new_res['p95_latency_ms']} ms")
    print(f"ENCODE LATENCY:     {new_res['encode_latency_ms']} ms")
    print(f"DROPPED STALE %:    {new_res['dropped_pct']}%")
    print("=" * 60)

