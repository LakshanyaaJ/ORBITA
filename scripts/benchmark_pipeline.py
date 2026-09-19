"""
ORBITA End-to-End Pipeline Latency Benchmark Script (Section 41)
==============================================================
Profiles the entire video-to-result pipeline:
Video -> Capture -> Decode -> Preprocessing -> YOLO -> Pose -> HMR -> Hand Tracking
      -> Interactions -> Temporal Feature Fusion -> GRU HAR -> FSM -> Telemetry.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from core_ai.app.config import load_config
from core_ai.har.feature_fusion import TemporalFeatureWindow, build_feature_vector
from core_ai.har.temporal_model import ActionClassifier
from core_ai.interaction.hand_object import HandObjectInteractionTracker
from core_ai.perception.hand_tracker import HandTracker
from core_ai.perception.hmr_pipeline import HMRPipeline
from core_ai.perception.object_detector import ObjectDetector
from core_ai.perception.pose_estimator import PoseEstimator
from core_ai.reasoning.state_manager import StateManager
from core_ai.video.frame_buffer import LatestFrameBuffer


def benchmark_pipeline(
    video_path: str = "vdata/20260905_145858.mp4",
    num_frames: int = 120,
    target_imgsz: int = 480,
):
    print("=" * 72)
    print("ORBITA END-TO-END PIPELINE BENCHMARK (Section 41)")
    print("=" * 72)
    print(f"Video Source: {video_path}")
    print(f"Frames to Test: {num_frames}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Try finding any mp4 in vdata
        vdata_files = list(Path("vdata").glob("*.mp4"))
        if vdata_files:
            video_path = str(vdata_files[0])
            cap = cv2.VideoCapture(video_path)
            print(f"Fallback to available video: {video_path}")
        else:
            print("Error: No video file available in vdata/ directory.")
            return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"Video FPS: {video_fps:.1f} | Total Frames: {total_vid_frames}")

    # Load complete pipeline components
    cfg = load_config()
    cfg.detection.yolo_imgsz = target_imgsz

    pose_estimator = PoseEstimator(cfg.pose)
    hand_tracker = HandTracker(cfg.hand)
    detector = ObjectDetector(cfg.detection)
    hmr = HMRPipeline()
    interaction_tracker = HandObjectInteractionTracker(cfg.interaction)
    feature_window = TemporalFeatureWindow(cfg.har.window_frames)
    classifier = ActionClassifier(cfg.har)
    state_manager = StateManager(cfg)
    frame_buf = LatestFrameBuffer(name="bench-buf", maxsize=1)

    # Warmup all components once
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _ = pose_estimator.estimate(dummy_frame)
    _ = hand_tracker.track(None, dummy_frame)
    _ = detector.detect(dummy_frame)
    _ = hmr.process(dummy_frame)
    _ = classifier.predict(np.zeros((30, 64), dtype=np.float32))

    stage_timings: dict[str, list[float]] = {
        "capture_decode": [],
        "preprocessing": [],
        "pose_estimation": [],
        "hand_tracking": [],
        "yolo_detection": [],
        "hmr_mesh": [],
        "interaction_tracking": [],
        "feature_fusion": [],
        "har_gru": [],
        "fsm_reasoning": [],
        "serialization": [],
        "total_pipeline": [],
    }

    frame_ages: list[float] = []
    frame_drops: int = 0
    processed_count = 0
    start_bench_time = time.perf_counter()

    for i in range(num_frames):
        t_cycle_start = time.perf_counter()

        # 1. Capture & Decode
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            # Loop back for full benchmark count
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
        t_cap = (time.perf_counter() - t0) * 1000.0

        # Push to LatestFrameBuffer to measure frame age & drop behavior
        frame_buf.push(frame, timestamp=time.monotonic())
        latest_frame, cap_ts, f_id = frame_buf.get_latest()
        frame_age_ms = (time.monotonic() - cap_ts) * 1000.0
        frame_ages.append(frame_age_ms)

        # 2. Preprocessing / Resize
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        target_w = 960
        if max(h, w) > target_w:
            scale = target_w / max(h, w)
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame
        t_stamp = time.time()
        t_prep = (time.perf_counter() - t0) * 1000.0

        # 3. Pose Estimation
        t0 = time.perf_counter()
        poses = pose_estimator.estimate(frame_resized, t_stamp)
        pose = poses[0] if poses else None
        person_bbox = pose.bbox if pose else None
        t_pose = (time.perf_counter() - t0) * 1000.0

        # 4. Hand Tracking
        t0 = time.perf_counter()
        left, right = hand_tracker.track(pose, frame_resized, t_stamp, person_bbox=person_bbox)
        t_hand = (time.perf_counter() - t0) * 1000.0

        # 5. YOLO Detection + Tracking
        t0 = time.perf_counter()
        objects = detector.detect(frame_resized, t_stamp, hands=(left, right), frame_id=i)
        t_yolo = (time.perf_counter() - t0) * 1000.0

        # 6. HMR 3D Mesh Recovery
        t0 = time.perf_counter()
        hmr_result = hmr.process(frame_resized, person_bbox, keypoints_2d=pose.keypoints_px if pose else None)
        t_hmr = (time.perf_counter() - t0) * 1000.0

        # 7. Interaction Tracking
        t0 = time.perf_counter()
        interactions = interaction_tracker.update(left, right, objects, t_stamp)
        t_inter = (time.perf_counter() - t0) * 1000.0

        # 8. Feature Fusion
        t0 = time.perf_counter()
        fv = build_feature_vector(pose, left, right, objects, interactions, frame_resized.shape[:2])
        feature_window.push(fv)
        window = feature_window.get_window()
        t_feat = (time.perf_counter() - t0) * 1000.0

        # 9. HAR GRU Prediction
        t0 = time.perf_counter()
        prediction = classifier.predict(window)
        t_gru = (time.perf_counter() - t0) * 1000.0

        # 10. FSM Reasoning
        t0 = time.perf_counter()
        result = state_manager.process(
            prediction=prediction,
            detected_objects=objects,
            fps=video_fps,
            latency_ms=(time.perf_counter() - t_cycle_start) * 1000.0,
            left_hand=left,
            right_hand=right,
            interactions=interactions,
            frame_index=i,
        )
        t_fsm = (time.perf_counter() - t0) * 1000.0

        # 11. Serialization / Telemetry
        t0 = time.perf_counter()
        _ = result.to_dict() if result else {}
        t_serial = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_cycle_start) * 1000.0

        stage_timings["capture_decode"].append(t_cap)
        stage_timings["preprocessing"].append(t_prep)
        stage_timings["pose_estimation"].append(t_pose)
        stage_timings["hand_tracking"].append(t_hand)
        stage_timings["yolo_detection"].append(t_yolo)
        stage_timings["hmr_mesh"].append(t_hmr)
        stage_timings["interaction_tracking"].append(t_inter)
        stage_timings["feature_fusion"].append(t_feat)
        stage_timings["har_gru"].append(t_gru)
        stage_timings["fsm_reasoning"].append(t_fsm)
        stage_timings["serialization"].append(t_serial)
        stage_timings["total_pipeline"].append(t_total)

        processed_count += 1

    cap.release()
    total_elapsed_sec = time.perf_counter() - start_bench_time
    effective_fps = processed_count / max(0.001, total_elapsed_sec)

    print("\n" + "=" * 72)
    print(f"{'STAGE':<26} {'AVG (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'% TOTAL':<10}")
    print("-" * 72)

    total_mean = float(np.mean(stage_timings["total_pipeline"]))

    for stage, vals in stage_timings.items():
        if stage == "total_pipeline":
            continue
        s_avg = float(np.mean(vals))
        s_p50 = float(np.percentile(vals, 50))
        s_p95 = float(np.percentile(vals, 95))
        pct = (s_avg / total_mean) * 100.0 if total_mean > 0 else 0.0
        print(f"{stage:<26} {s_avg:8.2f}    {s_p50:8.2f}    {s_p95:8.2f}    {pct:6.1f}%")

    print("-" * 72)
    tot_avg = total_mean
    tot_p50 = float(np.percentile(stage_timings["total_pipeline"], 50))
    tot_p95 = float(np.percentile(stage_timings["total_pipeline"], 95))
    print(f"{'TOTAL END-TO-END':<26} {tot_avg:8.2f}    {tot_p50:8.2f}    {tot_p95:8.2f}    100.0%")
    print("=" * 72)

    avg_age = float(np.mean(frame_ages)) if frame_ages else 0.0
    p95_age = float(np.percentile(frame_ages, 95)) if frame_ages else 0.0

    print(f"Effective Inference Throughput: {effective_fps:.2f} FPS")
    print(f"Frame Age (Freshness):          Avg: {avg_age:.2f} ms | P95: {p95_age:.2f} ms")
    print(f"Dropped Frames in Buffer:       {frame_buf._frames_dropped} frames")
    print("=" * 72)

    return {
        "timings": stage_timings,
        "effective_fps": effective_fps,
        "total_mean": total_mean,
        "p95_latency": tot_p95,
        "frame_age_avg": avg_age,
        "frame_age_p95": p95_age,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORBITA End-to-End Pipeline Benchmark")
    parser.add_argument("--video", type=str, default="vdata/20260905_145858.mp4")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=480)
    args = parser.parse_args()

    benchmark_pipeline(
        video_path=args.video,
        num_frames=args.frames,
        target_imgsz=args.imgsz,
    )
