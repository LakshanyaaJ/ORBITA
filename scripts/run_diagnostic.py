"""
ORBITA Diagnostic Mode Runner
=============================
Runs real frame-by-frame diagnostic profiling on test video or camera input.
Outputs complete pipeline diagnostics: YOLO detections, Pose, Hands, 64D Feature Contract,
GRU Action Prediction + Source, FSM step status, and per-stage latency breakdown.

Usage:
  python scripts/run_diagnostic.py --video vdata/20260905_145858.mp4 --frames 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_ai.app.config import load_config
from core_ai.perception.object_detector import ObjectDetector
from core_ai.perception.pose_estimator import PoseEstimator
from core_ai.perception.hand_tracker import HandTracker
from core_ai.interaction.hand_object import HandObjectInteractionTracker
from core_ai.har.feature_fusion import TemporalFeatureWindow, build_feature_vector
from core_ai.har.temporal_model import ActionClassifier
from core_ai.reasoning.state_manager import StateManager


def print_startup_report(cfg, detector, pose_est, hand_tracker, classifier):
    print("=================================================================")
    print("  ORBITA SYSTEM READINESS & MODEL SELF-CHECK REPORT")
    print("=================================================================")
    print(f"  Project Root:        {cfg.camera.source}")
    print(f"  Detector Checkpoint: {getattr(detector, 'active_model_path', 'N/A')}")
    print(f"  Detector Classes:    {cfg.detection.classes}")
    print(f"  Pose Backend:        {pose_est.backend} (yolov8n-pose.pt)")
    print(f"  Hand Landmarker:     {'LOADED' if hand_tracker._mp_landmarker or hand_tracker._mp_legacy_hands else 'NOT LOADED'}")
    print(f"  HAR Checkpoint:      {classifier.active_checkpoint_path}")
    print(f"  HAR Model Loaded:    {classifier._model is not None} (Source: {'PyTorch GRU' if classifier._model else 'Heuristic Rules'})")
    print(f"  FSM Total Steps:     {len(cfg.experiment_steps)}")
    print("=================================================================\n")


def run_diagnostics(video_path: str, max_frames: int = 30):
    cfg = load_config()
    detector = ObjectDetector(cfg.detection)
    pose_est = PoseEstimator(cfg.pose)
    hand_tracker = HandTracker(cfg.hand)
    interaction_tracker = HandObjectInteractionTracker(cfg.interaction)
    feature_window = TemporalFeatureWindow(cfg.har.window_frames)
    classifier = ActionClassifier(cfg.har)
    state_manager = StateManager(cfg)

    print_startup_report(cfg, detector, pose_est, hand_tracker, classifier)

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    print(f"--- RUNNING DIAGNOSTIC TRACE ON {video_path} ({max_frames} frames) ---\n")

    frame_idx = 0
    total_t0 = time.time()
    stage_latencies = {"yolo": [], "pose": [], "hands": [], "features": [], "gru": [], "fsm": [], "total": []}

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_stamp = time.time()
        t_frame_start = time.perf_counter()

        # 1. Pose
        t0 = time.perf_counter()
        poses = pose_est.estimate(frame, t_stamp)
        pose = poses[0] if poses else None
        dt_pose = (time.perf_counter() - t0) * 1000.0

        # 2. Hands
        t0 = time.perf_counter()
        left, right = hand_tracker.track(pose, frame, t_stamp)
        dt_hands = (time.perf_counter() - t0) * 1000.0

        # 3. YOLO Object Detector
        t0 = time.perf_counter()
        objects = detector.detect(frame, t_stamp, hands=(left, right), frame_id=frame_idx)
        dt_yolo = (time.perf_counter() - t0) * 1000.0

        # 4. Interaction Tracker
        interactions = interaction_tracker.update(left, right, objects, t_stamp)

        # 5. Feature Fusion & 64-D Contract Validation
        t0 = time.perf_counter()
        fv = build_feature_vector(pose, left, right, objects, interactions, frame.shape[:2])
        assert fv.shape == (64,), f"Feature vector shape mismatch: {fv.shape}"
        assert np.isfinite(fv).all(), "Feature vector contains non-finite values (NaN/Inf)"
        feature_window.push(fv)
        win = feature_window.get_window()
        assert win.shape == (30, 64), f"Feature window shape mismatch: {win.shape}"
        assert np.isfinite(win).all(), "Feature window contains non-finite values (NaN/Inf)"
        dt_feat = (time.perf_counter() - t0) * 1000.0

        # 6. GRU Action Recognition
        t0 = time.perf_counter()
        prediction = classifier.predict(win)
        dt_gru = (time.perf_counter() - t0) * 1000.0

        # 7. FSM Procedure Validation
        t0 = time.perf_counter()
        res = state_manager.process(prediction, objects, fps=15.0)
        dt_fsm = (time.perf_counter() - t0) * 1000.0

        dt_total = (time.perf_counter() - t_frame_start) * 1000.0

        # Record latencies (skip initial warmup frames 1-3 for steady-state statistics)
        if frame_idx > 3:
            stage_latencies["pose"].append(dt_pose)
            stage_latencies["hands"].append(dt_hands)
            stage_latencies["yolo"].append(dt_yolo)
            stage_latencies["features"].append(dt_feat)
            stage_latencies["gru"].append(dt_gru)
            stage_latencies["fsm"].append(dt_fsm)
            stage_latencies["total"].append(dt_total)

        step_label = res.fsm_state.current_step.label if res.fsm_state.current_step else "N/A"
        obj_names = [o.class_name for o in objects]

        print(f"Frame {frame_idx:05d} | Latency: {dt_total:5.1f}ms (YOLO:{dt_yolo:4.1f}ms Pose:{dt_pose:4.1f}ms Hands:{dt_hands:4.1f}ms GRU:{dt_gru:4.1f}ms)")
        print(f"  YOLO Detections ({len(objects)}): {obj_names}")
        print(f"  Pose Keypoints:  {'YES (17 joints)' if pose else 'NO'}")
        print(f"  Hands Tracked:   Left={'YES' if left and left.is_visible else 'NO'} | Right={'YES' if right and right.is_visible else 'NO'}")
        print(f"  64D Contract:    VALID (finite=True, shape=(30,64))")
        print(f"  Action Pred:     {prediction.action} (conf={prediction.confidence:.2f}, source={prediction.source}, valid={prediction.model_valid})")
        print(f"  Decision Reason: {prediction.decision_reason}")
        print(f"  FSM Step:        [{res.fsm_state.current_step_idx}] {step_label} | Status: {res.fsm_state.status.name}")
        print("-" * 65)

    cap.release()
    total_elapsed = time.time() - total_t0

    print("\n=================================================================")
    print("  PERFORMANCE LATENCY BENCHMARK PROFILE")
    print("=================================================================")
    print(f"  Frames Processed:   {frame_idx}")
    print(f"  Total Time Elapsed: {total_elapsed:.2f}s")
    print(f"  Synchronous FPS:    {frame_idx / max(0.001, total_elapsed):.2f} FPS")
    print("-----------------------------------------------------------------")
    for stage, vals in stage_latencies.items():
        if vals:
            arr = np.array(vals)
            print(f"  {stage.upper():12s} -> Median: {np.median(arr):5.2f}ms | P95: {np.percentile(arr, 95):5.2f}ms | P99: {np.percentile(arr, 99):5.2f}ms")
    print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORBITA Diagnostic Runner")
    parser.add_argument("--video", default="vdata/20260905_145858.mp4", help="Video path")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to process")
    args = parser.parse_args()
    run_diagnostics(args.video, args.frames)
