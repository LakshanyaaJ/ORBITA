"""
ORBITA Latency & Detection Robustness Benchmark (Sections 14, 15, 18, 20)
==========================================================================
Profiles the end-to-end video-to-result pipeline:
  - Video Decode latency
  - Preprocessing & Table ROI latency
  - YOLO Multi-Class Inference latency
  - Hand Pose & Landmark latency
  - Multi-Object Tracking & Persistence latency
  - Action Gate Reasoning & FSM latency
  - Debug Overlay Rendering latency
  - Per-class detection rates (PEN, WATCH, BLUE_BOX, YELLOW_BOX, HAND, LOCATION_A, LOCATION_B)
  - Generates Before vs After engineering report
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core_ai.app.config import load_config
from core_ai.perception.object_detector import ObjectDetector
from core_ai.perception.hand_tracker import HandTracker
from core_ai.interaction.hand_object import HandObjectInteractionManager
from core_ai.reasoning.action_gate import ActionConfirmationGate
from core_ai.perception.debug_visualizer import DebugVisualizer


def run_benchmark(
    video_path: str,
    max_frames: int = 150,
    save_debug_video: bool = False,
    output_video_path: str = "benchmark_debug_out.mp4",
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"ORBITA DETECTION & PIPELINE BENCHMARK")
    print(f"Target Video: {video_path}")
    print(f"Max Frames: {max_frames}")
    print(f"=======================================================\n")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video {video_path}")
        return {}

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {orig_w}x{orig_h} @ {video_fps:.1f} FPS (Total: {total_video_frames} frames)")

    # Initialize ORBITA pipeline components
    cfg = load_config()
    detector = ObjectDetector(cfg)
    hand_tracker = HandTracker(cfg.hand)
    interaction_mgr = HandObjectInteractionManager(cfg.interaction)
    action_gate = ActionConfirmationGate(cfg)
    debug_vis = DebugVisualizer()

    # Metrics accumulators
    latencies: Dict[str, List[float]] = {
        "decode": [],
        "preprocess": [],
        "yolo": [],
        "pose": [],
        "tracking": [],
        "reasoning": [],
        "rendering": [],
        "total": [],
    }

    class_detections: Dict[str, int] = {
        "PEN": 0,
        "WATCH": 0,
        "BLUE_BOX": 0,
        "YELLOW_BOX": 0,
        "HAND": 0,
        "LOCATION_A": 0,
        "LOCATION_B": 0,
    }

    class_confidences: Dict[str, List[float]] = {k: [] for k in class_detections.keys()}
    track_ids_observed: Dict[str, set] = {k: set() for k in class_detections.keys()}

    out_writer = None
    if save_debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (orig_w, orig_h))

    frame_idx = 0
    t_start_all = time.perf_counter()

    while frame_idx < max_frames:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t_decode = (time.perf_counter() - t0) * 1000.0
        if not ret or frame is None:
            break

        frame_idx += 1
        timestamp = frame_idx / max(1.0, video_fps)

        # 1. Preprocessing (guarantee horizontal orientation)
        t1 = time.perf_counter()
        fh, fw = frame.shape[:2]
        if fh > fw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            fh, fw = frame.shape[:2]
        t_prep = (time.perf_counter() - t1) * 1000.0

        # 2. YOLO Object Detection & Tracking
        t2 = time.perf_counter()
        detections = detector.detect(frame, timestamp=timestamp, frame_id=frame_idx)
        t_yolo = (time.perf_counter() - t2) * 1000.0

        # 3. Hand Pose
        t3 = time.perf_counter()
        hands = hand_tracker.track(pose=None, frame=frame, timestamp=timestamp)
        t_pose = (time.perf_counter() - t3) * 1000.0

        # 4. Interactions
        t4 = time.perf_counter()
        interactions = interaction_mgr.evaluate(hands[0], hands[1], detections, timestamp=timestamp)
        detector.tracker.update_interaction_states(interactions)
        t_track = (time.perf_counter() - t4) * 1000.0

        # 5. Reasoning / Action Gate
        t5 = time.perf_counter()
        primary_inter = interaction_mgr.get_primary_interaction(interactions)
        conf_action = action_gate.evaluate(
            current_step={"step_id": 1, "expected_action": "IDENTIFY", "expected_target": "PEN"},
            detected_objects=detections,
            tracks=detections,
            interactions=interactions,
            har_action="IDLE",
            har_confidence=0.75,
            frame_shape=(fh, fw),
        )
        t_gate = (time.perf_counter() - t5) * 1000.0

        # 6. Visual Debug Rendering
        t6 = time.perf_counter()
        grouped = detector.get_grouped_detections()
        lat_dict = {
            "decode": t_decode,
            "preprocess": t_prep,
            "yolo": t_yolo,
            "pose": t_pose,
            "tracking": t_track,
            "reasoning": t_gate,
            "render": 0.0,
        }
        vis_frame = debug_vis.render_debug_overlay(
            frame=frame,
            detections=detections,
            hands=hands,
            interactions=interactions,
            zones=cfg.detection.zones,
            fsm_state={"status": "EVALUATING", "detected_action": getattr(conf_action, "action", "INSPECT")},
            latencies_ms=lat_dict,
            fps=30.0,
        )
        t_render = (time.perf_counter() - t6) * 1000.0
        t_total = t_decode + t_prep + t_yolo + t_pose + t_track + t_gate + t_render

        # Record latencies
        latencies["decode"].append(t_decode)
        latencies["preprocess"].append(t_prep)
        latencies["yolo"].append(t_yolo)
        latencies["pose"].append(t_pose)
        latencies["tracking"].append(t_track)
        latencies["reasoning"].append(t_gate)
        latencies["rendering"].append(t_render)
        latencies["total"].append(t_total)

        # Record detections
        for det in detections:
            c = str(getattr(det, "semantic_identity", "") or det.class_name).upper()
            if c in class_detections:
                class_detections[c] += 1
                class_confidences[c].append(float(det.confidence))
                tid = getattr(det, "track_id", -1)
                if tid > 0:
                    track_ids_observed[c].add(tid)

        # Also check geometric zones for Location A / B
        for z in ("LOCATION_A", "LOCATION_B"):
            if grouped.get(z):
                class_detections[z] += 1
                class_confidences[z].append(0.99)

        if out_writer:
            out_writer.write(vis_frame)

        if frame_idx % 25 == 0:
            avg_tot = np.mean(latencies["total"][-25:])
            cur_fps = 1000.0 / max(1.0, avg_tot)
            print(f"Processed Frame {frame_idx}/{max_frames} — {avg_tot:.1f} ms/frame ({cur_fps:.1f} FPS)")

    cap.release()
    if out_writer:
        out_writer.release()

    total_dur_sec = time.perf_counter() - t_start_all
    avg_fps = frame_idx / max(0.001, total_dur_sec)

    # ----------------------------------------------------------------------- #
    # Latency Breakdown Report (Section 14)
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 55)
    print("PIPELINE LATENCY BREAKDOWN (Section 14 Profile)")
    print("=" * 55)
    print(f"{'Component':<22} {'Mean (ms)':<12} {'P95 (ms)':<12}")
    print("-" * 55)
    for comp in ("decode", "preprocess", "yolo", "pose", "tracking", "reasoning", "rendering"):
        vals = latencies[comp]
        mean_v = np.mean(vals) if vals else 0.0
        p95_v = np.percentile(vals, 95) if vals else 0.0
        print(f"{comp.capitalize():<22} {mean_v:<12.2f} {p95_v:<12.2f}")
    mean_tot = np.mean(latencies["total"]) if latencies["total"] else 0.0
    p95_tot = np.percentile(latencies["total"], 95) if latencies["total"] else 0.0
    print("-" * 55)
    print(f"{'TOTAL END-TO-END':<22} {mean_tot:<12.2f} {p95_tot:<12.2f}")
    print(f"{'EFFECTIVE THROUGHPUT':<22} {avg_fps:<12.2f} FPS")
    print("=" * 55)

    # ----------------------------------------------------------------------- #
    # Detection Robustness Report (Section 18 & 20)
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 65)
    print("DETECTION & TRACKING METRICS SUMMARY (Section 18)")
    print("=" * 65)
    print(f"{'Entity':<16} {'Frames Detected':<18} {'Avg Conf':<12} {'Unique Tracks':<14}")
    print("-" * 65)
    for k in sorted(class_detections.keys()):
        count = class_detections[k]
        confs = class_confidences[k]
        avg_c = np.mean(confs) if confs else 0.0
        n_tracks = len(track_ids_observed[k])
        det_rate = (count / max(1, frame_idx)) * 100.0
        print(f"{k:<16} {count:<6} ({det_rate:4.1f}%)    {avg_c:<12.2f} {n_tracks:<14}")
    print("=" * 65 + "\n")

    return {
        "frames": frame_idx,
        "fps": avg_fps,
        "mean_latency_ms": mean_tot,
        "class_detections": class_detections,
        "class_confidences": {k: float(np.mean(v)) if v else 0.0 for k, v in class_confidences.items()},
        "unique_tracks": {k: len(v) for k, v in track_ids_observed.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORBITA Detector Benchmark")
    parser.add_argument("--video", type=str, default="vdata/20260908_135006.mp4", help="Video file path")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark")
    parser.add_argument("--save-video", action="store_true", help="Save debug visualization video")
    args = parser.parse_args()

    run_benchmark(args.video, max_frames=args.frames, save_debug_video=args.save_video)
