"""
ORBITA Detailed Pipeline Profiler
=================================
Accurately benchmarks and instruments every stage of the ORBITA AI Worker pipeline
using real video frames from vdata/20260905_145858.mp4.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core_ai.app.config import load_config
from core_ai.har.feature_fusion import TemporalFeatureWindow, build_feature_vector
from core_ai.har.temporal_model import ActionClassifier
from core_ai.interaction.hand_object import HandObjectInteractionTracker
from core_ai.perception.hand_tracker import HandTracker
from core_ai.perception.object_detector import ObjectDetector
from core_ai.perception.pose_estimator import PoseEstimator
from core_ai.reasoning.state_manager import StateManager
from core_ai.backend.api import _annotate_frame


def profile_pipeline(video_path: str = "vdata/20260905_145858.mp4", num_frames: int = 60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    cfg = load_config()
    pose_estimator = PoseEstimator(cfg.pose)
    hand_tracker = HandTracker(cfg.hand)
    detector = ObjectDetector(cfg.detection)
    interaction_tracker = HandObjectInteractionTracker(cfg.interaction)
    feature_window = TemporalFeatureWindow(cfg.har.window_frames)
    classifier = ActionClassifier(cfg.har)
    state_manager = StateManager(cfg)

    # Ensure models are warm
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _ = pose_estimator.estimate(dummy_frame)
    _ = hand_tracker.track(None, dummy_frame)
    _ = detector.detect(dummy_frame)
    _ = classifier.predict(np.zeros((30, 64), dtype=np.float32))

    metrics = {
        "capture": [],
        "preprocessing": [],
        "pose": [],
        "hands": [],
        "yolo_detector": [],
        "chroma_fallback": [],
        "object_tracker": [],
        "interaction": [],
        "feature_fusion": [],
        "har_gru": [],
        "fsm": [],
        "annotation": [],
        "telemetry": [],
        "total": [],
    }

    print(f"Profiling ORBITA pipeline over {num_frames} frames from {video_path}...")
    frame_idx = 0

    while frame_idx < num_frames:
        t_total_start = time.perf_counter()

        # 1. Frame Capture
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        t_cap = (time.perf_counter() - t0) * 1000.0

        # 2. Preprocessing
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        target_w = 960
        if max(h, w) > target_w:
            scale = target_w / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        t_stamp = time.time()
        t_prep = (time.perf_counter() - t0) * 1000.0

        # 3. Pose Estimation
        t0 = time.perf_counter()
        poses = pose_estimator.estimate(frame, t_stamp)
        pose = poses[0] if poses else None
        person_bbox = pose.bbox if pose else None
        t_pose = (time.perf_counter() - t0) * 1000.0

        # 4. Hand Detection & Tracking
        t0 = time.perf_counter()
        left, right = hand_tracker.track(pose, frame, t_stamp, person_bbox=person_bbox)
        t_hands = (time.perf_counter() - t0) * 1000.0

        # 5. Object Detection (Split into YOLO vs Chroma vs Tracker)
        # Instrument detector internals directly
        t0 = time.perf_counter()
        yolo_objs = detector._detect_yolo(frame, t_stamp) if detector._yolo else []
        t_yolo = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        found_classes = {o.class_name for o in yolo_objs}
        missing_boxes = {"RED_BOX", "YELLOW_BOX", "MAIN_BOX"} - found_classes
        if missing_boxes and hasattr(detector, "tracker") and hasattr(detector.tracker, "tracks"):
            tracked_classes = {
                t.class_name for t in detector.tracker.tracks.values()
                if getattr(t, "time_since_update", 0) <= 2
            }
            missing_boxes = missing_boxes - tracked_classes

        chroma_objs = []
        if missing_boxes:
            chroma_objs = detector._detect_chroma(
                frame,
                t_stamp,
                filter_classes=missing_boxes,
                min_area_override=3000,
                min_confidence_override=0.72,
                hands=(left, right),
            )
        raw_objects = list(yolo_objs)
        for co in chroma_objs:
            if not any(detector._iou(co, yo) > 0.20 for yo in yolo_objs):
                raw_objects.append(co)
        t_chroma = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        smoothed = detector._smooth_detections(raw_objects, t_stamp)
        objects = detector.tracker.update(smoothed, t_stamp)
        t_obj_tracker = (time.perf_counter() - t0) * 1000.0

        # 6. Interaction Tracking
        t0 = time.perf_counter()
        interactions = interaction_tracker.update(left, right, objects, t_stamp)
        t_interaction = (time.perf_counter() - t0) * 1000.0

        # 7. Feature Fusion
        t0 = time.perf_counter()
        fv = build_feature_vector(pose, left, right, objects, interactions, frame.shape[:2])
        feature_window.push(fv)
        window = feature_window.get_window()
        t_fusion = (time.perf_counter() - t0) * 1000.0

        # 8. GRU Inference
        t0 = time.perf_counter()
        prediction = classifier.predict(window)
        t_gru = (time.perf_counter() - t0) * 1000.0

        # 9. FSM State Manager
        t0 = time.perf_counter()
        result = state_manager.process(
            prediction=prediction,
            detected_objects=objects,
            fps=15.0,
            latency_ms=100.0,
            left_hand=left,
            right_hand=right,
            interactions=interactions,
        )
        t_fsm = (time.perf_counter() - t0) * 1000.0

        # 10. Frame Annotation
        t0 = time.perf_counter()
        vis = frame.copy()
        if objects:
            vis = detector.draw(vis, objects)
        if poses:
            vis = pose_estimator.draw(vis, poses)
        if left or right:
            vis = hand_tracker.draw(vis, left, right)
        if interactions and objects and (left or right):
            vis = interaction_tracker.draw_interactions(vis, interactions, objects, left, right)
        t_annotation = (time.perf_counter() - t0) * 1000.0

        # 11. Telemetry / Serialization
        t0 = time.perf_counter()
        _ = result.to_dict()
        t_telemetry = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_total_start) * 1000.0

        metrics["capture"].append(t_cap)
        metrics["preprocessing"].append(t_prep)
        metrics["pose"].append(t_pose)
        metrics["hands"].append(t_hands)
        metrics["yolo_detector"].append(t_yolo)
        metrics["chroma_fallback"].append(t_chroma)
        metrics["object_tracker"].append(t_obj_tracker)
        metrics["interaction"].append(t_interaction)
        metrics["feature_fusion"].append(t_fusion)
        metrics["har_gru"].append(t_gru)
        metrics["fsm"].append(t_fsm)
        metrics["annotation"].append(t_annotation)
        metrics["telemetry"].append(t_telemetry)
        metrics["total"].append(t_total)

        frame_idx += 1

    cap.release()

    print("\n" + "=" * 56)
    print(f"{'COMPONENT':<26} {'AVG':<14} {'P95':<14}")
    print("-" * 56)
    for comp, vals in metrics.items():
        if vals:
            avg_v = float(np.mean(vals))
            p95_v = float(np.percentile(vals, 95))
            print(f"{comp.upper():<26} {avg_v:6.2f} ms      {p95_v:6.2f} ms")
    print("=" * 56)


if __name__ == "__main__":
    profile_pipeline()
