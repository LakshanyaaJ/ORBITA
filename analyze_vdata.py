import time
from core_ai.app.config import load_config, CameraConfig
from core_ai.video.camera import Camera
from core_ai.perception.object_detector import ObjectDetector
from core_ai.perception.pose_estimator import PoseEstimator
from core_ai.perception.hand_tracker import HandTracker
from core_ai.interaction.hand_object import HandObjectInteractionTracker
from core_ai.har.temporal_model import ActionClassifier
from core_ai.har.feature_fusion import build_feature_vector, TemporalFeatureWindow
from core_ai.reasoning.state_manager import StateManager

cfg = load_config()
detector = ObjectDetector(cfg.detection)
pose_est = PoseEstimator(cfg.pose)
hand_t = HandTracker(cfg.hand)
int_t = HandObjectInteractionTracker(cfg.interaction)
classifier = ActionClassifier(cfg.har)
window = TemporalFeatureWindow(cfg.har.window_frames)

voices = []
mgr = StateManager(cfg, on_voice=lambda v: voices.append(v), experiment_id="EXP_ANALYZE")
mgr.reset()

import sys
from pathlib import Path

video_arg = sys.argv[1] if len(sys.argv) > 1 else "vdata/20260908_135006.mp4"
# If user passed only filename or full path
video_path = str(video_arg) if Path(video_arg).exists() else str(Path("vdata") / video_arg)
print(f"Target video: {video_path}")

cam_cfg = CameraConfig(source=video_path, width=1280, height=720, fps=30, sequential=True)
cam = Camera(cam_cfg)
cam.start(threaded=False)
total_frames = cam.total_video_frames
print(f"Total frames: {total_frames}", flush=True)

max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else (total_frames if total_frames > 0 else 2000)
frame_idx = 0
last_step = 0
while cam.is_running() and not cam.is_eof and frame_idx < max_frames:
    frame = cam.read()
    if frame is None:
        break
    frame_idx += 1
    t = frame_idx / 30.0

    poses = pose_est.estimate(frame, t)
    pose = poses[0] if poses else None
    left, right = hand_t.track(pose, frame, t)
    objects = detector.detect(frame, t, hands=(left, right))
    interactions = int_t.update(left, right, objects, t)
    detector.tracker.update_interaction_states(interactions)
    tracks = detector.tracker.get_active_tracks()

    fv = build_feature_vector(pose, left, right, objects, interactions, frame.shape[:2])
    window.push(fv)
    pred = classifier.predict(window.get_window())

    res = mgr.process(
        prediction=pred,
        detected_objects=objects,
        fps=30.0,
        left_hand=left,
        right_hand=right,
        interactions=interactions,
        tracks=tracks,
        camera_source="MP4",
        frame_index=frame_idx,
        total_frames=total_frames,
    )

    curr_step = res.fsm_state.current_step_idx
    if curr_step != last_step or frame_idx % 10 == 0:
        c_stat = res.confirmed_action.status.value if res.confirmed_action else "NONE"
        val_st = getattr(res.confirmed_action, "validation_state", "")
        val_reas = getattr(res.confirmed_action, "validation_reason", "")
        holding = list(dict.fromkeys([getattr(t, "class_name", "") for t in tracks if getattr(getattr(t, "state", ""), "value", "") in ("BEING_HELD", "CARRIED")]))
        inter_held = list(dict.fromkeys([i.object_class for i in interactions if i.is_holding or getattr(i, "state", 0) in (2, 3)]))
        print(f"F{frame_idx:03d}: Step {curr_step+1} | HAR={pred.action}({pred.confidence:.2f},{pred.target_object}) | HeldTrk={holding} InterHeld={inter_held} | Gate={c_stat}:{val_st} | Reason: {val_reas}", flush=True)
        if curr_step != last_step:
            print(f"  *** TRANSITION {last_step+1} -> {curr_step+1} ***", flush=True)
            last_step = curr_step

cam.stop()
print("\nVoices spoken:", flush=True)
for v in voices:
    print("  ->", v, flush=True)
