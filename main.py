"""
ORBITA — Top-level launcher
============================
Usage:
  python main.py --mode web --scenario A
  python main.py --mode web --scenario B
  python main.py --mode web --scenario C
  python main.py --mode desktop --scenario A
  python main.py --mode web --camera 0       # Use webcam

Modes:
  web       → FastAPI server + React dashboard at http://localhost:8000
  desktop   → OpenCV window (no browser needed)
  headless  → No display, logging only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import core_ai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("core_ai.main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ORBITA — Offline AI Experiment Copilot"
    )
    parser.add_argument(
        "--mode", choices=["web", "desktop", "headless"], default="web",
        help="Run mode: web (FastAPI+React), desktop (OpenCV), headless"
    )
    parser.add_argument(
        "--scenario", choices=["A", "B", "C"], default="A",
        help="Simulation scenario: A=nominal, B=wrong object, C=skipped step"
    )
    parser.add_argument(
        "--camera", default=None,
        help="Camera source: integer index or video path (overrides sim mode)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="API server port (web mode)"
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Disable TTS voice output"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to custom experiment config JSON"
    )
    return parser.parse_args()


def run_web(args):
    """Start FastAPI backend."""
    import uvicorn
    from core_ai.backend.api import app, _state

    # Pre-configure state before startup
    _state.mode = "webcam" if args.camera else "sim"
    _state.current_scenario = args.scenario

    logger.info("Starting ORBITA web server on http://0.0.0.0:%d", args.port)
    logger.info("Dashboard: http://localhost:%d", args.port)
    logger.info("Video feed: http://localhost:%d/video_feed", args.port)
    logger.info("Scenario: %s", args.scenario)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
    )


def run_desktop(args):
    """Run with OpenCV display window."""
    import cv2
    from core_ai.app.config import load_config
    from core_ai.simulation.simulator import ExperimentSimulator, Scenario
    from core_ai.perception.object_detector import ObjectDetector
    from core_ai.perception.pose_estimator import PoseEstimator
    from core_ai.perception.hand_tracker import HandTracker
    from core_ai.interaction.hand_object import HandObjectInteractionTracker
    from core_ai.har.feature_fusion import TemporalFeatureWindow, build_feature_vector
    from core_ai.har.temporal_model import ActionClassifier, ActionPrediction
    from core_ai.reasoning.state_manager import StateManager
    from core_ai.voice.tts import TTSEngine

    cfg = load_config(args.config)
    if args.no_voice:
        cfg.voice.enabled = False

    tts = TTSEngine(cfg.voice)
    def on_voice(msg):
        tts.speak(msg)

    detector = ObjectDetector(cfg.detection)
    pose_est = PoseEstimator(cfg.pose)
    hand_tracker = HandTracker(cfg.hand)
    interaction_tracker = HandObjectInteractionTracker(cfg.interaction)
    feature_window = TemporalFeatureWindow(cfg.har.window_frames)
    classifier = ActionClassifier(cfg.har)
    state_manager = StateManager(cfg, on_voice=on_voice)

    scenario = Scenario(args.scenario)
    sim = ExperimentSimulator(scenario)

    logger.info("Desktop mode. Press 'a'=ScenA  'b'=ScenB  'c'=ScenC  'r'=Reset  'q'=Quit")

    fps_display = 0.0
    frame_count = 0
    fps_timer = time.time()

    while True:
        t0 = time.time()
        sim_frame = sim.next_frame()
        frame = sim_frame.image

        # Perception
        objects = detector.detect(frame, sim_frame.timestamp)
        poses = pose_est.estimate(frame, sim_frame.timestamp)
        pose = poses[0] if poses else None
        left, right = hand_tracker.track(pose, frame, sim_frame.timestamp)
        interactions = interaction_tracker.update(left, right, objects, sim_frame.timestamp)

        # Feature fusion + classification
        fv = build_feature_vector(pose, left, right, objects, interactions, frame.shape[:2])
        feature_window.push(fv)
        window = feature_window.get_window()

        # Use simulator ground truth for demo clarity
        prediction = ActionPrediction(
            action=sim_frame.action_verb,
            confidence=sim_frame.confidence,
            next_action="IDLE",
            next_confidence=0.3,
            is_uncertain=sim_frame.confidence < cfg.har.action_confidence_min,
        )

        result = state_manager.process(prediction, objects, fps=fps_display)

        # Annotate
        vis = detector.draw(frame, objects)
        if pose:
            vis = pose_est.draw(vis, [pose])
        vis = hand_tracker.draw(vis, left, right)

        # Status overlay
        status = result.fsm_state.status.name
        colour_map = {
            "CORRECT": (0, 200, 80), "WAITING": (180, 180, 180),
            "WRONG_OBJECT": (0, 60, 220), "WRONG_ACTION": (0, 40, 200),
            "STEP_SKIPPED": (0, 60, 220), "UNCERTAIN": (0, 180, 230),
            "COMPLETED": (0, 220, 120),
        }
        colour = colour_map.get(status, (180, 180, 180))
        cv2.putText(vis, f"STATUS: {status}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps_display:.1f}", (530, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1, cv2.LINE_AA)

        try:
            cv2.imshow("ORBITA — Experiment Copilot", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                sim.reset(Scenario.A)
                state_manager.reset()
            elif key == ord('b'):
                sim.reset(Scenario.B)
                state_manager.reset()
            elif key == ord('c'):
                sim.reset(Scenario.C)
                state_manager.reset()
            elif key == ord('r'):
                state_manager.reset()
                sim.reset()
        except cv2.error:
            logger.warning("OpenCV GUI (cv2.imshow) is not available in this OpenCV build.")
            logger.info("Falling back automatically to Web Mission-Control mode...")
            cv2.destroyAllWindows()
            tts.stop()
            return run_web(args)

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

    cv2.destroyAllWindows()
    tts.stop()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("  ORBITA — Offline AI Experiment Copilot")
    logger.info("  SIH 26174 — Mode: %s | Scenario: %s", args.mode, args.scenario)
    logger.info("=" * 60)

    if args.mode == "web":
        run_web(args)
    elif args.mode == "desktop":
        run_desktop(args)
    elif args.mode == "headless":
        logger.info("Headless mode — logs only.")
        # Future: headless integration for Jetson deployment
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
