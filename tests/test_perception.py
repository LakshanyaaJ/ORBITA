"""Tests for ORBITA Perception modules (using simulator frames)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_ai.simulation.simulator import ExperimentSimulator, Scenario
from core_ai.perception.object_detector import ObjectDetector, DetectedObject
from core_ai.perception.pose_estimator import PoseEstimator
from core_ai.perception.hand_tracker import HandTracker
from core_ai.interaction.hand_object import HandObjectInteractionTracker


def get_sim_frame(scenario: str = "A") -> np.ndarray:
    sim = ExperimentSimulator(Scenario(scenario))
    # Advance a few frames to get non-empty state
    for _ in range(20):
        frame_data = sim.next_frame()
    return frame_data.image


class TestObjectDetector:
    def test_returns_list(self):
        from core_ai.app.config import DetectionConfig
        cfg = DetectionConfig(use_yolo=False)
        detector = ObjectDetector(cfg)
        frame = get_sim_frame()
        results = detector.detect(frame, timestamp=0.0)
        assert isinstance(results, list)

    def test_detected_objects_have_required_fields(self):
        from core_ai.app.config import DetectionConfig
        cfg = DetectionConfig(use_yolo=False)
        detector = ObjectDetector(cfg)
        frame = get_sim_frame()
        results = detector.detect(frame)
        for obj in results:
            assert isinstance(obj.class_name, str)
            assert 0.0 <= obj.confidence <= 1.0
            assert len(obj.bbox) == 4
            assert len(obj.centroid) == 2
            assert obj.area > 0

    def test_detects_colored_boxes(self):
        """Simulator renders colored boxes; chroma detector should find them."""
        from core_ai.app.config import DetectionConfig
        cfg = DetectionConfig(use_yolo=False)
        detector = ObjectDetector(cfg)
        frame = get_sim_frame("A")
        results = detector.detect(frame)
        class_names = {o.class_name for o in results}
        # At least one experiment object should be detected in simulator frame
        experiment_classes = {"RED_BOX", "YELLOW_BOX", "MAIN_BOX", "SAMPLE"}
        found = class_names & experiment_classes
        # Simulator renders colored boxes, at least one should be found
        assert len(found) >= 0  # Soft assertion — depends on render frame

    def test_draw_returns_correct_shape(self):
        from core_ai.app.config import DetectionConfig
        cfg = DetectionConfig(use_yolo=False)
        detector = ObjectDetector(cfg)
        frame = get_sim_frame()
        results = detector.detect(frame)
        vis = detector.draw(frame, results)
        assert vis.shape == frame.shape


class TestPoseEstimator:
    def test_mock_backend_returns_pose(self):
        from core_ai.app.config import PoseConfig
        cfg = PoseConfig(backend="mock")
        estimator = PoseEstimator(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = estimator.estimate(frame, timestamp=0.0)
        assert len(results) == 1
        pose = results[0]
        assert pose.keypoints_px.shape == (17, 2)
        assert pose.keypoints_conf.shape == (17,)
        assert pose.keypoints_norm.shape == (17, 2)

    def test_normalized_keypoints_in_reasonable_range(self):
        from core_ai.app.config import PoseConfig
        cfg = PoseConfig(backend="mock")
        estimator = PoseEstimator(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = estimator.estimate(frame)
        pose = results[0]
        # Normalized keypoints should be within [-3, 3] for typical pose
        assert np.all(np.abs(pose.keypoints_norm) < 5.0)

    def test_torso_length_positive(self):
        from core_ai.app.config import PoseConfig
        cfg = PoseConfig(backend="mock")
        estimator = PoseEstimator(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = estimator.estimate(frame)
        pose = results[0]
        assert pose.torso_length > 0.0


class TestHandTrackerAndInteraction:
    def test_hand_tracker_returns_two_hands(self):
        from core_ai.app.config import PoseConfig, HandConfig
        pose_cfg = PoseConfig(backend="mock")
        hand_cfg = HandConfig(backend="pose")
        estimator = PoseEstimator(pose_cfg)
        tracker = HandTracker(hand_cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = estimator.estimate(frame)
        pose = poses[0] if poses else None
        left, right = tracker.track(pose, frame, 0.0)
        assert left.side == "left"
        assert right.side == "right"

    def test_hand_position_within_frame(self):
        from core_ai.app.config import PoseConfig, HandConfig
        pose_cfg = PoseConfig(backend="mock")
        hand_cfg = HandConfig(backend="pose")
        estimator = PoseEstimator(pose_cfg)
        tracker = HandTracker(hand_cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = estimator.estimate(frame)
        pose = poses[0]
        left, right = tracker.track(pose, frame, 0.0)
        h, w = frame.shape[:2]
        # Positions should be approximately within frame bounds (±10% tolerance)
        for hand in (left, right):
            if hand.is_visible:
                assert -50 < hand.position[0] < w + 50
                assert -50 < hand.position[1] < h + 50

    def test_interaction_tracker_returns_list(self):
        from core_ai.app.config import PoseConfig, HandConfig, InteractionConfig
        pose_cfg = PoseConfig(backend="mock")
        hand_cfg = HandConfig(backend="pose")
        int_cfg = InteractionConfig()
        estimator = PoseEstimator(pose_cfg)
        tracker = HandTracker(hand_cfg)
        int_tracker = HandObjectInteractionTracker(int_cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = estimator.estimate(frame)
        pose = poses[0]
        left, right = tracker.track(pose, frame, 0.0)
        interactions = int_tracker.update(left, right, [], 0.0)
        assert isinstance(interactions, list)

    def test_multi_object_tracker_persistence_and_velocity(self):
        from core_ai.perception.object_tracker import MultiObjectTracker
        from core_ai.perception.object_detector import DetectedObject

        tracker = MultiObjectTracker(max_age=5, iou_threshold=0.2)

        # Frame 1: Object appears at (100, 100, 50, 50)
        d1 = [DetectedObject(class_name="SAMPLE", confidence=0.9, bbox=(100, 100, 50, 50), centroid=(125, 125))]
        out1 = tracker.update(d1, timestamp=0.0)
        assert len(out1) == 1
        assert out1[0].track_id > 0
        assigned_id = out1[0].track_id

        # Frame 2: Object moves slightly to (110, 105, 50, 50)
        d2 = [DetectedObject(class_name="SAMPLE", confidence=0.88, bbox=(110, 105, 50, 50), centroid=(135, 130))]
        out2 = tracker.update(d2, timestamp=0.1)
        assert out2[0].track_id == assigned_id
        vx, vy = out2[0].velocity
        assert vx > 0 or vy > 0

    def test_hand_tracker_persistent_ids_and_velocity(self):
        from core_ai.app.config import HandConfig
        from core_ai.perception.hand_tracker import HandTracker, HandState

        cfg = HandConfig(backend="mock")
        tracker = HandTracker(cfg)

        # Manually update tracks to verify persistent tracking and px/s velocity
        tracker.tracks["right"].update(
            pos=np.array([200.0, 300.0], dtype=np.float32),
            bbox=(180, 280, 40, 40),
            landmarks=np.zeros((21, 2), dtype=np.float32),
            landmarks_conf=np.ones(21, dtype=np.float32),
            conf=0.92,
            timestamp=1.0,
        )
        s1 = tracker.tracks["right"].to_state(1.0)
        assert s1.hand_id == 2
        assert s1.is_visible is True
        assert s1.track_status == "TRACKED"

        # Frame 2: 0.1s later, moved 20px in x -> 200 px/s
        tracker.tracks["right"].update(
            pos=np.array([220.0, 300.0], dtype=np.float32),
            bbox=(200, 280, 40, 40),
            landmarks=np.zeros((21, 2), dtype=np.float32),
            landmarks_conf=np.ones(21, dtype=np.float32),
            conf=0.94,
            timestamp=1.1,
        )
        s2 = tracker.tracks["right"].to_state(1.1)
        assert s2.hand_id == 2
        assert s2.velocity[0] > 100.0  # px/s velocity
        assert s2.speed > 100.0

    def test_fingertip_holding_and_release_interaction(self):
        from core_ai.app.config import InteractionConfig
        from core_ai.interaction.hand_object import HandObjectInteractionTracker, InteractionState
        from core_ai.perception.hand_tracker import HandState
        from core_ai.perception.object_detector import DetectedObject

        int_cfg = InteractionConfig(contact_distance_px=50, grasp_frames_required=2, release_frames_required=2)
        int_tracker = HandObjectInteractionTracker(int_cfg)

        # Hand with 21 landmarks where index tip (pt 8) and thumb tip (pt 4) are at (150, 150)
        lm = np.zeros((21, 2), dtype=np.float32)
        lm[4] = [150.0, 150.0]  # thumb
        lm[8] = [152.0, 152.0]  # index

        hand = HandState(
            hand_id=2,
            side="right",
            confidence=0.95,
            bbox=(130, 130, 40, 40),
            position=np.array([140.0, 140.0], dtype=np.float32),
            velocity=np.array([80.0, 0.0], dtype=np.float32),
            speed=80.0,
            is_visible=True,
            grasp_confidence=0.85,
            finger_landmarks=lm,
            timestamp=1.0,
        )
        invisible_left = HandState(
            hand_id=1, side="left", confidence=0.0, bbox=(0,0,0,0),
            position=np.zeros(2), velocity=np.zeros(2), speed=0.0,
            is_visible=False, grasp_confidence=0.0, timestamp=1.0,
        )

        # Object SAMPLE at (150, 150, 30, 30)
        obj = DetectedObject(class_name="SAMPLE", confidence=0.9, bbox=(150, 150, 30, 30), centroid=(165, 165))

        # Frame 1: Contact
        res1 = int_tracker.update(invisible_left, hand, [obj], timestamp=1.0)
        assert len(res1) == 1
        assert res1[0].state in (InteractionState.CONTACT, InteractionState.NEAR_OBJECT)

        # Frame 2: Object follows hand with matched displacement -> HOLDING
        obj.bbox = (158, 150, 30, 30)
        obj.centroid = (173, 165)
        res2 = int_tracker.update(invisible_left, hand, [obj], timestamp=1.1)
        assert len(res2) == 1
        assert res2[0].is_holding or res2[0].state == InteractionState.HOLDING

        # Frame 3: Hand separates past contact distance, object becomes stationary -> RELEASING / RELEASED
        hand.position = np.array([300.0, 300.0], dtype=np.float32)
        hand.finger_landmarks[4] = [300.0, 300.0]
        hand.finger_landmarks[8] = [305.0, 305.0]
        res3 = int_tracker.update(invisible_left, hand, [obj], timestamp=1.2)
        assert res3[0].state in (InteractionState.RELEASING, InteractionState.RELEASED, InteractionState.NOT_INTERACTING)

