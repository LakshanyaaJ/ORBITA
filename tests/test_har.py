"""Tests for ORBITA feature fusion and temporal HAR model."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orbita.har.feature_fusion import (
    FEATURE_DIM,
    TemporalFeatureWindow,
    build_feature_vector,
)
from orbita.har.temporal_model import (
    ActionClassifier,
    ActionPrediction,
    ACTIONS,
    _heuristic_predict,
)


class TestFeatureFusion:
    def test_feature_vector_correct_dimension(self):
        fv = build_feature_vector(
            pose=None,
            left_hand=None,
            right_hand=None,
            objects=[],
            interactions=[],
            frame_shape=(480, 640),
        )
        assert fv.shape == (FEATURE_DIM,)

    def test_feature_vector_dtype_float32(self):
        fv = build_feature_vector(None, None, None, [], [], (480, 640))
        assert fv.dtype == np.float32

    def test_feature_vector_all_zeros_on_empty_input(self):
        fv = build_feature_vector(None, None, None, [], [], (480, 640))
        assert np.all(fv == 0.0)

    def test_feature_window_shape(self):
        window = TemporalFeatureWindow(window_size=30, feature_dim=FEATURE_DIM)
        arr = window.get_window()
        assert arr.shape == (30, FEATURE_DIM)

    def test_feature_window_push(self):
        window = TemporalFeatureWindow(window_size=5, feature_dim=FEATURE_DIM)
        fv = np.ones(FEATURE_DIM, dtype=np.float32)
        window.push(fv)
        arr = window.get_window()
        # Last frame should be all ones
        np.testing.assert_array_equal(arr[-1], fv)

    def test_feature_window_fifo(self):
        window = TemporalFeatureWindow(window_size=3, feature_dim=FEATURE_DIM)
        for i in range(5):
            fv = np.full(FEATURE_DIM, float(i), dtype=np.float32)
            window.push(fv)
        arr = window.get_window()
        # Should contain last 3 frames: 2.0, 3.0, 4.0
        assert arr[0, 0] == 2.0
        assert arr[1, 0] == 3.0
        assert arr[2, 0] == 4.0

    def test_feature_window_reset(self):
        window = TemporalFeatureWindow(window_size=3, feature_dim=FEATURE_DIM)
        for _ in range(3):
            window.push(np.ones(FEATURE_DIM, dtype=np.float32))
        window.reset()
        arr = window.get_window()
        assert np.all(arr == 0.0)

    def test_object_onehot_populated(self):
        """Object presence should be encoded in the one-hot section [50:60]."""
        from orbita.perception.object_detector import DetectedObject
        obj = DetectedObject(
            class_name="RED_BOX",
            confidence=0.85,
            bbox=(100, 100, 30, 20),
            centroid=(115, 110),
        )
        fv = build_feature_vector(None, None, None, [obj], [], (480, 640))
        # RED_BOX is index 2 in OBJECT_CLASSES → feature index 52
        assert fv[52] > 0.0


class TestHeuristicPredict:
    def test_returns_action_prediction(self):
        window = np.zeros((30, FEATURE_DIM), dtype=np.float32)
        result = _heuristic_predict(window, confidence_threshold=0.55)
        assert isinstance(result, ActionPrediction)
        assert result.action in ACTIONS
        assert 0.0 <= result.confidence <= 1.0

    def test_idle_on_empty_features(self):
        window = np.zeros((30, FEATURE_DIM), dtype=np.float32)
        result = _heuristic_predict(window, confidence_threshold=0.55)
        assert result.action == "IDLE"

    def test_uncertain_when_confidence_low(self):
        window = np.zeros((30, FEATURE_DIM), dtype=np.float32)
        # Force very low confidence by setting all features to near-zero
        result = _heuristic_predict(window, confidence_threshold=0.99)
        # With threshold 0.99, heuristic confidence won't reach it
        assert result.is_uncertain or result.action == "IDLE"


class TestActionClassifier:
    def test_classifier_returns_prediction(self):
        from orbita.app.config import HARConfig
        cfg = HARConfig()
        classifier = ActionClassifier(cfg)
        window = np.zeros((30, FEATURE_DIM), dtype=np.float32)
        result = classifier.predict(window)
        assert isinstance(result, ActionPrediction)
        assert result.action in ACTIONS

    def test_classifier_uncertain_flag(self):
        from orbita.app.config import HARConfig
        cfg = HARConfig(action_confidence_min=0.99)  # Very high threshold
        classifier = ActionClassifier(cfg)
        window = np.zeros((30, FEATURE_DIM), dtype=np.float32)
        result = classifier.predict(window)
        # Empty features should produce IDLE
        assert result.action in ("IDLE", "UNCERTAIN") or result.is_uncertain
