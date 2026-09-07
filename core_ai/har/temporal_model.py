"""
ORBITA Temporal HAR Model
==========================
PyTorch GRU-based action recognition and next-action prediction.

Architecture:
  Input: (batch, window, 64) temporal feature sequence
  GRU: 64 → 128, 2 layers, dropout 0.3
  Action head:      Linear(128, NUM_ACTIONS) → softmax
  Next-action head: Linear(128, NUM_ACTIONS) → softmax (auxiliary)

Actions recognized:
  IDLE, OPEN, TAKE, PLACE, TRANSFER, CLOSE, ACTIVATE, PERFORM, STORE

NOTE:
  For the hackathon prototype, the model is initialized with
  kinematic heuristics rather than experimentally trained weights.
  This means the model will produce reasonable outputs for obvious
  actions (large object displacement + grasp) but may misclassify
  subtle actions. A trained checkpoint should replace this.

  The UNCERTAIN safeguard (confidence < threshold → return IDLE)
  prevents invalid FSM transitions from low-confidence predictions.

SCIENTIFIC HONESTY:
  Do NOT interpret the output probabilities as calibrated confidence
  scores unless the model has been calibrated on a held-out dataset.
  Raw softmax probabilities are not the same as true probabilities.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Action labels
ACTIONS = [
    "IDLE",
    "OPEN",
    "TAKE",
    "PLACE",
    "TRANSFER",
    "CLOSE",
    "ACTIVATE",
    "PERFORM",
    "STORE",
]
NUM_ACTIONS = len(ACTIONS)
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}


@dataclass
class ActionPrediction:
    action: str
    confidence: float
    next_action: str
    next_confidence: float
    is_uncertain: bool          # True if confidence < threshold
    target_object: Optional[str] = None


# --------------------------------------------------------------------------- #
# PyTorch model (only imported if torch is available)
# --------------------------------------------------------------------------- #
def _build_model():
    try:
        import torch
        import torch.nn as nn

        class OrbitaTemporalGRU(nn.Module):
            """
            Dual-head GRU for simultaneous action recognition and
            next-action prediction.
            """

            def __init__(
                self,
                input_dim: int = 64,
                hidden_dim: int = 128,
                num_layers: int = 2,
                num_actions: int = NUM_ACTIONS,
                dropout: float = 0.3,
            ):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.action_head = nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_actions),
                )
                self.next_action_head = nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_actions),
                )

            def forward(self, x):
                # x: (batch, seq, input_dim)
                out, _ = self.gru(x)
                last = out[:, -1, :]          # Last timestep
                action_logits = self.action_head(last)
                next_logits = self.next_action_head(last)
                return action_logits, next_logits

        return OrbitaTemporalGRU, torch
    except ImportError:
        return None, None


# --------------------------------------------------------------------------- #
# Heuristic fallback (no PyTorch / no model)
# --------------------------------------------------------------------------- #
def _heuristic_predict(
    window: np.ndarray, confidence_threshold: float
) -> ActionPrediction:
    """
    Rule-based action prediction from feature window.
    Used when PyTorch is not available or model not loaded.

    Feature layout:
      [42:46] left interaction (state, dist, vel, duration)
      [46:50] right interaction
      [50:60] object one-hot
      [60:62] primary object centroid
      [62:64] grasp confidence
    """
    # Use last 5 frames for stability
    recent = window[-5:]

    # Interaction states (index 42, 46)
    left_state_avg = float(recent[:, 42].mean()) * 6
    right_state_avg = float(recent[:, 46].mean()) * 6
    max_state = max(left_state_avg, right_state_avg)

    # Object presence
    obj_presence = recent[:, 50:57].mean(axis=0)
    # 50=PERSON, 51=MAIN_BOX, 52=RED_BOX, 53=YELLOW_BOX, 54=SAMPLE, 55=TOOL, 56=CHAMBER

    # Displacement of primary interaction object
    centroid_x = recent[:, 60]
    displacement = float(np.std(centroid_x))

    # Grasp confidence
    grasp = float(recent[:, 62:64].max())

    # --- Heuristic rules ---
    action = "IDLE"
    confidence = 0.60

    if max_state >= 5.0:  # MANIPULATING
        if obj_presence[4] > 0.3:   # SAMPLE
            action = "PERFORM"
            confidence = 0.72
        elif displacement > 0.03:
            action = "TRANSFER"
            confidence = 0.68
        else:
            action = "PLACE"
            confidence = 0.65

    elif max_state >= 4.0:  # GRASPING
        if obj_presence[2] > 0.4:   # RED_BOX
            action = "TAKE"
            confidence = 0.70
        elif obj_presence[3] > 0.4:  # YELLOW_BOX
            action = "TAKE"
            confidence = 0.68
        elif obj_presence[5] > 0.4:  # TOOL
            action = "ACTIVATE"
            confidence = 0.65
        else:
            action = "TAKE"
            confidence = 0.60

    elif max_state >= 3.0:  # CONTACT
        if obj_presence[1] > 0.3:   # MAIN_BOX
            action = "OPEN"
            confidence = 0.62

    elif max_state >= 1.0:  # APPROACHING / NEAR
        action = "IDLE"
        confidence = 0.55

    # Low overall activity → IDLE
    if recent[:, 34:42].mean() < 0.05:
        action = "IDLE"
        confidence = 0.70

    # Next action heuristic (simple sequence prediction)
    next_map = {
        "IDLE": "OPEN", "OPEN": "TAKE", "TAKE": "PLACE",
        "PLACE": "ACTIVATE", "ACTIVATE": "PERFORM",
        "PERFORM": "CLOSE", "CLOSE": "STORE", "STORE": "IDLE",
        "TRANSFER": "CLOSE",
    }
    next_action = next_map.get(action, "IDLE")
    next_confidence = confidence * 0.5

    is_uncertain = confidence < confidence_threshold
    if is_uncertain:
        action = "IDLE"

    return ActionPrediction(
        action=action,
        confidence=confidence,
        next_action=next_action,
        next_confidence=next_confidence,
        is_uncertain=is_uncertain,
    )


# --------------------------------------------------------------------------- #
# Main ActionClassifier
# --------------------------------------------------------------------------- #
class ActionClassifier:
    """
    Classifies the current action from a temporal feature window.

    Usage:
        classifier = ActionClassifier(config)
        prediction = classifier.predict(window_array)  # (30, 64)
    """

    def __init__(self, config):
        self.confidence_threshold = config.action_confidence_min
        self._config = config
        self._model = None
        self._torch = None
        self._active_checkpoint_path = ""

        ModelClass, torch_module = _build_model()
        self._ModelClass = ModelClass
        if ModelClass is not None:
            self._torch = torch_module
            self._try_load_model(ModelClass, config)

        if self._model is None:
            logger.info(
                "Temporal GRU model not available — using heuristic action classifier."
            )

    def is_ready(self) -> bool:
        """True if neural model or heuristic fallback is ready for inference."""
        return True

    @property
    def active_checkpoint_path(self) -> str:
        return self._active_checkpoint_path or "heuristic_fallback"

    def predict(self, window: np.ndarray) -> ActionPrediction:
        """
        Predict action from feature window.

        Args:
            window: np.ndarray of shape (window_size, feature_dim)

        Returns:
            ActionPrediction
        """
        if self._model is not None:
            return self._predict_torch(window)
        return _heuristic_predict(window, self.confidence_threshold)

    def _predict_torch(self, window: np.ndarray) -> ActionPrediction:
        try:
            torch = self._torch
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
            self._model.eval()
            with torch.no_grad():
                action_logits, next_logits = self._model(x)
                action_probs = torch.softmax(action_logits[0], dim=-1).numpy()
                next_probs = torch.softmax(next_logits[0], dim=-1).numpy()

            action_idx = int(np.argmax(action_probs))
            confidence = float(action_probs[action_idx])
            next_idx = int(np.argmax(next_probs))
            next_confidence = float(next_probs[next_idx])

            is_uncertain = confidence < self.confidence_threshold
            action = ACTIONS[action_idx] if not is_uncertain else "IDLE"

            return ActionPrediction(
                action=action,
                confidence=confidence,
                next_action=ACTIONS[next_idx],
                next_confidence=next_confidence,
                is_uncertain=is_uncertain,
            )
        except Exception as exc:
            logger.warning("Torch inference error: %s — falling back to heuristic.", exc)
            return _heuristic_predict(window, self.confidence_threshold)

    @property
    def active_checkpoint_path(self) -> str:
        return self._active_checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Dynamically reload GRU action recognition model weights."""
        if not os.path.exists(checkpoint_path):
            logger.warning("Cannot load GRU checkpoint from non-existent path: %s", checkpoint_path)
            return False
        if self._ModelClass is None or self._torch is None:
            logger.warning("PyTorch or ModelClass not available to load GRU checkpoint.")
            return False
        try:
            torch = self._torch
            f_dim = getattr(self._config, "feature_dim", 64)
            h_dim = getattr(self._config, "hidden_dim", 128)
            n_layers = getattr(self._config, "num_layers", 2)
            model = self._ModelClass(
                input_dim=f_dim,
                hidden_dim=h_dim,
                num_layers=n_layers,
            )
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            self._model = model
            self._active_checkpoint_path = str(Path(checkpoint_path).resolve())
            logger.info("Successfully hot-reloaded GRU checkpoint from %s", checkpoint_path)
            return True
        except Exception as exc:
            logger.error("Failed to hot-reload GRU checkpoint from %s: %s", checkpoint_path, exc)
            return False

    def _try_load_model(self, ModelClass, config) -> None:
        checkpoint_path = config.checkpoint_path
        if not os.path.exists(checkpoint_path):
            logger.info("GRU checkpoint not found at %s — using heuristic.", checkpoint_path)
            return
        try:
            torch = self._torch
            model = ModelClass(
                input_dim=config.feature_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            self._model = model
            self._active_checkpoint_path = str(Path(checkpoint_path).resolve())
            logger.info("GRU checkpoint loaded from %s", checkpoint_path)
        except Exception as exc:
            logger.warning("Failed to load GRU checkpoint: %s", exc)
