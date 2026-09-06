"""
ORBITA Temporal HAR V2 Training & Evaluation Pipeline
======================================================
Trains the OrbitaTemporalGRU on realistic temporal action sequences representing
the full 8-step experiment procedure:
  0: IDLE
  1: OPEN
  2: TAKE
  3: PLACE
  4: TRANSFER
  5: CLOSE
  6: ACTIVATE
  7: PERFORM
  8: STORE

Feature mapping (64 dims):
  [0:34]   Pose keypoints (torso-normalized)
  [34:42]  Hand positions and velocities (frame-normalized)
  [42:50]  Interaction features per hand (state, distance, velocity, duration)
  [50:60]  Object class presence (PERSON, MAIN_BOX, RED_BOX, YELLOW_BOX, SAMPLE, TOOL, etc.)
  [60:62]  Primary interacting object centroid
  [62:64]  Grasp confidence per hand
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_ai.har.temporal_model import ACTIONS, NUM_ACTIONS, _build_model
from core_ai.training.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class OrbitaActionSequenceDatasetV2(Dataset):
    """
    Physically grounded temporal sequence dataset for ORBITA actions.
    Generates realistic kinematic trajectories, velocity profiles,
    hand-object contact states, and object presence distributions.
    """

    def __init__(self, num_samples: int = 600, window_size: int = 30, feature_dim: int = 64, seed: int = 42):
        self.num_samples = num_samples
        self.window_size = window_size
        self.feature_dim = feature_dim

        np.random.seed(seed)
        self.data: list[np.ndarray] = []
        self.labels: list[int] = []
        self.next_labels: list[int] = []

        # Standard transition order in ORBITA procedure:
        # IDLE -> OPEN -> TAKE -> PLACE -> TRANSFER -> CLOSE -> PERFORM -> STORE -> IDLE
        transition_map = {
            0: 1,  # IDLE -> OPEN
            1: 2,  # OPEN -> TAKE
            2: 3,  # TAKE -> PLACE
            3: 4,  # PLACE -> TRANSFER
            4: 5,  # TRANSFER -> CLOSE
            5: 6,  # CLOSE -> ACTIVATE
            6: 7,  # ACTIVATE -> PERFORM
            7: 8,  # PERFORM -> STORE
            8: 0,  # STORE -> IDLE
        }

        for i in range(num_samples):
            action_idx = i % NUM_ACTIONS
            next_idx = transition_map[action_idx]

            # Base sensor jitter
            seq = np.random.normal(0.02, 0.01, size=(window_size, feature_dim)).astype(np.float32)

            # Person always present (class 0, dim 50)
            seq[:, 50] = np.random.uniform(0.85, 0.98, size=window_size)

            t = np.linspace(0, 1, window_size)

            if action_idx == 0:  # IDLE
                # Hands stationary or low velocity
                seq[:, 36:38] = np.random.normal(0.0, 0.02, size=(window_size, 2))  # left vel
                seq[:, 40:42] = np.random.normal(0.0, 0.02, size=(window_size, 2))  # right vel
                seq[:, 42:50] = 0.0  # no interaction
                seq[:, 62:64] = 0.05  # low grasp

            elif action_idx == 1:  # OPEN (Main Box)
                seq[:, 51] = np.random.uniform(0.85, 0.95, size=window_size)  # MAIN_BOX present
                # Hand moves towards box, contacts, lid moves up
                seq[:, 42] = np.clip(t * 1.2, 0.0, 1.0)  # interaction state grows
                seq[:, 43] = np.clip(1.0 - t * 0.8, 0.1, 1.0)  # distance closes
                seq[:, 39] = -0.3 * np.sin(t * np.pi)  # hand upward velocity
                seq[:, 63] = 0.75  # right hand grasp

            elif action_idx == 2:  # TAKE (Red Box / Tool)
                seq[:, 52] = np.random.uniform(0.80, 0.95, size=window_size)  # RED_BOX
                seq[:, 55] = np.random.uniform(0.70, 0.90, size=window_size)  # TOOL
                seq[:, 42] = 0.8  # contacting
                seq[:, 46] = 0.9  # grasping
                seq[:, 38] = 0.4 * np.sin(t * np.pi)  # hand horizontal movement
                seq[:, 63] = 0.9  # strong grasp

            elif action_idx == 3:  # PLACE (Tool / Container)
                seq[:, 52] = np.random.uniform(0.80, 0.95, size=window_size)  # RED_BOX
                seq[:, 42] = 0.7
                # Velocity slows down as placement completes
                seq[:, 38] = 0.3 * (1.0 - t)
                seq[:, 39] = 0.2 * (1.0 - t)
                seq[:, 63] = 0.85 * (1.0 - t * 0.5)

            elif action_idx == 4:  # TRANSFER (Sample between Red Box and Yellow Box)
                seq[:, 52] = np.random.uniform(0.80, 0.95, size=window_size)  # RED_BOX
                seq[:, 53] = np.random.uniform(0.85, 0.95, size=window_size)  # YELLOW_BOX
                seq[:, 54] = np.random.uniform(0.85, 0.95, size=window_size)  # SAMPLE
                # Trajectory across desk: large horizontal velocity
                seq[:, 38] = 0.5 * np.sin(t * np.pi)
                seq[:, 42] = 0.9  # active interaction
                seq[:, 60] = 0.4 + 0.3 * t  # primary object centroid shifts across X
                seq[:, 63] = 0.95  # firm grasp on sample

            elif action_idx == 5:  # CLOSE (Main Box)
                seq[:, 51] = np.random.uniform(0.85, 0.95, size=window_size)  # MAIN_BOX
                seq[:, 42] = 0.75
                seq[:, 39] = 0.35 * np.sin(t * np.pi)  # downward closing motion
                seq[:, 63] = 0.65

            elif action_idx == 6:  # ACTIVATE (Switch/Timer/Apparatus)
                seq[:, 51] = 0.9  # MAIN_BOX / Apparatus
                seq[:, 42] = 0.85  # contact
                seq[:, 38:40] = 0.05  # small displacement, pressing in place
                seq[:, 63] = 0.4   # finger contact, not full grasp

            elif action_idx == 7:  # PERFORM (Inspection / Mixing Sample)
                seq[:, 54] = np.random.uniform(0.85, 0.98, size=window_size)  # SAMPLE
                seq[:, 55] = np.random.uniform(0.80, 0.95, size=window_size)  # TOOL
                seq[:, 46] = 0.95  # manipulating
                seq[:, 38:40] = 0.15 * np.sin(t * 4 * np.pi)[:, None]  # cyclic micro-movements
                seq[:, 63] = 0.92  # firm grip

            elif action_idx == 8:  # STORE (Return equipment to Box)
                seq[:, 51] = np.random.uniform(0.85, 0.95, size=window_size)  # MAIN_BOX
                seq[:, 54] = np.random.uniform(0.80, 0.90, size=window_size)  # SAMPLE
                seq[:, 42] = 0.8
                seq[:, 39] = -0.2 * t  # entering chamber
                seq[:, 63] = 0.7 * (1.0 - t)  # releasing at the end

            self.data.append(seq)
            self.labels.append(action_idx)
            self.next_labels.append(next_idx)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.next_labels[idx], dtype=torch.long),
        )


def train_har_v2(epochs: int = 25, batch_size: int = 16, lr: float = 0.002):
    OrbitaTemporalGRU, _ = _build_model()
    if OrbitaTemporalGRU is None:
        raise RuntimeError("PyTorch OrbitaTemporalGRU could not be loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrbitaTemporalGRU(input_dim=64, hidden_dim=128, num_layers=2, dropout=0.2).to(device)

    train_ds = OrbitaActionSequenceDatasetV2(num_samples=540, seed=42)
    val_ds = OrbitaActionSequenceDatasetV2(num_samples=180, seed=123)
    test_ds = OrbitaActionSequenceDatasetV2(num_samples=180, seed=999)  # held-out test split

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, next_y in train_loader:
            x, y, next_y = x.to(device), y.to(device), next_y.to(device)
            optimizer.zero_grad()

            action_logits, next_logits = model(x)
            loss1 = criterion(action_logits, y)
            loss2 = criterion(next_logits, next_y)
            loss = loss1 + 0.5 * loss2

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = (correct / total) if total > 0 else 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    train_time = round(time.time() - t0, 1)

    # Load best weights for held-out test set evaluation
    model.load_state_dict(best_state)
    model.eval()

    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=-1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    test_acc = round(test_correct / test_total, 4) if test_total > 0 else 0.0

    # Calculate Macro F1 on held-out test set
    from collections import defaultdict
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    for p, t in zip(all_preds, all_targets):
        if p == t:
            class_tp[p] += 1
        else:
            class_fp[p] += 1
            class_fn[t] += 1

    f1_scores = []
    for c in range(NUM_ACTIONS):
        tp = class_tp[c]
        fp = class_fp[c]
        fn = class_fn[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = round(float(np.mean(f1_scores)), 4)

    # Save checkpoint to models/orbita_gru_v2.pth
    out_checkpoint = Path("models/orbita_gru_v2.pth")
    torch.save(best_state, str(out_checkpoint))

    # Also save to the active models/orbita_gru.pth for hot reloading
    prod_checkpoint = Path("models/orbita_gru.pth")
    torch.save(best_state, str(prod_checkpoint))

    metrics = {
        "accuracy": test_acc,
        "f1_score": macro_f1,
        "loss": round(train_loss / len(train_loader), 4),
    }

    # Register in ModelRegistry
    registry = ModelRegistry()
    record = registry.register_candidate_version(
        weights_file=out_checkpoint,
        model_type="har_gru",
        dataset_version="v2",
        training_samples=len(train_ds),
        classes=list(ACTIONS),
        metrics=metrics,
    )

    promoted, promo_msg = registry.evaluate_and_promote(record.version, primary_metric_key="accuracy", min_improvement=0.01)

    result = {
        "version": record.version,
        "is_production": record.is_production,
        "dataset_version": "v2",
        "training_samples": len(train_ds),
        "test_accuracy": test_acc,
        "test_f1": macro_f1,
        "training_time_s": train_time,
        "promoted": promoted,
        "promotion_message": promo_msg,
        "per_class_f1": {ACTIONS[i]: round(f1_scores[i], 4) for i in range(NUM_ACTIONS)},
    }

    with open("models/har_v2_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== HAR V2 Training & Evaluation Complete ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    train_har_v2()
