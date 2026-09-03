"""
ORBITA Visual Simulator
========================
Generates synthetic experiment video frames for demonstration
without requiring real hardware or trained models.

Renders:
  - Payload workbench background
  - Colored experiment boxes (MAIN_BOX, RED_BOX, YELLOW_BOX)
  - Animated articulated arm (simplified astronaut)
  - Hand position indicators
  - Dynamic interaction states

Scenarios:
  A — Nominal execution (all steps correct)
  B — Wrong object error (YELLOW grabbed instead of RED at step 2)
  C — Skipped step (step 3 skipped, jump to step 4)

The simulator is deterministic given scenario + frame_number.
It directly injects perception-compatible data so the full pipeline
can be tested without a camera.

NOTE:
  These are 2D synthetic frames, not microgravity simulation.
  True microgravity validation requires future parabolic flight or
  BAS analog data. This simulator exists for proof-of-concept demonstration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np


class Scenario(Enum):
    A = "A"    # Nominal — all steps correct
    B = "B"    # Wrong object error at step 2
    C = "C"    # Step skipped (step 3 skipped)


# Duration in frames for each step in the simulator
STEP_DURATIONS = {
    "A": [90, 90, 90, 90, 90, 120, 90, 90],     # 8 steps × ~3s each
    "B": [90, 90, 90, 90, 90, 120, 90, 90],     # step 2 will use WRONG object
    "C": [90, 90,  0, 90, 90, 120, 90, 90],     # step 3 duration=0 = skipped
}

# Canonical step actions
STEP_ACTIONS = [
    ("OPEN", "MAIN_BOX"),
    ("TAKE", "RED_BOX"),
    ("TAKE", "YELLOW_BOX"),
    ("OPEN", "RED_BOX"),
    ("TAKE", "SAMPLE"),
    ("PERFORM", "SAMPLE"),
    ("CLOSE", "MAIN_BOX"),
    ("STORE", "MAIN_BOX"),
]


@dataclass
class SimFrame:
    """Simulator output for one frame."""
    image: np.ndarray
    timestamp: float
    step_idx: int                  # 0-indexed step
    progress: float                # 0.0 – 1.0 within step
    action_verb: str               # e.g. "TAKE"
    action_object: str             # e.g. "RED_BOX"
    is_error: bool                 # True during scenario B/C error frames
    error_type: Optional[str]      # "WRONG_OBJECT" | "STEP_SKIPPED" | None
    confidence: float              # Simulated HAR confidence


class ExperimentSimulator:
    """
    Generates synthetic experiment video frames.

    Usage:
        sim = ExperimentSimulator(Scenario.A)
        sim.reset()
        while True:
            frame_data = sim.next_frame()
            # Use frame_data.image for display
            # Use frame_data.action_verb + action_object for FSM testing
    """

    WIDTH = 640
    HEIGHT = 480

    def __init__(self, scenario: Scenario = Scenario.A, fps: int = 30):
        self.scenario = scenario
        self.fps = fps
        self._frame_idx: int = 0
        self._global_frame: int = 0
        self._step_idx: int = 0
        self._step_frame: int = 0
        self._scenario_complete: bool = False

        # Build step timeline
        self._durations = STEP_DURATIONS[scenario.value]
        self._step_actions = list(STEP_ACTIONS)

        # For scenario B: override step 2 with wrong object
        if scenario == Scenario.B:
            self._step_actions[1] = ("TAKE", "YELLOW_BOX")  # Wrong! should be RED

        # Pre-calculate object positions (fixed in frame)
        self._object_positions = {
            "MAIN_BOX":   (120, 300),
            "RED_BOX":    (200, 200),
            "YELLOW_BOX": (300, 200),
            "SAMPLE":     (240, 260),
            "TOOL":       (450, 280),
        }

    def reset(self, scenario: Optional[Scenario] = None) -> None:
        if scenario:
            self.scenario = scenario
            self._durations = STEP_DURATIONS[scenario.value]
            self._step_actions = list(STEP_ACTIONS)
            if scenario == Scenario.B:
                self._step_actions[1] = ("TAKE", "YELLOW_BOX")
        self._frame_idx = 0
        self._global_frame = 0
        self._step_idx = 0
        self._step_frame = 0
        self._scenario_complete = False

    def next_frame(self) -> SimFrame:
        """Generate next simulator frame."""
        t = time.time()

        # Advance step
        if self._step_idx < len(self._durations):
            dur = self._durations[self._step_idx]
            if dur == 0:
                # Skipped step — fast-forward
                self._step_idx += 1
                self._step_frame = 0

            if self._step_frame >= max(dur, 1):
                self._step_idx += 1
                self._step_frame = 0

        # Check if complete
        if self._step_idx >= len(self._step_actions):
            self._scenario_complete = True
            self._step_idx = len(self._step_actions) - 1

        step_idx = self._step_idx
        step_dur = max(self._durations[step_idx], 1) if step_idx < len(self._durations) else 1
        progress = min(self._step_frame / step_dur, 1.0)

        verb, obj = self._step_actions[step_idx]
        is_error = (self.scenario == Scenario.B and step_idx == 1)
        error_type = "WRONG_OBJECT" if is_error else None

        # Skipped step detection
        if self.scenario == Scenario.C and step_idx == 2:
            error_type = "STEP_SKIPPED"

        # Generate frame image
        image = self._render_frame(step_idx, progress, verb, obj, is_error)

        frame = SimFrame(
            image=image,
            timestamp=t,
            step_idx=step_idx,
            progress=progress,
            action_verb=verb,
            action_object=obj,
            is_error=is_error,
            error_type=error_type,
            confidence=0.75 if not is_error else 0.70,
        )

        self._step_frame += 1
        self._global_frame += 1
        return frame

    # ----------------------------------------------------------------------- #
    # Rendering
    # ----------------------------------------------------------------------- #
    def _render_frame(
        self,
        step_idx: int,
        progress: float,
        verb: str,
        obj: str,
        is_error: bool,
    ) -> np.ndarray:
        frame = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

        # Background — deep space dark + workbench
        self._draw_background(frame)
        self._draw_workbench(frame)
        self._draw_objects(frame, step_idx, progress, verb, obj)
        self._draw_arm(frame, step_idx, progress, verb, obj)
        self._draw_hud(frame, step_idx, progress, verb, obj, is_error)

        return frame

    def _draw_background(self, frame: np.ndarray) -> None:
        # Deep space gradient
        for y in range(self.HEIGHT):
            r = max(0, 8 - y // 30)
            g = max(0, 12 - y // 25)
            b = max(0, 35 - y // 12)
            frame[y, :] = [b, g, r]

        # Stars
        rng = np.random.default_rng(42)
        stars = rng.integers(0, [self.WIDTH, self.HEIGHT], size=(60, 2))
        for sx, sy in stars:
            brightness = rng.integers(80, 220)
            frame[sy, sx] = [brightness, brightness, brightness]

    def _draw_workbench(self, frame: np.ndarray) -> None:
        # Payload rack (blue-grey box)
        rack_x, rack_y, rack_w, rack_h = 80, 160, 480, 240
        cv2.rectangle(frame, (rack_x, rack_y), (rack_x + rack_w, rack_y + rack_h),
                      (60, 70, 80), -1)
        cv2.rectangle(frame, (rack_x, rack_y), (rack_x + rack_w, rack_y + rack_h),
                      (100, 130, 160), 2)

        # Grid lines on workbench
        for gx in range(rack_x + 40, rack_x + rack_w, 80):
            cv2.line(frame, (gx, rack_y), (gx, rack_y + rack_h), (70, 80, 90), 1)
        for gy in range(rack_y + 40, rack_y + rack_h, 60):
            cv2.line(frame, (rack_x, gy), (rack_x + rack_w, gy), (70, 80, 90), 1)

        # Label
        cv2.putText(frame, "PAYLOAD RACK", (rack_x + 5, rack_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 140, 180), 1, cv2.LINE_AA)

    def _draw_objects(
        self,
        frame: np.ndarray,
        step_idx: int,
        progress: float,
        verb: str,
        obj: str,
    ) -> None:
        # Object colours
        obj_styles = {
            "MAIN_BOX":   ((120, 90, 30), (100, 80, 50), "MAIN"),
            "RED_BOX":    ((0, 0, 200),   (30, 30, 180), "RED"),
            "YELLOW_BOX": ((0, 200, 220), (20, 180, 200), "YEL"),
            "SAMPLE":     ((0, 180, 80),  (20, 160, 60), "SMP"),
            "TOOL":       ((160, 100, 40), (140, 80, 30), "TOOL"),
        }
        for oname, (pos) in self._object_positions.items():
            if oname not in obj_styles:
                continue
            fill, border, label = obj_styles[oname]
            ox, oy = pos
            w, h = (50, 35) if oname == "MAIN_BOX" else (30, 22)

            # Animate picked object (move toward arm)
            if verb.upper() in ("TAKE", "OPEN") and obj.upper() == oname.upper():
                pick_progress = min(progress * 2, 1.0)
                arm_x, arm_y = self._arm_tip_position(step_idx, progress)
                ox = int(ox + (arm_x - ox) * pick_progress)
                oy = int(oy + (arm_y - oy) * pick_progress)

            cv2.rectangle(frame, (ox - w, oy - h), (ox + w, oy + h), fill, -1)
            cv2.rectangle(frame, (ox - w, oy - h), (ox + w, oy + h), border, 2)
            cv2.putText(frame, label, (ox - 12, oy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1, cv2.LINE_AA)

    def _arm_tip_position(self, step_idx: int, progress: float) -> tuple[int, int]:
        """Compute animated arm tip (simulated hand) position."""
        # Base: resting position
        base_x, base_y = 320, 420

        # Target: object being interacted with
        if step_idx < len(STEP_ACTIONS):
            _, obj = STEP_ACTIONS[step_idx]
            target = self._object_positions.get(obj, (300, 240))
        else:
            target = (300, 240)

        tx, ty = target
        # Smooth interpolation: reach at 60% progress, hold to 90%, retract
        if progress < 0.1:
            t = progress / 0.1
        elif progress < 0.6:
            t = 1.0
        elif progress < 0.9:
            t = 1.0
        else:
            t = 1.0 - (progress - 0.9) / 0.1

        # Ease in/out
        t = t * t * (3 - 2 * t)

        x = int(base_x + (tx - base_x) * t)
        y = int(base_y + (ty - base_y) * t)
        return x, y

    def _draw_arm(
        self,
        frame: np.ndarray,
        step_idx: int,
        progress: float,
        verb: str,
        obj: str,
    ) -> None:
        """Draw simplified 2-segment arm."""
        base = (320, 460)
        tip_x, tip_y = self._arm_tip_position(step_idx, progress)
        tip = (tip_x, tip_y)

        # Elbow at midpoint + offset
        elbow = (
            (base[0] + tip[0]) // 2 - 30,
            (base[1] + tip[1]) // 2 - 20,
        )

        # Draw arm segments
        cv2.line(frame, base, elbow, (80, 100, 80), 8, cv2.LINE_AA)
        cv2.line(frame, elbow, tip, (80, 120, 80), 6, cv2.LINE_AA)

        # Shoulder
        cv2.circle(frame, base, 10, (100, 130, 100), -1)
        # Elbow joint
        cv2.circle(frame, elbow, 7, (100, 140, 100), -1)
        # Hand (glove)
        hand_color = (0, 200, 150) if not self._scenario_complete else (0, 255, 100)
        cv2.circle(frame, tip, 12, hand_color, -1)
        cv2.circle(frame, tip, 12, (200, 255, 200), 2)

    def _draw_hud(
        self,
        frame: np.ndarray,
        step_idx: int,
        progress: float,
        verb: str,
        obj: str,
        is_error: bool,
    ) -> None:
        """Overlay HUD text on frame."""
        # Progress bar
        bar_x, bar_y, bar_w = 80, 30, 480
        n_steps = len(STEP_ACTIONS)
        filled = int(bar_w * step_idx / n_steps)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (40, 50, 60), -1)
        color = (0, 200, 100) if not is_error else (0, 80, 220)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 12), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (100, 130, 160), 1)

        # Step label
        step_names = [a[0] + " " + a[1].replace("_", " ") for a in STEP_ACTIONS]
        step_label = step_names[step_idx] if step_idx < len(step_names) else "COMPLETE"
        cv2.putText(frame, f"STEP {step_idx + 1}: {step_label}", (bar_x, bar_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 1, cv2.LINE_AA)

        # Scenario badge
        badge_color = (0, 80, 200) if not is_error else (0, 40, 200)
        cv2.putText(frame, f"SIM:{self.scenario.value}", (self.WIDTH - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_color, 1, cv2.LINE_AA)

        # ORBITA watermark
        cv2.putText(frame, "ORBITA", (self.WIDTH // 2 - 35, self.HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 90, 130), 1, cv2.LINE_AA)

        # Error flash
        if is_error:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.WIDTH, self.HEIGHT), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.putText(frame, "! PROCEDURE ERROR", (80, self.HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2, cv2.LINE_AA)
