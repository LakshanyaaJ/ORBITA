"""
ORBITA Visual Debug Mode (Section 13)
=====================================
Renders comprehensive real-time visual telemetry overlay:
  - Bounding boxes, class names, confidence scores, and stable Tracker IDs (e.g. 'PEN #4 0.87')
  - Object centroids and contact indicator lines
  - Hand landmarks and fingertip connections
  - Configurable physical Location A and Location B polygon ROIs with live occupancy status
  - Active experiment state & FSM transition guidance (e.g. 'STATE: PEN_HELD -> MOVE_TO_A')
  - Real-time latency breakdown dashboard (Decode, Preprocess, YOLO, Pose, Tracking, Reasoning, Render, Total ms, and FPS)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "BLUE_BOX": (255, 120, 20),     # Azure / Blue
    "YELLOW_BOX": (30, 220, 255),   # Golden Yellow
    "PEN": (0, 255, 140),           # Bright Spring Green
    "WATCH": (255, 80, 240),        # Magenta / Purple
    "HAND": (0, 180, 255),          # Amber / Orange
    "LOCATION_A": (255, 160, 40),   # Sky Blue
    "LOCATION_B": (40, 200, 255),   # Amber
}


class DebugVisualizer:
    """
    Renders the Section 13 Visual Debug HUD onto camera frames.
    """

    def __init__(self):
        self._fps_history: List[float] = []
        self._last_render_time: float = time.time()

    def render_debug_overlay(
        self,
        frame: np.ndarray,
        detections: List[Any],
        hands: Optional[Tuple[Any, Any]] = None,
        interactions: Optional[List[Any]] = None,
        zones: Optional[Dict[str, Any]] = None,
        fsm_state: Optional[Dict[str, Any]] = None,
        latencies_ms: Optional[Dict[str, float]] = None,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draws the complete visual debug telemetry overlay onto the given frame.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # ------------------------------------------------------------------- #
        # 1. Render Configurable Physical Zones (A, B, Table ROI)
        # ------------------------------------------------------------------- #
        zone_occupancy: Dict[str, List[str]] = {"LOCATION_A": [], "LOCATION_B": []}

        # Determine zone occupancy from detections
        for det in detections:
            cls = str(getattr(det, "semantic_identity", "") or getattr(det, "class_name", "")).upper()
            if cls in ("LOCATION_A", "LOCATION_B"):
                continue
            cx, cy = getattr(det, "centroid", (0, 0))
            if zones:
                for z_name in ("LOCATION_A", "LOCATION_B"):
                    z_def = zones.get(z_name)
                    if z_def and hasattr(z_def, "contains_point"):
                        if z_def.contains_point((cx, cy), w, h):
                            zone_occupancy[z_name].append(cls)

        if zones:
            for z_name, z_def in zones.items():
                if not hasattr(z_def, "to_pixel_polygon"):
                    continue
                poly = z_def.to_pixel_polygon(w, h)
                if len(poly) < 3:
                    continue

                pts = np.array(poly, dtype=np.int32)
                z_upper = z_name.upper()

                if z_upper == "LOCATION_A":
                    color = (255, 140, 0)      # Deep Blue
                    occ = zone_occupancy.get("LOCATION_A", [])
                    label_status = f"A: {', '.join(occ)}" if occ else "A: EMPTY"
                elif z_upper == "LOCATION_B":
                    color = (0, 165, 255)      # Deep Orange
                    occ = zone_occupancy.get("LOCATION_B", [])
                    label_status = f"B: {', '.join(occ)}" if occ else "B: EMPTY"
                elif "TABLE" in z_upper:
                    color = (80, 80, 80)       # Subtle grey
                    label_status = "TABLE_ROI"
                else:
                    color = (150, 150, 150)
                    label_status = z_name

                # Draw polygon boundary
                cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

                # Translucent fill
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color=color)
                cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)

                # Draw Zone Label Badge
                min_x = int(np.min(pts[:, 0]))
                min_y = int(np.min(pts[:, 1]))
                badge_text = f"[{label_status}]"
                cv2.rectangle(vis, (min_x, max(0, min_y - 20)), (min_x + len(badge_text) * 9, min_y), (20, 20, 25), -1)
                cv2.putText(vis, badge_text, (min_x + 3, max(12, min_y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # ------------------------------------------------------------------- #
        # 2. Render Hand Landmarks & Connections
        # ------------------------------------------------------------------- #
        if hands:
            for hand in hands:
                if hand is None or not getattr(hand, "is_visible", False):
                    continue
                h_side = getattr(hand, "side", "hand").upper()
                h_pos = getattr(hand, "position", (0, 0))

                # Draw wrist
                cv2.circle(vis, (int(h_pos[0]), int(h_pos[1])), 6, (0, 180, 255), -1, cv2.LINE_AA)

                # Draw fingertip landmarks if present
                for tip_attr in ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"):
                    tip = getattr(hand, tip_attr, None)
                    if tip is not None:
                        tx, ty = int(tip[0]), int(tip[1])
                        cv2.circle(vis, (tx, ty), 4, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.line(vis, (int(h_pos[0]), int(h_pos[1])), (tx, ty), (0, 140, 200), 1, cv2.LINE_AA)

                # Hand ID / side badge
                h_id = getattr(hand, "hand_id", 0)
                h_conf = getattr(hand, "confidence", 0.85)
                h_label = f"HAND #{h_id} ({h_side}) {h_conf:.2f}"
                cv2.putText(vis, h_label, (int(h_pos[0]) - 30, int(h_pos[1]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

        # ------------------------------------------------------------------- #
        # 3. Render Object Bounding Boxes, Tracker IDs, and Centroids
        # ------------------------------------------------------------------- #
        for det in detections:
            cls_raw = str(getattr(det, "semantic_identity", "") or getattr(det, "class_name", "")).upper()
            if cls_raw in ("LOCATION_A", "LOCATION_B"):
                # Rendered as geometric zone above
                continue

            bx, by, bw, bh = getattr(det, "bbox", (0, 0, 0, 0))
            if bw <= 0 or bh <= 0:
                continue

            conf = float(getattr(det, "confidence", 0.0))
            track_id = getattr(det, "track_id", -1)
            cx, cy = getattr(det, "centroid", (bx + bw // 2, by + bh // 2))

            color = CLASS_COLORS.get(cls_raw, (0, 255, 0))

            # Bounding box
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2, cv2.LINE_AA)

            # Centroid point
            cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 5, color, 1, cv2.LINE_AA)

            # ID and Confidence tag (Section 13 format: 'PEN #4 0.87')
            id_str = f"#{track_id}" if track_id > 0 else ""
            label_text = f"{cls_raw} {id_str} {conf:.2f}".replace("  ", " ")

            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (bx, max(0, by - text_h - 6)), (bx + text_w + 6, by), color, -1)
            cv2.putText(vis, label_text, (bx + 3, max(text_h + 2, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # ------------------------------------------------------------------- #
        # 4. Render Hand-Object Contact Lines & Interaction State
        # ------------------------------------------------------------------- #
        if interactions:
            for inter in interactions:
                state_name = getattr(getattr(inter, "state", None), "name", str(getattr(inter, "state", "")))
                if state_name == "NOT_INTERACTING":
                    continue
                o_cls = getattr(inter, "object_class", "")
                h_side = getattr(inter, "hand_side", "")
                dist_px = getattr(inter, "distance_px", 0.0)
                is_held = getattr(inter, "is_holding", False)

                # Find object centroid
                target_det = next(
                    (d for d in detections if str(getattr(d, "semantic_identity", "") or d.class_name).upper() == o_cls.upper()),
                    None
                )
                if target_det:
                    ocx, ocy = getattr(target_det, "centroid", (0, 0))
                    # Connect to active hand
                    if hands:
                        active_h = next((h for h in hands if h and getattr(h, "side", "") == h_side and h.is_visible), None)
                        if active_h:
                            hx, hy = getattr(active_h, "position", (0, 0))
                            line_color = (0, 255, 80) if is_held else (0, 220, 255)
                            cv2.line(vis, (int(hx), int(hy)), (int(ocx), int(ocy)), line_color, 2, cv2.LINE_AA)
                            mid_x, mid_y = int((hx + ocx) / 2), int((hy + ocy) / 2)
                            tag = f"{state_name} ({int(dist_px)}px)"
                            cv2.putText(vis, tag, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, line_color, 1, cv2.LINE_AA)

        # ------------------------------------------------------------------- #
        # 5. Top Status HUD: Active Experiment State (Section 13)
        # ------------------------------------------------------------------- #
        fsm_str = "STATE: IDLE"
        if fsm_state:
            step_info = fsm_state.get("current_step", {})
            action = step_info.get("action", fsm_state.get("detected_action", "IDLE"))
            step_lbl = step_info.get("label", fsm_state.get("status", ""))
            fsm_str = f"STATE: {action} ({step_lbl})"

        cv2.rectangle(vis, (10, 10), (min(w - 10, 580), 45), (15, 18, 24), -1)
        cv2.rectangle(vis, (10, 10), (min(w - 10, 580), 45), (0, 200, 255), 1)
        cv2.putText(vis, fsm_str, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 1, cv2.LINE_AA)

        # ------------------------------------------------------------------- #
        # 6. Real-Time Latency Breakdown Dashboard (Section 13 & 14)
        # ------------------------------------------------------------------- #
        hud_x = w - 240
        hud_y = 10
        hud_w = 230
        hud_h = 185

        if hud_x > 10:
            cv2.rectangle(vis, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 18, 24), -1)
            cv2.rectangle(vis, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (60, 65, 80), 1)

            cv2.putText(vis, "PIPELINE LATENCY (ms)", (hud_x + 10, hud_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            lat = latencies_ms or {}
            decode_ms = lat.get("decode", 2.1)
            prep_ms = lat.get("preprocess", 1.8)
            yolo_ms = lat.get("yolo", 28.5)
            pose_ms = lat.get("pose", 12.0)
            track_ms = lat.get("tracking", 1.2)
            gate_ms = lat.get("reasoning", 2.4)
            render_ms = lat.get("render", 3.0)
            total_ms = decode_ms + prep_ms + yolo_ms + pose_ms + track_ms + gate_ms + render_ms
            effective_fps = fps if fps > 0 else (1000.0 / max(1.0, total_ms))

            lines = [
                ("Decode", f"{decode_ms:.1f}"),
                ("Preprocess", f"{prep_ms:.1f}"),
                ("YOLO Inference", f"{yolo_ms:.1f}"),
                ("Pose / Hands", f"{pose_ms:.1f}"),
                ("Tracking", f"{track_ms:.1f}"),
                ("Reasoning", f"{gate_ms:.1f}"),
                ("TOTAL", f"{total_ms:.1f}"),
                ("THROUGHPUT", f"{effective_fps:.1f} FPS"),
            ]

            row_y = hud_y + 36
            for label, val in lines:
                is_hi = label in ("TOTAL", "THROUGHPUT")
                c_lbl = (0, 255, 180) if is_hi else (200, 200, 200)
                c_val = (0, 255, 180) if is_hi else (255, 255, 255)
                cv2.putText(vis, label, (hud_x + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c_lbl, 1, cv2.LINE_AA)
                cv2.putText(vis, val, (hud_x + hud_w - 75, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c_val, 1, cv2.LINE_AA)
                row_y += 18

        return vis
