# ORBITA Camera Standardization & Demonstration Guidelines

## 1. Overview & Demonstration Objective
To minimize environmental domain shift and maximize object detection and temporal action recognition reliability during the SIH demonstration, the mobile phone camera must follow a standardized mounting geometry and environmental protocol.

While ORBITA is trained with physical data augmentation to tolerate reasonable variations in lighting, angle, and position, adhering to this baseline setup ensures predictable telemetry, stable multi-object tracking, and robust hand-object interaction detection.

---

## 2. Standardized Camera Setup Specifications

```
             ┌─────────────────────────────┐
             │       MOBILE PHONE CAM      │
             │   (Resolution: 1080p / 720p)│
             └──────────────┬──────────────┘
                            │
               Height: 45–60 cm above desk
               Pitch Angle: 45° to 60° downward
                            │
                            ▼
             ┌─────────────────────────────┐
             │     EXPERIMENT WORKSPACE    │
             │   Distance: 60–80 cm        │
             │   Field of View: Full Desk  │
             └─────────────────────────────┘
```

| Parameter | Recommended Value | Permissible Range | Engineering Justification |
| :--- | :--- | :--- | :--- |
| **Mounting Type** | Fixed tripod / clamp mount | Rigid mount only | Eliminates camera motion jitter; ensures consistent velocity tracking. |
| **Mounting Height** | **50 cm** above desk surface | 45 cm – 65 cm | Balances overall scene context (operator torso) with small-object resolution. |
| **Camera Pitch Angle** | **50° downward** | 45° – 60° | Reduces specular desk glare while maintaining 3D depth perception of containers. |
| **Distance to Desk Center** | **70 cm** | 60 cm – 85 cm | Captures full workspace (boxes, tool, sample, operator hands) in one frame. |
| **Orientation** | **Landscape (16:9)** | 16:9 / 4:3 | Aligns with horizontal workspace layout; maximizes lateral tracking area. |
| **Camera Resolution** | **1280 × 720 (720p)** | 720p – 1080p | Optimal sweet spot for Jetson Orin Nano / CPU real-time 30 FPS transmission. |
| **Target Frame Rate** | **30 FPS** | 25 – 30 FPS | Essential for temporal GRU action recognition (30-frame window = 1.0s action). |

---

## 3. Lighting & Optical Environment

1. **Illumination Level**:
   - Recommended: **400–700 Lux** (standard laboratory/office overhead lighting).
   - Avoid direct sharp spotlights that cast hard shadows or specular reflections on the desk surface.
   - Diffuse overhead lighting prevents desk saturation artifacts that cause blue reflection false positives.

2. **Desk Surface & Background**:
   - Non-reflective / matte desk surface is strongly recommended.
   - If the desk has a glossy finish, place a neutral, non-patterned matte experiment mat (grey or dark) under the equipment.

3. **Color Balance & Exposure**:
   - Fix camera auto-exposure and auto-white balance where possible to prevent sudden brightness jumps during operator arm movements.

---

## 4. Physical Object Arrangement

To ensure the deterministic FSM and perception layer can track items throughout the 8-step protocol:

1. **Main Chamber (`MAIN_BOX`)**:
   - Placed in the right-center quadrant of the desk workspace ($x \approx 0.65 - 0.80$).
   - Kept stationary throughout the procedure.

2. **Reagent Containers (`RED_BOX` & `YELLOW_BOX`)**:
   - `RED_BOX`: Left-center quadrant ($x \approx 0.35 - 0.50$, $y \approx 0.20 - 0.35$).
   - `YELLOW_BOX`: Center quadrant ($x \approx 0.55 - 0.70$, $y \approx 0.20 - 0.35$).
   - Maintain at least **10 cm separation** between containers to avoid bounding-box overlap during non-interacting states.

3. **Sample Vial (`SAMPLE`)**:
   - Initial position: Inside or adjacent to `RED_BOX`.
   - Ensure the sample is handled such that fingers do not completely occlude the object for $>15$ consecutive frames.

4. **Experiment Instrument (`TOOL`)**:
   - Initial position: Tool tray or designated placement area adjacent to `RED_BOX`.
   - Held vertically or angled during transfer and manipulation steps.

---

## 5. Live Telemetry Verification Checklist Before Running

Before initiating a verified experiment run on the live dashboard:

- [ ] Camera connected via QR code or `/cam` web app at ~30 FPS.
- [ ] Latency displayed on dashboard is $<100\text{ ms}$ (typically 60–75 ms).
- [ ] All 5 physical items visible in FOV: `MAIN_BOX`, `RED_BOX`, `YELLOW_BOX`, `SAMPLE`, `TOOL`.
- [ ] Operator torso and hands detected without persistent flickering.
- [ ] Hybrid assist saturation threshold confirmed ($S \ge 80$) under current lighting.
