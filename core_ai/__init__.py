"""
ORBITA — Offline AI Experiment Copilot
SIH 26174 — AI Human Activity Recognition for On-board BAS Experiments

Architecture:
  CAMERA → PERCEPTION → INTERACTION → HAR → FSM+RULES → VOICE/LOG → DASHBOARD
"""

import sys

__version__ = "1.0.0"
__author__ = "ORBITA Team"

# Maintain full backwards compatibility for any modules or tests referencing `orbita`
sys.modules.setdefault("orbita", sys.modules[__name__])
