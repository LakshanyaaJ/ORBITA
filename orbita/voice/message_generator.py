"""
ORBITA Message Generator
========================
Template-based message generation for all FSM states.

Responsibilities:
  - Convert FSM status + step info into human-readable voice and HUD text
  - Provide consistent, clear astronaut-appropriate language
  - Format next-step suggestions

All messages are pre-defined templates. No LLM required.
"""

from __future__ import annotations

from typing import Optional

from orbita.reasoning.fsm import FSMState, FSMStatus
from orbita.app.config import ExperimentStep


class MessageGenerator:
    """Generates voice and HUD messages from FSM state."""

    CORRECT_TEMPLATES = [
        "{label} complete. {next_prompt}",
        "Step {step_id} done. {next_prompt}",
        "Confirmed. {next_prompt}",
    ]

    WAITING_TEMPLATE = "Awaiting: {label}."

    ERROR_TEMPLATES = {
        FSMStatus.WRONG_OBJECT: "Incorrect object detected. {recovery}",
        FSMStatus.WRONG_ACTION: "Incorrect action. {recovery}",
        FSMStatus.STEP_SKIPPED: "Step skipped. {recovery}",
        FSMStatus.OUT_OF_SEQUENCE: "Action out of sequence. {recovery}",
        FSMStatus.REPEATED_ACTION: "Step already completed. Please proceed to next step.",
        FSMStatus.UNCERTAIN: "Unable to identify action clearly. Please hold position.",
    }

    def generate_voice(
        self,
        fsm_state: FSMState,
        template_idx: int = 0,
    ) -> str:
        status = fsm_state.status

        if status == FSMStatus.COMPLETED:
            return "Experiment complete. All steps successfully performed. Excellent work."

        if status == FSMStatus.CORRECT:
            step = fsm_state.current_step
            next_step = fsm_state.next_step
            label = step.label if step else "Step"
            step_id = step.id if step else "?"
            next_prompt = f"Next: {next_step.label}." if next_step else "Experiment nearing completion."
            tmpl = self.CORRECT_TEMPLATES[template_idx % len(self.CORRECT_TEMPLATES)]
            return tmpl.format(label=label, step_id=step_id, next_prompt=next_prompt)

        if status == FSMStatus.WAITING:
            step = fsm_state.current_step
            return self.WAITING_TEMPLATE.format(label=step.label if step else "action")

        template = self.ERROR_TEMPLATES.get(status)
        if template:
            recovery = fsm_state.recovery_message or ""
            return template.format(recovery=recovery).strip().rstrip(".")  + "."

        return ""

    def generate_hud(self, fsm_state: FSMState) -> str:
        status = fsm_state.status
        step = fsm_state.current_step

        hud_map = {
            FSMStatus.COMPLETED: "✓ EXPERIMENT COMPLETE",
            FSMStatus.CORRECT: f"✓ COMPLETED → NEXT: {fsm_state.next_step.label.upper() if fsm_state.next_step else 'DONE'}",
            FSMStatus.WAITING: f"STEP {step.id if step else '?'}: {step.label.upper() if step else 'READY'}",
            FSMStatus.WRONG_OBJECT: f"⚠ WRONG OBJECT",
            FSMStatus.WRONG_ACTION: f"⚠ WRONG ACTION",
            FSMStatus.STEP_SKIPPED: f"⚠ STEP SKIPPED",
            FSMStatus.OUT_OF_SEQUENCE: f"⚠ OUT OF SEQUENCE",
            FSMStatus.REPEATED_ACTION: "⚠ ALREADY DONE",
            FSMStatus.UNCERTAIN: "? UNCERTAIN — HOLD",
        }
        return hud_map.get(status, "READY")
