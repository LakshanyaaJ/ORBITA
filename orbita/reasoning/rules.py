"""
ORBITA Rule Engine
==================
Explicit, testable, deterministic rules that run alongside the FSM.

Purpose: Provide additional classification signals beyond pure FSM
state matching. Rules are evaluated on every detected action and
can override or supplement FSM decision.

All rules are explicit IF-THEN statements with no ML component.
Each rule produces a RuleMatch that contributes to the final
ValidationResult.

Rules implemented:
  R01: WRONG_OBJECT          — right action class, wrong object
  R02: STEP_SKIPPED          — jump forward in sequence detected
  R03: OUT_OF_SEQUENCE       — past-step action detected
  R04: REPEATED_ACTION       — same step submitted twice
  R05: UNCERTAIN_THRESHOLD   — confidence too low to validate
  R06: CRITICAL_SAFETY       — forbidden action (undefined in experiment)
  R07: TIMEOUT               — step exceeded allowed time (flag only)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from orbita.app.config import ExperimentStep

logger = logging.getLogger(__name__)


class RuleResult(Enum):
    PASS = auto()
    FAIL = auto()
    WARN = auto()


@dataclass
class RuleMatch:
    rule_id: str
    result: RuleResult
    message: str
    error_type: Optional[str] = None
    recovery: Optional[str] = None


def evaluate_rules(
    detected_action: str,
    detected_object: str,
    confidence: float,
    current_step: ExperimentStep,
    all_steps: list[ExperimentStep],
    completed_step_ids: list[int],
    step_start_time: float,
    confidence_threshold: float = 0.55,
) -> list[RuleMatch]:
    """
    Evaluate all rules against the current detection.

    Returns:
        List of RuleMatch objects. Empty if all rules pass.
    """
    results: list[RuleMatch] = []

    # R05: Confidence threshold
    if confidence < confidence_threshold:
        results.append(RuleMatch(
            rule_id="R05",
            result=RuleResult.WARN,
            message=f"Action confidence {confidence:.2f} below threshold {confidence_threshold:.2f}.",
            error_type="UNCERTAIN",
            recovery="Unable to identify action clearly. Please hold your position.",
        ))
        return results  # No further rules if uncertain

    # Skip IDLE
    if detected_action in ("IDLE", "", "UNKNOWN"):
        return results

    expected_verb, expected_obj = _split(current_step.action)
    full_expected = current_step.action.upper()

    # Build full detected label
    full_detected = f"{detected_action}_{detected_object}".upper() if detected_object else detected_action.upper()
    detected_verb = detected_action.upper()
    detected_obj = detected_object.upper()

    # R04: Repeated action
    if current_step.id in completed_step_ids:
        results.append(RuleMatch(
            rule_id="R04",
            result=RuleResult.WARN,
            message=f"Step {current_step.id} ({current_step.action}) already completed.",
            error_type="REPEATED_ACTION",
            recovery=f"Step {current_step.id} is already done. Proceed to the next step.",
        ))
        return results

    # R01: Wrong object (verb matches, object does not)
    if (detected_verb == expected_verb
            and detected_obj
            and expected_obj
            and detected_obj != expected_obj
            and detected_obj not in expected_obj):
        results.append(RuleMatch(
            rule_id="R01",
            result=RuleResult.FAIL,
            message=f"Wrong object: expected '{expected_obj}', got '{detected_obj}'.",
            error_type="WRONG_OBJECT",
            recovery=f"Incorrect object. Please use the {expected_obj.replace('_', ' ').lower()}.",
        ))

    # R02: Step skipped (action matches future step)
    if not results:
        current_idx = next(
            (i for i, s in enumerate(all_steps) if s.id == current_step.id), 0
        )
        for future_step in all_steps[current_idx + 1:]:
            fv, fo = _split(future_step.action)
            if detected_verb == fv and (not fo or detected_obj == fo or detected_obj in future_step.action):
                results.append(RuleMatch(
                    rule_id="R02",
                    result=RuleResult.FAIL,
                    message=f"Step skipped: detected step {future_step.id} before completing step {current_step.id}.",
                    error_type="STEP_SKIPPED",
                    recovery=f"Please complete step {current_step.id}: {current_step.label} first.",
                ))
                break

    # R03: Out of sequence (action matches past step)
    if not results:
        current_idx = next(
            (i for i, s in enumerate(all_steps) if s.id == current_step.id), 0
        )
        for past_step in all_steps[:current_idx]:
            pv, po = _split(past_step.action)
            if detected_verb == pv and (not po or detected_obj == po or detected_obj in past_step.action):
                results.append(RuleMatch(
                    rule_id="R03",
                    result=RuleResult.WARN,
                    message=f"Out-of-sequence: action matches completed step {past_step.id}.",
                    error_type="OUT_OF_SEQUENCE",
                    recovery=f"Step {past_step.id} is done. Please proceed with step {current_step.id}: {current_step.label}.",
                ))
                break

    # R06: Critical safety — action not in experiment definition at all
    all_actions = {s.action.split("_")[0] for s in all_steps}
    if detected_verb not in all_actions and detected_verb not in ("IDLE", "UNKNOWN"):
        results.append(RuleMatch(
            rule_id="R06",
            result=RuleResult.WARN,
            message=f"Action '{detected_action}' not defined in experiment procedure.",
            error_type="UNDEFINED_ACTION",
            recovery="This action is not part of the experiment. Please refer to the procedure.",
        ))

    # R07: Timeout
    elapsed = time.time() - step_start_time
    if elapsed > current_step.timeout_seconds:
        results.append(RuleMatch(
            rule_id="R07",
            result=RuleResult.WARN,
            message=f"Step {current_step.id} timeout: {elapsed:.1f}s > {current_step.timeout_seconds}s.",
            error_type="TIMEOUT",
            recovery=f"Step is taking longer than expected. Continue with: {current_step.label}.",
        ))

    return results


def _split(action: str) -> tuple[str, str]:
    parts = action.upper().split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")
