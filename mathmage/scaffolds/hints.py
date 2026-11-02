"""Agent-time leakage screening with no access to reference answers.

Passing these checks is not an empirical leakage measurement: that requires a
separate solver panel and is deliberately recorded as unmeasured here.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


HINT_LEVELS = ("strategic", "decomposition", "socratic", "diagnosis")


@dataclass
class HintAudit:
    accepted: bool
    structural_pass: bool
    residual_work_pass: bool = False
    reasons: list[str] = field(default_factory=list)
    residual_steps: list[str] = field(default_factory=list)
    empirical_leakage: str = "unmeasured"

    def to_dict(self):
        return asdict(self)


def structural_audit(text: str) -> HintAudit:
    reasons = []
    if not text.strip():
        reasons.append("empty hint")
    if len(text.split()) > 120:
        reasons.append("hint exceeds 120 words")
    if re.search(r"\\(?:boxed|fbox)\s*\{|\b(?:final answer|answer is|answer equals)\b", text, re.I):
        reasons.append("explicit answer marker")
    if "```" in text:
        reasons.append("executable or complete-solution block")
    return HintAudit(False, not reasons, reasons=reasons)


def residual_audit(text: str, verdict: dict) -> HintAudit:
    result = structural_audit(text)
    steps = verdict.get("remaining_nontrivial_steps", [])
    steps = [str(step).strip() for step in steps if str(step).strip()] if isinstance(steps, list) else []
    result.residual_steps = steps
    result.residual_work_pass = (verdict.get("reveals_answer") is False
                                 and verdict.get("one_step_completion") is False
                                 and len(set(steps)) >= 2)
    if not result.residual_work_pass:
        result.reasons.append("independent audit did not establish at least two nontrivial remaining steps")
    result.accepted = result.structural_pass and result.residual_work_pass
    return result

