"""Serializable, answer-key-free state for a team run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


POLICY_DEFAULTS = {"max_policy_actions": 14, "max_item_cost_usd": 0.75,
                   "voc_epsilon": 0.002, "hint_audit": "structural", "max_calls": 40}


@dataclass
class CheckEvidence:
    status: str = "unavailable"  # passed | failed | partial | unavailable | error
    source: str = "none"
    verified_claims: list[str] = field(default_factory=list)
    failed_claims: list[str] = field(default_factory=list)
    full_answer_verified: bool = False
    details: str = ""


@dataclass
class Candidate:
    id: str
    member: str
    model_key: str
    text: str
    answer: str | None
    certificate: str
    phase: str
    parent_id: str | None = None
    evidence: CheckEvidence = field(default_factory=CheckEvidence)
    valid: bool = True
    error: str | None = None


@dataclass
class ZoneEstimate:
    label: str
    probabilities: dict[str, float]
    uncertain: bool
    calibrated: bool
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamConfig:
    roster: str = "sat"
    condition: str = "full"
    seed: int = 0
    members: list[str] | None = None
    checker: str | None = None
    default_judge: str | None = None
    offline_artifacts: dict[str, Any] | str | None = None
    max_strategies: int = 2
    fresh_samples: int = 1
    max_hint_levels: int = 2
    tools_enabled: bool = True
    # No automatic image pull: a local image is an explicit prerequisite.
    sandbox_image: str = "python:3.11-alpine"
    max_calls: int | None = None
    strategy_override: list[str] | None = None
    # --- zone-aware value-of-computation router (the main MathMage system) ---
    router: str = "posterior"          # "posterior" (label-free posterior + VOC policy) | "rule" (operational rules)
    mu: float | None = None            # None = frozen development choice, or 0 for uncalibrated mock runs
    voc_epsilon: float | None = None
    max_policy_actions: int | None = None
    max_item_cost_usd: float | None = None
    hint_audit: str | None = None       # None uses frozen policy_config, then POLICY_DEFAULTS
    tool_candidate: bool = True        # the independent program's answer joins the candidate pool
    sandbox: str = "auto"              # "auto" | "docker" | "local"

    def __post_init__(self):
        if self.condition not in {"full", *(f"A{i}" for i in range(1, 9)), "C1", "C2", "R1"}:
            raise ValueError(f"unknown condition: {self.condition}")
        required_integer_limits = (self.max_strategies, self.fresh_samples, self.max_hint_levels)
        optional_integer_limits = (self.max_calls, self.max_policy_actions)
        if (any(isinstance(value, bool) or not isinstance(value, int) for value in required_integer_limits)
                or any(value is not None and (isinstance(value, bool) or not isinstance(value, int))
                       for value in optional_integer_limits)):
            raise ValueError("call, strategy, hint and action limits must be integers")
        if self.max_strategies < 0 or self.fresh_samples < 0 or (self.max_calls is not None and self.max_calls < 1):
            raise ValueError("call and strategy limits must be nonnegative (max_calls positive)")
        if not 1 <= self.max_hint_levels <= 4:
            raise ValueError("max_hint_levels must be between 1 and 4")
        if self.router not in {"posterior", "rule"}:
            raise ValueError("router must be 'posterior' or 'rule'")
        if self.condition == "R1" or self.strategy_override is not None:
            # R1 is the operational-rule router; forced mechanism replays (Z3, Z6, offline strategy
            # validation) execute named strategy templates, which live on the rule path.
            self.router = "rule"
        if self.hint_audit not in {None, "structural", "residual"}:
            raise ValueError("hint_audit must be 'structural' or 'residual'")
        if self.sandbox not in {"auto", "docker", "local"}:
            raise ValueError("sandbox must be 'auto', 'docker', or 'local'")
        finite_limits = (self.max_item_cost_usd, self.voc_epsilon)
        if self.mu is not None:
            finite_limits += (self.mu,)
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value))
               for value in finite_limits):
            raise ValueError("policy budgets must be finite numbers")
        if ((self.mu is not None and self.mu < 0)
                or (self.max_item_cost_usd is not None and self.max_item_cost_usd <= 0)
                or (self.voc_epsilon is not None and self.voc_epsilon < 0)
                or (self.max_policy_actions is not None and self.max_policy_actions < 0)):
            raise ValueError("policy limits must be nonnegative")


@dataclass
class TeamResult:
    item_id: str
    condition: str
    seed: int
    candidates: list[Candidate]
    stage0: list[str]
    final_id: str | None
    zone: ZoneEstimate
    checkpoint: dict[str, Any]
    meta: dict[str, Any]
    trace: list[dict[str, Any]]
    costs: dict[str, Any]

    @property
    def final(self) -> Candidate | None:
        return next((c for c in self.candidates if c.id == self.final_id), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
