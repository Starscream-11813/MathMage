"""Zone-aware value-of-computation policy (research plan, Algorithm 1).

At every step the policy scores each feasible action by its myopic value of
computation (VOC): the expected increase in the probability that the team's
current best answer is correct, computed from the label-free posterior
(:mod:`mathmage.team.posterior`). It executes the action with the largest VOC per
expected dollar and stops when that ratio falls below a threshold ``mu``, when
no action has value, or when the item budget is spent.

Because actions are chosen by VOC/cost, which does not depend on ``mu``, the
sequence of actions is the same for every ``mu``: a stricter policy is a prefix of
a looser one. One maximal run (``mu = 0``) therefore contains every cheaper
policy, and the whole accuracy-cost frontier is read off its logged snapshots.

VOC is exact for the posterior's one-response or one-evidence experiment. A
coaching action can reject a hint and a review can trigger revision: those are
multi-call actions whose gain is approximated by these one-step experiments.
Their cost must include the possible continuation, and held-out experiments must
test the approximation. Threshold replay applies to the normal policy, not to
uniform-budget, fixed-order, or terminal-judge ablations.

Zones emerge from the posterior rather than from hand rules: high confidence
means no action pays (Autonomous); probability mass split across existing
answers makes reviews and judging pay (Latent); low coverage makes generation pay,
through scaffolding when the calibrated coaching lift beats fresh sampling
(Proximal) and through fresh search otherwise (Unreachable).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from mathmage.llm.types import approx_tokens
from mathmage.team.posterior import Evidence, Obs, PosteriorParams, PosteriorResult, TeamPosterior

REVIEW_OUTCOMES = ["review_objection_verified", "review_objection_unverified", "review_valid", "review_unsure"]
GENERATION = {"sample", "coach", "tool_alt", "diversify"}


@dataclass
class Action:
    kind: str
    member: Optional[int] = None
    tutor: Optional[int] = None
    cluster: Optional[int] = None
    options: tuple = ()
    cost: float = 0.0
    voc: float = 0.0

    @property
    def ratio(self) -> float:
        return self.voc / max(self.cost, 1e-7)

    def label(self) -> str:
        parts = [self.kind]
        if self.member is not None:
            parts.append(f"m{self.member}")
        if self.tutor is not None:
            parts.append(f"t{self.tutor}")
        if self.cluster is not None:
            parts.append(f"c{self.cluster}")
        if self.options:
            parts.append("o" + "-".join(map(str, self.options)))
        return ":".join(parts)

    def to_dict(self) -> dict:
        return {**asdict(self), "options": list(self.options), "ratio": self.ratio, "label": self.label()}


@dataclass
class PolicyConfig:
    mu: float = 0.0                 # stop when the best VOC per dollar falls below mu
    epsilon: float = 0.002          # minimum expected accuracy gain worth acting on
    max_actions: int = 14
    max_item_cost_usd: float = 0.75
    max_reviews_per_cluster: int = 1
    max_samples_per_member: int = 3
    max_hints_per_solver: int = 2
    max_tool_alt: int = 1
    max_diversify: int = 1
    max_judges: int = 1
    allow: set = field(default_factory=lambda: {"sample", "coach", "tool_alt", "review", "judge", "diversify"})
    uniform_budget: bool = False    # ablation A1: ignore early stopping, spend the full budget
    screen_coaching: bool = True    # A5 open discussion bypasses the hint audit


def estimate_cost(models, key: str, tokens_in: int, tokens_out: int) -> float:
    return models.model(key).price.cost(tokens_in, tokens_out)


class Policy:
    def __init__(self, params: PosteriorParams, config: Optional[PolicyConfig] = None):
        self.params = params
        self.config = config or PolicyConfig()
        self.posterior = TeamPosterior(params)

    # ------------------------------------------------------------------ state
    def posterior_of(self, obs: list[Obs], evidence: list[Evidence], C: int, *, selectable=None) -> PosteriorResult:
        return self.posterior.compute(obs, evidence, C, selectable=selectable)

    def zone(self, result: PosteriorResult, actions: list[Action], gamma_a: float = 0.9, gamma_l: float = 0.5) -> str:
        """Operational team zone at decision time (a summary of the posterior and best action)."""
        if result.confidence >= gamma_a:
            return "A"
        if result.coverage >= gamma_l:
            return "L"
        generation = [a for a in actions if a.kind in GENERATION and a.voc > 0]
        if generation and max(generation, key=lambda a: a.ratio).kind == "coach":
            return "P"
        return "U"

    # ------------------------------------------------------------------ scoring
    def score(self, actions: list[Action], obs: list[Obs], evidence: list[Evidence], C: int,
              agent_of_member, lift_of_pair, result: Optional[PosteriorResult] = None) -> list[Action]:
        result = result or self.posterior_of(obs, evidence, C)
        tp = self.posterior
        for a in actions:
            if a.kind == "sample":
                a.voc = tp.voc_sample(obs, evidence, C, agent_of_member(a.member), 0.0, result)
            elif a.kind == "coach":
                # The callback returns a single lift or (lift, tutor's cluster, tutor-right lift).
                spec = lift_of_pair(a.member, a.tutor)
                lift, source, source_lift = spec if isinstance(spec, tuple) else (spec, -1, None)
                a.voc = tp.voc_sample(obs, evidence, C, agent_of_member(a.member), lift, result, source, source_lift)
                # A rejected hint produces no solver response. The acceptance
                # propensity is an offline estimate (or explicit prior), assumed
                # independent of correctness here; it is not a pseudo failure.
                if self.config.screen_coaching:
                    a.voc *= self.params.hint_acceptance
            elif a.kind == "diversify":
                a.voc = tp.voc_sample(obs, evidence, C, agent_of_member(a.member),
                                      self.params.lift.get("diversify", 0.0), result)
            elif a.kind == "tool_alt":
                a.voc = tp.voc_sample(obs, evidence, C, "tool", 0.0, result)
            elif a.kind == "review":
                a.voc = tp.voc_evidence(obs, evidence, C, REVIEW_OUTCOMES, a.cluster, result)
            elif a.kind == "judge":
                a.voc = tp.voc_judge(obs, evidence, C, list(a.options), result)
            else:
                raise ValueError(a.kind)
        return actions

    def choose(self, actions: list[Action], spent_usd: float, n_actions: int) -> tuple[Optional[Action], str]:
        cfg = self.config
        if n_actions >= cfg.max_actions:
            return None, "action budget exhausted"
        affordable = [a for a in actions if spent_usd + a.cost <= cfg.max_item_cost_usd]
        if not affordable:
            return None, "item cost budget exhausted"
        best = max(affordable, key=lambda a: (a.ratio, -a.cost))
        if cfg.uniform_budget:
            return best, "uniform budget (no early stopping)"
        if best.voc < cfg.epsilon:
            return None, "no action has value"
        if best.ratio < cfg.mu:
            return None, "value per dollar below mu"
        return best, "highest value per dollar"


def prompt_tokens(*texts: str) -> int:
    return sum(approx_tokens(t) for t in texts if t) + 120
