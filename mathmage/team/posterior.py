"""Label-free sequential posterior over answer clusters and item difficulty.

The hidden correct answer starts unobserved. Each usable response either joins
the correct cluster, reveals the previously unseen correct answer, or produces a
wrong answer. Wrong answers choose among previous wrong clusters and a new one
by normalized collision *odds*. Thus probabilities remain valid for any number
of clusters. Abstention is independent of correctness and carries no difficulty
information. A same-agent beta-Bernoulli predictive reinforcement models residual
correctness dependence after conditioning on difficulty. With varying lifts this
is an autoregressive approximation, rather than an exchangeable random effect.

The response predictive and posterior update use exactly the same transitions.
One-response VOC is therefore coherent within this assumed model. It is not an
exact model of multi-call coaching, rejected hints, or review-triggered revisions.
Evidence likelihoods remain conditional-independence approximations and require
held-out validation, especially when a response author also judges the pool.

Two refinements (model v3), both fitted on training data only:

* Trap items. Wrong-answer coincidence varies by item: some problems have an
  attractive wrong answer. An item-level latent type T in {ordinary, trap} with
  prior weight ``trap_weight`` selects the collision odds (``kappa`` or
  ``kappa_trap``). T is integrated exactly, so agreement on an answer is weaker
  evidence of correctness when the agreement itself suggests a trap.
* Tutor-dependent coaching. A coached response's ability lift depends on whether
  the tutor's current answer is the truth: ``source_lift`` when the truth hypothesis
  is the tutor's cluster, ``lift`` otherwise. Coaching therefore has value in
  proportion to the posterior that the tutor knows the answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import comb
from typing import Iterable, Optional

import numpy as np
from scipy.special import expit, logsumexp


DEFAULT_EVIDENCE = {
    "review_objection_verified": (0.03, 0.30),
    "review_objection_unverified": (0.18, 0.36),
    "review_valid": (0.62, 0.30),
    "review_unsure": (0.17, 0.04),
    "check_failed": (0.08, 0.35),
    "check_passed": (0.45, 0.20),
}
MODEL_VERSION = "sequential_cluster_v3"


@dataclass
class AgentProfile:
    theta: float
    a: float = 1.0
    abstain: float = 0.03


@dataclass
class PosteriorParams:
    agents: dict[str, AgentProfile]
    model_version: str = MODEL_VERSION
    b_mean: float = 0.0
    b_sd: float = 1.6
    kappa: float = 0.12
    kappa_same: float = 0.30
    # Item-level trap mixture: weight of the trap type and its collision probability.
    # trap_weight = 0 recovers the single-type model exactly.
    trap_weight: float = 0.0
    kappa_trap: float = 0.0
    # Legacy name: response failure-odds multiplier, NOT an extra hypothesis prior.
    # A separate unit prior for each newly observed cluster is not sequentially coherent.
    none_weight: float = 1.0
    same_agent_correlation: float = 0.15
    hint_acceptance: float = 0.8
    lift: dict[str, float] = field(default_factory=lambda: {"coached": 0.6, "diversify": 0.2,
                                                           "revision": 0.3, "discussion": 0.3})
    evidence: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_EVIDENCE))
    judge_accuracy: float = 0.70
    grid: int = 121
    calibrated: bool = False
    provenance: dict = field(default_factory=dict)

    def identity(self, key: str) -> str:
        if key in self.agents:
            return key
        if key.startswith("tool:") and "tool" in self.agents:
            return "tool"
        base = key.split(":", 1)[-1]
        return base if base in self.agents else key

    def agent(self, key: str) -> AgentProfile:
        return self.agents.get(self.identity(key), AgentProfile(theta=0.0))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = {k: list(v) for k, v in self.evidence.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PosteriorParams":
        d = dict(d)
        if d.get("calibrated") and d.get("model_version") != MODEL_VERSION:
            raise ValueError("posterior model semantics changed; recalibrate and freeze a new artifact")
        d["agents"] = {k: AgentProfile(**v) for k, v in d["agents"].items()}
        d["evidence"] = {k: tuple(v) for k, v in d.get("evidence", DEFAULT_EVIDENCE).items()}
        return cls(**d)


@dataclass(frozen=True)
class Obs:
    """One answer from an actual model key (not a role instance); -1 abstains.

    ``source`` is the answer cluster of the tutor that coached this response (-1:
    no tutor dependence). Under truth hypothesis ``h == source`` the response uses
    ``source_lift``; under every other hypothesis it uses ``lift``.
    """

    agent: str
    cluster: int
    lift: float = 0.0
    source: int = -1
    source_lift: Optional[float] = None

    def lift_under(self, h: int) -> float:
        if self.source >= 0 and self.source_lift is not None and h == self.source:
            return self.source_lift
        return self.lift


@dataclass(frozen=True)
class Evidence:
    kind: str
    cluster: int
    options: tuple[int, ...] = ()


@dataclass
class PosteriorResult:
    q: np.ndarray
    q_none: float
    grid: np.ndarray
    joint: np.ndarray
    log_marginal: float
    # Joint posterior per item type (ordinary, trap), shape (T, C+1, grid); sums to ``joint``.
    parts: Optional[np.ndarray] = None
    # Likelihoods retain historical answers even when an ablation retires them
    # from the decision pool. This mask changes decision utility, never belief.
    selectable: Optional[frozenset[int]] = None

    @property
    def trap_probability(self) -> float:
        if self.parts is None or len(self.parts) < 2:
            return 0.0
        return float(self.parts[1].sum())

    @property
    def coverage(self) -> float:
        if self.selectable is not None:
            return float(sum(self.q[c] for c in self.selectable))
        return 1.0 - self.q_none

    @property
    def best(self) -> int:
        eligible = range(len(self.q)) if self.selectable is None else sorted(self.selectable)
        return max(eligible, key=lambda c: self.q[c], default=-1)

    @property
    def confidence(self) -> float:
        return float(self.q[self.best]) if self.best >= 0 else 0.0

    def b_posterior(self) -> np.ndarray:
        return self.joint.sum(axis=0)

    def summary(self) -> dict:
        b = self.b_posterior()
        return {"q": [round(float(x), 6) for x in self.q], "q_none": round(float(self.q_none), 6),
                "best": self.best, "confidence": round(self.confidence, 6),
                "b_mean": round(float((b * self.grid).sum()), 4),
                "b_sd": round(float(np.sqrt(max(0.0, (b * self.grid ** 2).sum() - (b * self.grid).sum() ** 2))), 4),
                "trap_probability": round(self.trap_probability, 6)}


class TeamPosterior:
    def __init__(self, params: PosteriorParams):
        if params.model_version != MODEL_VERSION:
            raise ValueError("unsupported posterior model version; recalibration required")
        if not (0 <= params.kappa < 1 and 0 <= params.kappa_same < 1):
            raise ValueError("collision probabilities must lie in [0, 1)")
        if not (0 <= params.trap_weight < 1) or (params.trap_weight > 0 and not params.kappa <= params.kappa_trap < 1):
            raise ValueError("trap weight must lie in [0, 1) and kappa_trap in [kappa, 1)")
        if not (0 <= params.same_agent_correlation < 1):
            raise ValueError("same_agent_correlation must lie in [0, 1)")
        if not 0 <= params.hint_acceptance <= 1:
            raise ValueError("hint_acceptance must lie in [0,1]")
        if params.none_weight <= 0 or params.b_sd <= 0 or params.grid < 3:
            raise ValueError("positive none_weight, b_sd and grid >= 3 required")
        if not (0 < params.judge_accuracy < 1):
            raise ValueError("judge_accuracy must lie strictly between zero and one")
        for a in params.agents.values():
            if a.a <= 0 or not 0 <= a.abstain <= 1:
                raise ValueError("agent discrimination must be positive and abstain in [0,1]")
        self.params = params
        nodes, weights = np.polynomial.hermite_e.hermegauss(params.grid)
        self.grid = params.b_mean + params.b_sd * nodes
        self.log_prior_b = np.log(weights / weights.sum())

    def _p(self, obs: Obs) -> np.ndarray:
        prof = self.params.agent(obs.agent)
        return expit(prof.a * (prof.theta + obs.lift - self.grid) - np.log(self.params.none_weight))

    def _correct_prob(self, agent: str, lift: float, h: int,
                      members: dict[int, list[str]]) -> np.ndarray:
        """Correctness conditional on usable response and current truth hypothesis."""
        p = self._p(Obs(agent, -1, lift))
        rho = self.params.same_agent_correlation
        if rho == 0:
            return p
        identity = self.params.identity(agent)
        counts = {c: sum(self.params.identity(a) == identity for a in agents)
                  for c, agents in members.items()}
        n = sum(counts.values())
        concentration = (1 - rho) / rho
        return (concentration * p + counts.get(h, 0)) / (concentration + n)

    def components(self) -> list[tuple[float, float, float]]:
        """(log prior weight, different-agent kappa, same-agent kappa) per item type."""
        p = self.params
        if p.trap_weight <= 0:
            return [(0.0, p.kappa, p.kappa_same)]
        return [(float(np.log1p(-p.trap_weight)), p.kappa, p.kappa_same),
                (float(np.log(p.trap_weight)), p.kappa_trap, max(p.kappa_same, p.kappa_trap))]

    def _wrong_distribution(self, agent: str, h: int, members: dict[int, list[str]],
                            kappa: Optional[float] = None, kappa_same: Optional[float] = None
                            ) -> tuple[dict[int, float], float]:
        """Categorical wrong response: existing wrong clusters plus a new answer."""
        kappa = self.params.kappa if kappa is None else kappa
        kappa_same = self.params.kappa_same if kappa_same is None else kappa_same
        identity = self.params.identity(agent)
        odds = {}
        for c, agents in members.items():
            if c == h:
                continue
            same = any(self.params.identity(a) == identity for a in agents)
            k = kappa_same if same else kappa
            odds[c] = k / (1 - k)
        z = 1.0 + sum(odds.values())
        return {c: x / z for c, x in odds.items()}, 1.0 / z

    def _sequence(self, obs: list[Obs], C: int, kappa: float, kappa_same: float) -> np.ndarray:
        """Unnormalized log joint over (truth hypothesis, difficulty) for one item type."""
        logjoint = np.full((C + 1, len(self.grid)), -np.inf)
        logjoint[C] = self.log_prior_b
        members: dict[int, list[str]] = {}
        for o in obs:
            abstain = self.params.agent(o.agent).abstain
            if o.cluster == -1:
                # An unavailable response has no correctness/difficulty evidence.
                logjoint += np.log(max(abstain, 1e-300))
                continue
            new = o.cluster not in members
            nxt = np.full_like(logjoint, -np.inf)
            for h in [*members, C]:
                p = np.clip(self._correct_prob(o.agent, o.lift_under(h), h, members), 1e-15, 1 - 1e-15)
                old = logjoint[h] + np.log(max(1 - abstain, 1e-300))
                wrong, fresh = self._wrong_distribution(o.agent, h, members, kappa, kappa_same)
                if new:
                    # A newly named cluster is correct only if the truth was unseen.
                    if h == C:
                        nxt[o.cluster] = np.logaddexp(nxt[o.cluster], old + np.log(p))
                    nxt[h] = np.logaddexp(nxt[h], old + np.log1p(-p) + np.log(fresh))
                elif h == o.cluster:
                    nxt[h] = old + np.log(p)
                elif wrong.get(o.cluster, 0) > 0:
                    nxt[h] = old + np.log1p(-p) + np.log(wrong[o.cluster])
            logjoint = nxt
            members.setdefault(o.cluster, []).append(o.agent)
        return logjoint

    def compute(self, observations: Iterable[Obs], evidence: Iterable[Evidence] = (),
                n_clusters: Optional[int] = None, *, selectable: Optional[Iterable[int]] = None) -> PosteriorResult:
        obs = list(observations)
        C = n_clusters if n_clusters is not None else max((o.cluster for o in obs), default=-1) + 1
        if C < 0 or any(o.cluster < -1 or o.cluster >= C for o in obs):
            raise ValueError("observations must use -1 or a cluster below n_clusters")
        if any(o.source >= C for o in obs):
            raise ValueError("a coaching source must be an observed answer cluster")
        selectable = frozenset(selectable) if selectable is not None else None
        if selectable is not None and any(c < 0 or c >= C for c in selectable):
            raise ValueError("selectable answers must be observed clusters")
        evidence_loglik = sum((self._evidence_loglik(e, C) for e in evidence), np.zeros(C + 1))
        stacked = np.stack([log_weight + self._sequence(obs, C, kappa, kappa_same) + evidence_loglik[:, None]
                            for log_weight, kappa, kappa_same in self.components()])
        z = logsumexp(stacked)
        if not np.isfinite(z):
            raise ValueError("observations have zero probability under the configured response model")
        parts = np.exp(stacked - z)
        joint = parts.sum(axis=0)
        q_h = joint.sum(axis=1)
        return PosteriorResult(q=q_h[:C], q_none=float(q_h[C]), grid=self.grid, joint=joint, log_marginal=float(z),
                               parts=parts, selectable=selectable)

    def _evidence_loglik(self, e: Evidence, C: int) -> np.ndarray:
        if not 0 <= e.cluster < C:
            raise ValueError("evidence must target an observed answer cluster")
        if e.kind == "judge_pick":
            if len(set(e.options)) != len(e.options) or e.cluster not in e.options or any(c < 0 or c >= C for c in e.options):
                raise ValueError("judge options must be distinct observed clusters and include the pick")
            m = len(e.options)
            if m == 1:
                return np.zeros(C + 1)
            acc = self.params.judge_accuracy
            out = np.full(C + 1, np.log(1.0 / m))
            for h in e.options:
                out[h] = np.log(acc if h == e.cluster else (1 - acc) / (m - 1))
            return out
        pc, pw = self.params.evidence[e.kind]
        if not (0 < pc <= 1 and 0 < pw <= 1):
            raise ValueError("evidence likelihoods must lie in (0,1]")
        out = np.full(C + 1, np.log(pw))
        out[e.cluster] = np.log(pc)
        return out

    def sample_outcomes(self, result: PosteriorResult, agent: str, lift: float, C: int,
                        clusters_agents: dict[int, list[str]], source: int = -1,
                        source_lift: Optional[float] = None) -> list[tuple[int, float]]:
        """Exact predictive for one modeled response; C=new cluster, -1=abstain."""
        if result.joint.shape != (C + 1, len(self.grid)):
            raise ValueError("posterior dimensions do not match the response experiment")
        parts = result.parts if result.parts is not None else result.joint[None]
        components = self.components()
        if len(parts) != len(components):
            raise ValueError("posterior item-type components do not match the parameters")
        modeled = Obs(agent, -1, lift, source, source_lift)
        probs = np.zeros(C + 1)
        for joint, (_, kappa, kappa_same) in zip(parts, components):
            for h in range(C + 1):
                p = self._correct_prob(agent, modeled.lift_under(h), h, clusters_agents)
                probs[h if h < C else C] += float((joint[h] * p).sum())
                wrong, fresh = self._wrong_distribution(agent, h, clusters_agents, kappa, kappa_same)
                mass = float((joint[h] * (1 - p)).sum())
                for c, prob in wrong.items():
                    probs[c] += mass * prob
                probs[C] += mass * fresh
        a = self.params.agent(agent).abstain
        return [(c, float((1 - a) * prob)) for c, prob in enumerate(probs)] + [(-1, a)]

    def success_probability(self, result: PosteriorResult, agent: str, lift: float,
                            clusters_agents: dict[int, list[str]], source: int = -1,
                            source_lift: Optional[float] = None) -> float:
        """Posterior predictive probability that one more usable response is correct."""
        modeled = Obs(agent, -1, lift, source, source_lift)
        total = 0.0
        for h in range(result.joint.shape[0]):
            p = self._correct_prob(agent, modeled.lift_under(h), h, clusters_agents)
            total += float((result.joint[h] * p).sum())
        return total

    def voc_sample(self, obs: list[Obs], evidence: list[Evidence], C: int, agent: str, lift: float = 0.0,
                   result: Optional[PosteriorResult] = None, source: int = -1,
                   source_lift: Optional[float] = None) -> float:
        result = result or self.compute(obs, evidence, C)
        members: dict[int, list[str]] = {}
        for o in obs:
            if o.cluster >= 0:
                members.setdefault(o.cluster, []).append(o.agent)
        expected = 0.0
        for cluster, prob in self.sample_outcomes(result, agent, lift, C, members, source, source_lift):
            if prob <= 0:
                continue
            # A new live response is selectable even when it repeats a retired
            # answer. Abstention leaves the current decision pool unchanged.
            eligible = (None if result.selectable is None else
                        result.selectable | ({cluster} if cluster >= 0 else set()))
            after = self.compute(obs + [Obs(agent, cluster, lift, source, source_lift)], evidence,
                                 C + 1 if cluster == C else C, selectable=eligible)
            expected += prob * after.confidence
        return max(0.0, expected - result.confidence)

    def voc_evidence(self, obs: list[Obs], evidence: list[Evidence], C: int, kinds: list[str], cluster: int,
                     result: Optional[PosteriorResult] = None) -> float:
        if C == 0:
            return 0.0
        likelihoods = np.asarray([self.params.evidence[k] for k in kinds])
        if not np.allclose(likelihoods.sum(axis=0), 1.0):
            raise ValueError("VOC evidence outcomes must form an exhaustive probability partition")
        result = result or self.compute(obs, evidence, C)
        q = np.append(result.q, result.q_none)
        eligible = list(range(C)) if result.selectable is None else sorted(result.selectable)
        expected = 0.0
        for kind in kinds:
            like = np.exp(self._evidence_loglik(Evidence(kind, cluster), C))
            prob = float((q * like).sum())
            if prob > 0:
                expected += max((float(q[c] * like[c]) for c in eligible), default=0.0)
        return max(0.0, expected - result.confidence)

    def voc_judge(self, obs: list[Obs], evidence: list[Evidence], C: int, options: list[int],
                  result: Optional[PosteriorResult] = None) -> float:
        if len(options) < 2:
            return 0.0
        result = result or self.compute(obs, evidence, C)
        q = np.append(result.q, result.q_none)
        eligible = list(range(C)) if result.selectable is None else sorted(result.selectable)
        expected = 0.0
        for pick in options:
            like = np.exp(self._evidence_loglik(Evidence("judge_pick", pick, tuple(options)), C))
            expected += max((float(q[c] * like[c]) for c in eligible), default=0.0)
        return max(0.0, expected - result.confidence)

    def member_zones(self, result: PosteriorResult, agent: str, tau: float = 0.5, K: int = 8,
                     scaffold_lift: Optional[float] = None) -> dict[str, float]:
        """Fresh-response zone probabilities; K uses the declared dependence model.

        The proximal zone asks what the member reaches with guidance from a tutor who
        knows the answer (the "more capable other"), so the default scaffold is the
        tutor-right lift when it was calibrated. Zone sizes depend on K: more fresh
        attempts absorb part of what coaching would otherwise reach.
        """
        if not 0 < tau < 1 or K < 1:
            raise ValueError("tau must be in (0,1) and K must be positive")
        conditional_p = self._p(Obs(agent, -1))
        abstain = self.params.agent(agent).abstain
        p = (1 - abstain) * conditional_p
        rho = self.params.same_agent_correlation
        if rho == 0:
            no_success = (1 - p) ** K
        else:
            concentration = (1 - rho) / rho
            # Mix over how many of the K requests yield a usable response.
            no_success, all_wrong = np.zeros_like(p), np.ones_like(p)
            for usable in range(K + 1):
                no_success += comb(K, usable) * (1 - abstain) ** usable * abstain ** (K - usable) * all_wrong
                all_wrong *= (concentration * (1 - conditional_p) + usable) / (concentration + usable)
        lift = (self.params.lift.get("coached_right", self.params.lift.get("coached", 0.0))
                if scaffold_lift is None else scaffold_lift)
        c = (1 - abstain) * self._p(Obs(agent, -1, lift))
        zone = np.where(p >= tau, 0, np.where(1 - no_success >= tau, 1, np.where(c >= tau, 2, 3)))
        b = result.b_posterior()
        return {z: float(b[zone == i].sum()) for i, z in enumerate("ALPU")}
