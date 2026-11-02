"""Offline calibration of the label-free team posterior (training items only).

Inputs are graded training records; outputs are frozen :class:`PosteriorParams`.

* Abilities and difficulty spread: marginal MAP with discrimination fixed to one
  and b_mean fixed to zero, identifying scale and location. Correctness is
  conditional on usable answers; repeated responses estimate residual dependence.
* kappa / kappa_same: the observed rate at which two wrong answers to the same item
  coincide, for responses from different agents and from the same agent.
* none_weight: kept at one; its legacy name now denotes a response failure-odds
  multiplier, not an extra prior on a changing set of candidate hypotheses.
* Evidence likelihoods, judge accuracy, and coaching lift: counts with prior
  pseudo-counts (the defaults in :mod:`mathmage.team.posterior`), so small samples
  shrink toward conservative values instead of producing extreme likelihood ratios.

Model v3 additions (training items only):

* Trap mixture: ordinary and trap item types with their own collision odds and a
  type weight, fitted by marginal MAP over items. Extra wrong-answer panels (for
  example public logs split into disjoint team-sized groups) share their item's
  type, so no response is counted twice and the panel size matches deployment.
* Tutor-dependent coaching lifts: the ability lift of a coached response when the
  tutor's own answer was right and when it was not, fitted under the same IRT
  model the posterior uses (item difficulty posterior, same-agent reinforcement).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, logsumexp

from mathmage.team.posterior import DEFAULT_EVIDENCE, AgentProfile, Obs, PosteriorParams, TeamPosterior

REVIEW_KINDS = ("review_objection_verified", "review_objection_unverified", "review_valid", "review_unsure")


@dataclass
class Response:
    agent: str
    answer: Optional[str]          # None = no usable answer
    correct: bool


@dataclass
class ItemRecord:
    item_id: str
    answer_format: str
    responses: list[Response]
    family_id: str = ""


@dataclass
class EvidenceRecord:
    kind: str                       # an evidence kind of DEFAULT_EVIDENCE
    target_correct: bool


@dataclass
class CoachingRecord:
    item_id: str
    solver: str                     # agent identity of the coached solver
    correct: bool                   # the coached response was correct
    tutor_right: bool               # the tutor's own current answer was correct (False if it had none)


@dataclass
class CalibrationData:
    items: list[ItemRecord] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    judge_trials: list[bool] = field(default_factory=list)          # judge picked the correct candidate
    coaching: list[tuple[bool, bool]] = field(default_factory=list)  # (coached correct, baseline correct)
    revision: list[tuple[bool, bool]] = field(default_factory=list)  # (revised correct, original correct)
    hint_acceptance_trials: list[bool] = field(default_factory=list)  # one per authored hint, not replay recipient
    coaching_records: list[CoachingRecord] = field(default_factory=list)
    # Extra wrong-answer panels used only for collision odds; several may share an item_id.
    kappa_items: list[ItemRecord] = field(default_factory=list)


def _clusters(item: ItemRecord) -> list[int]:
    from mathmage.grading.equiv import cluster

    return cluster([r.answer for r in item.responses], item.answer_format)


def fit_irt(items: list[ItemRecord], grid: int = 61, prior_theta_sd: float = 2.0) -> dict:
    """Identified marginal MAP 1PL with beta-Bernoulli within-agent dependence.

    A free a, theta and b_sd would leave a likelihood scale symmetry. Fix a=1 and
    b_mean=0. The correlation prior is retained when repeats cannot identify it.
    """
    if not items or not any(it.responses for it in items):
        raise ValueError("IRT calibration needs responses")
    agents = sorted({r.agent for it in items for r in it.responses})
    index = {a: i for i, a in enumerate(agents)}
    nodes, weights = np.polynomial.hermite_e.hermegauss(grid)
    logw = np.log(weights / weights.sum())
    data = [(np.array([index[r.agent] for r in it.responses if r.answer is not None], dtype=int),
             np.array([float(r.correct) for r in it.responses if r.answer is not None]))
            for it in items if any(r.answer is not None for r in it.responses)]
    if not data:
        raise ValueError("IRT calibration needs a usable answer")
    repeated = any(len(set(ag)) < len(ag) for ag, _ in data)

    def unpack(x):
        rho = float(0.95 * expit(x[len(agents) + 1])) if repeated else 0.15
        return x[:len(agents)], np.exp(x[len(agents)]), rho

    def neg_log_post(x):
        theta, b_sd, rho = unpack(x)
        b = b_sd * nodes
        total = 0.0
        for ag, y in data:
            p = expit(theta[ag][:, None] - b[None, :])
            if repeated:
                concentration = (1 - rho) / rho
                seen, successes = defaultdict(int), defaultdict(float)
                for j, agent in enumerate(ag):
                    p[j] = (concentration * p[j] + successes[agent]) / (concentration + seen[agent])
                    seen[agent] += 1
                    successes[agent] += y[j]
            p = np.clip(p, 1e-9, 1 - 1e-9)
            ll = (y[:, None] * np.log(p) + (1 - y[:, None]) * np.log1p(-p)).sum(axis=0)
            total += logsumexp(ll + logw)
        prior = (-0.5 * (theta / prior_theta_sd) ** 2).sum()
        prior += -0.5 * ((np.log(b_sd) - np.log(1.6)) / 0.5) ** 2
        if repeated:
            prior += -0.5 * ((rho - 0.15) / 0.15) ** 2
        return -(total + prior)

    x0 = np.concatenate([np.zeros(len(agents)), [np.log(1.6)]] + ([[logit(0.15 / 0.95)]] if repeated else []))
    res = minimize(neg_log_post, x0, method="L-BFGS-B")
    theta, b_sd, rho = unpack(res.x)
    return {"agents": agents, "theta": dict(zip(agents, map(float, theta))), "a": 1.0, "b_sd": float(b_sd),
            "same_agent_correlation": rho, "correlation_estimated": repeated,
            "identification": "b_mean=0; a=1; conditional-on-usable-answer likelihood",
            "converged": bool(res.success), "neg_log_post": float(res.fun)}


def _wrong_events(item: ItemRecord) -> list[tuple[int, int, Optional[int]]]:
    """Per wrong response after the first: (#other-agent clusters, #same-agent clusters, joined type)."""
    events = []
    labels = _clusters(item)
    members = {}
    for r, c in zip(item.responses, labels):
        if r.answer is None or r.correct:
            continue
        same = {k: r.agent in v for k, v in members.items()}
        if members:
            events.append((sum(not v for v in same.values()), sum(same.values()),
                           None if c not in same else int(same[c])))
        members.setdefault(c, []).append(r.agent)
    return events


def _collision_loglik(events, kappa: float, kappa_same: float) -> float:
    odds = (kappa / (1 - kappa), kappa_same / (1 - kappa_same))
    total = 0.0
    for diff, same, target in events:
        total -= np.log(1 + diff * odds[0] + same * odds[1])
        if target is not None:
            total += np.log(odds[target])
    return total


def _log_beta(x: float, prior: tuple[float, float]) -> float:
    return (prior[0] - 1) * np.log(x) + (prior[1] - 1) * np.log1p(-x)


def fit_kappa_mixture(groups: dict[str, list[ItemRecord]], prior=(1.0, 6.0), prior_same=(2.0, 4.0),
                      prior_trap=(2.0, 4.0), prior_weight=(1.0, 4.0)) -> dict:
    """Marginal MAP of the ordinary/trap collision mixture; the type is shared within an item.

    ``groups`` maps an item id to every response panel observed for it (our team's
    responses and any disjoint extra panels). Returns the fitted parameters and the
    log-likelihood gain over the single-type fit (a diagnostic, not a test).
    """
    event_groups = [events for events in ([e for item in records for e in _wrong_events(item)]
                                          for records in groups.values()) if events]
    if not event_groups:
        return {"kappa": prior[0] / sum(prior), "kappa_same": prior_same[0] / sum(prior_same),
                "trap_weight": 0.0, "kappa_trap": 0.0, "n_items": 0, "n_events": 0, "loglik_gain": 0.0,
                "converged": True}

    def unpack(x):
        weight = float(expit(x[0]))
        kappa = float(0.95 * expit(x[1]))
        trap = float(kappa + (0.95 - kappa) * expit(x[2]))
        same = float(0.95 * expit(x[3]))
        return weight, kappa, trap, same

    def objective(x):
        weight, kappa, trap, same = unpack(x)
        total = 0.0
        for events in event_groups:
            total += np.logaddexp(np.log1p(-weight) + _collision_loglik(events, kappa, max(same, kappa)),
                                  np.log(weight) + _collision_loglik(events, trap, max(same, trap)))
        total += (_log_beta(weight, prior_weight) + _log_beta(kappa, prior) + _log_beta(trap, prior_trap)
                  + _log_beta(same, prior_same))
        return -total

    starts = [(0.2, 0.03, 0.3, 0.25), (0.1, 0.05, 0.5, 0.3), (0.3, 0.02, 0.15, 0.2), (0.05, 0.08, 0.6, 0.35)]
    best = None
    for weight, kappa, trap, same in starts:
        x0 = [logit(weight), logit(kappa / 0.95), logit((trap - kappa) / (0.95 - kappa)), logit(same / 0.95)]
        result = minimize(objective, x0, method="L-BFGS-B")
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not np.isfinite(best.fun) or not np.all(np.isfinite(best.x)):
        raise ValueError("collision mixture optimization produced nonfinite parameters or objective")
    weight, kappa, trap, same = unpack(best.x)

    def single(k):
        return -(sum(_collision_loglik(events, k, max(same, k)) for events in event_groups))
    single_fit = minimize_scalar(single, bounds=(1e-4, 0.95), method="bounded")
    mixture_loglik = -objective(best.x) - (_log_beta(weight, prior_weight) + _log_beta(kappa, prior)
                                           + _log_beta(trap, prior_trap) + _log_beta(same, prior_same))
    return {"kappa": kappa, "kappa_same": max(same, kappa), "trap_weight": weight, "kappa_trap": trap,
            "n_items": len(event_groups), "n_events": sum(len(events) for events in event_groups),
            "loglik_gain": float(mixture_loglik + single_fit.fun), "single_kappa": float(single_fit.x),
            "converged": bool(best.success)}


def fit_coaching_lifts(records: list[CoachingRecord], items: dict[str, ItemRecord], theta: dict[str, float],
                       b_sd: float, rho: float, grid: int = 61, prior_mean: float = 0.5,
                       prior_sd: float = 1.5) -> dict:
    """IRT-consistent coaching lifts: pooled, tutor right, and tutor wrong.

    Each coached outcome is scored against the item's difficulty posterior (from the
    graded responses of known agents) and the solver's earlier attempts through the
    same beta-Bernoulli reinforcement the posterior applies.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(grid)
    b, logw = b_sd * nodes, np.log(weights / weights.sum())
    prepared = []
    for record in records:
        item = items.get(record.item_id)
        if item is None or record.solver not in theta:
            continue
        loglik = logw.copy()
        seen, successes = defaultdict(int), defaultdict(float)
        for r in item.responses:
            if r.answer is None or r.agent not in theta:
                continue
            p = expit(theta[r.agent] - b)
            if rho > 0:
                concentration = (1 - rho) / rho
                p = (concentration * p + successes[r.agent]) / (concentration + seen[r.agent])
            p = np.clip(p, 1e-15, 1 - 1e-15)
            loglik += np.log(p) if r.correct else np.log1p(-p)
            seen[r.agent] += 1
            successes[r.agent] += bool(r.correct)
        posterior = np.exp(loglik - logsumexp(loglik))
        earlier = [r for r in item.responses if r.agent == record.solver and r.answer is not None]
        prepared.append((record.tutor_right, bool(record.correct), posterior, theta[record.solver],
                         len(earlier), sum(bool(r.correct) for r in earlier)))

    def fit(rows):
        if not rows:
            return prior_mean, 0

        def negative(lift):
            total = 0.0
            for _, correct, posterior, ability, n, k in rows:
                p = expit(ability + lift - b)
                if rho > 0:
                    concentration = (1 - rho) / rho
                    p = (concentration * p + k) / (concentration + n)
                prob = float(np.clip((posterior * p).sum(), 1e-9, 1 - 1e-9))
                total += np.log(prob) if correct else np.log1p(-prob)
            return -(total - 0.5 * ((lift - prior_mean) / prior_sd) ** 2)
        return float(minimize_scalar(negative, bounds=(-4.0, 6.0), method="bounded").x), len(rows)

    pooled, n = fit(prepared)
    right, n_right = fit([row for row in prepared if row[0]])
    wrong, n_wrong = fit([row for row in prepared if not row[0]])
    return {"coached": pooled, "coached_right": right, "coached_wrong": wrong,
            "n": n, "n_right": n_right, "n_wrong": n_wrong,
            "success_right": sum(row[1] for row in prepared if row[0]),
            "success_wrong": sum(row[1] for row in prepared if not row[0])}


def fit_kappa(items: list[ItemRecord], prior=(1.0, 6.0), prior_same=(2.0, 4.0)) -> tuple[float, float]:
    """Fit the sequential categorical wrong-cluster model, not pair collision rate.

    Conditional on being wrong, the odds of each previous wrong cluster are
    k/(1-k), with a unit weight for a fresh answer. Pairwise collision frequencies
    are not this parameter when there are several previous wrong clusters.
    """
    events = [event for it in items for event in _wrong_events(it)]

    def loss(x):
        k, ks = x
        odds = np.asarray([k / (1 - k), ks / (1 - ks)])
        total = 0.0
        for diff, same, target in events:
            total += np.log(1 + diff * odds[0] + same * odds[1])
            if target is not None:
                total -= np.log(odds[target])
        total -= prior[0] * np.log(k) + prior[1] * np.log1p(-k)
        total -= prior_same[0] * np.log(ks) + prior_same[1] * np.log1p(-ks)
        return total

    result = minimize(loss, [prior[0] / sum(prior), prior_same[0] / sum(prior_same)],
                      bounds=[(1e-5, 0.95), (1e-5, 0.95)], method="L-BFGS-B")
    kappa, kappa_same = result.x
    return float(kappa), float(max(kappa_same, kappa))


def _observations(item: ItemRecord) -> tuple[list[Obs], int, int]:
    """Posterior observations plus the index of the true hypothesis (C means none)."""
    labels = _clusters(item)
    obs = [Obs(r.agent, labels[i] if r.answer is not None else -1) for i, r in enumerate(item.responses)]
    C = max(labels, default=-1) + 1
    correct_clusters = {labels[i] for i, r in enumerate(item.responses) if r.correct and r.answer is not None}
    if len(correct_clusters) > 1:
        raise ValueError("posterior assumes one correct equivalence cluster per item")
    true = next(iter(correct_clusters), C)
    if any(r.answer is not None and not r.correct and labels[i] == true for i, r in enumerate(item.responses)):
        raise ValueError("equivalent responses have inconsistent correctness labels")
    return obs, C, true


def fit_none_weight(items: list[ItemRecord], params: PosteriorParams) -> float:
    """Optional residual failure-odds fit; not used after jointly fitting abilities.

    Restores the input parameter after optimization. Fitting the common ability
    intercept already calibrates response odds, so calibrate() fixes this to one.
    """
    prepared = [_observations(it) for it in items]
    original = params.none_weight

    def neg_score(log_w):
        params.none_weight = float(np.exp(log_w))
        tp = TeamPosterior(params)
        score = 0.0
        for obs, C, true in prepared:
            r = tp.compute(obs, [], C)
            q = np.append(r.q, r.q_none)
            score += np.log(max(q[true], 1e-9))
        return -score - 0.5 * (log_w / 1.5) ** 2

    try:
        res = minimize_scalar(neg_score, bounds=(-4.0, 4.0), method="bounded")
    finally:
        params.none_weight = original
    return float(np.exp(res.x))


def _abstain(items: list[ItemRecord]) -> dict[str, float]:
    n, k = defaultdict(int), defaultdict(int)
    for it in items:
        for r in it.responses:
            n[r.agent] += 1
            k[r.agent] += int(r.answer is None)
    return {a: (k[a] + 0.5) / (n[a] + 10.0) for a in n}


def fit_evidence(records: Iterable[EvidenceRecord], pseudo: float = 4.0) -> dict[str, tuple[float, float]]:
    records = list(records)
    out = dict(DEFAULT_EVIDENCE)
    groups = {"review": REVIEW_KINDS, "check": ("check_failed", "check_passed")}
    for name, kinds in groups.items():
        for correct in (True, False):
            sub = [r for r in records if r.kind in kinds and r.target_correct == correct]
            total = len(sub)
            for kind in kinds:
                prior = DEFAULT_EVIDENCE[kind][0 if correct else 1]
                count = sum(r.kind == kind for r in sub)
                value = (count + pseudo * prior) / (total + pseudo * sum(DEFAULT_EVIDENCE[k][0 if correct else 1]
                                                                          for k in kinds))
                pc, pw = out[kind]
                out[kind] = (value, pw) if correct else (pc, value)
    return out


def _logit_lift(pairs: list[tuple[bool, bool]], shrink: float = 20.0, default: float = 0.0) -> float:
    if not pairs:
        return default
    treated = (sum(a for a, _ in pairs) + 1.0) / (len(pairs) + 2.0)
    base = (sum(b for _, b in pairs) + 1.0) / (len(pairs) + 2.0)
    est = float(logit(treated) - logit(base))
    w = len(pairs) / (len(pairs) + shrink)
    return w * est + (1 - w) * default


def calibrate(data: CalibrationData, *, provenance: Optional[dict] = None, base: Optional[PosteriorParams] = None
              ) -> tuple[PosteriorParams, dict]:
    """Fit all posterior parameters from training records. Returns (params, diagnostics)."""
    if not data.items:
        raise ValueError("calibration needs graded training items")
    if len({it.item_id for it in data.items}) != len(data.items):
        raise ValueError("merge response draws for the same item_id before calibration")
    for item in data.items:
        _observations(item)
    irt = fit_irt(data.items)
    if not irt["converged"]:
        raise ValueError("IRT optimization did not converge; posterior cannot be frozen as calibrated")
    single_kappa, single_kappa_same = fit_kappa(data.items)
    groups = defaultdict(list)
    for item in [*data.items, *data.kappa_items]:
        groups[item.item_id].append(item)
    mixture = fit_kappa_mixture(groups)
    fitted = [mixture.get(key, np.nan) for key in ("kappa", "kappa_same", "trap_weight", "kappa_trap", "loglik_gain")]
    if not mixture.get("converged") or not all(np.isfinite(value) for value in fitted):
        raise ValueError("collision mixture optimization did not converge to finite parameters; recalibrate before freezing")
    abstain = _abstain(data.items)
    base = base or PosteriorParams(agents={})
    agents = {a: AgentProfile(theta=irt["theta"][a], a=irt["a"], abstain=abstain.get(a, 0.03)) for a in irt["agents"]}
    params = PosteriorParams(agents=agents, b_mean=0.0, b_sd=irt["b_sd"], kappa=mixture["kappa"],
                             kappa_same=mixture["kappa_same"], trap_weight=mixture["trap_weight"],
                             kappa_trap=mixture["kappa_trap"],
                             none_weight=1.0, same_agent_correlation=irt["same_agent_correlation"],
                             hint_acceptance=(sum(data.hint_acceptance_trials) + 4.0) / (len(data.hint_acceptance_trials) + 5.0),
                             lift=dict(base.lift), evidence=fit_evidence(data.evidence),
                             judge_accuracy=float((sum(data.judge_trials) + 7.0) / (len(data.judge_trials) + 10.0)),
                             grid=base.grid, calibrated=True, provenance=dict(provenance or {}))
    coaching = None
    if data.coaching_records:
        coaching = fit_coaching_lifts(data.coaching_records, {it.item_id: it for it in data.items}, irt["theta"],
                                      irt["b_sd"], irt["same_agent_correlation"])
        params.lift.update({key: coaching[key] for key in ["coached", "coached_right", "coached_wrong"]})
    else:
        params.lift["coached"] = _logit_lift(data.coaching, default=base.lift.get("coached", 0.6))
    params.lift["revision"] = _logit_lift(data.revision, default=base.lift.get("revision", 0.3))
    diagnostics = {"irt": irt, "kappa": params.kappa, "kappa_same": params.kappa_same, "none_weight": params.none_weight,
                   "kappa_mixture": mixture, "kappa_single_type": {"kappa": single_kappa, "kappa_same": single_kappa_same},
                   "coaching_lifts": coaching,
                   "n_items": len(data.items), "n_responses": sum(len(i.responses) for i in data.items),
                   "n_kappa_panels": len(data.kappa_items),
                   "n_evidence": len(data.evidence), "n_judge_trials": len(data.judge_trials),
                   "n_coaching_pairs": len(data.coaching), "n_coaching_records": len(data.coaching_records),
                   "n_revision_pairs": len(data.revision),
                   "n_hint_acceptance_trials": len(data.hint_acceptance_trials),
                   "calibration_scope": {"abilities": "fitted; conditional on usable response",
                       "same_agent_correlation": "fitted" if irt["correlation_estimated"] else "prior only",
                       "collision_types": ("ordinary/trap mixture fitted per item on training panels"
                                           + (" incl. disjoint public team panels" if data.kappa_items else "")),
                       "evidence": {kind: sum(r.kind == kind for r in data.evidence) for kind in DEFAULT_EVIDENCE},
                       "judge": "binary offered-correct-answer trials; multiway extrapolation" if data.judge_trials else "prior only",
                       "coached_lift": ("IRT-consistent replay fit, separately for tutor right/wrong; action model approximation"
                                        if data.coaching_records else
                                        "conditional replay-vs-self-revision contrast; action model approximation"
                                        if data.coaching else "prior only"),
                       "hint_acceptance": "distinct authored-hint audit outcomes" if data.hint_acceptance_trials else "prior only",
                       "revision_lift": "self-revision trials; gated-review extrapolation" if data.revision else "prior only",
                       "diversify_lift": "prior only", "discussion_lift": "prior only"}}
    params.provenance["calibration_scope"] = diagnostics["calibration_scope"]
    return params, diagnostics


def reliability(items: list[ItemRecord], params: PosteriorParams, bins: int = 5) -> dict:
    """Calibration of the posterior's decision confidence on held-out graded items."""
    if bins < 1:
        raise ValueError("bins must be positive")
    tp = TeamPosterior(params)
    conf, correct, brier = [], [], []
    for it in items:
        obs, C, true = _observations(it)
        r = tp.compute(obs, [], C)
        conf.append(r.confidence)
        correct.append(float(r.best == true))
        q = np.append(r.q, r.q_none)
        onehot = np.zeros_like(q)
        onehot[true] = 1.0
        brier.append(float(((q - onehot) ** 2).sum()))
    conf, correct = np.array(conf), np.array(correct)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if sel.any():
            rows.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                         "confidence": float(conf[sel].mean()), "accuracy": float(correct[sel].mean())})
            ece += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return {"n": int(len(conf)), "ece": float(ece), "brier": float(np.mean(brier)) if brier else None,
            "accuracy": float(correct.mean()) if len(correct) else None,
            "mean_confidence": float(conf.mean()) if len(conf) else None, "bins": rows}
