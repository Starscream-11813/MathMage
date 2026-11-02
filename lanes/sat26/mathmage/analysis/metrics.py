"""Coverage/selection/generation accounting with explicit denominators."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable


def decomposition(rows: Iterable[dict], weights: Iterable[float] | None = None) -> dict:
    rows = list(rows)
    n = len(rows)
    if not n:
        return {"n": 0, "accuracy": None, "coverage": None, "selection": None,
                "generation": None, "harm": None, "rescue": None}
    covered = [r for r in rows if r["covered"]]
    blocked = [r for r in rows if not r["covered"]]
    initially_right = [r for r in rows if r["initial_correct"]]
    initially_wrong = [r for r in rows if not r["initial_correct"]]
    weighted = weights is not None
    weights = list(weights) if weighted else [1.] * n
    if len(weights) != n or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("weights must be finite, positive, and match observations")
    def mean(field, condition=lambda r: True):
        selected = [(r, w) for r, w in zip(rows, weights) if condition(r)]
        if not selected or any(r.get(field) is None for r, _ in selected):
            return None
        return sum(float(r[field]) * w for r, w in selected) / sum(w for _, w in selected)
    accuracy = mean("final_correct")
    pi = mean("covered")
    s = mean("final_correct", lambda r: r["covered"])
    g = mean("final_correct", lambda r: not r["covered"])
    initial = mean("initial_correct")
    right_final = mean("final_correct", lambda r: r["initial_correct"])
    harm = 1 - right_final if right_final is not None else None
    rescue = mean("final_correct", lambda r: not r["initial_correct"])
    return {
        "n": n, "accuracy": accuracy, "initial_accuracy": initial,
        "aggregation": "weighted_observations" if weighted else "observation_micro",
        "coverage": pi, "selection": s, "generation": g,
        "n_covered": len(covered), "n_blocked": len(blocked),
        "n_initially_correct": len(initially_right), "n_initially_wrong": len(initially_wrong),
        "harm": harm, "rescue": rescue,
        "covered_rescue": mean("final_correct", lambda r: r["covered"] and not r["initial_correct"]),
        "net_gain": accuracy - initial,
        "decomposed_accuracy": pi * (s or 0) + (1 - pi) * (g or 0),
        "rescue_harm_gain": (1 - initial) * (rescue or 0) - initial * (harm or 0),
        "pool_coverage": sum(w * bool(r.get("pool_covered", r["covered"])) for r, w in zip(rows, weights)) / sum(weights),
        "cost_usd_per_problem": mean("cost_usd"),
        "marginal_cost_usd_per_problem": mean("marginal_cost_usd"),
        "calls_per_problem": mean("calls"),
        "tokens_per_problem": (mean("tokens_in") + mean("tokens_out")) if mean("tokens_in") is not None and mean("tokens_out") is not None else None,
    }


def macro_decomposition(rows: Iterable[dict]) -> dict:
    """Equal benchmark mass, with all conditional rates under that same measure.

    Macro selection is coverage-weighted across benchmarks: averaging conditional
    selection rates independently would break the coverage/selection identity.
    Count fields remain raw observation counts, not weighted denominators.
    """
    rows = list(rows)
    sizes = defaultdict(int)
    for row in rows:
        sizes[row["benchmark"]] += 1
    if not rows:
        return decomposition([])
    result = decomposition(rows, [1 / sizes[row["benchmark"]] for row in rows])
    return {**result, "aggregation": "benchmark_macro", "n_benchmarks": len(sizes),
            "conditional_weighting": "conditional probabilities under equal benchmark mass; counts are raw"}


def grouped(rows: Iterable[dict], field: str) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get(field, "unknown"))].append(r)
    return {k: decomposition(v) for k, v in sorted(groups.items())}


def macro_accuracy(rows: Iterable[dict], field: str = "final_correct") -> float:
    """Equal benchmark weights, as in SAT's reported five-benchmark mean."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["benchmark"]].append(float(r[field]))
    if not groups:
        raise ValueError("no observations")
    return sum(sum(v) / len(v) for v in groups.values()) / len(groups)
