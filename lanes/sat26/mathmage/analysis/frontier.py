"""Anytime accuracy-cost frontiers from maximal policy runs.

The value-of-computation policy chooses actions by VOC per dollar, which does not
depend on the stopping threshold mu. A policy with threshold mu therefore stops at
the first logged step whose best action has ratio < mu (or where no action was
taken). One maximal run (mu = 0) per item yields every cheaper policy exactly.

Each graded step carries: calls, cost_usd (logical, including shared stage-0 work),
ratio of the chosen action (None at the final step), and correct (whether that
step's best answer is right; attached by the grader, never seen by agents).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def stop_index(steps: list[dict], mu: float) -> int:
    """Index of the step at which a policy with threshold ``mu`` stops."""
    if not steps:
        raise ValueError("cannot replay an empty anytime trajectory")
    if mu < 0 or not np.isfinite(mu):
        raise ValueError("mu must be finite and nonnegative")
    for i, step in enumerate(steps):
        chosen = step.get("chosen")
        if chosen is None or chosen.get("ratio", 0.0) < mu:
            return i
    return len(steps) - 1


def operating_point(graded_items: Iterable[dict], mu: float, weight_by: Optional[str] = "benchmark") -> dict:
    """Macro accuracy and mean logical cost/calls of the policy with threshold mu."""
    rows = []
    for item in graded_items:
        if not item.get("frontier_replay_valid", item.get("meta", {}).get("frontier_replay_valid", True)):
            raise ValueError("trajectory is not a replayable maximal threshold-policy run")
        steps = item["anytime"]
        if not steps:
            continue
        s = steps[stop_index(steps, mu)]
        rows.append({"benchmark": item.get("benchmark", "all"), "correct": float(bool(s["correct"])),
                     "cost": float(s["cost_usd"]), "calls": float(s["calls"])})
    if not rows:
        return {"mu": mu, "n": 0, "accuracy": None, "cost_usd": None, "calls": None}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["benchmark"] if weight_by else "all", []).append(r)
    acc = float(np.mean([np.mean([r["correct"] for r in g]) for g in groups.values()]))
    return {"mu": float(mu), "n": len(rows), "accuracy": acc,
            "micro_accuracy": float(np.mean([r["correct"] for r in rows])),
            "cost_usd": float(np.mean([r["cost"] for r in rows])),
            "calls": float(np.mean([r["calls"] for r in rows]))}


def frontier(graded_items: list[dict], mus: Optional[Iterable[float]] = None) -> list[dict]:
    """Descriptive curve, covering every threshold-induced prefix by default.

    Thresholds just above each logged ratio enumerate all distinct operating
    policies; a coarse log grid can miss a narrow but feasible budget point.
    Accuracy-based selection on this curve is an oracle unless frozen on dev data.
    """
    if mus is None:
        ratios = [s["chosen"]["ratio"] for it in graded_items for s in it["anytime"] if s.get("chosen")]
        if any(r < 0 or not np.isfinite(r) for r in ratios):
            raise ValueError("logged VOC/cost ratios must be finite and nonnegative")
        mus = [0.0] + sorted({float(np.nextafter(r, np.inf)) for r in ratios})
    points = [operating_point(graded_items, mu) for mu in mus]
    return sorted([p for p in points if p["n"]], key=lambda p: (p["cost_usd"], -p["accuracy"]))


def pareto(points: list[dict]) -> list[dict]:
    """Non-dominated subset (lower cost, higher accuracy)."""
    out, best = [], -1.0
    for p in sorted(points, key=lambda p: (p["cost_usd"], -p["accuracy"])):
        if p["accuracy"] > best + 1e-12:
            out.append(p)
            best = p["accuracy"]
    return out


def budget_matched(graded_items: list[dict], cost_usd: float, *, mu: Optional[float] = None,
                   development_items: Optional[list[dict]] = None, descriptive_oracle: bool = False) -> dict:
    """Evaluate a frozen threshold, or select it on separate development items.

    Selecting the best test accuracy under a test-cost cap is a descriptive
    oracle, not an unbiased compute-matched comparison. It requires an explicit
    opt-in and is labeled in the returned result. A development-selected policy
    can exceed the target test cost; report that rather than silently retuning it.
    """
    if cost_usd < 0 or not np.isfinite(cost_usd):
        raise ValueError("cost_usd must be finite and nonnegative")
    if mu is not None and (development_items is not None or descriptive_oracle):
        raise ValueError("choose exactly one threshold selection rule")
    if development_items is not None and descriptive_oracle:
        raise ValueError("choose development selection or descriptive oracle")
    if mu is None:
        if development_items is None and not descriptive_oracle:
            raise ValueError("supply frozen mu or development_items; test-label oracle requires descriptive_oracle=True")
        selection_items = graded_items if descriptive_oracle else development_items
        if development_items is not None:
            test_ids = {i.get("item_id") for i in graded_items if i.get("item_id")}
            dev_ids = {i.get("item_id") for i in development_items if i.get("item_id")}
            if test_ids & dev_ids:
                raise ValueError("development threshold selection must not overlap test items")
        feasible = [p for p in frontier(selection_items) if p["cost_usd"] <= cost_usd + 1e-12]
        if not feasible:
            return {"cost_usd": None, "target_cost_usd": cost_usd, "accuracy": None,
                    "selection": "no feasible development policy" if not descriptive_oracle else "no feasible oracle policy"}
        chosen = max(feasible, key=lambda p: (p["accuracy"], -p["cost_usd"]))
        mu = chosen["mu"]
        selection = "descriptive test-label oracle; not confirmatory" if descriptive_oracle else "development selected threshold"
    else:
        selection = "frozen threshold"
    result = operating_point(graded_items, mu)
    return {**result, "target_cost_usd": cost_usd,
            "within_budget": result["cost_usd"] is not None and result["cost_usd"] <= cost_usd + 1e-12,
            "selection": selection, "confirmatory": not descriptive_oracle}
