"""Family-cluster bootstrap, paired tests, and multiplicity adjustment."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import numpy as np
from scipy.stats import binomtest

from mathmage.analysis.metrics import macro_accuracy


def cluster_interval(rows: Iterable[dict], statistic: Callable = macro_accuracy,
                     samples: int = 2000, seed: int = 0, expected_seeds: dict | None = None) -> dict:
    """Resample families within benchmark-membership strata and crossed seed blocks.

    A family shared by benchmarks is sampled as one cluster within its membership
    stratum. Seed draws are shared across benchmarks with the same seed schedule.
    Different schedules (e.g. one physics seed versus three AIME seeds) get their
    own crossed draws. Partially missing item/seed panels are rejected.
    """
    rows = list(rows)
    if not rows or samples < 1:
        raise ValueError("rows and positive bootstrap sample count required")
    families = defaultdict(list)
    observed_keys = set()
    benchmark_seeds, item_seeds = defaultdict(set), defaultdict(set)
    for r in rows:
        key = (r["benchmark"], r["item_id"], r.get("seed", 0))
        if key in observed_keys:
            raise ValueError(f"duplicate observation in bootstrap population: {key}")
        observed_keys.add(key)
        benchmark_seeds[r["benchmark"]].add(r.get("seed", 0))
        item_seeds[(r["benchmark"], r["item_id"])].add(r.get("seed", 0))
        families[r.get("family_id", r["item_id"])].append(r)
    if expected_seeds is not None and set(expected_seeds) != set(benchmark_seeds):
        raise ValueError("declared seed schedule must match the observed benchmark set")
    for (benchmark, item), observed in item_seeds.items():
        scheduled = set(expected_seeds[benchmark]) if expected_seeds is not None else benchmark_seeds[benchmark]
        if observed != scheduled:
            raise ValueError(f"incomplete seed panel for {benchmark}/{item}: expected {sorted(scheduled)}, observed {sorted(observed)}")
    ids = sorted(families)
    seeds = sorted({r.get("seed", 0) for r in rows})
    schedules = {benchmark: tuple(sorted(values)) for benchmark, values in benchmark_seeds.items()}
    strata = defaultdict(list)
    for family, observations in families.items():
        strata[tuple(sorted({r["benchmark"] for r in observations}))].append(family)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        seed_draws = {panel: rng.choice(panel, len(panel), replace=True).tolist() for panel in sorted(set(schedules.values()))}
        draw = []
        for stratum in sorted(strata):
            for family in rng.choice(strata[stratum], len(strata[stratum]), replace=True):
                for benchmark in stratum:
                    for sampled_seed in seed_draws[schedules[benchmark]]:
                        draw.extend(r for r in families[family] if r["benchmark"] == benchmark and r.get("seed", 0) == sampled_seed)
        estimate = statistic(draw)
        if estimate is not None and np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        raise ValueError("bootstrap could not produce complete draws")
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": float(statistic(rows)), "low": float(lo), "high": float(hi),
            "level": 0.95, "samples": len(estimates), "families": len(ids), "seeds": len(seeds),
            "seed_panels": {benchmark: list(panel) for benchmark, panel in schedules.items()},
            "seed_schedule_validation": "declared schedule" if expected_seeds is not None else "balanced observed schedule; entirely absent planned seeds require manifest validation",
            "method": "benchmark-membership-stratified family clusters and schedule-specific crossed-seed percentile bootstrap"}


def paired_contrast(left: Iterable[dict], right: Iterable[dict], samples: int = 2000, seed: int = 0) -> dict:
    def index(rows):
        out = {}
        for r in rows:
            key = (r["item_id"], r.get("seed", 0))
            if key in out:
                raise ValueError(f"duplicate paired observation: {key}")
            out[key] = r
        return out
    a, b = index(left), index(right)
    if a.keys() != b.keys() or not a:
        raise ValueError("paired contrasts require exactly the same nonempty item/seed set")
    rows, wins, losses = [], 0, 0
    for k, x in a.items():
        y = b[k]
        if (x["family_id"], x["benchmark"]) != (y["family_id"], y["benchmark"]):
            raise ValueError("paired provenance differs")
        xv, yv = bool(x["final_correct"]), bool(y["final_correct"])
        wins += xv and not yv
        losses += yv and not xv
        rows.append({**x, "delta": int(xv) - int(yv)})
    ci = cluster_interval(rows, lambda d: macro_accuracy(d, "delta"), samples, seed)
    # Exact McNemar assumes independent pairs; repeated seeds/related items violate
    # that assumption. Supply only the clustered interval in those cases.
    independent = len({r["family_id"] for r in rows}) == len(rows)
    p = float(binomtest(wins, wins + losses).pvalue) if independent and wins + losses else (1.0 if independent else None)
    return {**ci, "wins": wins, "losses": losses, "mcnemar_exact_p": p,
            "mcnemar_note": "unweighted independent item-pair test (micro); the interval targets benchmark macro accuracy" if independent else "omitted: repeated seeds or related items; use clustered interval"}


def holm(pvalues: list[float]) -> list[float]:
    if any(not 0 <= p <= 1 for p in pvalues):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(pvalues)
    out, previous = [0.0] * len(pvalues), 0.0
    for rank, idx in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[idx]))
        out[idx] = previous
    return out
