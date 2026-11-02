"""Rebuild all measured tables from separate graded observation files.

    python -m mathmage.analysis.make_tables --logs runs/e1_live/graded --output outputs/tables/e1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mathmage.analysis.metrics import decomposition, grouped, macro_accuracy, macro_decomposition
from mathmage.analysis.stats import cluster_interval, paired_contrast
from mathmage.experiments.artifacts import atomic_json


def load_rows(path: Path) -> list[dict]:
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    out = []
    for p in files:
        try:
            value = json.loads(p.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and {"final_correct", "covered", "initial_correct", "benchmark"} <= value.keys():
            out.append(value)
    keys = [(r.get("mode"), r["condition"], r["item_id"], r.get("seed", 0)) for r in out]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate graded observations; choose a single run directory")
    return out


def build_report(rows: list[dict], output: Path, samples=1000, expected_seeds: dict | None = None) -> dict:
    if not rows:
        raise ValueError("no graded observations found")
    modes = {r.get("mode", "unknown") for r in rows}
    if len(modes) != 1:
        raise ValueError("mock and live observations cannot be combined")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table, zones, anchors, contrasts = [], [], [], []
    conditions = sorted({r["condition"] for r in rows})
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        interval = cluster_interval(subset, samples=samples, expected_seeds=expected_seeds)
        table.append({"condition": condition, "benchmark": "macro_average", **macro_decomposition(subset),
                      "ci_low": interval["low"], "ci_high": interval["high"]})
        micro_ci = cluster_interval(subset, lambda d: decomposition(d)["accuracy"], samples=samples,
                                    expected_seeds=expected_seeds)
        table.append({"condition": condition, "benchmark": "micro_average", **decomposition(subset),
                      "ci_low": micro_ci["low"], "ci_high": micro_ci["high"]})
        for benchmark in sorted({r["benchmark"] for r in subset}):
            block = [r for r in subset if r["benchmark"] == benchmark]
            ci = cluster_interval(block, samples=samples)
            latency = [r.get("resource_latency_s", r.get("latency_s")) for r in block]
            wall = [r.get("wall_latency_s") for r in block]
            def quantile(values, q):
                return float(np.quantile(values, q)) if values and all(v is not None for v in values) else None
            table.append({"condition": condition, "benchmark": benchmark, **decomposition(block),
                          "ci_low": ci["low"], "ci_high": ci["high"],
                          "resource_service_latency_median_s": quantile(latency, .5),
                          "resource_service_latency_p95_s": quantile(latency, .95),
                          "wall_latency_median_s": quantile(wall, .5), "wall_latency_p95_s": quantile(wall, .95)})
        zones.extend({"condition": condition, "zone": z, **metrics} for z, metrics in grouped(subset, "zone").items())
        for name, field in [("member_vote", "initial_correct"), ("routing_oracle", "covered")]:
            ci = cluster_interval(subset, lambda d: macro_accuracy(d, field), samples=samples)
            anchors.append({"condition": condition, "anchor": name, **ci})
        members = sorted({m for r in subset for m in r.get("stage0_correct", {})})
        for member in members:
            obs = [{**r, "member_correct": r["stage0_correct"][member]} for r in subset if member in r.get("stage0_correct", {})]
            ci = cluster_interval(obs, lambda d: macro_accuracy(d, "member_correct"), samples=samples)
            anchors.append({"condition": condition, "anchor": member, **ci})
        if condition != "full" and "full" in conditions:
            full = [r for r in rows if r["condition"] == "full"]
            contrasts.append({"condition": condition, "comparator": "full", **paired_contrast(subset, full, samples=samples)})
    for name, records in [("accuracy", table), ("zones", zones), ("anchors", anchors), ("contrasts", contrasts)]:
        frame = pd.DataFrame(records)
        frame.to_csv(output / f"{name}.csv", index=False)
        (output / f"{name}.tex").write_text(frame.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    report = {"mode": next(iter(modes)), "status": "synthetic validation only" if modes == {"mock"} else "observed result",
              "n_observations": len(rows), "n_families": len({r["family_id"] for r in rows}),
              "conditions": conditions, "interval_method": "95% benchmark-membership-stratified family clusters and schedule-specific crossed-seed bootstrap",
              "primary_aggregation": "equal benchmark weight (macro average)",
              "decomposition_note": "Macro conditional rates use equal benchmark mass jointly; micro rows and pooled zone strata use observation weights. Raw count columns are not macro-weighted denominators.",
              "seed_schedule_validation": "declared schedule" if expected_seeds is not None else "balanced observed panels only; entirely missing planned seeds require manifest validation",
              "latency_note": "Resource service latency sums original call durations, including cached/inherited calls; it is not elapsed wall time. Missing wall measurements remain null.",
              "tools_status": {condition: {status: sum(r["condition"] == condition and r.get("tools_status", "unrecorded") == status for r in rows)
                                           for status in sorted({r.get("tools_status", "unrecorded") for r in rows if r["condition"] == condition})}
                               for condition in conditions},
              "comparison_note": "Published SAT/Xolver scores use different rosters/splits and are reference-only; no significance test against their rounded averages.",
              "accuracy": table, "zones": zones, "anchors": anchors, "contrasts": contrasts}
    atomic_json(output / "report.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/tables/rebuilt"))
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--expected-seeds", type=Path, help="JSON mapping benchmark names to the complete planned seed lists")
    args = ap.parse_args()
    expected_seeds = json.loads(args.expected_seeds.read_text(encoding="utf-8")) if args.expected_seeds else None
    report = build_report(load_rows(args.logs), args.output, args.bootstrap_samples, expected_seeds)
    print(json.dumps({k: v for k, v in report.items() if k not in {"accuracy", "zones", "anchors", "contrasts"}}, indent=2))


if __name__ == "__main__":
    main()
