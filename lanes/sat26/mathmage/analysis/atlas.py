"""Descriptive solo-sampling evidence from public MathArena configurations.

These logs contain no matched scaffold experiments. In particular, low solo success
cannot distinguish a proximal item from an unreachable one. The posterior zones
below describe solo success probability under each public configuration only.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import beta

from mathmage.data.loaders import ERRATA, matharena_outputs
from mathmage.data.registry import load_registry

COMPETITIONS = ("aime_2025", "aime_2026", "hmmt_feb_2025", "hmmt_feb_2026")
ZONES = ("autonomous", "latent", "unassisted_below_sampling_threshold")
REQUIRED_COLUMNS = ["problem_idx", "model_name", "model_config", "idx_answer", "correct"]


def posterior_zones(successes: int, trials: int, *, tau: float = 0.5, horizon_k: int = 8,
                    alpha: float = 0.05, confidence: float = 0.8,
                    prior_alpha: float = 1.0, prior_beta: float = 1.0) -> dict:
    """Beta-Binomial solo zones; ``trials`` is observed n, not the horizon K.

    The latent interval is [tau_K, tau), where tau_K = 1-(1-tau)**(1/K).
    The one-sided zero-success bound is frequentist and uses observed n.
    """
    if not isinstance(trials, int) or not isinstance(successes, int) or not 0 <= successes <= trials or trials < 1:
        raise ValueError("require integer trials >= 1 and 0 <= successes <= trials")
    if not 0 < tau < 1 or not isinstance(horizon_k, int) or horizon_k < 1:
        raise ValueError("require 0 < tau < 1 and integer horizon_k >= 1")
    if not 0 < alpha < 1 or not 0 < confidence <= 1 or prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("invalid posterior prior, confidence, or alpha")
    tau_k = -math.expm1(math.log1p(-tau) / horizon_k)
    a, b = successes + prior_alpha, trials - successes + prior_beta
    below = float(beta.cdf(tau_k, a, b))
    autonomous = float(beta.sf(tau, a, b))
    masses = (autonomous, max(0.0, 1 - below - autonomous), below)
    winner = int(np.argmax(masses))
    return {
        "n_trials": trials, "n_successes": successes, "empirical_pass1": successes / trials,
        "tau": tau, "horizon_k": horizon_k, "tau_k": tau_k,
        "posterior_mean": a / (a + b),
        "posterior_low": float(beta.ppf(alpha / 2, a, b)),
        "posterior_high": float(beta.ppf(1 - alpha / 2, a, b)),
        **{f"p_{zone}": mass for zone, mass in zip(ZONES, masses)},
        "inferred_zone": ZONES[winner] if masses[winner] >= confidence else "inconclusive",
        "zone_confidence": masses[winner],
        "zero_success_upper_bound": -math.expm1(math.log(alpha) / trials) if successes == 0 else None,
    }


def _family_interval(group: pd.DataFrame, samples: int, seed: int, alpha: float) -> tuple:
    """Resample families, keeping all item rates in each sampled family together."""
    families = group.groupby("family_id").empirical_pass1.agg(["sum", "count"])
    if len(families) < 2 or samples < 1:
        return None, None
    rng = np.random.default_rng(seed)
    draw = rng.integers(len(families), size=(samples, len(families)))
    rates = families["sum"].to_numpy()[draw].sum(axis=1) / families["count"].to_numpy()[draw].sum(axis=1)
    return tuple(float(v) for v in np.quantile(rates, [alpha / 2, 1 - alpha / 2]))


def summarize_items(items: pd.DataFrame, *, bootstrap_samples: int = 1000, seed: int = 0,
                    alpha: float = 0.05) -> pd.DataFrame:
    """Per-public-configuration aggregates, weighting items equally for mean pass@1."""
    rows = []
    keys = ["benchmark", "model_name", "model_config", "tau", "horizon_k"]
    for key, group in items.groupby(keys, sort=True, dropna=False):
        lo, hi = _family_interval(group, bootstrap_samples, seed, alpha)
        rows.append({
            **dict(zip(keys, key)), "n_items": len(group), "n_families": group.family_id.nunique(),
            "n_trials": int(group.n_trials.sum()), "n_successes": int(group.n_successes.sum()),
            "attempt_accuracy": float(group.n_successes.sum() / group.n_trials.sum()),
            "mean_item_pass1": float(group.empirical_pass1.mean()),
            "mean_item_pass1_ci_low": lo, "mean_item_pass1_ci_high": hi,
            "n_zero_success_items": int((group.n_successes == 0).sum()),
            "min_observed_trials": int(group.n_trials.min()), "max_observed_trials": int(group.n_trials.max()),
            **{f"mean_p_{zone}": float(group[f"p_{zone}"].mean()) for zone in ZONES},
            **{f"n_{zone}": int((group.inferred_zone == zone).sum()) for zone in (*ZONES, "inconclusive")},
        })
    return pd.DataFrame(rows)


def analyze_attempts(frames: Mapping[str, pd.DataFrame], *, item_metadata: Mapping[str, dict] | None = None,
                     tau: float = 0.5, horizon_k: int = 8, alpha: float = 0.05,
                     confidence: float = 0.8, bootstrap_samples: int = 1000, seed: int = 0,
                     exclude_errata: bool = True) -> tuple[dict[str, pd.DataFrame], dict]:
    """Analyze source correctness labels without regrading or using reference answers."""
    metadata = item_metadata or {}
    counts, sources, rows = {}, {}, []
    for benchmark, raw in sorted(frames.items()):
        missing = set(REQUIRED_COLUMNS) - set(raw.columns)
        if missing:
            raise ValueError(f"{benchmark}: missing columns {sorted(missing)}")
        frame = raw[REQUIRED_COLUMNS].copy()
        if frame[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError(f"{benchmark}: missing model, attempt, item, or correctness values")
        if not all(isinstance(v, (bool, np.bool_)) for v in frame.correct):
            raise ValueError(f"{benchmark}: correct must contain booleans")
        frame["item_id"] = [f"{benchmark}/{int(v):02d}" for v in frame.problem_idx]
        duplicate_keys = ["item_id", "model_name", "model_config", "idx_answer"]
        if frame.duplicated(duplicate_keys).any():
            raise ValueError(f"{benchmark}: duplicate public attempt identifiers")
        excluded = frame.item_id.isin(ERRATA) if exclude_errata else pd.Series(False, index=frame.index)
        counts[benchmark] = {"source_attempts": len(frame), "excluded_errata_attempts": int(excluded.sum())}
        frame = frame[~excluded]
        sources[benchmark] = {
            "repository": raw.attrs.get("source_repo", f"MathArena/{benchmark}_outputs"),
            "revision": raw.attrs.get("source_revision"),
            "attempt_dates": "not supplied by source dataset",
        }
        for (iid, name, config), group in frame.groupby(["item_id", "model_name", "model_config"], sort=True):
            meta = metadata.get(iid, {})
            n, s = len(group), int(group.correct.sum())
            rows.append({
                "benchmark": benchmark, "item_id": iid, "family_id": meta.get("family_id", f"fam-{iid}"),
                "item_date": meta.get("item_date", ""), "model_name": name, "model_config": config,
                "configuration_scope": "public configuration; not matched to live roster",
                "correctness_source": "MathArena public labels; not regraded",
                "known_erratum": iid in ERRATA,
                **posterior_zones(s, n, tau=tau, horizon_k=horizon_k, alpha=alpha, confidence=confidence),
            })
    if not rows:
        raise ValueError("no public attempts remain to analyze")
    items = pd.DataFrame(rows)
    summary = summarize_items(items, bootstrap_samples=bootstrap_samples, seed=seed, alpha=alpha)
    sensitivity = []
    a = items.n_successes.to_numpy() + 1
    b = items.n_trials.to_numpy() - items.n_successes.to_numpy() + 1
    for trial_tau in (0.4, 0.5, 0.6):
        for horizon in (4, 8, 16):
            variants = items.copy()
            tau_k = -math.expm1(math.log1p(-trial_tau) / horizon)
            below, autonomous = beta.cdf(tau_k, a, b), beta.sf(trial_tau, a, b)
            masses = np.column_stack([autonomous, np.maximum(0, 1 - below - autonomous), below])
            for column, zone in enumerate(ZONES):
                variants[f"p_{zone}"] = masses[:, column]
            winners = masses.argmax(axis=1)
            variants["zone_confidence"] = masses.max(axis=1)
            variants["inferred_zone"] = np.where(masses.max(axis=1) >= confidence,
                                                   np.asarray(ZONES)[winners], "inconclusive")
            variants["tau"], variants["horizon_k"], variants["tau_k"] = trial_tau, horizon, tau_k
            sensitivity.append(summarize_items(variants, bootstrap_samples=0, alpha=alpha))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_source_attempts": sum(c["source_attempts"] for c in counts.values()),
        "n_analyzed_attempts": int(items.n_trials.sum()), "n_item_configuration_pairs": len(items),
        "n_public_configurations": len(items[["model_name", "model_config"]].drop_duplicates()),
        "source_counts": counts, "sources": sources,
        "settings": {"tau": tau, "horizon_k": horizon_k, "prior": "Beta(1,1)", "alpha": alpha,
                     "zone_confidence": confidence, "bootstrap_samples": bootstrap_samples,
                     "bootstrap_seed": seed, "exclude_errata": exclude_errata},
        "interpretation": [
            "Public model/configuration names are retained; configurations are not matched to the live roster.",
            "Source correctness labels are used verbatim; they are not grader-validation ground truth.",
            "Autonomous: p>=tau; latent: tau_K<=p<tau; unassisted_below_sampling_threshold: p<tau_K.",
            "tau_K=1-(1-tau)^(1/K); K is a sampling horizon, not the number of observed attempts n.",
            "A label is inconclusive when no zone has posterior mass >= the confidence threshold.",
            "No scaffold data exist here: proximal and unreachable status cannot be identified.",
            "Zero observed successes does not imply zero success probability; its exact one-sided bound uses actual n.",
            "Posterior and exact-binomial bounds assume independent, exchangeable attempts within a public configuration.",
            "Bootstrap intervals resample item families and describe item variation, not shared-prompt or model dependence.",
            "Source attempt timestamps are unavailable; item dates are benchmark dates from the local registry.",
        ],
    }
    return {"item_posteriors": items, "model_summary": summary,
            "threshold_sensitivity": pd.concat(sensitivity, ignore_index=True)}, report


def build_atlas(*, competitions=COMPETITIONS, allow_download: bool = False, **kwargs):
    """Read only cached datasets by default. Downloads require explicit opt-in."""
    registry = load_registry()
    metadata = {it.item_id: {"family_id": it.family_id, "item_date": it.item_date} for it in registry.all()}
    frames = {name: matharena_outputs(name, columns=REQUIRED_COLUMNS, local_files_only=not allow_download)
              for name in competitions}
    return analyze_attempts(frames, item_metadata=metadata, **kwargs)
