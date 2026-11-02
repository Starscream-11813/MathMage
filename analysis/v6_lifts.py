"""S8 (pre-registered with v6): hint lifts refitted on the crossed hint trial, and proximal-zone shares (no calls).

The lifts use the frozen IRT-consistent estimator (mathmage/psychometrics/calibrate.py: fit_coaching_lifts) under
each team's frozen abilities, difficulty spread and same-author correlation. Each coached re-solve is scored
against its own state's stage-0 panel, as at test time. Fits use the training replays alone (the published
lifts), the trial alone, and both together. The zone shares then integrate the zones of development (Definition 1)
over the difficulty prior, as for the paper's ZPD figure, with the right tutor's lift replaced by each refit.

    python analysis/v6_lifts.py --package sat26
    python analysis/v6_lifts.py --package lane
Writes results/tables/v6_lifts_<package>.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = sys.argv[sys.argv.index("--package") + 1] if "--package" in sys.argv else "lane"
_argv = sys.argv
sys.argv = [sys.argv[0], "--package", PACKAGE]
_spec = importlib.util.spec_from_file_location("v6_fit", REPO / "scripts" / "v6_fit.py")
FIT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FIT)
sys.argv = _argv

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import norm  # noqa: E402

from mathmage.experiments.artifacts import read_frozen  # noqa: E402
from mathmage.psychometrics.calibrate import ItemRecord, Response, fit_coaching_lifts  # noqa: E402

F6 = FIT.F6
TAU = 0.5
GRID = np.linspace(-9, 11, 4001)


def no_success(prof, rho, K, b):
    """P(no correct usable answer in K fresh attempts), as in analysis/fig_zpd.py and TeamPosterior.member_zones."""
    conditional = expit(prof["a"] * (prof["theta"] - b))
    abstain = prof["abstain"]
    if rho == 0:
        return (1 - (1 - abstain) * conditional) ** K
    conc = (1 - rho) / rho
    none, wrong = np.zeros_like(b), np.ones_like(b)
    for usable in range(K + 1):
        none += comb(K, usable) * (1 - abstain) ** usable * abstain ** (K - usable) * wrong
        wrong = wrong * (conc * (1 - conditional) + usable) / (conc + usable)
    return none


def zones(prof, rho, lift, K, b):
    p = (1 - prof["abstain"]) * expit(prof["a"] * (prof["theta"] - b))
    reach = 1 - no_success(prof, rho, K, b)
    c = (1 - prof["abstain"]) * expit(prof["a"] * (prof["theta"] + lift - b))
    return np.where(p >= TAU, 0, np.where(reach >= TAU, 1, np.where(c >= TAU, 2, 3)))


def shares(pp, lift):
    prior = norm.pdf(GRID, pp["b_mean"], pp["b_sd"])
    prior /= prior.sum()
    out = {}
    for key, prof in pp["agents"].items():
        if "@" in key or key == "tool":
            continue
        out[key] = {K: {zz: float(prior[zones(prof, pp["same_agent_correlation"], lift, K, GRID) == i].sum())
                        for i, zz in enumerate("ALPU")} for K in (1, 2, 3, 4, 8)}
    return out


def main():
    plan = F6.load_plan(PACKAGE)
    frozen = read_frozen(F6.SPECS[PACKAGE]["artifact"], live=True)
    pp = frozen["posterior_params"]
    theta = {k: v["theta"] for k, v in pp["agents"].items()}
    mech = FIT.mechanism_data(PACKAGE, plan)
    train = FIT.training_data(PACKAGE)
    states = {}
    for record in mech["records"]:
        grade = F6.correct_of(record["item_id"])
        state = f"{record['item_id']}#s{int(record['seed'])}"
        states[state] = ItemRecord(state, F6.item_of(record["item_id"]).answer_format,
                                   [Response(FIT.agent_of(c), c["answer"] if FIT.usable(c) else None,
                                             bool(FIT.usable(c) and grade(c["answer"])))
                                    for c in record["stage0_candidates"]], "")
    items = {**train["items"], **states}
    args = (theta, pp["b_sd"], pp["same_agent_correlation"])
    fits = {"training replays (published)": fit_coaching_lifts(train["coaching"], items, *args),
            "hint trial": fit_coaching_lifts(mech["coaching"], items, *args),
            "training + trial": fit_coaching_lifts(train["coaching"] + mech["coaching"], items, *args)}
    out = {"package": PACKAGE, "plan_sha256": plan["sha256"], "artifact_sha256": frozen["sha256"],
           "frozen_lift": pp["lift"], "n_trial_records": len(mech["coaching"]),
           "n_trial_right": sum(r.tutor_right for r in mech["coaching"]),
           "n_training_records": len(train["coaching"]),
           "hint_delivery": {"delivered": int(sum(mech["hints"])), "offered": len(mech["hints"])},
           "fits": fits,
           "horizon": {K: float(np.log(TAU / (1 - TAU)) - np.log((1 - (1 - TAU) ** (1 / K)) / (1 - TAU) ** (1 / K)))
                       for K in (1, 2, 3, 4, 8)},
           "shares": {name: shares(pp, fit.get("coached_right", fit.get("coached", 0.0)))
                      for name, fit in fits.items()}}
    path = REPO / "results" / "tables" / f"v6_lifts_{PACKAGE}.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("package", "n_trial_records", "n_trial_right", "hint_delivery")}, indent=1))
    for name, fit in fits.items():
        print(name, {k: round(float(v), 3) for k, v in fit.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
