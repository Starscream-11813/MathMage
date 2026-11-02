"""Stopping rules on identical trajectories, and sensitivity to the frozen price (read-only; no API calls).

    python analysis/arr4_stopping.py            # tables + figure (LuaLaTeX)
    python analysis/arr4_stopping.py --draft

Every sealed MathMage run records its anytime trajectory: the actions in the controller's order with the
posterior confidence, cumulative cost and correctness after each. Because the action order does not depend
on the stopping rule (the prefix property of Proposition 5 in THEORY.md), alternative stopping rules can be
replayed exactly on the same trajectories:

  VOC price mu      stop at the first step whose best value-per-dollar ratio is below mu (MathMage)
  confidence tau    stop at the first step whose posterior confidence is at least tau
  fixed k           stop after k actions (k = 0 is the stage-0 posterior pick; k = max is the maximal run)

Each rule traces an accuracy-cost curve. Curves are compared at matched mean cost (linear interpolation in
cost), on every evaluation set. Also reported: accuracy and cost at mu*/4 ... 4 mu*, to show how much the
conclusions depend on the development-frozen price.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import v4_results as V4R  # noqa: E402
from mathmage.analysis.frontier import stop_index  # noqa: E402

R = V4R.R
OUT_T, OUT_F = REPO / "results/tables", REPO / "results/figures"
SAMPLES = 2000


def trajectories(rows):
    return [{"benchmark": r["benchmark"], "item_id": r["item_id"], "family_id": r.get("family_id", r["item_id"]),
             "seed": int(r.get("seed", 0)), "steps": r["anytime"]} for r in rows]


def outcome(traj, pick):
    units = []
    for t in traj:
        step = t["steps"][pick(t["steps"])]
        units.append({"benchmark": t["benchmark"], "item_id": t["item_id"], "family_id": t["family_id"],
                      "seed": t["seed"], "final_correct": float(bool(step["correct"])), "cost_usd": float(step["cost_usd"])})
    return units


def macro(units):
    return V4R._macro(np.array([u["final_correct"] for u in units]),
                      np.array([sorted({u["benchmark"] for u in units}).index(u["benchmark"]) for u in units]),
                      len({u["benchmark"] for u in units}))


def curve(traj, rule, params):
    points = []
    for p in params:
        units = outcome(traj, lambda steps: rule(steps, p))
        points.append({"param": p, "accuracy": macro(units), "cost": float(np.mean([u["cost_usd"] for u in units]))})
    return points


def by_confidence(steps, tau):
    for i, s in enumerate(steps):
        if float(s.get("confidence") or 0) >= tau:
            return i
    return len(steps) - 1


def by_count(steps, k):
    return min(k, len(steps) - 1)


def interpolate(points, cost):
    """Accuracy of a curve at a given mean cost (linear in cost between the bracketing points)."""
    pts = sorted(points, key=lambda p: p["cost"])
    if cost <= pts[0]["cost"]:
        return pts[0]["accuracy"]
    for a, b in zip(pts, pts[1:]):
        if a["cost"] <= cost <= b["cost"]:
            w = (cost - a["cost"]) / (b["cost"] - a["cost"]) if b["cost"] > a["cost"] else 0
            return a["accuracy"] + w * (b["accuracy"] - a["accuracy"])
    return pts[-1]["accuracy"]


def analyse(name, rows, mu_star):
    traj = trajectories(rows)
    max_steps = max(len(t["steps"]) for t in traj) - 1
    mus = sorted(set([0.0] + list(np.geomspace(mu_star / 64, mu_star * 64, 25)) + [mu_star * f for f in (.25, .5, 1, 2, 4)]))
    voc = curve(traj, lambda steps, mu: stop_index(steps, mu), mus)
    conf = curve(traj, by_confidence, [round(x, 3) for x in np.linspace(0, 1, 41)])
    fixed = curve(traj, by_count, list(range(0, max_steps + 1)))
    mm = outcome(traj, lambda steps: stop_index(steps, mu_star))
    cost_star = float(np.mean([u["cost_usd"] for u in mm]))
    acc_star = macro(mm)
    # the confidence rule's threshold that matches MathMage's mean cost, then a paired bootstrap at that point
    taus = [p for p in conf if p["cost"] >= cost_star]
    tau = min(taus, key=lambda p: p["cost"])["param"] if taus else conf[-1]["param"]
    matched = outcome(traj, lambda steps: by_confidence(steps, tau))
    ks = [p for p in fixed if p["cost"] >= cost_star]
    k = min(ks, key=lambda p: p["cost"])["param"] if ks else fixed[-1]["param"]
    matched_k = outcome(traj, lambda steps: by_count(steps, k))
    sensitivity = {f"{f:g} x mu*": {"accuracy": macro(outcome(traj, lambda steps, m=mu_star * f: stop_index(steps, m))),
                                    "cost": float(np.mean([u["cost_usd"] for u in outcome(
                                        traj, lambda steps, m=mu_star * f: stop_index(steps, m))]))}
                   for f in (0.25, 0.5, 1, 2, 4)}
    return {"name": name, "n": len(traj), "mu_star": mu_star, "mu_star_point": {"accuracy": acc_star, "cost": cost_star},
            "curves": {"voc": voc, "confidence": conf, "fixed": fixed},
            "at_mu_star_cost": {"confidence (interpolated)": interpolate(conf, cost_star),
                                "fixed k (interpolated)": interpolate(fixed, cost_star)},
            "matched_confidence": {"tau": tau, "cost": float(np.mean([u["cost_usd"] for u in matched])),
                                   "contrast": V4R.contrast(mm, matched, SAMPLES)},
            "matched_fixed": {"k": k, "cost": float(np.mean([u["cost_usd"] for u in matched_k])),
                              "contrast": V4R.contrast(mm, matched_k, SAMPLES)},
            "sensitivity": sensitivity}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--figure-only", action="store_true", help="redraw from arr4_stopping.json")
    args = parser.parse_args()
    if args.figure_only:
        figure(json.loads((OUT_T / "arr4_stopping.json").read_text(encoding="utf-8")), args.draft)
        return
    e1, e2 = R.STUDIES
    a1, a2 = (json.loads(s.artifact.read_text(encoding="utf-8")) for s in (e1, e2))
    rows1, _, _ = R.load_phase(e1.root, e1.phase)
    rows2, _, _ = R.load_phase(e2.root, e2.phase)
    new1 = V4R.mathmage_rows(["new_e1_mathmage_nov", "new_e1_mathmage_apr"])
    new2 = V4R.mathmage_rows(["new_e2_mathmage_nov", "new_e2_mathmage_apr"])
    results = [analyse("SAT roster · E1 suite", rows1, float(a1["mu_star"])),
               analyse("SAT roster · 153 new contest problems", new1, float(a1["mu_star"])),
               analyse("Open roster · E2 suite", rows2, float(a2["mu_star"])),
               analyse("Open roster · new and 2026 contests", new2, float(a2["mu_star"]))]
    (OUT_T / "arr4_stopping.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    lines = ["# Stopping rules on identical trajectories (no API calls)", "",
             "Same actions in the same order; only the stopping rule differs. Matched-cost contrasts use the "
             "confidence threshold (or action count) whose mean cost is closest at or above MathMage's.", "",
             "| Set | n | MathMage at mu* | confidence rule at same cost | fixed-k at same cost | mu* − confidence (matched) | "
             "mu* − fixed k (matched) |", "|---|---|---|---|---|---|---|"]
    for r in results:
        mc, mf = r["matched_confidence"], r["matched_fixed"]
        fmt = lambda c: (f"{100 * c['estimate']:+.1f} [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}], "  # noqa: E731
                         f"{c['wins']}/{c['losses']}, p = {c['p']:.3f}")
        lines.append(f"| {r['name']} | {r['n']} | {100 * r['mu_star_point']['accuracy']:.1f} @ ${r['mu_star_point']['cost']:.4f} | "
                     f"{100 * r['at_mu_star_cost']['confidence (interpolated)']:.1f} | "
                     f"{100 * r['at_mu_star_cost']['fixed k (interpolated)']:.1f} | "
                     f"{fmt(mc['contrast'])} (tau = {mc['tau']}, ${mc['cost']:.4f}) | "
                     f"{fmt(mf['contrast'])} (k = {mf['k']}, ${mf['cost']:.4f}) |")
    lines += ["", "## Sensitivity to the frozen price", "", "| Set | " + " | ".join(results[0]["sensitivity"]) + " |",
              "|---|" + "---|" * len(results[0]["sensitivity"])]
    for r in results:
        lines.append(f"| {r['name']} | " + " | ".join(f"{100 * v['accuracy']:.1f} @ ${v['cost']:.4f}"
                                                     for v in r["sensitivity"].values()) + " |")
    (OUT_T / "arr4_stopping.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    figure(results, args.draft)


SHORT = {"SAT roster · E1 suite": "SAT · E1 suite", "SAT roster · 153 new contest problems": "SAT · 153 new problems",
         "Open roster · E2 suite": "Open · E2 suite", "Open roster · new and 2026 contests": "Open · new and 2026"}


def figure(results, draft):
    from mathmage.analysis import style
    style.use(publication=not draft, base_size=8.0)
    import matplotlib.pyplot as plt
    P = style.P
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for ax, r in zip(axes, results):
        for key, color, label in (("fixed", P["faint"], "fixed number of actions"),
                                  ("confidence", P["cyan_deep"], "confidence threshold"),
                                  ("voc", P["purple"], "value-of-computation price")):
            pts = sorted(r["curves"][key], key=lambda p: p["cost"])
            ax.plot([p["cost"] for p in pts], [100 * p["accuracy"] for p in pts], lw=1.1, color=color, label=label,
                    marker="o" if key == "fixed" else None, ms=2)
        ax.plot(r["mu_star_point"]["cost"], 100 * r["mu_star_point"]["accuracy"], "*", ms=7, color=P["pink_deep"],
                label=r"MathMage at $\mu^\star$" if not draft else "MathMage at mu*", zorder=5)
        name = SHORT.get(r["name"], r["name"])
        ax.set_title(name.replace("·", r"$\cdot$") if not draft else name, fontsize=7, loc="left")
        ax.set_xlabel("cost per problem (USD)", fontsize=6.6)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=.5)
    axes[0].set_ylabel("macro accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.4, bbox_to_anchor=(.5, -.04))
    fig.tight_layout(rect=(0, .1, 1, 1))
    style.save(fig, "arr4_stopping", OUT_F)



if __name__ == "__main__":
    main()
