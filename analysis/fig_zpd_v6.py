"""The zone of proximal development with the fourth wave's randomized hint trial (read-only; no API calls).

    python analysis/fig_zpd_v6.py            # publication figure (LuaLaTeX)
    python analysis/fig_zpd_v6.py --draft    # fast preview

Panels (a) and (b) are those of analysis/fig_zpd.py, from the frozen fitted models; (b) adds the SAT-style 2026 team's
members, from its own calibration. Panel (c) replaces the training replays with the crossed, hash-assigned hint
trial of the fourth wave (analysis/v6_results.py, S8): the share of initially wrong members rescued by one fresh
attempt, two fresh attempts, a placebo re-solve, a wrong tutor's hint and a right tutor's hint (intention to treat;
the hollow marker is delivered hints only), with problem-clustered 95% intervals. Every arm is shown on the units
that have a right tutor (a teammate was right), so the bars compare the same solvers and states.
Writes results/figures/zpd_v6.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MATHMAGE_PALETTE"] = "paper"            # the shared nine colours; must precede the style import

import numpy as np
from scipy.special import expit
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import fig_zpd as Z  # noqa: E402
from mathmage.analysis import style  # noqa: E402

OUT_F, OUT_T = REPO / "results/figures", REPO / "results/tables"
Z.NAMES.update({"gpt-5-mini-low": "GPT-5-mini"})
Z.LOGOS.update({"gpt-5-mini-low": "gpt"})
SAT26 = Z.RUNS / "program_v6" / "offline_sat26"
ARMS = [("fresh attempt (k=1)", "one attempt"), ("best of two fresh attempts", "two attempts"),
        ("placebo re-solve", "placebo"), ("wrong tutor's hint (ITT)", "wrong tutor"),
        ("right tutor's hint (ITT)", "right tutor")]


def sat26_shares():
    """Proximal shares of the SAT-style 2026 members under their own calibration (as fig_zpd.numbers)."""
    pp = json.loads((SAT26 / "artifacts.json").read_text(encoding="utf-8"))["posterior_params"]
    right = pp["lift"].get("coached_right", pp["lift"]["coached"])
    prior = norm.pdf(Z.GRID, pp["b_mean"], pp["b_sd"])
    prior /= prior.sum()
    out = {}
    for key, prof in pp["agents"].items():
        if "@" in key or key == "tool":
            continue
        out[key] = {K: float(prior[Z.zones(prof, pp["same_agent_correlation"], right, K, Z.GRID) == 2].sum())
                    for K in (1, 2, 3, 4, 6, 8)}
    return out


def figure(nums, trial, draft):
    style.use(publication=not draft, base_size=8.0)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    P = style.P
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.05), gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})
    a, bax, c = axes

    # (a) as in fig_zpd.py: o3-mini (SAT-style team) under its fitted model
    team = nums["teams"]["SAT-style"]
    art = json.loads((Z.TEAMS["SAT-style"] / "artifacts.json").read_text(encoding="utf-8"))["posterior_params"]
    prof, rho = art["agents"]["o3-mini-low"], team["rho"]
    one = lambda b: (1 - prof["abstain"]) * expit(prof["theta"] - b)                       # noqa: E731
    two = lambda b: 1 - Z.no_success(prof, rho, 2, b)                                      # noqa: E731
    hint = lambda b: (1 - prof["abstain"]) * expit(prof["theta"] + team["lift_right"] - b)  # noqa: E731
    bad = lambda b: (1 - prof["abstain"]) * expit(prof["theta"] + team["lift_wrong"] - b)   # noqa: E731
    b = np.linspace(-3.6, 4.2, 900)
    bA, bP, b2 = Z.boundary(one), Z.boundary(hint), Z.boundary(two)
    a.add_patch(Rectangle((-3.6, 0), bA + 3.6, 1.08, color=style.tint(P["peri_inner"], .74), lw=0, zorder=0))
    a.add_patch(Rectangle((bA, 0), bP - bA, 1.08, facecolor=style.tint(P["pink"], .55), edgecolor="white",
                          hatch="////", lw=0, zorder=0))
    a.add_patch(Rectangle((bP, 0), 4.2 - bP, 1.08, color=style.tint(P["peri_outer"], .62), lw=0, zorder=0))
    dens = norm.pdf(b, 0, team["b_sd"])
    a.fill_between(b, 0, .20 * dens / dens.max(), color=P["lav2"], alpha=.28, lw=0, zorder=1)
    a.plot(b, one(b), color=P["ink"], lw=1.3, label="one attempt", zorder=3)
    a.plot(b, two(b), color=P["cyan_deep"], lw=1.3, ls=(0, (4, 2)), label="two attempts", zorder=3)
    a.plot(b, hint(b), color=P["pink_deep"], lw=1.4, label="hint (right tutor)", zorder=3)
    a.plot(b, bad(b), color=P["pink_deep"], lw=1.0, ls=(0, (1, 1.6)), label="hint (wrong tutor)", zorder=3)
    a.axhline(Z.TAU, color=P["faint"], lw=.6, ls=(0, (1, 2)), zorder=2)
    for x, y, txt, col in ((-2.35, .36, "alone", P["purple"]), ((bA + bP) / 2 - .03, .99, "P", P["violet"]),
                           (2.9, .36, "beyond reach", P["lav4"])):
        a.text(x, y, txt, ha="center", va="top", fontsize=6.6, color=col)
    a.annotate("", xy=(b2, Z.TAU), xytext=(bP, Z.TAU),
               arrowprops=dict(arrowstyle="-|>", color=P["indigo"], lw=.8, mutation_scale=6,
                               connectionstyle="arc3,rad=-0.55", shrinkA=0, shrinkB=0))
    a.text(b2 + .12, Z.TAU + .025, "two attempts reach $\\tau$ later", fontsize=5.4, color=P["indigo"],
           ha="left", va="bottom")
    a.text(-3.45, .115, "difficulty prior", fontsize=5.6, color=P["lav4"], va="bottom")
    a.set_xlim(-3.6, 4.2)
    a.set_ylim(0, 1.06)
    a.set_xlabel("problem difficulty $b$ (logits)")
    a.set_ylabel("success probability")
    a.set_yticks([0, .5, 1])
    a.set_title("(a) " + (Z.logo("o3-mini-low") if not draft else "") + "o3-mini under the fitted model",
                fontsize=7.6, loc="left")
    a.legend(loc="upper right", fontsize=5.7, handlelength=2.0, borderaxespad=.3, labelspacing=.22,
             frameon=True, facecolor="white", edgecolor="none", framealpha=.85)

    # (b) proximal share against K, both SAT-style teams and the open team
    Ks = [1, 2, 3, 4, 6, 8]
    series = [("SAT-style", "o3-mini-low", P["purple"], "o", "-"), ("SAT-style", "haiku-4.5", P["lav3"], "s", "-"),
              ("SAT-style", "deepseek-v3", P["lav1"], "D", "-"),
              ("open-weight", "gpt-oss-120b-low", P["cyan_deep"], "^", "-")]
    for team_name, key, col, mk, ls in series:
        m = nums["teams"][team_name]["members"][key]
        bax.plot(Ks, [100 * m["shares"][str(K)]["P"] if str(K) in m["shares"] else 100 * m["shares"][K]["P"]
                      for K in Ks], marker=mk, ms=3.0, lw=1.1, ls=ls, color=col, zorder=3,
                 label=(Z.logo(key, 5.6) if not draft else "") + Z.NAMES[key])
    shares26 = sat26_shares()
    if any(v[1] > 0 for v in shares26.values()):
        for key, col in (("gpt-5-mini-low", P["pink_deep"]),):
            bax.plot(Ks, [100 * shares26[key][K] for K in Ks], marker="v", ms=3.0, lw=1.1, ls=(0, (3, 1.5)),
                     color=col, zorder=3, label=(Z.logo(key, 5.6) if not draft else "") + "GPT-5-mini")
    bax.set_xscale("log", base=2)
    bax.set_xticks(Ks, [str(k) for k in Ks])
    bax.minorticks_off()
    bax.set_ylim(-.8, 16)
    bax.set_xlabel("fresh attempts instead of one hint ($K$)")
    bax.set_ylabel("proximal-zone share (\\%)" if not draft else "proximal-zone share (%)")
    bax.set_title("(b) The zone is budget-relative", fontsize=7.6, loc="left")
    bax.legend(fontsize=5.7, loc="upper right", handlelength=1.6, labelspacing=.25)

    # (c) the randomized hint trial: rescue of initially wrong members by arm
    teams = [(p, lab, col) for p, lab, col in (("sat26", "2026 team", P["pink_deep"]),
                                               ("lane", "open-weight", P["purple"])) if p in trial]
    top = 0.0
    width = .36
    for j, (package, label, col) in enumerate(teams):
        # like with like: every arm on the units that have a right tutor (falls back to all units in old outputs)
        rates = trial[package].get("rates (right-tutor units)") or trial[package]["rates"]
        for i, (key, _) in enumerate(ARMS):
            r = rates.get(key)
            if not r:
                continue
            x = i + (j - (len(teams) - 1) / 2) * width
            c.bar(x, 100 * r["mean"], width * .92, color=style.tint(col, .55), edgecolor=col, lw=.6, zorder=2,
                  label=label if i == 0 else None)
            c.errorbar(x, 100 * r["mean"], yerr=[[100 * (r["mean"] - r["low"])], [100 * (r["high"] - r["mean"])]],
                       color=col, elinewidth=.7, capsize=1.3, zorder=3)
            top = max(top, 100 * r["high"])
        delivered = rates.get("right tutor's hint (delivered)")
        if delivered:
            x = len(ARMS) - 1 + (j - (len(teams) - 1) / 2) * width
            c.plot(x, 100 * delivered["mean"], marker="o", ms=3.6, ls="none", mfc="white", mec=col, mew=.9, zorder=4,
                   label="delivered hints" if j == len(teams) - 1 else None)
            top = max(top, 100 * delivered["mean"])
    c.set_xticks(range(len(ARMS)), [label for _, label in ARMS], fontsize=5.8, rotation=28, ha="right")
    c.set_ylim(0, 10 * np.ceil((top + 14) / 10))       # headroom for the legend above the bars
    c.set_ylabel("wrong members rescued (\\%)" if not draft else "wrong members rescued (%)")
    c.set_title("(c) Randomized hint trial", fontsize=7.6, loc="left")
    c.legend(fontsize=5.6, loc="upper right", handlelength=1.2, labelspacing=.2, frameon=False)
    c.grid(alpha=.5, axis="y")
    fig.tight_layout(w_pad=1.0)
    style.save(fig, "zpd_v6", OUT_F)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    args = ap.parse_args()
    nums = Z.numbers()
    results = json.loads((OUT_T / "v6_results.json").read_text(encoding="utf-8"))
    figure(nums, results.get("S8") or {}, args.draft)


if __name__ == "__main__":
    main()
