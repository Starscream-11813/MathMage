"""The zone of proximal development of a language-model team, from the frozen fitted models (no API calls).

    python analysis/fig_zpd.py            # publication figure (LuaLaTeX) + numbers
    python analysis/fig_zpd.py --draft    # fast preview

Panels: (a) zones in item-response coordinates for one member, under the fitted S-IRT model; (b) the share of
problems in the proximal zone as the number K of fresh attempts offered instead of one hint grows (the zone
computation of mathmage/team/posterior.py:member_zones, with abstention and same-model correlation);
(c) hinted re-solve success on the training problems, split by whether the tutor was right.
Writes results/figures/zpd.{pdf,png} and results/tables/zpd_numbers.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from mathmage.analysis import style  # noqa: E402

OUT_F, OUT_T = REPO / "results/figures", REPO / "results/tables"
RUNS = REPO / "runs/live_20261018"
TEAMS = {"SAT-style": RUNS / "program_v3/offline_sat", "open-weight": RUNS / "program_open/offline_open"}
NAMES = {"o3-mini-low": "o3-mini", "haiku-4.5": "Claude Haiku 4.5", "deepseek-v3": "DeepSeek-V3",
         "gpt-oss-120b-low": "gpt-oss-120b"}
LOGOS = {"o3-mini-low": "gpt", "haiku-4.5": "claude", "deepseek-v3": "deepseek", "gpt-oss-120b-low": "gpt"}


def logo(key: str, height: float = 6.0) -> str:
    """Keep model labels portable without manuscript logo assets."""
    return ""
TAU = 0.5
GRID = np.linspace(-9, 11, 4001)


def no_success(prof, rho, K, b):
    """P(no correct usable answer in K fresh attempts), as in TeamPosterior.member_zones."""
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


def family_rate(trials, draws=4000, seed=0):
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    fams = defaultdict(list)
    for t in trials:
        fams[t.get("family_id", t["item_id"])].append(float(bool(t["correct"])))
    keys = list(fams)
    if not keys:
        return {"k": 0, "n": 0, "rate": 0.0, "low": 0.0, "high": 0.0}
    boot = [np.mean([x for key in rng.choice(keys, len(keys)) for x in fams[key]]) for _ in range(draws)]
    k = int(sum(sum(v) for v in fams.values()))
    return {"k": k, "n": len(trials), "rate": k / len(trials), "low": float(np.quantile(boot, .025)),
            "high": float(np.quantile(boot, .975))}


def numbers():
    out = {"tau": TAU, "teams": {}}
    for team, root in TEAMS.items():
        art = json.loads((root / "artifacts.json").read_text(encoding="utf-8"))
        pp = art["posterior_params"]
        rho = pp["same_agent_correlation"]
        right = pp["lift"].get("coached_right", pp["lift"]["coached"])
        wrong = pp["lift"].get("coached_wrong", pp["lift"]["coached"])
        prior = norm.pdf(GRID, pp["b_mean"], pp["b_sd"])
        prior /= prior.sum()
        members = {}
        for key, prof in pp["agents"].items():
            if "@" in key or key == "tool":
                continue
            shares = {}
            for K in (1, 2, 3, 4, 6, 8):
                z = zones(prof, rho, right, K, GRID)
                shares[K] = {zz: float(prior[z == i].sum()) for i, zz in enumerate("ALPU")}
            members[key] = {"theta": prof["theta"], "abstain": prof["abstain"], "shares": shares}
        evidence = json.loads((root / "training_evidence.json").read_text(encoding="utf-8"))
        trials = [{**t, "correct": t.get("correct")} for t in evidence["tutoring_trials"]]
        trials = [t for t in trials if t["correct"] is not None]
        out["teams"][team] = {
            "rho": rho, "b_sd": pp["b_sd"], "lift_right": right, "lift_wrong": wrong,
            "hint_acceptance": pp.get("hint_acceptance"), "members": members,
            "replay": {"tutor right": family_rate([t for t in trials if t.get("tutor_correct")]),
                       "tutor wrong": family_rate([t for t in trials if not t.get("tutor_correct")])}}
    # plain S-IRT sampling horizons H_K = logit(tau) - logit(tau_K), tau_K = 1 - (1 - tau)^(1/K)
    out["horizon"] = {K: float(logit(TAU) - logit(1 - (1 - TAU) ** (1 / K))) for K in (1, 2, 3, 4, 6, 8)}
    return out


def boundary(f, lo=-9, hi=11):
    """Difficulty where a decreasing success curve crosses tau."""
    b = np.linspace(lo, hi, 20001)
    v = f(b)
    i = np.argmax(v < TAU)
    return float(b[i])


def figure(nums, draft):
    style.use(publication=not draft, base_size=8.0)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    P = style.P
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.05), gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})
    a, bax, c = axes

    # (a) zones in IRT coordinates for o3-mini (SAT-style team), with the fitted lifts
    team = nums["teams"]["SAT-style"]
    art = json.loads((TEAMS["SAT-style"] / "artifacts.json").read_text(encoding="utf-8"))["posterior_params"]
    prof, rho = art["agents"]["o3-mini-low"], team["rho"]
    one = lambda b: (1 - prof["abstain"]) * expit(prof["theta"] - b)                       # noqa: E731
    two = lambda b: 1 - no_success(prof, rho, 2, b)                                        # noqa: E731
    hint = lambda b: (1 - prof["abstain"]) * expit(prof["theta"] + team["lift_right"] - b)  # noqa: E731
    bad = lambda b: (1 - prof["abstain"]) * expit(prof["theta"] + team["lift_wrong"] - b)   # noqa: E731
    b = np.linspace(-3.6, 4.2, 900)
    bA, bP, b2 = boundary(one), boundary(hint), boundary(two)
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
    a.axhline(TAU, color=P["faint"], lw=.6, ls=(0, (1, 2)), zorder=2)
    for x, y, txt, col in ((-2.35, .36, "alone", P["purple"]), ((bA + bP) / 2 - .03, .99, "P", P["violet"]),
                           (2.9, .36, "beyond reach", P["lav4"])):
        a.text(x, y, txt, ha="center", va="top", fontsize=6.6, color=col, fontweight="bold")
    a.annotate("", xy=(b2, TAU), xytext=(bP, TAU),
               arrowprops=dict(arrowstyle="-|>", color=P["indigo"], lw=.8, mutation_scale=6,
                               connectionstyle="arc3,rad=-0.55", shrinkA=0, shrinkB=0))
    # one line between the threshold and the legend, where every curve has already fallen below tau
    a.text(b2 + .12, TAU + .025, "two attempts reach $\\tau$ later", fontsize=5.4, color=P["indigo"],
           ha="left", va="bottom")
    a.text(-3.45, .115, "difficulty prior", fontsize=5.6, color=P["lav4"], va="bottom")
    a.set_xlim(-3.6, 4.2)
    a.set_ylim(0, 1.06)
    a.set_xlabel("problem difficulty $b$ (logits)")
    a.set_ylabel("success probability")
    a.set_yticks([0, .5, 1])
    a.set_title("(a) " + (logo("o3-mini-low") if not draft else "") + "o3-mini under the fitted model", fontsize=7.6,
                loc="left")
    a.legend(loc="upper right", fontsize=5.7, handlelength=2.0, borderaxespad=.3, labelspacing=.22,
             frameon=True, facecolor="white", edgecolor="none", framealpha=.85)

    # (b) proximal share against K fresh attempts
    Ks = [1, 2, 3, 4, 6, 8]
    series = [("SAT-style", "o3-mini-low", P["purple"], "o"), ("SAT-style", "haiku-4.5", P["lav3"], "s"),
              ("SAT-style", "deepseek-v3", P["lav1"], "D"), ("open-weight", "gpt-oss-120b-low", P["cyan_deep"], "^")]
    for team_name, key, col, mk in series:
        m = nums["teams"][team_name]["members"][key]
        y = [100 * m["shares"][K]["P"] for K in Ks]
        bax.plot(Ks, y, marker=mk, ms=3.0, lw=1.1, color=col, zorder=3,
                 label=(logo(key, 5.6) if not draft else "") + NAMES[key])
    bax.set_xscale("log", base=2)
    bax.set_xticks(Ks, [str(k) for k in Ks])
    bax.minorticks_off()
    bax.set_ylim(-.8, 16)
    bax.set_xlabel("fresh attempts instead of one hint ($K$)")
    bax.set_ylabel("proximal-zone share (%)")
    bax.set_title("(b) The zone is budget-relative", fontsize=7.6, loc="left")
    bax.legend(fontsize=5.7, loc="upper right", handlelength=1.6, labelspacing=.25)

    # (c) hinted re-solve success on training problems
    bars = [("SAT-style", "tutor right", P["cyan"], P["indigo"]), ("SAT-style", "tutor wrong", P["pink"], P["violet"]),
            ("open-weight", "tutor right", P["cyan"], P["indigo"]), ("open-weight", "tutor wrong", P["pink"], P["violet"])]
    xs = [0, .9, 2.3, 3.2]
    for x, (team_name, kind, fill, line) in zip(xs, bars):
        r = nums["teams"][team_name]["replay"][kind]
        c.bar(x, 100 * r["rate"], .8, color=fill, edgecolor=line, lw=.6, zorder=2)
        c.errorbar(x, 100 * r["rate"], yerr=[[100 * (r["rate"] - r["low"])], [100 * (r["high"] - r["rate"])]],
                   color=line, elinewidth=.8, capsize=1.6, zorder=3)
        lift = nums["teams"][team_name]["lift_right" if kind == "tutor right" else "lift_wrong"]
        c.text(x, 100 * r["high"] + 3, f"{r['k']}/{r['n']}\n{lift:+.2f}", ha="center", va="bottom", fontsize=5.6,
               color=P["ink"], linespacing=1.0)
    c.set_xticks(xs, ["right", "wrong", "right", "wrong"], fontsize=6.2)
    for lbl in c.get_xticklabels():
        lbl.set_va("baseline")
    c.tick_params(axis="x", pad=9)
    c.text(.45, -27, "SAT-style team", ha="center", fontsize=6.2, color=P["muted"])
    c.text(2.75, -27, "open-weight team", ha="center", fontsize=6.2, color=P["muted"])
    c.set_ylim(0, 100)
    c.set_ylabel("hinted re-solve correct (%)")
    c.set_title("(c) Hinted recovery by tutor", fontsize=7.6, loc="left")
    fig.tight_layout(w_pad=1.0)
    style.save(fig, "zpd", OUT_F)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    args = ap.parse_args()
    nums = numbers()
    (OUT_T / "zpd_numbers.json").write_text(json.dumps(nums, indent=1), encoding="utf-8")
    for team, t in nums["teams"].items():
        print(team, "lift right/wrong", round(t["lift_right"], 3), round(t["lift_wrong"], 3), "replay",
              {k: f"{v['k']}/{v['n']}" for k, v in t["replay"].items()})
        for key, m in t["members"].items():
            print("   ", key, {K: round(100 * s["P"], 1) for K, s in m["shares"].items()})
    print("horizons", {K: round(h, 3) for K, h in nums["horizon"].items()})
    figure(nums, args.draft)


if __name__ == "__main__":
    main()
