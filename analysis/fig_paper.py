"""Every matplotlib figure of the ACL paper, in the house palette (no API calls, no new statistics).

    python analysis/fig_paper.py                 # all figures
    python analysis/fig_paper.py cost zpd        # a subset

Each figure is redrawn from saved results (v5_summary.json, summary.json, zpd fits, arr4_*.json, the frozen
artifacts) or, for the theory illustrations, from the stated formulas with illustrative parameters. Outputs go to
results/figures/ and are copied to results/figures/publication/ under the original figure names.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ["MATHMAGE_PALETTE"] = "paper"            # must precede the style import
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np  # noqa: E402

from mathmage.analysis import style  # noqa: E402

TABLES, OUT, ACL = REPO / "results/tables", REPO / "results/figures", REPO / "results/figures/publication"
P = style.P
# text drawn in a light palette colour switches to its darker partner, so it stays legible on white
TEXT = {P["cyan"]: P["indigo"], P["pink"]: P["violet"], P["lav1"]: P["lav4"], P["lav2"]: P["lav4"]}
ZONE_TEXT = {"A": P["purple"], "L": P["indigo"], "P": P["violet"], "U": P["lav4"]}


def read(name):
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def publish(stem: str, name: str | None = None) -> None:
    ACL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / f"{stem}.pdf", ACL / f"{name or stem}.pdf")
    print("figure", name or stem)


def _plt():
    style.use(publication=True, base_size=8.0)
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------------------------------- main text
def cost():
    """Accuracy against cost on the 2025-26 contests (Figure 2)."""
    plt = _plt()
    from matplotlib.ticker import FuncFormatter
    v5 = read("v5_summary.json")
    names = {"mu*": r"\textsc{MathMage}", "vote": "majority vote", "debate": "debate", "USC": "universal SC",
             "MoA": "mixture-of-agents", "SC@10 (DeepSeek-V3)": "self-consistency",
             "SC@5 (pooled seeds)": "self-consistency"}
    # where each label sits: beside the marker (left/right) or beyond the end of its interval (above/below),
    # chosen so that no label crosses an error bar
    place = {"SAT roster": {"mu*": "right", "vote": "below", "debate": "above", "USC": "left", "MoA": "right",
                            "SC@10 (DeepSeek-V3)": "below"},
             "open roster": {"mu*": "right", "vote": "below", "debate": "above", "USC": "above", "MoA": "above",
                             "SC@5 (pooled seeds)": "below"}}
    note_at = {"SAT roster": (0.98, 0.97, "top", "right"), "open roster": (0.03, 0.97, "top", "left")}
    ticks = {"SAT roster": ([0.02, 0.03, 0.04, 0.05], "{:.2f}", (0.018, 0.059)),
             "open roster": ([0.0012, 0.0016, 0.0020, 0.0024], "{:.4f}", (0.00105, 0.00275))}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.05))
    for ax, (roster, title) in zip(axes, (("SAT roster", r"SAT-style team $\cdot$ 216 problems"),
                                          ("open roster", r"Open-weight team $\cdot$ 1,968 evaluations"))):
        methods = v5["S1"][roster]["methods"]
        for key, m in methods.items():
            mm = key == "mu*"
            color = (P["purple"] if mm else P["cyan_deep"] if key.startswith("SC") else
                     P["pink_deep"] if key == "debate" else P["lav3"])
            x, y, lo, hi = m["cost"], 100 * m["macro"], 100 * m["low"], 100 * m["high"]
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="*" if mm else "o",
                        ms=9.5 if mm else 4.2, color=color, mec="white" if not mm else color, mew=0.5,
                        ecolor=style.tint(color, 0.35), elinewidth=0.8, capsize=1.6, zorder=5 if mm else 3)
            side = place[roster][key]
            anchor, offset, ha, va = {"right": ((x, y), (6, 0), "left", "center"),
                                      "left": ((x, y), (-6, 0), "right", "center"),
                                      "above": ((x, hi), (0, 3), "center", "bottom"),
                                      "below": ((x, lo), (0, -3), "center", "top")}[side]
            ax.annotate(names[key], anchor, xytext=offset, textcoords="offset points", ha=ha, va=va,
                        fontsize=6.8, color=P["purple"] if mm else P["ink"], fontweight="bold" if mm else "normal")
        deb, mm = methods["debate"], methods["mu*"]
        gain, saving = 100 * (mm["macro"] - deb["macro"]), 100 * (1 - mm["cost"] / deb["cost"])
        nx, ny, nva, nha = note_at[roster]
        ax.text(nx, ny, rf"$+{gain:.1f}$ points over debate" "\n" rf"at {saving:.0f}\% lower cost",
                transform=ax.transAxes, ha=nha, va=nva, fontsize=6.4, color=P["violet"], linespacing=1.15,
                multialignment=nha)
        positions, fmt, xlim = ticks[roster]
        ax.set_xticks(positions, [r"\$" + fmt.format(v) for v in positions])
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=7.6, loc="left")
        ax.set_xlabel("cost per problem")
        ax.grid(True, color=P["grid"], lw=0.5)
        ax.margins(y=0.12)
    axes[0].set_ylabel(r"macro accuracy (\%)")
    fig.tight_layout(w_pad=1.6)
    style.save(fig, "cost", OUT)
    plt.close(fig)
    publish("cost")


def zpd():
    """Figure 3: zones in IRT coordinates, budget-relative proximal share, hinted recovery."""
    import fig_zpd
    nums = fig_zpd.numbers()
    fig_zpd.figure(nums, draft=False)
    publish("zpd")


# ------------------------------------------------------------------------------------------- appendix: results
def forest():
    import synthesis as S
    style.use(publication=True)
    v3, v4, v5 = S.load()
    rows = S.sets(v3, v4, v5)
    short = {"Original suite (E1)": "E1", "April 2025 contests (primary)": "April 2025 (primary)",
             "Strictly post-cutoff for every member": "post-cutoff for all members",
             "All 2025-26 contests": "2025--26 contests", "Original suite (E2)": "E2",
             "April 2025 + 2026 contests (primary)": "April 2025 + 2026 (primary)",
             "All 2025-26 contests (strictly post-cutoff)": "2025--26 contests"}
    S.fig_forest(rows, OUT, v4, rename=short)
    publish("synthesis_forest", "forest")


def stopping():
    import arr4_stopping
    arr4_stopping.SHORT = {"SAT roster · E1 suite": "SAT-style · E1",
                           "SAT roster · 153 new contest problems": "SAT-style · new contests",
                           "Open roster · E2 suite": "Open-weight · E2",
                           "Open roster · new and 2026 contests": "Open-weight · new contests"}
    arr4_stopping.figure(read("arr4_stopping.json"), draft=False)
    publish("arr4_stopping", "stopping")


def adaptive():
    import arr4_offline
    result = read("arr4_offline.json")
    titles = {"E1 suite": "SAT-style · E1", "post-cutoff (93)": "SAT-style · post-cutoff",
              "E2 AIME": "Open-weight · E2 AIME", "(1,968)": "Open-weight · 2025--26"}
    renamed = {}
    for name, value in result["adaptive"].items():
        new = next((t for k, t in titles.items() if k in name), name)
        renamed[new] = value
    result["adaptive"] = renamed
    arr4_offline.figure(result, draft=False)
    publish("arr4_adaptive", "adaptive")


def voc():
    """Predicted value of computation against realized change in correctness, per action type (maximal runs)."""
    plt = _plt()
    from matplotlib.lines import Line2D
    v3 = read("summary.json")
    # one marker shape per action type, one colour per team; marker area grows with the number of actions
    kinds = [("judge", "judge", "D"), ("tool_alt", "program", "s"), ("sample", "fresh attempt", "o"),
             ("diversify", "diversified re-solve", "^"), ("coach", "hint", "v"), ("review", "review", "P")]
    teams = (("e1", r"E1 $\cdot$ SAT-style", P["lav3"]), ("e2", r"E2 $\cdot$ open-weight", P["cyan_deep"]))
    fig, ax = plt.subplots(figsize=(6.9, 2.45))
    hi = lo = 0.0
    for study, _, color in teams:
        for kind, _, marker in kinds:
            a = v3[study]["actions"].get(kind)
            if not a or a["n"] < 15:
                continue
            n, r, h = a["n"], a["rescue"], a["harm"]
            mean = (r - h) / n                               # each action changes correctness by -1, 0 or +1
            se = np.sqrt(max((r + h) - n * mean ** 2, 0.0) / (n - 1) / n)
            lo, hi = min(lo, mean - 1.96 * se), max(hi, mean + 1.96 * se)
            ax.errorbar(a["mean_voc"], mean, yerr=1.96 * se, fmt=marker, ms=3.2 + 3.4 * np.sqrt(n / 213), color=color,
                        mec="white", mew=0.5, ecolor=style.tint(color, 0.35), elinewidth=0.8, capsize=1.5, zorder=3)
            dx, dy, ha, va = {("e2", "judge"): (-6, 3, "right", "bottom"),
                              ("e1", "coach"): (-5, -2, "right", "top")}.get((study, kind), (5, -1, "left", "top"))
            ax.annotate(str(n), (a["mean_voc"], mean), xytext=(dx, dy), textcoords="offset points", fontsize=5.4,
                        color=P["muted"], ha=ha, va=va)
    top = 0.20
    ax.plot([0, top], [0, top], color=P["faint"], lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.text(0.165, 0.172, "predicted = realized", fontsize=6.0, color=P["muted"], rotation=24, ha="center",
            va="bottom")
    ax.axhline(0, color=P["ink"], lw=0.5, zorder=0)
    ax.set_xlim(0, top)
    ax.set_ylim(lo * 1.08, hi * 1.08)
    ax.set_xlabel("mean predicted value of computation")
    ax.set_ylabel("realized change in\ncorrectness per action")
    ax.grid(True, color=P["grid"], lw=0.5)
    handles = [Line2D([], [], ls="", marker=m, color=P["muted"], ms=4.5, label=lab) for _, lab, m in kinds]
    handles += [Line2D([], [], ls="", marker="o", color=c, ms=5, label=lab) for _, lab, c in teams]
    ax.legend(handles=handles, fontsize=6.0, loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.0,
              labelspacing=0.35, frameon=False)
    fig.tight_layout()
    style.save(fig, "voc_calibration", OUT)
    plt.close(fig)
    publish("voc_calibration")


# ------------------------------------------------------------------------------------------- appendix: theory
def _zone_bands(ax, edges, ymin, ymax, letters=True):
    for zone, (lo, hi) in zip("ALPU", zip(edges[:-1], edges[1:])):
        ax.axvspan(lo, hi, color=style.tint(style.ZONE_FILL[zone], 0.55), lw=0, zorder=0)
        if letters:
            ax.text((lo + hi) / 2, ymin + 0.04 * (ymax - ymin), rf"\textbf{{{zone}}}", ha="center",
                    va="bottom", fontsize=8, color=ZONE_TEXT[zone], zorder=3)


def zones_concept():
    import fig_zones_concept
    fig_zones_concept.main()
    publish("zones_concept")


def sweet():
    """Propositions 1 and 6 with illustrative parameters (theta = 0, a = 1, Delta = 5, K = 8, tau = 1/2)."""
    plt = _plt()
    from scipy.optimize import minimize_scalar
    from scipy.special import expit, logit
    K, delta = 8, 5.0
    b = np.linspace(-3, 8, 700)
    one, best, scaf = expit(-b), 1 - (1 - expit(-b)) ** K, expit(delta - b)
    horizon = -logit(1 - 0.5 ** (1 / K))                     # r_K crosses 1/2 here (H_K with tau = 1/2)
    edges = (-3, 0, horizon, delta, 8)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.35))
    _zone_bands(a1, edges, 0, 1.06)
    a1.plot(b, one, color=P["ink"], lw=1.4, label=r"one attempt $p$")
    a1.plot(b, best, color=P["cyan_deep"], lw=1.4, ls=(0, (4, 2)), label=rf"best of {K} $r_{K}$")
    a1.plot(b, scaf, color=P["pink_deep"], lw=1.4, label=r"scaffolded $c$")
    a1.axhline(0.5, color=P["faint"], lw=0.6, ls=(0, (1, 2)), zorder=1)
    a1.set(xlim=(-3, 8), ylim=(0, 1.06), xlabel=r"difficulty $b$", ylabel="success probability")
    a1.set_title("(a) Zones are bands of difficulty", fontsize=7.6, loc="left")
    # the upper-right corner is free: every curve has fallen below 0.65 beyond b = 4.5
    a1.legend(loc="upper right", fontsize=5.8, handlelength=1.6, labelspacing=0.2, borderaxespad=0.3,
              framealpha=0.9, facecolor="white", edgecolor="none", frameon=True)
    _zone_bands(a2, edges, -0.05, 1.0, letters=False)
    gains = [(r"sampling: $r_8-p$", best - one, P["cyan_deep"], (0, (4, 2)),
              lambda x: -((1 - (1 - expit(-x)) ** K) - expit(-x))),
             (r"scaffolding: $c-p$", scaf - one, P["pink_deep"], "-", lambda x: -(expit(delta - x) - expit(-x))),
             (r"scaffold over sampling: $c-r_8$", scaf - best, P["purple"], "-",
              lambda x: -(expit(delta - x) - (1 - (1 - expit(-x)) ** K)))]
    for label, g, color, ls, f in gains:
        r = minimize_scalar(f, bounds=(-3, 8), method="bounded")
        a2.plot(b, g, color=color, lw=1.4, ls=ls, label=label)
        a2.plot([r.x], [-r.fun], "o", ms=3.6, color=color, zorder=5)
    a2.axhline(0, color=P["faint"], lw=0.6, ls=(0, (1, 2)))
    a2.set(xlim=(-3, 8), ylim=(-0.05, 1.12), xlabel=r"difficulty $b$", ylabel="gain in success probability")
    a2.set_title("(b) Two sweet spots", fontsize=7.6, loc="left")
    # the upper-left corner is free: every gain is below 0.75 before b = 1.2
    a2.legend(loc="upper left", fontsize=5.8, handlelength=1.6, labelspacing=0.2, borderaxespad=0.3,
              framealpha=0.9, facecolor="white", edgecolor="none", frameon=True)
    fig.tight_layout(w_pad=1.4)
    style.save(fig, "theory_sweet", OUT)
    plt.close(fig)
    publish("theory_sweet")


def discern():
    """Proposition 7 (required peer validity) beside the rescue-harm accounting identity."""
    plt = _plt()
    from mathmage.analysis.concepts import vstar
    z = np.linspace(-8, 8, 801)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
    a1.axvspan(0, 8, color=style.tint(style.ZONE_FILL["A"], 0.85), lw=0, zorder=0)
    a1.text(7.7, 0.04, "solver already right\nmore often than not", ha="right", va="bottom", fontsize=6.2,
            color=P["purple"])
    for (e, d), color, ls in (((2, 2), P["purple"], "-"), ((4, 4), P["purple"], (0, (4, 2))),
                              ((4, 1), P["pink_deep"], "-"), ((1, 4), P["cyan_deep"], "-")):
        a1.plot(z, vstar(z, e, d), color=color, ls=ls, lw=1.4, label=rf"$e={e},\ d={d}$")
    a1.set(xlim=(-8, 8), ylim=(0, 1), xlabel=r"unassisted logit margin $z=a(\theta-b)$",
           ylabel=r"required peer validity $v^\star(z)$")
    a1.set_title("(a) Mastered answers need nearly infallible peers", fontsize=7.6, loc="left")
    a1.legend(loc="upper left", fontsize=6.2, handlelength=1.8, labelspacing=0.25)
    h = np.linspace(0, 0.5, 200)
    a2.fill_between(h, h, 0.62, color=style.tint(P["cyan"], 0.75), lw=0, zorder=0)
    for p0, color in ((0.3, P["cyan_deep"]), (0.5, P["purple"]), (0.7, P["pink_deep"]), (0.9, P["indigo"])):
        slope = p0 / (1 - p0)
        hh = h[slope * h <= 0.62]
        a2.plot(hh, slope * hh, color=color, lw=1.3)
        if slope * 0.5 <= 0.62:                     # the line leaves through the right edge: label above its end
            a2.text(0.492, slope * 0.492 + 0.012, rf"$p={p0}$", color=TEXT.get(color, color), fontsize=6.4,
                    ha="right", va="bottom")
        else:                                       # it leaves through the top: label to the right of its end
            a2.text(0.62 / slope + 0.008, 0.605, rf"$p={p0}$", color=TEXT.get(color, color), fontsize=6.4,
                    ha="left", va="top")
    a2.set(xlim=(0, 0.5), ylim=(0, 0.62), xlabel=r"harm rate $h$ (right $\to$ wrong)",
           ylabel=r"rescue rate $r$ (wrong $\to$ right)")
    a2.set_title(r"(b) $\Delta\mathrm{acc}=(1-p)\,r-p\,h$", fontsize=7.6, loc="left")
    fig.tight_layout(w_pad=1.4)
    style.save(fig, "theory_discern", OUT)
    plt.close(fig)
    publish("theory_discern")


def surfaces():
    """The two theory surfaces, with the fitted right-tutor lift marked on the first (no new statistics)."""
    style.use(publication=True, base_size=8.0)
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from scipy.special import expit
    from mathmage.analysis import concepts
    fig = concepts.surfaces()
    for t in fig.findobj(Text):                       # light text colours -> their darker partners
        rgb = style.to_rgb(t.get_color()) if isinstance(t.get_color(), str) else None
        for light, dark in TEXT.items():
            if rgb is not None and np.allclose(rgb, style.to_rgb(light), atol=0.02):
                t.set_color(dark)
    # the SAT-style team's fitted lift of a right tutor's hint, on the base plane of panel (a)
    lift = read("zpd_numbers.json")["teams"]["SAT-style"]["lift_right"]
    z, e = np.meshgrid(np.linspace(-8, 3, 120), np.linspace(0, 8, 100))
    floor = float((expit(z + e) - (1 - (1 - expit(z)) ** 8)).min()) - 0.02
    ax = fig.axes[0]
    ax.plot([-8, 3], [lift, lift], [floor, floor], color=P["indigo"], lw=1.0, zorder=11)   # explained in the caption
    style.save(fig, "theory_surfaces", OUT)
    plt.close(fig)
    publish("theory_surfaces")


def prefix():
    """Proposition 5 illustrated: a frozen price keeps a prefix of the maximal run; realized answers can regress."""
    plt = _plt()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9, 2.1), gridspec_kw={"width_ratios": [1, 1]})
    ratios, mu = np.array([7.5, 2.8, 4.3, 1.2]), 3.0
    ax.axhspan(mu, 8.5, color=style.tint(P["lav1"], 0.8), lw=0, zorder=0)
    ax.plot(np.arange(4), ratios, color=P["purple"], marker="o", ms=3.6, lw=1.3, zorder=3)
    ax.axhline(mu, color=P["pink_deep"], ls=(0, (4, 2)), lw=1.0)
    ax.scatter([1], [ratios[1]], color=P["pink_deep"], s=36, zorder=5)
    ax.annotate("stop: first ratio below the price", xy=(1, 2.8), xytext=(1.35, 6.4), fontsize=6.4,
                color=P["ink"], arrowprops=dict(arrowstyle="-|>", color=P["muted"], lw=0.6, mutation_scale=6,
                                                connectionstyle="arc3,rad=0.35"))
    ax.text(3.1, mu + 0.18, r"price $\mu$", fontsize=6.4, color=P["violet"], ha="right", va="bottom")
    ax.set(xlabel="action index in the maximal run", ylabel="value per dollar", xticks=range(4), ylim=(0, 8.5))
    ax.set_title("(a) A frozen price keeps a prefix", fontsize=7.6, loc="left")
    costs = np.array([0.010, 0.020, 0.040, 0.065, 0.080])
    correct = np.array([0, 0, 1, 0, 1], float)
    bx.step(costs, correct, where="post", color=P["cyan_deep"], lw=1.3)
    bx.scatter(costs, correct, color=P["purple"], s=14, zorder=4)
    bx.scatter([costs[1]], [0], color=P["pink_deep"], s=36, zorder=5)
    bx.annotate(r"run at $\mu$ stops here", xy=(costs[1], 0), xytext=(0.024, 0.42), fontsize=6.4,
                arrowprops=dict(arrowstyle="-|>", color=P["muted"], lw=0.6, mutation_scale=6,
                                connectionstyle="arc3,rad=0.35"))
    bx.set(xlabel="cumulative cost (illustrative, \\$)", ylabel="decision correct?", ylim=(-0.15, 1.25),
           yticks=[0, 1], yticklabels=["wrong", "right"])
    bx.set_title("(b) Realized correctness need not be monotone", fontsize=7.6, loc="left")
    fig.tight_layout(w_pad=2.0)
    style.save(fig, "theory_prefix", OUT)
    plt.close(fig)
    publish("theory_prefix")


FIGURES = {"cost": cost, "zpd": zpd, "forest": forest, "stopping": stopping, "adaptive": adaptive,
           "voc": voc, "zones_concept": zones_concept, "sweet": sweet, "discern": discern,
           "surfaces": surfaces, "prefix": prefix}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGURES):
        FIGURES[name]()
