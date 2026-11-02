"""Conceptual and theory figures (no experimental data).

Every figure here explains an idea of the method: the four zones, the team
architecture, the label-free posterior, the selection/generation accounting, the
anytime cost frontier, and the formal results of the research plan. Theory curves
are computed from the stated formulas; nothing here is a measurement.

    python -m mathmage.analysis.concepts            # all figures -> outputs/figures/concepts
    python -m mathmage.analysis.concepts zones      # one figure
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from mathmage import paths
from mathmage.analysis import style
from mathmage.analysis.style import P, ZONE_FILL, ZONE_LINE, tint

OUT = paths.FIGURES / "concepts"


def _canvas(w, h, xlim, ylim):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


# ============================================================================ C1 zones
def zones():
    from mathmage.analysis.diagram import blob, equation_box, label, pointer, scaled

    fig, ax = _canvas(7.0, 4.55, (-8.2, 8.2), (-5.15, 4.95))
    c = (0.0, 0.4)
    u = [(0.6, -3.1), (-2.8, -2.95), (-5.0, -0.6), (-4.35, 2.6), (-1.2, 4.25), (2.4, 3.9), (5.05, 1.4), (4.45, -2.0)]
    p = [(0.9, -2.22), (-2.4, -2.05), (-3.62, 0.0), (-2.9, 2.2), (-0.5, 3.05), (2.1, 2.95), (3.78, 1.1), (3.2, -1.38)]
    l = [(0.5, -1.36), (-1.7, -1.22), (-2.48, 0.3), (-1.76, 1.75), (0.0, 2.1), (1.82, 1.74), (2.36, 0.38), (1.92, -0.9)]
    a = [(0.3, -0.45), (-0.82, -0.5), (-1.25, 0.35), (-0.76, 1.05), (0.35, 1.2), (1.15, 0.75), (1.15, -0.06)]
    blob(ax, scaled(u, c, 1.07), tint(ZONE_FILL["U"], 0.45), z=1)
    blob(ax, u, ZONE_FILL["U"], z=1.1)
    blob(ax, scaled(p, c, 1.05), tint(ZONE_FILL["P"], 0.35), hatch="////", z=1.2)
    blob(ax, p, ZONE_FILL["P"], hatch="////", z=1.3)
    blob(ax, scaled(l, c, 1.06), tint(ZONE_FILL["L"], 0.3), z=1.4)
    blob(ax, l, ZONE_FILL["L"], z=1.5)
    blob(ax, scaled(a, c, 1.1), tint(ZONE_FILL["A"], 0.2), z=1.6)
    blob(ax, a, ZONE_FILL["A"], z=1.7)

    def tag(x, y, text, color, *, glow=True, size=13):
        kw = dict(ha="center", va="center", fontsize=size, color=color, zorder=5)
        if glow:
            kw["bbox"] = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85)
        ax.text(x, y, text, **kw)

    tag(-0.15, 0.2, r"$\mathsf{A}_m$", "white", glow=False, size=14)
    tag(-1.85, -0.35, r"$\mathsf{L}_m$", ZONE_LINE["L"])
    tag(-2.75, 1.45, r"$\mathsf{P}_m$", ZONE_LINE["P"])
    tag(-0.45, -2.66, r"$\mathsf{U}_m$", ZONE_LINE["U"])

    # Difficulty axis: zones are bands of model-relative difficulty (Proposition 1).
    from mathmage.analysis.diagram import arrow
    arrow(ax, (0.55, 0.62), (0.95, 4.55), color=P["ink"], lw=0.9, rad=-0.04, scale=9, z=8)
    for y, txt in ((1.16, r"$p=\tau$"), (2.07, r"$p=\tau_K$"), (3.0, r"$c(\bar\lambda)=\tau$")):
        ax.plot([0.55, 0.85], [y, y], color=P["ink"], lw=0.8, zorder=8)
        ax.text(0.98, y, txt, fontsize=7.6, va="center", ha="left", color=P["ink"], zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))
    ax.text(1.12, 4.55, "harder for model $m$", fontsize=7.4, va="center", ha="left", color=P["ink"],
            family="sans-serif", zorder=8)

    # Labels and pointers (house style).
    label(ax, -8.0, 4.7, "Latent", "Some of $K$ unassisted attempts succeed.\nBottleneck: selecting the right answer.\n"
          "Act: sample, then select on evidence.", color=ZONE_LINE["L"])
    pointer(ax, (-4.7, 3.3), (-1.55, 1.25), rad=-0.3)
    label(ax, -8.0, -1.2, "Autonomous", "Solved reliably by one attempt.\nRisk: persuaded out of a correct answer.\n"
          "Act: accept; revise only on verified errors.", color=ZONE_LINE["A"])
    pointer(ax, (-5.75, -1.42), (-0.35, -0.12), rad=-0.28)
    label(ax, 8.0, 4.7, "Proximal", "Missed by $K$ attempts, solved with a\ncapped scaffold: the ZPD proper.\n"
          "Act: least-informative help, matched tutor.", ha="right", color=ZONE_LINE["P"])
    pointer(ax, (4.75, 3.3), (2.75, 2.15), rad=0.28)
    label(ax, 8.0, -1.2, "Unreachable", "Missed even with capped help; an\noperational label, never “impossible.”\n"
          "Act: verified search, escalate, or stop.", ha="right", color=P["purple"])
    pointer(ax, (5.75, -1.42), (4.22, -0.55), rad=0.35)

    equation_box(ax, -7.9, -5.0, 15.8, 1.15,
                 r"$\mathsf{A}=\{p\geq\tau\}\qquad \mathsf{L}=\{p<\tau\leq 1-(1-p)^K\}\qquad "
                 r"\mathsf{P}=\{r^K<\tau\leq c(\bar\lambda)\}\qquad \mathsf{U}=\{r^K<\tau,\ c(\bar\lambda)<\tau\}$",
                 size=10.2)
    return fig


# ============================================================================ theory helpers
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1 - p))


def vstar(z, e, d):
    """Discernment threshold v*(z) of Proposition 3 (numerically stable form)."""
    from scipy.special import log_expit

    lr = (np.log1p(-np.exp(-d)) - np.log1p(-np.exp(-e)) + log_expit(z) + log_expit(d - z)
          - log_expit(z + e) - log_expit(-z))
    return _sig(lr)


def _zone_bands(ax, ymin, ymax, edges=(-3, 0, 2.402, 5, 8), alpha=0.32):
    for zone, (lo, hi) in zip("ALPU", zip(edges[:-1], edges[1:])):
        ax.axvspan(lo, hi, color=tint(ZONE_FILL[zone], 0.55), lw=0, zorder=0)
        ax.text((lo + hi) / 2, ymin + 0.035 * (ymax - ymin), zone, ha="center", va="bottom", fontsize=9,
                family="sans-serif", weight="bold", color=ZONE_LINE[zone], zorder=3)


# ============================================================================ T1 bands + sweet spots
def sweet_spots():
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar

    K, delta = 8, 5.0
    b = np.linspace(-3, 8, 600)
    one = _sig(-b)
    best = 1 - (1 - one) ** K
    scaf = _sig(delta - b)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.75))
    fig.subplots_adjust(wspace=0.32, left=0.075, right=0.995, bottom=0.17, top=0.88)
    _zone_bands(a1, 0, 1.06)
    a1.plot(b, one, color=P["ink"], lw=1.6, label="one attempt")
    a1.plot(b, best, color=P["cyan_deep"], lw=1.6, ls=(0, (4, 2)), label=f"best of {K}")
    a1.plot(b, scaf, color=P["pink_deep"], lw=1.6, label="scaffolded")
    a1.axhline(0.5, color=P["faint"], lw=0.7, ls=(0, (1, 2)), zorder=1)
    a1.set(xlim=(-3, 8), ylim=(0, 1.06), xlabel=r"item difficulty $b$ (ability $\theta=0$)",
           ylabel="success probability", title="(a) Zones are difficulty bands")
    a1.legend(loc="center right", bbox_to_anchor=(1.0, 0.7), handlelength=1.8, borderaxespad=0.2)

    _zone_bands(a2, -0.05, 1.0)
    gains = [("best of 8 $-$ one", best - one, P["cyan_deep"], (0, (4, 2))),
             ("scaffolded $-$ one", scaf - one, P["pink_deep"], "-"),
             ("scaffolded $-$ best of 8", scaf - best, P["purple"], "-")]
    for name, g, color, ls in gains:
        a2.plot(b, g, color=color, lw=1.6, ls=ls, label=name)
    peaks = []
    for f in (lambda x: -((1 - (1 - _sig(-x)) ** K) - _sig(-x)),
              lambda x: -(_sig(delta - x) - _sig(-x)),
              lambda x: -(_sig(delta - x) - (1 - (1 - _sig(-x)) ** K))):
        r = minimize_scalar(f, bounds=(-3, 8), method="bounded")
        peaks.append((r.x, -r.fun))
    # Closed forms (Proposition 2): b* = -logit(1 - K^{-1/(K-1)}), tanh(e/4) at b = delta/2.
    assert abs(peaks[0][0] + _logit(1 - K ** (-1 / (K - 1)))) < 1e-3
    assert abs(peaks[1][0] - delta / 2) < 1e-3 and abs(peaks[1][1] - np.tanh(delta / 4)) < 1e-6
    names = []
    for (x, y), color, (name, *_rest) in zip(peaks, (P["cyan_deep"], P["pink_deep"], P["purple"]), gains):
        a2.plot([x], [y], "o", ms=4.5, color=color, zorder=5)
        names.append(f"{name} (max {y:.2f} at $b={x:.2f}$)")
    for line, name in zip(a2.get_lines()[:3], names):
        line.set_label(name)
    a2.axhline(0, color=P["faint"], lw=0.7, ls=(0, (1, 2)))
    a2.set(xlim=(-3, 8), ylim=(-0.05, 1.2), xlabel=r"item difficulty $b$", ylabel="gain in success probability",
           title="(b) Two sweet spots")
    a2.legend(loc="upper right", handlelength=1.8, borderaxespad=0.15, fontsize=6.6, labelspacing=0.3)
    return fig


# ============================================================================ T2 discernment threshold
def discernment():
    import matplotlib.pyplot as plt

    z = np.linspace(-8, 8, 801)
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.9)
    ax.axvspan(0, 8, color=tint(ZONE_FILL["A"], 0.85), lw=0, zorder=0)
    ax.text(7.8, 0.04, "solver already succeeds\nmore often than not", ha="right", va="bottom", fontsize=7,
            color=ZONE_LINE["A"], family="sans-serif")
    curves = [((2, 2), P["purple"], "-"), ((4, 4), P["purple"], (0, (4, 2))),
              ((4, 1), P["pink_deep"], "-"), ((1, 4), P["cyan_deep"], "-")]
    for (e, d), color, ls in curves:
        ax.plot(z, vstar(z, e, d), color=color, ls=ls, lw=1.6, label=f"$e={e},\\ d={d}$")
        assert np.all(np.diff(vstar(z, e, d)) >= -1e-12)  # Proposition 3(i): monotone
    ax.set(xlim=(-8, 8), ylim=(0, 1), xlabel=r"unassisted logit margin $z=a(\theta-b)$",
           ylabel=r"required peer validity $v^\star(z)$", title="Mastered answers need nearly infallible peers")
    ax.legend(loc="upper left", fontsize=7, handlelength=2.0)
    return fig


# ============================================================================ T3 rescue-harm plane
def rescue_harm(points=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.9)
    h = np.linspace(0, 0.5, 200)
    ax.fill_between(h, h, 0.62, color=tint(P["cyan"], 0.8), lw=0, zorder=0)
    for p0, color in ((0.3, P["cyan_deep"]), (0.5, P["purple"]), (0.7, P["pink_deep"]), (0.9, P["indigo"])):
        slope = p0 / (1 - p0)
        hh = h[slope * h <= 0.62]
        ax.plot(hh, slope * hh, color=color, lw=1.4)
        if slope * 0.42 <= 0.55:
            ax.text(0.42, slope * 0.42 + 0.014, f"$p={p0}$", color=color, fontsize=7.5, ha="center", va="bottom",
                    rotation=np.degrees(np.arctan(slope * 0.5 / 0.62 * 3.0 / 3.6)) * 0.0)
        else:
            ax.text(0.47 / slope + 0.012, 0.47, f"$p={p0}$", color=color, fontsize=7.5, ha="left", va="center")
    ax.text(0.27, 0.6, "net gain when $p=0.5$", fontsize=7, family="sans-serif", color=P["muted"], va="top")
    if points:
        for pt in points:
            ax.plot(pt["harm"], pt["rescue"], pt.get("marker", "o"), ms=5, color=pt.get("color", P["purple"]),
                    zorder=5)
            ax.annotate(pt["label"], (pt["harm"], pt["rescue"]), xytext=(4, 3), textcoords="offset points",
                        fontsize=6.8)
    ax.set(xlim=(0, 0.5), ylim=(0, 0.62), xlabel=r"harm $h$ (correct $\to$ wrong)",
           ylabel=r"rescue $r$ (wrong $\to$ correct)",
           title=r"$\Delta\mathrm{acc}=(1-p)\,r-p\,h$")
    return fig


def _contour_segments(X, Y, Z, level):
    """Contour polylines (computed offscreen) for drawing crisp 3D lines on a surface."""
    from contourpy import contour_generator

    gen = contour_generator(X, Y, Z)
    return [np.asarray(seg) for seg in gen.lines(level) if len(seg) > 1]


# ============================================================================ T4 3D surfaces
def surfaces():
    import matplotlib.pyplot as plt
    from matplotlib import cm  # noqa: F401
    from matplotlib.colors import TwoSlopeNorm

    K = 8
    fig = plt.figure(figsize=(6.9, 3.1))
    # (a) advantage of one scaffolded attempt over oracle best-of-K
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    z = np.linspace(-8, 3, 120)
    e = np.linspace(0, 8, 100)
    Z, E = np.meshgrid(z, e)
    adv = _sig(Z + E) - (1 - (1 - _sig(Z)) ** K)
    norm = TwoSlopeNorm(vmin=adv.min(), vcenter=0.0, vmax=adv.max())
    ax.plot_surface(Z, E, adv, cmap=style.CMAP_CYAN_PINK, norm=norm, rstride=2, cstride=2, linewidth=0,
                    antialiased=True, alpha=0.96)
    for seg in _contour_segments(Z, E, adv, 0.0):
        ax.plot(seg[:, 0], seg[:, 1], np.full(len(seg), 0.012), color=P["ink"], lw=1.1, zorder=10)
    ax.contour(Z, E, adv, levels=[0.0], colors=[P["purple"]], linewidths=0.9, zdir="z", offset=adv.min() - 0.02)
    ax.plot([-8, 3], [np.log(K)] * 2, [adv.min() - 0.02] * 2, color=P["pink_deep"], lw=0.9, ls=(0, (3, 2)))
    ax.text(1.2, np.log(K) - 1.15, adv.min() - 0.02, r"$e=\ln K$", fontsize=7, color=P["pink_deep"], zorder=10)
    ax.set_xlabel(r"margin $z$", labelpad=-4)
    ax.set_ylabel(r"scaffold lift $e$", labelpad=-4)
    ax.set_zlabel("advantage", labelpad=-6)
    ax.set_title("(a) Scaffold beats best of 8 (pink)", fontsize=8.5, pad=0)
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_zlim(adv.min() - 0.02, adv.max())
    # (b) discernment threshold landscape (e = d)
    bx = fig.add_subplot(1, 2, 2, projection="3d")
    z2 = np.linspace(-8, 8, 120)
    s = np.linspace(0.3, 6, 100)
    Z2, S2 = np.meshgrid(z2, s)
    V = vstar(Z2, S2, S2)
    bx.plot_surface(Z2, S2, V, cmap=style.CMAP_PERI, rstride=2, cstride=2, linewidth=0, antialiased=True,
                    alpha=0.96, vmin=0, vmax=1)
    for seg in _contour_segments(Z2, S2, V, 0.5):
        bx.plot(seg[:, 0], seg[:, 1], np.full(len(seg), 0.512), color=P["ink"], lw=1.0, zorder=10)
    bx.text(-7.5, 5.6, 0.62, r"$v^\star=0.5$", fontsize=7, color=P["ink"], zorder=11)
    bx.set_xlabel(r"margin $z$", labelpad=-4)
    bx.set_ylabel(r"$e=d$", labelpad=-4)
    bx.zaxis.set_rotate_label(False)
    bx.set_zlabel(r"$v^\star$", labelpad=-6, rotation=0)
    bx.set_title("(b) Required peer validity rises with mastery", fontsize=8.5, pad=0)
    bx.view_init(elev=24, azim=-122)
    bx.tick_params(labelsize=6, pad=-2)
    bx.set_zlim(0, 1)
    for axis3 in (ax, bx):
        for a in (axis3.xaxis, axis3.yaxis, axis3.zaxis):
            a.set_pane_color((1, 1, 1, 0))
            a._axinfo["grid"]["color"] = P["grid"]
            a._axinfo["grid"]["linewidth"] = 0.5
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.04, top=0.95, wspace=0.05)
    return fig


def posterior_flow():
    """The learning/inference boundary, without claiming empirical calibration."""
    from mathmage.analysis.diagram import arrow, card

    fig, ax = _canvas(7.2, 4.2, (0, 14.4), (0, 8.4))
    ax.text(.3, 8.1, "A posterior without test answers", fontsize=15, weight="bold", va="top")
    ax.text(.3, 7.35, "Proposed inference flow; probabilities depend on a calibrated generative model.",
            fontsize=8.4, color=P["muted"])
    card(ax, .3, 4.5, 3.65, 2.1, "Test-time observations",
         "Independent answer clusters\nProgram outputs and abstentions\nLocated reviewer evidence",
         accent=P["cyan_deep"], body_size=8)
    card(ax, .3, 1.5, 3.65, 2.1, "Frozen training estimates",
         "Abilities and item difficulty\nError dependence and evidence\nAction effects and stopping rule",
         accent=P["pink_deep"], body_size=8)
    card(ax, 5.05, 3, 4.0, 2.55, "Update the state",
         "Mass on each observed answer\nMass on an unseen correct answer\nUncertainty about item difficulty\nNo reference answer is accessed",
         accent=P["purple"], body_size=8)
    card(ax, 10.35, 3, 3.65, 2.55, "Act or stop",
         "Score feasible actions\nExpected gain / expected cost\nCompare with frozen threshold\nRetain original candidates",
         accent=P["cyan_deep"], body_size=8)
    arrow(ax, (4.05, 5.45), (4.95, 4.85), color=P["cyan_deep"])
    arrow(ax, (4.05, 2.55), (4.95, 3.65), color=P["pink_deep"])
    arrow(ax, (9.15, 4.3), (10.25, 4.3))
    ax.text(5.05, 2.15, "Training labels fit the model; held-out development data select the operating point.",
            fontsize=7.7, color=P["muted"])
    ax.text(.3, .55, "Model-based confidence is not proof validity or a measured developmental zone.\n"
            "Check held-out reliability, correlated errors, family shift, and misleading evidence before drawing claims.",
            fontsize=8.1, color=P["muted"], linespacing=1.4)
    return fig


def anytime_prefix():
    """Illustrative discrete thresholds: no outcome-driven threshold selection."""
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.35), gridspec_kw={"width_ratios": [1.05, 1]})
    ratios = np.array([7.5, 2.8, 4.3, 1.2])
    threshold = 3.0
    ax.plot(np.arange(4), ratios, color=P["purple"], marker="o", ms=4)
    ax.axhline(threshold, color=P["pink_deep"], ls="--", lw=1)
    ax.scatter([1], [ratios[1]], color=P["pink_deep"], s=50, zorder=5)
    ax.annotate("stop at the first failure", xy=(1, 2.8), xytext=(.15, 5.7),
                fontsize=8, arrowprops={"arrowstyle": "->", "color": P["muted"]})
    ax.text(2.1, 3.25, r"frozen $\mu=3$", fontsize=8, color=P["pink_deep"])
    ax.set(xlabel="Action index in the maximal run", ylabel="Expected gain / expected dollar",
           xticks=range(4), ylim=(0, 8.5))
    ax.set_title("(a) A threshold keeps a prefix", loc="left", fontsize=9)
    # The nonmonotone realized correctness path illustrates why a threshold must
    # be chosen on development data, rather than by the best test-time outcome.
    costs = np.array([.01, .02, .04, .065, .08])
    correct = np.array([False, False, True, False, True])
    bx.step(costs, correct.astype(float), where="post", color=P["cyan_deep"], lw=1.6)
    bx.scatter(costs, correct.astype(float), color=P["purple"], s=18, zorder=4)
    bx.scatter([costs[1]], [0], color=P["pink_deep"], s=50, zorder=5)
    bx.set(xlabel="Accumulated logical cost (illustrative USD)", ylabel="Realized answer correctness",
           ylim=(-.15, 1.25), yticks=[0, 1], yticklabels=["wrong", "right"])
    bx.set_title("(b) Realized answers can regress", loc="left", fontsize=9)
    fig.text(.07, .96, "One maximal run, multiple stopping policies", fontsize=13, weight="bold")
    fig.text(.07, .035, "Illustrative values, not experimental results. Replay requires threshold-independent action selection\n"
             "and selection at each snapshot. Choose the threshold on development data, then freeze it for test.",
             fontsize=7.4, color=P["muted"], linespacing=1.3)
    fig.subplots_adjust(left=.08, right=.98, top=.78, bottom=.31, wspace=.55)
    return fig


FIGURES = {"zones": zones, "sweet_spots": sweet_spots, "discernment": discernment,
           "rescue_harm": rescue_harm, "surfaces": surfaces,
           "posterior_flow": posterior_flow, "anytime_prefix": anytime_prefix}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    args = ap.parse_args(argv)
    style.use()
    import matplotlib.pyplot as plt

    for name in args.names or list(FIGURES):
        fig = FIGURES[name]()
        style.save(fig, f"C_{name}", OUT)
        plt.close(fig)
        print("saved", name)


if __name__ == "__main__":
    main(sys.argv[1:])
