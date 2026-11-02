"""v6 figures from results/tables/v6_results.json (read-only; run analysis/v6_results.py first).

    python analysis/fig_v6.py            # publication figures (LuaLaTeX, Latin Modern)
    python analysis/fig_v6.py --draft    # fast previews

  v6_fresh     accuracy against cost on the 187 fresh problems, both teams (95% intervals; SC@k curves)
  v6_hints     the crossed hint trial: rescue of initially wrong members by arm, both teams
  v6_voc       predicted value of computation against the realized change in decision correctness
  v6_learning  S10: held-out stage-0 accuracy and ECE as the calibration pool grows
Figures use the paper palette (the shared nine colours); coloured text uses the darker shades.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MATHMAGE_PALETTE"] = "paper"            # must precede the style import

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mathmage.analysis import style  # noqa: E402

TABLES, FIGURES = REPO / "results/tables", REPO / "results/figures"
TEAMS = (("H9", "sat26", "2026 team · 187 fresh problems"),
         ("H10", "lane", "Open-weight team · 187 fresh problems × 8 seeds"))
DEEP_CYAN, DEEP_PINK = "#1F94CC", "#BC45D8"     # legible text shades of cyan and pink (as in the zones figure)


def pct(label, draft):
    return label if draft else label.replace("%", r"\%")


def dollars(draft):
    sign = "$" if draft else r"\$"
    return lambda value, _pos=None: f"{sign}{value:g}"


def method_styles(draft):
    """Marker colour, marker, fill and text colour for each method, from the nine colours."""
    P = style.P
    return {
        "mu*": ("MathMage", P["purple"], "*", True, P["purple"], 9.0),
        "maximal": ("$\\mu{=}0$", P["purple"], "*", False, P["purple"], 9.0),
        "vote": ("majority vote", P["lav2"], "o", True, P["lav4"], 4.4),
        "weighted vote + program": ("weighted vote + program", P["lav1"], "o", True, P["lav4"], 4.4),
        "USC": ("universal SC", P["lav3"], "s", True, P["lav4"], 4.0),
        "MoA": ("mixture-of-agents", P["violet"], "D", True, P["violet"], 3.8),
        "debate": ("debate", P["pink"], "^", True, DEEP_PINK, 4.6),
        "SC@10 (DeepSeek-V3)": ("DeepSeek-V3 SC@10", P["cyan"], "o", True, DEEP_CYAN, 4.4),
        "SC@5 (GPT-5-mini)": ("GPT-5-mini SC@5", P["cyan"], "o", True, DEEP_CYAN, 4.4),
        "SC@5 (pooled seeds)": ("SC@5", P["cyan"], "o", True, DEEP_CYAN, 4.4),
    }


def place_labels(ax, points, obstacles, fontsize, order, prefer=None):
    """Greedy label placement in display space: each label takes the first candidate offset whose estimated box
    overlaps no marker, error bar, curve or earlier label and stays inside the axes (else the least overlap).
    ``prefer`` maps a key to candidate offsets tried first (e.g. keep a label on one side of its marker)."""
    fig = ax.figure
    k = fig.dpi / 72.0
    frame = ax.get_window_extent()
    boxes = list(obstacles)
    default = [(7, 0, "left"), (-7, 0, "right"), (0, 9, "center"), (0, -9, "center"), (7, 7, "left"),
               (7, -7, "left"), (-7, 7, "right"), (-7, -7, "right"), (12, 0, "left"), (-12, 0, "right"),
               (0, 15, "center"), (0, -15, "center"), (10, 13, "left"), (10, -13, "left"), (-10, 13, "right"),
               (-10, -13, "right"), (18, 0, "left"), (-18, 0, "right")]
    prefer = prefer or {}

    def area(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    placed = {}
    for key in order:
        if key not in points:
            continue
        (x, y), text, color = points[key]
        px, py = ax.transData.transform((x, y))
        width, height = 0.47 * fontsize * k * len(text.replace("$", "").replace("\\mu{=}", "m=")), 1.05 * fontsize * k
        best = None
        for dx, dy, ha in prefer.get(key, []) + default:
            cx, cy = px + dx * k, py + dy * k
            x0 = cx if ha == "left" else cx - width if ha == "right" else cx - width / 2
            box = (x0, cy - height / 2, x0 + width, cy + height / 2)
            outside = (max(0, frame.x0 - box[0]) + max(0, box[2] - frame.x1)) * height + \
                (max(0, frame.y0 - box[1]) + max(0, box[3] - frame.y1)) * width
            score = sum(area(box, other) for other in boxes) + 4 * outside
            if best is None or score < best[0]:
                best = (score, dx, dy, ha, box)
            if score == 0:
                break
        _, dx, dy, ha, box = best
        boxes.append(box)
        placed[key] = (dx, dy, ha)
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=fontsize, color=color, zorder=6)
    return placed


def fig_fresh(out, draft):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    P = style.P
    teams = [t for t in TEAMS if t[0] in out]
    if not teams:
        return
    styles = method_styles(draft)
    fig, axes = plt.subplots(1, len(teams), figsize=(7.0, 2.6), squeeze=False)
    plotted = []
    for ax, (key, package, title) in zip(axes[0], teams):
        methods = out[key]["methods"]
        curves = []
        for family, dash in (("DeepSeek-V3", (0, (3, 1.6))), ("GPT-5-mini", "-"), ("pooled", "-")):
            curve = out.get(f"_curves_{package}", {}).get(family)
            if curve:
                pts = sorted((v[0], 100 * v[1]) for v in curve.values())
                ax.plot([c for c, _ in pts], [a for _, a in pts], color=P["cyan"], lw=1.0, ls=dash, marker="o", ms=2.0,
                        zorder=2)
                curves.append(pts)
        shown = [k for k in ("vote", "weighted vote + program", "USC", "MoA", "debate", "SC@10 (DeepSeek-V3)",
                             "SC@5 (GPT-5-mini)", "SC@5 (pooled seeds)", "maximal", "mu*") if methods.get(k)]
        price_curve = out.get(f"_price_curve_{package}") or []
        if price_curve:                                          # every price, replayed from the maximal run
            pts = sorted((p["cost"], 100 * p["accuracy"]) for p in price_curve)
            ax.plot([c for c, _ in pts], [a for _, a in pts], color=P["purple"], lw=.9, alpha=.85, zorder=3)
            curves.append(pts)
        elif methods.get("mu*") and methods.get("maximal"):     # fallback: join the two operating points
            a, b = methods["mu*"], methods["maximal"]
            ax.plot([a["cost"], b["cost"]], [100 * a["macro"], 100 * b["macro"]], color=P["purple"], lw=.8,
                    ls=(0, (1.5, 1.5)), zorder=3)
        for name in shown:
            m = methods[name]
            label, color, marker, filled, _, size = styles[name]
            ax.errorbar(m["cost"], 100 * m["macro"], yerr=[[100 * (m["macro"] - m["low"])], [100 * (m["high"] - m["macro"])]],
                        fmt=marker, ms=size, color=color, mfc=color if filled else "white", mec=color, mew=.9,
                        elinewidth=.7, capsize=1.5, zorder=5 if name in ("mu*", "maximal") else 4)
        costs = [methods[n]["cost"] for n in shown] + [c for pts in curves for c, _ in pts]
        lows = [100 * methods[n]["low"] for n in shown] + [a for pts in curves for _, a in pts]
        highs = [100 * methods[n]["high"] for n in shown] + [a for pts in curves for _, a in pts]
        ax.set_xscale("log")
        ax.set_xlim(min(costs) / 1.35, max(costs) * 1.9)
        ax.set_ylim(min(lows) - 2, max(highs) + 2)
        lo, hi = ax.get_xlim()
        ticks = [t for t in (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1) if lo <= t <= hi]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(dollars(draft)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title, fontsize=7.4, loc="left")
        ax.set_xlabel("cost per problem (log scale)")
        ax.grid(alpha=.5)
        plotted.append((ax, methods, shown, curves))
    axes[0][0].set_ylabel(pct("macro accuracy (%)", draft))
    fig.tight_layout(w_pad=1.6)
    k = fig.dpi / 72.0
    for ax, methods, shown, curves in plotted:
        obstacles = []
        for name in shown:
            m = methods[name]
            px, py = ax.transData.transform((m["cost"], 100 * m["macro"]))
            r = (5.0 if name in ("mu*", "maximal") else 3.2) * k
            obstacles.append((px - r, py - r, px + r, py + r))
            _, lo_y = ax.transData.transform((m["cost"], 100 * m["low"]))
            _, hi_y = ax.transData.transform((m["cost"], 100 * m["high"]))
            obstacles.append((px - 1.8 * k, lo_y, px + 1.8 * k, hi_y))
        for pts in curves:
            for (c0, a0), (c1, a1) in zip(pts, pts[1:]):
                for t in [i / 8 for i in range(9)]:
                    x0, y0 = ax.transData.transform((c0, a0))
                    x1, y1 = ax.transData.transform((c1, a1))
                    x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                    obstacles.append((x - 1.5 * k, y - 1.5 * k, x + 1.5 * k, y + 1.5 * k))
        points = {name: ((methods[name]["cost"], 100 * methods[name]["macro"]), styles[name][0], styles[name][4])
                  for name in shown if name not in ("mu*", "maximal")}     # MathMage's markers are in the legend
        order = ["debate", "MoA", "USC", "vote", "weighted vote + program", "SC@5 (GPT-5-mini)",
                 "SC@10 (DeepSeek-V3)", "SC@5 (pooled seeds)"]
        place_labels(ax, points, obstacles, 6.0, order)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="*", ms=8, ls="none", color=P["purple"], label="MathMage at $\\mu^\\star$"),
               Line2D([], [], marker="*", ms=8, ls="none", color=P["purple"], mfc="white",
                      label="MathMage at $\\mu{=}0$"),
               Line2D([], [], color=P["purple"], lw=.9, label="MathMage, every price")]
    axes[0][0].legend(handles=handles, loc="lower right", fontsize=6.0, frameon=False, handlelength=1.6,
                      labelcolor=P["purple"])
    style.save(fig, "v6_fresh", FIGURES)
    plt.close(fig)


ARMS = [("fresh attempt (k=1)", "one fresh attempt"), ("best of two fresh attempts", "best of two fresh attempts"),
        ("placebo re-solve", "placebo re-solve"), ("wrong tutor's hint (ITT)", "wrong tutor's hint"),
        ("right tutor's hint (ITT)", "right tutor's hint")]


def fig_hints(out, draft):
    import matplotlib.pyplot as plt
    P = style.P
    trials = [(p, out["S8"][p]) for p in ("sat26", "lane") if p in (out.get("S8") or {})]
    if not trials:
        return
    fig, axes = plt.subplots(1, len(trials), figsize=(7.0, 1.9), squeeze=False, sharex=True)
    colors = [P["lav1"], P["lav3"], P["cyan"], P["pink"], P["purple"]]
    for ax, (package, trial) in zip(axes[0], trials):
        rates_all = trial.get("rates (right-tutor units)") or trial["rates"]
        for y, ((key, label), color) in enumerate(zip(ARMS, colors)):
            r = rates_all.get(key)
            if not r:
                continue
            ax.errorbar(100 * r["mean"], y, xerr=[[100 * (r["mean"] - r["low"])], [100 * (r["high"] - r["mean"])]],
                        fmt="o", ms=4.2, color=color, elinewidth=.8, capsize=1.6)
            ax.annotate(f"{100 * r['mean']:.1f}", (100 * r["mean"], y), xytext=(0, 4), textcoords="offset points",
                        fontsize=5.6, color=P["purple"], ha="center")
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([label for _, label in ARMS], fontsize=6.4)
        ax.set_ylim(-.6, len(ARMS) - .4)
        name = "2026 team" if package == "sat26" else "Open-weight team"
        ax.set_title(f"{name} · members with a right tutor", fontsize=7.2, loc="left")
        ax.set_xlabel(pct("rescued (%)", draft))
        ax.grid(alpha=.5, axis="x")
    fig.tight_layout(w_pad=1.2)
    style.save(fig, "v6_hints", FIGURES)
    plt.close(fig)


def fig_voc(out, draft):
    import matplotlib.pyplot as plt
    P = style.P
    teams = [(p, out["S9"][p]) for p in ("sat26", "lane") if p in (out.get("S9") or {})]
    if not teams:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    markers = {"sample": "o", "tool_alt": "s", "diversify": "D", "review": "^", "judge": "v", "coach (right tutor)": "P"}
    for package, color in (("sat26", P["pink"]), ("lane", P["purple"])):
        rows = dict(teams).get(package)
        if not rows:
            continue
        for kind, v in rows.items():
            ci = v["realized_ci"]
            ax.errorbar(100 * v["mean_predicted"], 100 * v["mean_realized"],
                        yerr=[[100 * (v["mean_realized"] - ci["low"])], [100 * (ci["high"] - v["mean_realized"])]],
                        fmt=markers.get(kind, "o"), ms=4, color=color, elinewidth=.7, capsize=1.3)
    lim = ax.get_xlim()
    ax.plot([0, max(lim[1], ax.get_ylim()[1])], [0, max(lim[1], ax.get_ylim()[1])], color=P["ink"], lw=.6, ls="--")
    ax.axhline(0, color=P["grid"], lw=.8)
    ax.set_xlabel(pct("predicted value of computation (pp)", draft))
    ax.set_ylabel(pct("realized change in correctness (pp)", draft))
    ax.grid(alpha=.5)
    fig.tight_layout()
    style.save(fig, "v6_voc", FIGURES)
    plt.close(fig)


def fig_learning(draft):
    """S10: held-out stage-0 decision accuracy and ECE as the calibration pool grows (cross-fitted)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    P = style.P
    curves = {}
    for roster, label, color, marker in (("sat", "SAT-style team", P["pink"], "o"),
                                         ("open", "Open-weight team", P["purple"], "s")):
        path = TABLES / f"v6_learning_{roster}.json"
        if path.exists():
            curves[roster] = (label, color, marker, json.loads(path.read_text(encoding="utf-8")))
    if not curves:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for label, color, marker, data in curves.values():
        sizes = sorted((k for k in data["curve"] if k != "all"), key=int)
        xs = [16 + int(k) for k in sizes] + [16 + data["pool_items"]]
        rows = [data["curve"][k] for k in sizes] + [data["curve"]["all"]]
        for ax, field in zip(axes, ("accuracy", "ece")):
            scale = 100 if field == "accuracy" else 1
            ax.plot(xs, [scale * r[field] for r in rows], color=color, marker=marker, ms=3, lw=1.2, label=label)
    for ax, title in zip(axes, (pct("held-out decision accuracy (%)", draft), "held-out ECE (10 bins)")):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([16, 32, 64, 128, 256]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("calibration problems (16 training + pool)")
        ax.set_title(title, fontsize=7.4, loc="left")
        ax.grid(alpha=.5)
    axes[0].legend(fontsize=6.4, frameon=False)
    fig.tight_layout(w_pad=1.5)
    style.save(fig, "v6_learning", FIGURES)
    plt.close(fig)


def fig_forest_v6(out):
    """The appendix forest (analysis/synthesis.py) with the fourth wave's two fresh sets appended (Holm within each team's
    five prespecified contrasts). Written as results/figures/v6/synthesis_forest.pdf."""
    sys.path.insert(0, str(REPO / "analysis"))
    import synthesis as S
    if not (out.get("H9") and out.get("H10")):
        return
    v3, v4, v5 = S.load()
    rows = S.sets(v3, v4, v5)
    for key, title, roster, n, sc in (("H9", "Fresh set, 2026 team", "SAT", 187, "SC@10 (DeepSeek-V3)"),
                                      ("H10", "Fresh set", "open", 1496, "SC@5 (pooled seeds)")):
        family = {"vote": "majority vote", sc: "self-consistency (matched cost)", "USC": "universal self-consistency",
                  "debate": "debate", "MoA": "mixture of agents"}
        contrasts = {family[name.split(" vs ", 1)[1]]: S.tagged(c, "holm") for name, c in out[key]["contrasts"].items()
                     if c and name.split(" vs ", 1)[1] in family}
        rows.append((title, roster, n, "v6", contrasts, out[key]["methods"]))
    short = {"Original suite (E1)": "E1", "April 2025 contests (primary)": "April 2025 (primary)",
             "Strictly post-cutoff for every member": "post-cutoff for all members",
             "All 2025-26 contests": "2025--26 contests", "Original suite (E2)": "E2",
             "April 2025 + 2026 contests (primary)": "April 2025 + 2026 (primary)",
             "All 2025-26 contests (strictly post-cutoff)": "2025--26 contests",
             "Fresh set, 2026 team": "fresh set, 2026 team (primary)", "Fresh set": "fresh set (primary)"}
    S.fig_forest(rows, FIGURES / "v6", v4, rename=short)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    style.use(publication=not args.draft, base_size=8.0)
    fig_learning(args.draft)
    path = TABLES / "v6_results.json"
    if not path.exists():
        return
    out = json.loads(path.read_text(encoding="utf-8"))
    fig_fresh(out, args.draft)
    fig_hints(out, args.draft)
    fig_voc(out, args.draft)
    fig_forest_v6(out)


if __name__ == "__main__":
    main()
