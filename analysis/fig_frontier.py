"""Redraw the paper frontier from the included aggregate snapshot; no model calls.

    python analysis/fig_frontier.py --draft

This renders existing estimates. Statistical recomputation requires recorded runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MATHMAGE_PALETTE"] = "paper"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT_FIG = REPO / "results/figures"


def figure(rows, s, draft):
    from mathmage.analysis import style
    style.use(publication=not draft, base_size=7.6)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    P = style.P
    colour = {"vote": P["lav1"], "SC": P["cyan"], "USC": P["indigo"], "MoA": P["violet"], "debate": P["pink"]}
    names = {"vote": "majority vote", "SC": "self-consistency", "USC": "universal SC", "MoA": "mixture-of-agents",
             "debate": "debate"}
    marker = {"SAT-style": "o", "2026": "s", "open": "^"}
    size = {"o": 4.6, "s": 4.4, "^": 5.0}
    fig, ax = plt.subplots(figsize=(3.03, 2.5))
    x0, x1, y0, y1 = 0.6, 2.25, -18.6, 9.5
    ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.fill_between([x0, 1], 0, y1, color=style.tint(P["pink"], 0.86), lw=0, zorder=0)
    ax.fill_between([1, x1], y0, 0, color=style.tint(P["lav1"], 0.84), lw=0, zorder=0)
    ax.axhline(0, color=P["lav2"], lw=0.6, zorder=1)
    ax.axvline(1, color=P["lav2"], lw=0.6, zorder=1)
    for r in rows:                                   # the program's effect: base baseline -> shown the program
        if r["program"]:
            base = next(b for b in rows if b["set"] == r["set"] and b["method"] == r["method"].split(" + ")[0])
            ax.annotate("", xy=(r["ratio"], r["delta"]), xytext=(base["ratio"], base["delta"]),
                        arrowprops=dict(arrowstyle="-|>", color=colour[r["family"]], lw=0.6, mutation_scale=5,
                                        shrinkA=2.4, shrinkB=2.6, connectionstyle="arc3,rad=-0.25"), zorder=2)
    for r in rows:
        mk, c = marker[r["team"]], colour[r["family"]]
        ax.plot(r["ratio"], r["delta"], mk, ms=size[mk], mfc="white" if r["program"] else c, mec=c, mew=0.9, zorder=3)
    ax.plot(1, 0, "*", ms=10, color=P["purple"], mec="white", mew=0.5, zorder=5)
    corner = dict(fontsize=6.3, style="italic", color=P["lav4"], linespacing=1.15)
    ax.text(0.615, 8.9, f"more accurate\nand cheaper: {s['above_cheaper']}", ha="left", va="top", **corner)
    ax.text(2.2, -18.1, f"less accurate and costlier: {s['below_costlier']}", ha="right", va="bottom", **corner)
    ax.xaxis.set_major_locator(FixedLocator([0.7, 1.0, 1.5, 2.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"{v:g}$\\times$"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([-15, -10, -5, 0, 5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"${v:+.0f}$" if v else "0"))
    ax.set_xlabel("cost per problem relative to MathMage (log scale)", fontsize=7.0)
    ax.set_ylabel("$\\Delta$ accuracy (points)", labelpad=1)
    ax.grid(alpha=.35)
    handles = [Line2D([], [], ls="none", marker="o", ms=4.2, color=colour[f], label=names[f]) for f in colour]
    handles.append(Line2D([], [], ls="none", marker="*", ms=7.5, color=P["purple"], mec="white", mew=0.4,
                          label="MathMage at $\\mu^\\star$"))      # sixth entry: keeps the teams in the third column
    handles += [Line2D([], [], ls="none", marker=m, ms=size[m], mfc="white", mec=P["muted"], mew=0.8,
                       label=lab) for m, lab in (("o", "SAT-style team"), ("s", "2026 team"), ("^", "open-weight team"))]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.43, -0.21), ncol=3, fontsize=6.3,
                    frameon=False, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.0, handlelength=1.0,
                    labelspacing=0.3)
    for t, h in zip(leg.get_texts(), handles):
        t.set_color(P["ink"])
    fig.tight_layout(pad=0.25)
    out = style.save(fig, "frontier", OUT_FIG)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true", help="render without TeX")
    args = parser.parse_args()
    data = json.loads((REPO / "results/tables/v7_frontier.json").read_text(encoding="utf-8"))
    print(figure(data["pairs"], data["summary"], args.draft))


if __name__ == "__main__":
    main()
