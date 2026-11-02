"""Accuracy against cost on the 2025-26 contest sets, every method (read-only; uses v5_summary.json).

    python analysis/fig_cost.py            # publication figure (LuaLaTeX)
    python analysis/fig_cost.py --draft
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mathmage.analysis import style  # noqa: E402

TABLES, FIGURES = REPO / "results/tables", REPO / "results/figures"
NAMES = {"mu*": "MathMage", "vote": "majority vote", "debate": "debate", "USC": "universal SC", "MoA": "mixture of agents",
         "SC@10 (DeepSeek-V3)": "self-consistency", "SC@5 (pooled seeds)": "self-consistency"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    style.use(publication=not args.draft, base_size=8.0)
    import matplotlib.pyplot as plt
    P = style.P
    v5 = json.loads((TABLES / "v5_summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    offsets = {"SAT roster": {"mu*": (6, -2), "vote": (6, -3), "debate": (-6, 6), "USC": (6, -7), "MoA": (6, 3),
                              "SC@10 (DeepSeek-V3)": (6, -7)},
               "open roster": {"mu*": (6, -2), "vote": (-2, -14), "debate": (-6, 6), "USC": (6, 3), "MoA": (6, 2),
                               "SC@5 (pooled seeds)": (6, -6)}}
    for ax, (roster, title) in zip(axes, (("SAT roster", "SAT-style team · 216 problems from 2025–26 contests"),
                                          ("open roster", "Open-weight team · 1,968 evaluations, all post-cutoff"))):
        methods = v5["S1"][roster]["methods"]
        for key, m in methods.items():
            mathmage = key == "mu*"
            color = P["purple"] if mathmage else P["cyan_deep"] if key.startswith("SC") else P["pink_deep"] if key == "debate" \
                else P["lav2"]
            ax.errorbar(m["cost"], 100 * m["macro"], yerr=[[100 * (m["macro"] - m["low"])], [100 * (m["high"] - m["macro"])]],
                        fmt="*" if mathmage else "o", ms=9 if mathmage else 4.5, color=color, elinewidth=.7, capsize=1.5,
                        zorder=4 if mathmage else 3)
            dx, dy = offsets[roster].get(key, (6, 0))
            ax.annotate(NAMES.get(key, key), (m["cost"], 100 * m["macro"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.6, color=color, ha="left" if dx > 0 else "right",
                        fontweight="bold" if mathmage else "normal")
        ax.set_title(title, fontsize=7.4, loc="left")
        ax.set_xlabel("dollars per problem")
        ax.grid(alpha=.5)
    axes[0].set_ylabel("macro accuracy (%)" if args.draft else r"macro accuracy (\%)")
    fig.tight_layout(w_pad=1.5)
    style.save(fig, "cost_contamination_free", FIGURES)
    plt.close(fig)


if __name__ == "__main__":
    main()
