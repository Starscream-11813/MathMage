"""Vector research figures using the shared Latin Modern TrueType house style.

PDF export uses Matplotlib's PGF backend and LuaLaTeX. The font files and their
license are pinned in assets/fonts; no font substitution is performed silently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

from mathmage import paths
from mathmage.analysis import style as house_style

ZONE = house_style.ZONE_LINE
ACCENT = house_style.P["purple"]


def style(*, publication=True):
    house_style.use(publication=publication, base_size=10)


def pipeline(output: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set(xlim=(0, 12), ylim=(0, 5.5))
    ax.axis("off")
    ax.text(.25, 5.12, "MathMage: evidence guides collaboration", fontsize=20, weight="bold")
    ax.text(.25, 4.72, "Intended architecture: independent proposals, posterior updates, and a frozen stopping policy", fontsize=11, color="#626B79")

    def box(x, y, w, h, title, body, color=ACCENT):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.12",
                                   facecolor=color, alpha=.08, edgecolor=color, linewidth=1.1))
        ax.text(x + .14, y + h - .26, title, fontsize=11, weight="bold", color=color, va="top")
        ax.text(x + .14, y + h - .68, body, fontsize=9.8, linespacing=1.65, va="top")

    def arrow(start, end, color=ACCENT):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1, color=color))

    box(.3, 1.65, 2.1, 2.5, "1  Propose", "Isolated member contexts\nSolutions + certificates\nTraining strategy cards")
    box(2.9, 1.65, 2.1, 2.5, "2  Check + route", "Independent program\nAnswer-cluster posterior\nUncertainty kept explicit")
    arrow((2.53, 2.85), (2.76, 2.85))
    lanes = [("A", "Confident answer: consider stopping"), ("L", "Located objections + rebuttal"),
             ("P", "Graded hints + matched tutor"), ("U", "Fresh search + diversification")]
    for index, (zone, label) in enumerate(lanes):
        y = 3.73 - index * .67
        ax.add_patch(FancyBboxPatch((5.55, y - .28), 3.1, .52, boxstyle="round,pad=.04",
                                   facecolor=ZONE[zone], edgecolor=ZONE[zone], alpha=.12))
        ax.text(5.7, y, zone, weight="bold", color=ZONE[zone], va="center")
        ax.text(6.02, y, label, fontsize=9.5, va="center")
        arrow((5.13, 2.85), (5.47, y), ZONE[zone])
        arrow((8.72, y), (9.13, 2.85), ZONE[zone])
    box(9.3, 1.65, 2.15, 2.5, "3  Act or stop", "Value per expected dollar\nThreshold frozen on dev\nOriginal candidates retained")
    ax.plot([.35, 11.6], [1.1, 1.1], color="#DADCE5", lw=.8)
    ax.text(.35, .69, "Offline only", color=ACCENT, weight="bold")
    ax.text(1.75, .69, "Family-disjoint train/dev data  |  tutoring matrix  |  strategy bank  |  frozen artifacts", fontsize=10)
    ax.text(.35, .15, "Architecture diagram, not a measured result. Routing labels are operational; empirical zones and hint leakage need separate measurements.",
            fontsize=8.5, color="#626B79")
    fig.savefig(output)
    plt.close(fig)


def atlas_uncertainty(atlas_dir: Path, output: Path):
    import matplotlib.pyplot as plt
    from scipy.stats import beta

    frame = pd.read_csv(atlas_dir / "item_posteriors.csv")
    manifest = json.loads((atlas_dir / "atlas_manifest.json").read_text(encoding="utf-8"))
    settings = manifest["settings"]
    if settings.get("prior") != "Beta(1,1)":
        raise ValueError("atlas posterior figure currently requires the declared Beta(1,1) prior")
    alpha = float(settings["alpha"])
    percent = r"\%" if mpl.rcParams["text.usetex"] else "%"
    zero = frame[frame.n_successes == 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle("What public attempts can establish", x=.09, ha="left", fontsize=18, weight="bold")
    x = np.linspace(0, 1, 601)
    for i, n in enumerate(sorted(zero.n_trials.unique())):
        color = [ACCENT, ZONE["L"], ZONE["A"]][i % 3]
        axes[0].plot(x, beta.pdf(x, 1, int(n) + 1), color=color, lw=2, label=f"0 successes in {int(n)} attempts")
        bound = 1 - alpha ** (1 / int(n))
        axes[0].axvline(bound, color=color, ls="--", lw=1, alpha=.7)
        axes[0].text(bound + .015, 1.2 + i * .65, f"{100 * (1 - alpha):.0f}{percent} upper: {bound:.2f}", color=color, fontsize=9)
    ymax = max([int(n) + 1 for n in zero.n_trials.unique()] + [1]) * 1.12
    axes[0].set(xlabel="Unassisted success probability", ylabel="Posterior density", xlim=(0, 1), ylim=(0, ymax))
    axes[0].set_title("Zero observed successes leave wide uncertainty", loc="left", pad=14, fontsize=11)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    counts = frame.inferred_zone.value_counts()
    labels = ["autonomous", "inconclusive", "latent", "unassisted_below_sampling_threshold"]
    display = ["Autonomous", "Inconclusive", "Latent", "Below sampling threshold"]
    values = [int(counts.get(label, 0)) for label in labels]
    colors = [ZONE["A"], "#B6BCC8", ZONE["L"], ZONE["U"]]
    bars = axes[1].barh(range(4), values, color=colors, height=.56)
    for bar, value in zip(bars, values):
        axes[1].text(value + len(frame) * .025, bar.get_y() + bar.get_height() / 2,
                     f"{value:,}", va="center", fontsize=10)
    axes[1].set(yticks=range(4), yticklabels=display, xlabel="Item/configuration pairs", xlim=(0, max(max(values), 1) * 1.22))
    axes[1].invert_yaxis()
    axes[1].set_title(f"Posterior labels at {100 * settings['zone_confidence']:.0f}{percent} confidence", loc="left", pad=14, fontsize=11)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    axes[1].grid(axis="x", alpha=.12)
    axes[1].set_axisbelow(True)
    fig.text(.09, .025, f"Public MathArena source labels; {manifest['n_analyzed_attempts']:,} attempts, {len(frame):,} item/configuration pairs. "
             f"Beta(1,1) prior; K={settings['horizon_k']}, threshold={settings['tau']:g}.\nDashed lines: exact one-sided binomial bounds. No scaffold data: proximal and unreachable status are unidentifiable.",
             fontsize=8.2, color="#626B79", linespacing=1.4)
    fig.subplots_adjust(left=.09, right=.98, top=.78, bottom=.23, wspace=.63)
    fig.savefig(output)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=paths.FIGURES)
    ap.add_argument("--atlas", type=Path, default=paths.TABLES / "atlas")
    ap.add_argument("--preview", action="store_true", help="PNG preview only; publication PDF uses LuaLaTeX")
    args = ap.parse_args()
    style(publication=not args.preview)
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if args.preview else ".pdf"
    pipeline(args.output / ("pipeline" + suffix))
    atlas_uncertainty(args.atlas, args.output / ("atlas_uncertainty" + suffix))


if __name__ == "__main__":
    main()
