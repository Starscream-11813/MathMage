"""The zones-of-development concept figure in the paper's notation and type (no data, no API calls).

    python analysis/fig_zones_concept.py

Reuses the conceptual drawing (mathmage.analysis.concepts.zones), then rewrites it for the paper: Definition 1's
notation (r_K for the reach of K fresh attempts, c for success after one scaffold), roman type throughout
(the conceptual drawing uses sans), upright zone symbols, and legible text colours in the house palette.
The original drawing in outputs/figures/concepts is left untouched.
Output: results/figures/zones_concept.{pdf,png}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MATHMAGE_PALETTE", "paper")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from mathmage.analysis import concepts, style  # noqa: E402

OUT = REPO / "results/figures"
P = style.P
RELABEL = {
    r"$p=\tau_K$": r"$r_K=\tau$",
    r"$c(\bar\lambda)=\tau$": r"$c=\tau$",
    "Missed by $K$ attempts, solved with a\ncapped scaffold: the ZPD proper.\n"
    "Act: least-informative help, matched tutor.":
        "Missed by $K$ attempts, solved with one\nscaffold: the ZPD proper.\n"
        "Act: least-informative help, a right tutor.",
    "Missed even with capped help; an\noperational label, never “impossible.”\n"
    "Act: verified search, escalate, or stop.":
        "Missed even with a scaffold; a label\nabout this budget, never “impossible.”\n"
        "Act: verified search, escalate, or stop.",
}
BANNER = (r"$\mathrm{A}=\{p\geq\tau\}\qquad \mathrm{L}=\{p<\tau\leq r_K\}\qquad "
          r"\mathrm{P}=\{r_K<\tau\leq c\}\qquad \mathrm{U}=\{r_K<\tau,\ c<\tau\}$")
DARKER = {P["cyan"]: P["indigo"], P["pink"]: P["violet"], P["lav2"]: P["lav4"]}
# each zone's heading and in-figure tag in a deep shade of the zone's own colour (same hue, legible on white):
# sky for the cyan latent zone, orchid for the pink proximal zone, deep lavender for the dense autonomous core and
# a lighter lavender for the pale unreachable rim
ZONE_TEXT = {"L": "#1F94CC", "P": "#BC45D8", "A": P["purple"], "U": P["lav3"]}
ZONE_OF = {"Latent": "L", "Proximal": "P", "Autonomous": "A", "Unreachable": "U",
           r"$\mathrm{L}_m$": "L", r"$\mathrm{P}_m$": "P", r"$\mathrm{U}_m$": "U"}


def main() -> None:
    style.use()
    # large areas take light tints of the palette; the exact colours stay on lines, markers and accents
    style.ZONE_FILL.update({"L": style.tint(P["cyan"], 0.40), "P": style.tint(P["pink"], 0.45)})
    fig = concepts.zones()
    seen = set()
    for t in fig.findobj(Text):
        s = t.get_text()
        if s in RELABEL:
            t.set_text(RELABEL[s])
            seen.add(s)
        elif s.startswith(r"$\mathsf{A}=\{p\geq\tau\}"):
            t.set_text(BANNER)
            seen.add("banner")
        elif "\\mathsf" in s:
            t.set_text(s.replace("\\mathsf", "\\mathrm"))
        t.set_fontfamily("serif")                     # roman type only
        try:
            rgb = to_rgb(t.get_color())
        except ValueError:
            continue
        for light, dark in DARKER.items():
            if np.allclose(rgb, to_rgb(light), atol=0.02):
                t.set_color(dark)
    missing = (set(RELABEL) | {"banner"}) - seen
    if missing:
        raise SystemExit(f"labels not found (the conceptual figure changed?): {missing}")
    coloured = set()
    for t in fig.findobj(Text):
        zone = ZONE_OF.get(t.get_text())
        if zone:
            t.set_color(ZONE_TEXT[zone])
            coloured.add(t.get_text())
    if coloured != set(ZONE_OF):
        raise SystemExit(f"zone labels not found: {set(ZONE_OF) - coloured}")
    style.save(fig, "zones_concept", OUT)
    print("saved", OUT / "zones_concept.pdf")


if __name__ == "__main__":
    main()
