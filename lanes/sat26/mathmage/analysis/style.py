"""House style for every MathMage figure.

Typography: Latin Modern (the default LaTeX look) as TrueType, typeset by LuaLaTeX
through Matplotlib's PGF backend, so figure text gets real ligatures, GPOS kerning,
optical sizes (7, 8, 9, 10 pt designs), and Unicode math in Latin Modern Math. PDFs
embed CID TrueType fonts and stay sharp at any zoom. Build the fonts once with
``python scripts/build_fonts.py``.

Palette: the cyan, pink, lavender, and purple family used throughout the research figures.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from mathmage import paths

FONT_DIR = paths.FONTS / "latinmodern-ttf"

# ----------------------------------------------------------------------------- palette
PALETTE = {
    "cyan": "#74D2FA",        # baby cyan
    "cyan_deep": "#31A9CD",   # deep cyan for lines and text
    "pink": "#EC8DFF",        # baby pink / magenta
    "pink_deep": "#C54DF5",   # deep pink for lines and text
    "lav1": "#A7A8F0",        # lavender ramp, light to dark
    "lav2": "#9194E4",
    "lav3": "#8180DE",
    "lav4": "#6E65CE",
    "purple": "#5445B5",
    "violet": "#7A4DFF",
    "indigo": "#211CBF",
    "peri_inner": "#6B66FF",  # periwinkle zone ramp: inner, middle, outer
    "peri_mid": "#A4A6FF",
    "peri_outer": "#D1D3FF",
    "ink": "#111827",         # body text
    "muted": "#5B6474",
    "faint": "#9AA1AE",
    "grid": "#E4E5F4",
    "paper": "#FFFFFF",
    "insight_bg": "#EFEDF7",
    "insight_title": "#CCC4F2",
    "insight_frame": "#B1ABE4",
    "blush": "#F6EEF7",
    "cite": "#25C0EE",
}
P = PALETTE

# Zones: the mastered core is the darkest periwinkle (as D_solo in the zones
# diagram); Latent is cyan (sampling/selection); Proximal is the pink accent (the ZPD
# proper, scaffolding); Unreachable is the pale outer periwinkle.
ZONE_FILL = {"A": P["peri_inner"], "L": P["cyan"], "P": P["pink"], "U": P["peri_outer"]}
ZONE_LINE = {"A": P["purple"], "L": P["cyan_deep"], "P": P["pink_deep"], "U": P["lav2"]}
ZONE_NAME = {"A": "Autonomous", "L": "Latent", "P": "Proximal", "U": "Unreachable"}

METHOD = {
    "mathmage": P["purple"],
    "mathmage_alt": P["violet"],
    "mathmage_light": P["lav3"],
    "sat": P["pink_deep"],
    "sat_light": P["pink"],
    "self_consistency": P["cyan_deep"],
    "self_consistency_light": P["cyan"],
    "member_vote": P["lav2"],
    "best_member": P["lav1"],
    "oracle": P["ink"],
}

CMAP_PERI = LinearSegmentedColormap.from_list(
    "mathmage_peri", ["#FFFFFF", P["peri_outer"], P["peri_mid"], P["peri_inner"], P["purple"], P["indigo"]])
CMAP_CYAN_PINK = LinearSegmentedColormap.from_list(
    "mathmage_cyan_pink", [P["cyan_deep"], P["cyan"], "#F7F7FF", P["pink"], P["pink_deep"]])
CMAP_ZPD = LinearSegmentedColormap.from_list(
    "mathmage_zpd", [P["peri_outer"], P["cyan"], P["lav1"], P["pink"], P["pink_deep"]])
for _cmap in (CMAP_PERI, CMAP_CYAN_PINK, CMAP_ZPD):
    try:
        mpl.colormaps.register(_cmap)
    except ValueError:
        pass


def tint(color: str, amount: float) -> tuple:
    """Mix a color with white (amount 0 = color, 1 = white)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# ----------------------------------------------------------------------------- typography
def _sizes(prefix: str, shape: str, sizes=(7, 8, 9, 10)) -> str:
    lo = {7: None, 8: 7.6, 9: 8.6, 10: 9.6}
    hi = {7: 7.6, 8: 8.6, 9: 9.6, 10: None}
    parts = []
    for s in sizes:
        rng = f"{lo[s] if lo[s] else ''}-{hi[s] if hi[s] else ''}"
        parts.append(f"{{Size={rng},Font={prefix}{s}-{shape}.ttf}}")
    return "SizeFeatures={" + ",".join(parts) + "}"


def preamble() -> str:
    d = FONT_DIR.as_posix() + "/"
    roman = (r"\setmainfont{LMRomanTT10-Regular.ttf}[Path={" + d + r"},Renderer=HarfBuzz,Ligatures=TeX,Kerning=On,"
             + _sizes("LMRomanTT", "Regular") + ","
             + r"BoldFont=LMRomanTT10-Bold.ttf,BoldFeatures={" + _sizes("LMRomanTT", "Bold") + "},"
             + r"ItalicFont=LMRomanTT10-Italic.ttf,ItalicFeatures={" + _sizes("LMRomanTT", "Italic") + "},"
             + r"BoldItalicFont=LMRomanTT10-BoldItalic.ttf]")
    sans = (r"\setsansfont{LMSansTT10-Regular.ttf}[Path={" + d + r"},Renderer=HarfBuzz,Ligatures=TeX,Kerning=On,"
            r"SizeFeatures={{Size=-8.6,Font=LMSansTT8-Regular.ttf},{Size=8.6-9.6,Font=LMSansTT9-Regular.ttf},"
            r"{Size=9.6-,Font=LMSansTT10-Regular.ttf}},"
            r"BoldFont=LMSansTT10-Bold.ttf,ItalicFont=LMSansTT10-Oblique.ttf]")
    math = r"\setmathfont{LMMathTT-Regular.ttf}[Path={" + d + "}]"
    return "\n".join([r"\usepackage{fontspec}", r"\usepackage{unicode-math}", roman, sans, math,
                      r"\usepackage{xcolor}"])


def use(publication: bool = True, base_size: float = 9.0) -> None:
    """Activate the house style. ``publication`` renders through LuaLaTeX (PGF)."""
    if publication:
        if shutil.which("lualatex") is None:
            raise RuntimeError("figure export requires LuaLaTeX (MiKTeX or TeX Live)")
        missing = [f for f in ("LMRomanTT10-Regular.ttf", "LMSansTT10-Regular.ttf", "LMMathTT-Regular.ttf")
                   if not (FONT_DIR / f).exists()]
        if missing:
            raise FileNotFoundError(f"missing {missing}; run: python scripts/build_fonts.py")
        mpl.use("pgf")
        mpl.rcParams.update({"pgf.texsystem": "lualatex", "pgf.rcfonts": False,
                             "pgf.preamble": preamble(), "text.usetex": True})
    else:
        mpl.use("Agg")
        from matplotlib import font_manager
        for font in FONT_DIR.glob("*.ttf"):
            font_manager.fontManager.addfont(str(font))
        mpl.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})
    ink, muted = P["ink"], P["muted"]
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["LM Roman TT 10"],
        "font.sans-serif": ["LM Sans TT 10"],
        "font.size": base_size, "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size, "xtick.labelsize": base_size - 1, "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "axes.edgecolor": P["faint"], "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": muted, "ytick.color": muted, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.grid": False, "grid.color": P["grid"], "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=[P["purple"], P["cyan_deep"], P["pink_deep"], P["lav2"],
                                              P["violet"], P["cyan"], P["pink"], P["lav1"]]),
        "lines.linewidth": 1.5, "patch.linewidth": 0.6,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "savefig.dpi": 300,
        "figure.dpi": 150, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, stem: str | Path, directory: Path | None = None, png: bool = True) -> list[Path]:
    """Save ``stem``.pdf (vector, TrueType) and a 300-dpi PNG preview."""
    directory = Path(directory) if directory else paths.FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    out = [directory / f"{Path(stem).name}.pdf"]
    fig.savefig(out[0])
    if png:
        out.append(directory / f"{Path(stem).name}.png")
        fig.savefig(out[1], dpi=300)
    return out
