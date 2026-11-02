"""Drawing primitives for conceptual diagrams in the house style.

Mirrors the TikZ zones diagram: organic smooth blobs, white
north-east hatching for scaffolded regions, bold sans titles over small sans
descriptions, curved pointers that end in a dot, and rounded filled boxes for
definitions and equations.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath
from scipy.interpolate import splev, splprep

from mathmage.analysis.style import P, tint


def blob_path(points, n: int = 480) -> MPath:
    """A smooth closed curve through control points (periodic cubic spline)."""
    pts = np.asarray(points, dtype=float)
    pts = np.vstack([pts, pts[:1]])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
    x, y = splev(np.linspace(0, 1, n), tck)
    verts = np.column_stack([x, y])
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 2) + [MPath.CLOSEPOLY]
    return MPath(verts, codes)


def blob(ax, points, color, *, alpha=1.0, hatch=None, hatch_color="white", edge=None, lw=0.0, z=1):
    path = blob_path(points)
    ax.add_patch(PathPatch(path, facecolor=color, edgecolor=edge or "none", lw=lw, alpha=alpha, zorder=z))
    if hatch:
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=hatch_color, lw=0, hatch=hatch, zorder=z + 0.01))
    return path


def scaled(points, center, factor):
    pts = np.asarray(points, dtype=float)
    c = np.asarray(center, dtype=float)
    return c + (pts - c) * factor


def pointer(ax, start, end, *, rad=0.25, color=None, lw=1.0, dot=0.075, z=6, head=True):
    """Curved pointer from a label to a target, ending in a dot (house style)."""
    color = color or P["ink"]
    style = "-|>,head_length=0.32,head_width=0.16" if head else "-"
    ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                                 mutation_scale=10, lw=lw, color=color, shrinkA=2, shrinkB=dot * 40 + 1,
                                 zorder=z))
    ax.add_patch(Circle(end, dot, color=color, zorder=z + 0.1))


def arrow(ax, start, end, *, rad=0.0, color=None, lw=1.1, z=4, style="-|>", scale=11, ls="-", shrink=2):
    color = color or P["purple"]
    ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                                 mutation_scale=scale, lw=lw, color=color, linestyle=ls,
                                 shrinkA=shrink, shrinkB=shrink, zorder=z))


def label(ax, x, y, title, body=None, *, ha="left", color=None, title_size=10.5, body_size=7.8,
          line=1.32, z=7):
    """Bold sans title with a smaller sans description underneath."""
    color = color or P["ink"]
    ax.text(x, y, title, ha=ha, va="top", family="sans-serif", weight="bold", fontsize=title_size, color=color,
            zorder=z)
    if body:
        ax.text(x, y - title_size * 0.052, body, ha=ha, va="top", family="sans-serif", fontsize=body_size,
                color=P["muted"], linespacing=line, zorder=z)


def rounded_box(ax, x, y, w, h, *, face, edge=None, lw=0.9, pad=0.08, rounding=0.18, alpha=1.0, z=2, ls="-"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={rounding}", facecolor=face,
                           edgecolor=edge or "none", lw=lw, alpha=alpha, zorder=z, linestyle=ls)
    ax.add_patch(patch)
    return patch


def equation_box(ax, x, y, w, h, text, *, face=None, color="white", size=10.5, z=3, rounding=0.3):
    rounded_box(ax, x, y, w, h, face=face or P["peri_inner"], rounding=rounding, z=z)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size, zorder=z + 1)


def card(ax, x, y, w, h, title, body, *, accent=None, face=None, title_size=9.5, body_size=7.6, hatch=None,
         z=3, body_line=1.35, title_color=None):
    """A rounded card with a colored title tab: the 'insight box' look."""
    accent = accent or P["insight_frame"]
    face = face if face is not None else tint(accent, 0.86)
    rounded_box(ax, x, y, w, h, face=face, edge=accent, lw=0.9, z=z)
    if hatch:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18", facecolor="none",
                                    edgecolor="white", lw=0, hatch=hatch, zorder=z + 0.05))
    ax.text(x + 0.12, y + h - 0.1, title, ha="left", va="top", family="sans-serif", weight="bold",
            fontsize=title_size, color=title_color or P["ink"], zorder=z + 1)
    if body:
        ax.text(x + 0.12, y + h - 0.1 - title_size * 0.05, body, ha="left", va="top", family="sans-serif",
                fontsize=body_size, color=P["muted"], linespacing=body_line, zorder=z + 1)


def chip(ax, x, y, text, *, face, color="white", size=7.4, pad=0.06, z=6, weight="bold", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", family="sans-serif", fontsize=size, color=color, weight=weight,
            zorder=z, bbox=dict(boxstyle=f"round,pad={pad * 4}", facecolor=face, edgecolor="none"))
