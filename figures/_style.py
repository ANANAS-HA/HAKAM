"""Shared matplotlib style + palette for the GP2 report figures.

Import `apply_style()` at the top of every Fn_*.py script so all 8 figures
share the same fonts, spines, grid, and colors.
"""
from __future__ import annotations

import matplotlib as mpl


def apply_style() -> None:
    """Project-wide matplotlib defaults — call at the top of every plot script."""
    mpl.rcParams.update({
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.1,
        "font.family":        "DejaVu Sans",
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.labelsize":     10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "--",
        "legend.frameon":     False,
        "legend.fontsize":    9,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
    })


# Project palette — Okabe-Ito-derived, colorblind-safe.
COLORS = {
    "vanilla":   "#999999",  # gray — vanilla student baseline
    "epoch1":    "#56B4E9",  # sky blue — partial training
    "epoch3":    "#0072B2",  # deep blue — final student (highlight)
    "teacher":   "#E69F00",  # orange — teacher (target)
    "delta_pos": "#009E73",  # green — student wins
    "delta_neg": "#D55E00",  # red — student trails
    "neutral":   "#888888",
    "highlight": "#CC79A7",  # magenta — callouts
}


# Standard figure sizes (inches), tuned for 6.5" Word content width.
FIGSIZE_SINGLE     = (6.5, 3.5)
FIGSIZE_WIDE       = (6.5, 4.0)
FIGSIZE_DOUBLE     = (6.5, 5.0)
FIGSIZE_SIDEBYSIDE = (7.0, 3.0)
