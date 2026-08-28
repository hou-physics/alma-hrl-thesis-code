"""Shared plotting style for thesis figures.

Centralizes:
  - matplotlib rcParams for publication-quality output (serif math, proper
    tick direction, minor ticks, thin grid)
  - CMasher colormap picks for diverging (null vs signal), sequential
    (moment-0 maps), and qualitative (scan parameter categories)
  - helper functions for WCS axes (via astropy; APLpy wrapper optional)
  - consistent figure sizes matching ApJ two-column layout

All thesis plots should import this module FIRST and then apply
`setup_thesis_style()`.
"""
from __future__ import annotations

from typing import Optional

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt

# Figure sizes (inches) matching common journal layouts
FIG_SINGLE = (3.5, 2.6)   # ApJ single-column width
FIG_DOUBLE = (7.2, 4.0)   # ApJ double-column / full-page width
FIG_SQUARE = (4.5, 4.0)

# Curated palettes — perceptually uniform, print-safe, colorblind-safe
CMAP_SEQUENTIAL = "cividis"       # for CO tracer moment-0 maps (vmin≈0 strategy)
CMAP_MOMENT0 = "RdBu_r"            # HRL moment-0: diverging, used ZERO-CENTERED
                                   # (vmin=-vmax, vmax=+vmax) so the negative range
                                   # (interferometric bowls / over-subtraction) is
                                   # visible. Adopted 2026-06-28 (Bertoldi request).
CMAP_DIVERGING = cmr.fusion        # for residuals / null-vs-signal
CMAP_QUALITATIVE = cmr.take_cmap_colors(
    "cmr.rainforest", 8, cmap_range=(0.15, 0.85), return_fmt="hex"
)

# Accent colors used repeatedly (null, threshold, science, control)
COLOR_THRESHOLD = "#2b8cbe"   # cool teal-blue (threshold-group samples)
COLOR_CONTROL = "#d95f02"     # muted warm orange (control)
COLOR_P95 = "#c1272d"         # dark red (threshold line)
COLOR_NAIVE = "#808080"       # neutral gray (naïve 3σ reference)


def setup_thesis_style() -> None:
    """Apply publication-quality rcParams in-place.

    Call once at the top of any plotting script.
    """
    mpl.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "dejavuserif",

        # Axes & spines
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#202020",
        "axes.labelcolor": "#202020",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,

        # Ticks inward, with minor ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,

        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,

        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # editable text in PDFs
        "ps.fonttype": 42,
    })


def apply_clean_spines(ax: plt.Axes, keep: Optional[tuple] = None) -> None:
    """Remove top/right spines (keep='all' keeps every spine)."""
    if keep == "all":
        return
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
