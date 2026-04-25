"""
fig_style.py — Shared IEEE two-column figure dimensions and style constants.

All figures in this project must use one of three standard sizes so they fit
cleanly into the IEEE/ACM two-column format without rescaling:

    SINGLE  3.5 × 2.8 in  — one panel, sits inside a single column
    DOUBLE  7.0 × 2.8 in  — two panels side-by-side, spans both columns
    TRIPLE  7.0 × 3.0 in  — three panels side-by-side, spans both columns

Usage in any plot module:
    import fig_style as fs
    fs.apply()
    fig, ax  = plt.subplots(figsize=fs.SINGLE)
    fig, axs = plt.subplots(1, 2, figsize=fs.DOUBLE)
    fig, axs = plt.subplots(1, 3, figsize=fs.TRIPLE)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Standard figure sizes (width × height, inches) ──────────────────────────
SINGLE = (3.5, 3.2)   # fits inside one IEEE column
DOUBLE = (7.0, 4.5)   # spans both columns, two panels
TRIPLE = (7.0, 6.0)   # spans both columns, three panels

# ── rcParams for IEEE-quality figures ───────────────────────────────────────
_RC = {
    "font.family":       "serif",
    "font.size":         7,
    "axes.labelsize":    8,
    "axes.titlesize":    8,
    "legend.fontsize":   6.5,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "lines.linewidth":   1.3,
    "lines.markersize":  4,
    "axes.linewidth":    0.7,
    "grid.linewidth":    0.4,
    "patch.linewidth":   0.5,
    "figure.dpi":        150,   # screen; savefig always uses dpi=300
}

def apply(style="seaborn-v0_8-whitegrid"):
    """Apply IEEE style to the current matplotlib session."""
    if style:
        plt.style.use(style)
    plt.rcParams.update(_RC)
