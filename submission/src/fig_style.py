"""
fig_style.py — Shared IEEE figure style constants.

Each figure uses its own natural dimensions rather than a single standard size.
This avoids squishing artefacts from forcing all figures to the same width.

Usage in any plot module:
    import matplotlib
    matplotlib.use("Agg")
    import fig_style as fs
    fs.apply()
    fig, ax  = plt.subplots(figsize=(10, 4.5))
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── rcParams for IEEE-quality figures ───────────────────────────────────────
_RC = {
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "lines.linewidth":   1.8,
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
