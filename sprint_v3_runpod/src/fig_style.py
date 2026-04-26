"""fig_style.py — Consistent matplotlib style for sprint_v3."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLE = (3.5, 2.6)
DOUBLE = (7.0, 2.6)
TRIPLE = (10.5, 2.6)


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.2,
    })
