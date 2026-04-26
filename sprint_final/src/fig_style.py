"""fig_style.py — IEEE two-column figure sizes and rcParams. Used by all experiment plot functions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLE = (3.5, 3.2)
DOUBLE = (7.0, 4.5)
TRIPLE = (7.0, 6.0)
QUAD   = (7.0, 7.5)

_RC = {
    "font.family": "serif", "font.size": 7,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.3, "lines.markersize": 4,
    "axes.linewidth": 0.7, "grid.linewidth": 0.4,
    "patch.linewidth": 0.5, "figure.dpi": 150,
}

def apply(style="seaborn-v0_8-whitegrid"):
    if style:
        plt.style.use(style)
    plt.rcParams.update(_RC)


apply_style = apply
