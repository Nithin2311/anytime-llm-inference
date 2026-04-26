"""
plots.py — Publication-quality single-column plots for sprint_v2.

All figures: 3.5 in wide (IEEE single column), one axes per PNG, no tables.
Colorblind-safe Wong 2011 palette. No seaborn. No fig_style.py.

Usage (from plot_runner.py or experiments):
    from plots import plot_wcet_heatmap, save_fig
    fig = plot_wcet_heatmap(results_dict)
    save_fig(fig, "results/wcet_heatmap.png")
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style constants ──────────────────────────────────────────────────────────
W   = 3.5    # single-column width (inches)
H   = 2.6    # default height
H_T = 3.0    # taller plots
H_S = 2.2    # shorter plots

# Wong 2011 colorblind-safe palette
BLUE    = "#0072B2"
ORANGE  = "#E69F00"
GREEN   = "#009E73"
PINK    = "#CC79A7"
SKY     = "#56B4E9"
VERMIL  = "#D55E00"
GOLD    = "#F0E442"
BLACK   = "#000000"

PALETTE = [BLUE, ORANGE, GREEN, PINK, SKY, VERMIL]

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "legend.fontsize":   7.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "lines.linewidth":   1.4,
    "lines.markersize":  4.5,
    "axes.linewidth":    0.7,
    "grid.linewidth":    0.4,
    "grid.color":        "#cccccc",
    "figure.dpi":        200,
    "savefig.dpi":       200,
})


# ── Base factory ─────────────────────────────────────────────────────────────
def _ax(h=H):
    fig, ax = plt.subplots(figsize=(W, h))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, length=3, width=0.7)
    return fig, ax


def _ax_both(h=H):
    """Axes with both-axis grid (for scatter/time-series)."""
    fig, ax = _ax(h)
    ax.grid(axis="both", linewidth=0.4, color="#cccccc", zorder=0)
    return fig, ax


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# E00  WCET Reprofiling
# ══════════════════════════════════════════════════════════════════════════════

def plot_wcet_heatmap(results: dict) -> plt.Figure:
    """
    Heatmap: rows = exit layer, cols = seq length, cells = mean latency (ms).
    results keyed as results[str(seq)][str(layer)]["mean_ms"].
    """
    seq_lengths  = sorted(int(k) for k in results)
    layer_labels = ["L5", "L11", "L16", "Full"]
    layer_keys   = ["5", "11", "16", "None"]

    matrix = np.zeros((len(layer_labels), len(seq_lengths)))
    for ci, seq in enumerate(seq_lengths):
        for ri, lk in enumerate(layer_keys):
            cell = results[str(seq)].get(lk, {})
            matrix[ri, ci] = cell.get("mean_ms", 0)

    fig, ax = plt.subplots(figsize=(W, H))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="upper")
    ax.set_xticks(range(len(seq_lengths)))
    ax.set_xticklabels([str(s) for s in seq_lengths], fontsize=8)
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels, fontsize=8)
    ax.set_xlabel("Input sequence length (tokens)")
    ax.set_ylabel("Exit point")
    ax.set_title("Mean per-token latency (ms)")

    for ri in range(len(layer_labels)):
        for ci in range(len(seq_lengths)):
            v = matrix[ri, ci]
            color = "white" if v > matrix.max() * 0.6 else "black"
            ax.text(ci, ri, f"{v:.1f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ms", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_wcet_cdf(results: dict) -> plt.Figure:
    """
    ECDF of full-pass latency for multiple sequence lengths.
    results[str(seq)]["None"]["samples_ms"] (list of floats) — or use percentile summary.
    Falls back to drawing from (mean, std) Gumbel if raw samples absent.
    """
    seq_lengths = sorted(int(k) for k in results)
    fig, ax = _ax_both(H)

    for i, seq in enumerate(seq_lengths):
        cell = results[str(seq)].get("None", {})
        samples = cell.get("samples_ms", None)
        if samples is not None and len(samples) > 0:
            s = np.sort(samples)
            ecdf = np.arange(1, len(s) + 1) / len(s)
        else:
            loc   = cell.get("gumbel_loc",   cell.get("mean_ms",  10))
            scale = cell.get("gumbel_scale",  cell.get("std_ms",   0.5))
            s     = np.linspace(loc - 2*scale, loc + 8*scale, 400)
            from scipy.stats import gumbel_r
            ecdf  = gumbel_r.cdf(s, loc=loc, scale=scale)

        ax.plot(s, ecdf * 100, color=PALETTE[i % len(PALETTE)],
                label=f"seq={seq}", lw=1.4)

    ax.set_xlabel("Per-token latency (ms)")
    ax.set_ylabel("Cumulative probability (%)")
    ax.set_title("Full-pass latency CDF by sequence length")
    ax.set_ylim(0, 101)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E01  EVT GEV + AD
# ══════════════════════════════════════════════════════════════════════════════

def plot_evt_gev_xi(results: dict) -> plt.Figure:
    """
    Grouped bar chart of GEV shape parameter ξ per exit layer, one group per seq.
    |ξ| < 0.15 band shaded.
    results[str(seq)][str(layer)]["gev"]["xi"]
    """
    seq_lengths = sorted(int(k) for k in results)
    layer_keys  = ["5", "11", "16", "None"]
    layer_labels = ["L5", "L11", "L16", "Full"]
    n_layers = len(layer_keys)
    n_seq    = len(seq_lengths)

    fig, ax = _ax(H)
    bar_w  = 0.18
    x_base = np.arange(n_layers)

    for si, seq in enumerate(seq_lengths):
        xis = []
        for lk in layer_keys:
            cell = results[str(seq)].get(lk, {})
            xi   = cell.get("gev", {}).get("xi", 0)
            xis.append(xi)
        offset = (si - (n_seq - 1) / 2.0) * bar_w
        ax.bar(x_base + offset, xis, bar_w,
               color=PALETTE[si % len(PALETTE)],
               label=f"seq={seq}", alpha=0.85, edgecolor="none")

    ax.axhspan(-0.15, 0.15, color=GREEN, alpha=0.12, zorder=0)
    ax.axhline(0,    color="black", lw=0.7, ls="--")
    ax.axhline( 0.15, color=GREEN, lw=0.8, ls="--", alpha=0.6)
    ax.axhline(-0.15, color=GREEN, lw=0.8, ls="--", alpha=0.6,
               label="Gumbel valid (|ξ|<0.15)")

    ax.set_xticks(x_base)
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Exit point")
    ax.set_ylabel("GEV shape ξ")
    ax.set_title("GEV tail index — Gumbel validity check")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


def plot_evt_pwcet_comparison(results: dict) -> plt.Figure:
    """
    Horizontal grouped bars: Gumbel vs GEV pWCET(1e-6) for each (seq, layer).
    Shows where GEV diverges from Gumbel.
    Uses seq=128 as representative.
    """
    seq = 128
    layer_keys   = ["5", "11", "16", "None"]
    layer_labels = ["L5", "L11", "L16", "Full"]

    gumbel_vals, gev_vals = [], []
    for lk in layer_keys:
        cell = results.get(str(seq), {}).get(lk, {})
        gumbel_vals.append(cell.get("gumbel", {}).get("wcet_1e6",  0))
        gev_vals.append(   cell.get("gev",    {}).get("wcet_1e6",  0))

    y = np.arange(len(layer_labels))
    fig, ax = _ax(H)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    ax.barh(y - 0.18, gumbel_vals, 0.32, color=BLUE,   label="Gumbel", alpha=0.9)
    ax.barh(y + 0.18, gev_vals,    0.32, color=ORANGE, label="GEV",    alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(layer_labels)
    ax.set_xlabel("pWCET(10⁻⁶) [ms]")
    ax.set_title(f"Gumbel vs GEV pWCET — seq={seq}")
    ax.legend(frameon=False)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E02  Threshold Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════

def plot_threshold_cv(rows: list) -> plt.Figure:
    """
    Calibration accuracy and held-out accuracy with CI band vs τ.
    rows: list of dicts with keys tau, cal_acc, holdout_acc, ci_lower, ci_upper, exit_rate_pct.
    """
    taus     = [r["tau"]           for r in rows]
    cal_acc  = [r.get("cal_acc",   r.get("calibration_accuracy", 0))  for r in rows]
    ho_acc   = [r.get("holdout_acc", r.get("held_out_accuracy", 0))   for r in rows]
    ci_lo    = [r.get("ci_lower",  0) for r in rows]
    ci_hi    = [r.get("ci_upper",  0) for r in rows]
    best_tau = rows[np.argmax(cal_acc)]["tau"] if rows else None

    fig, ax = _ax(H)
    ax.fill_between(taus, ci_lo, ci_hi, alpha=0.2, color=ORANGE,
                    label="Held-out 95% CI")
    ax.plot(taus, ho_acc,  "-o",  color=ORANGE, lw=1.4, label="Held-out acc")
    ax.plot(taus, cal_acc, "--s", color=BLUE,   lw=1.2, label="Calibration acc")

    if best_tau is not None:
        ax.axvline(best_tau, color=GREEN, lw=1.0, ls=":",
                   label=f"Best τ = {best_tau:.2f}")

    ax.set_xlabel("Confidence threshold τ")
    ax.set_ylabel("PubMedQA accuracy (%)")
    ax.set_title("Cross-validated threshold selection")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def plot_threshold_exit_rate(rows: list) -> plt.Figure:
    """Exit rate vs τ."""
    taus  = [r["tau"]              for r in rows]
    exits = [r.get("exit_rate_pct", r.get("exit_rate", 0)) for r in rows]
    miss  = [r.get("miss_rate_pct", r.get("miss_rate", 0)) for r in rows]

    fig, ax = _ax(H_S)
    ax.plot(taus, exits, "-o", color=GREEN,  label="Exit rate")
    ax.plot(taus, miss,  "-^", color=VERMIL, label="Miss rate")
    ax.axhline(1.0, color=BLACK, lw=0.8, ls="--", alpha=0.5,
               label="1% miss threshold")
    ax.set_xlabel("Confidence threshold τ")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Exit and miss rate vs. threshold")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E03  Forced-Exit Extended
# ══════════════════════════════════════════════════════════════════════════════

def plot_forced_exit_accuracy(rows: list) -> plt.Figure:
    """Accuracy (parseable only) vs deadline."""
    deadlines = [r["deadline_ms"]           for r in rows]
    acc_stat  = [r.get("stateless_accuracy", r.get("accuracy_stateless", 0)) for r in rows]
    acc_kv    = [r.get("kv_accuracy",        r.get("accuracy_kv",        0)) for r in rows]

    fig, ax = _ax(H)
    if any(v > 0 for v in acc_stat):
        ax.plot(deadlines, acc_stat, "-o",  color=BLUE,   label="Stateless",  lw=1.4)
    if any(v > 0 for v in acc_kv):
        ax.plot(deadlines, acc_kv,   "-s",  color=ORANGE, label="KV-cached",  lw=1.4)
    ax.set_xlabel("Deadline D (ms)")
    ax.set_ylabel("Accuracy on parseable tokens (%)")
    ax.set_title("Forced-exit accuracy vs. deadline")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_forced_exit_miss(rows: list) -> plt.Figure:
    """Deadline miss rate vs deadline."""
    deadlines = [r["deadline_ms"]                   for r in rows]
    miss_stat = [r.get("stateless_miss_rate_pct",
                       r.get("miss_rate_stateless", 0)) for r in rows]
    miss_kv   = [r.get("kv_miss_rate_pct",
                       r.get("miss_rate_kv",        0)) for r in rows]

    fig, ax = _ax(H_S)
    if any(v > 0 for v in miss_stat):
        ax.plot(deadlines, miss_stat, "-o",  color=BLUE,   label="Stateless", lw=1.4)
    if any(v > 0 for v in miss_kv):
        ax.plot(deadlines, miss_kv,   "-s",  color=ORANGE, label="KV-cached", lw=1.4)
    ax.axhline(1.0, color=BLACK, lw=0.8, ls="--", alpha=0.5,
               label="1% threshold")
    ax.set_xlabel("Deadline D (ms)")
    ax.set_ylabel("Miss rate (%)")
    ax.set_title("Deadline miss rate vs. deadline")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E04  POT Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def plot_pot_sensitivity(results: dict) -> plt.Figure:
    """
    pWCET(1e-6) vs POT fraction for each seq length (full pass only).
    results[str(seq)]["fractions"][fraction_str]["wcet_1e6"]
    """
    seq_lengths = sorted(int(k) for k in results if k != "fractions")

    fig, ax = _ax(H)
    for i, seq in enumerate(seq_lengths):
        seq_data = results[str(seq)]
        fracs = sorted(float(f) for f in seq_data if f != "metadata")
        wcets = [seq_data[str(f)].get("wcet_1e6", seq_data[str(f)].get("gumbel", {}).get("wcet_1e6", 0))
                 for f in fracs]
        ax.plot([f * 100 for f in fracs], wcets, "-o",
                color=PALETTE[i % len(PALETTE)], label=f"seq={seq}", lw=1.4)

    ax.set_xlabel("POT tail fraction (%)")
    ax.set_ylabel("pWCET(10⁻⁶) [ms]")
    ax.set_title("pWCET sensitivity to POT threshold — full pass")
    ax.axvline(20, color=BLACK, lw=0.8, ls="--", alpha=0.5, label="Default 20%")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E05  Deadline Sweep Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_deadline_sweep(rows: list) -> plt.Figure:
    """
    Miss rate vs deadline: stateless vs KV-cached.
    rows: list of dicts with deadline_ms, stateless_miss_pct, kv_miss_pct.
    """
    Ds        = [r["deadline_ms"]            for r in rows]
    stat_miss = [r.get("stateless_miss_pct",
                        r.get("miss_rate_stateless", 0)) for r in rows]
    kv_miss   = [r.get("kv_miss_pct",
                        r.get("miss_rate_kv",        0)) for r in rows]

    fig, ax = _ax(H)
    ax.semilogy()
    ax.plot(Ds, [max(v, 0.05) for v in stat_miss], "-o",
            color=BLUE,   label="Stateless",  lw=1.4)
    ax.plot(Ds, [max(v, 0.05) for v in kv_miss],   "-s",
            color=ORANGE, label="KV-cached",  lw=1.4)
    ax.axhline(1.0, color=BLACK, lw=0.8, ls="--", alpha=0.5,
               label="1% miss threshold")
    ax.set_xlabel("Deadline D (ms)")
    ax.set_ylabel("Miss rate (%) — log scale")
    ax.set_title("Deadline sweep: router comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E06  Accuracy Large-Scale
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_tau(sweep_rows: list, tau_default: float = 0.55) -> plt.Figure:
    """Accuracy + 95% CI band vs τ (500-query)."""
    taus  = [r["tau"]      for r in sweep_rows]
    accs  = [r.get("accuracy", r.get("accuracy_pct", 0)) for r in sweep_rows]
    ci_lo = [r.get("ci_lower", 0)  for r in sweep_rows]
    ci_hi = [r.get("ci_upper", 0)  for r in sweep_rows]

    fig, ax = _ax(H)
    ax.fill_between(taus, ci_lo, ci_hi, alpha=0.20, color=BLUE, label="95% CI")
    ax.plot(taus, accs, "-o", color=BLUE, lw=1.5, label="Accuracy")
    ax.axvline(tau_default, color=ORANGE, lw=1.0, ls="--",
               label=f"Default τ = {tau_default}")
    ax.set_xlabel("Confidence threshold τ")
    ax.set_ylabel("PubMedQA accuracy (%)")
    ax.set_title("Accuracy vs. threshold — 500 queries")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_exit_rate_vs_tau(sweep_rows: list, tau_default: float = 0.55) -> plt.Figure:
    """Early-exit rate vs τ."""
    taus  = [r["tau"]                for r in sweep_rows]
    exits = [r.get("exit_rate_pct",
                    r.get("exit_rate", 0)) for r in sweep_rows]
    miss  = [r.get("miss_rate_pct",
                    r.get("miss_rate", 0)) for r in sweep_rows]

    fig, ax = _ax(H_S)
    ax.plot(taus, exits, "-o", color=GREEN,  label="Exit rate")
    ax.plot(taus, miss,  "-^", color=VERMIL, label="Miss rate")
    ax.axvline(tau_default, color=ORANGE, lw=1.0, ls="--",
               label=f"Default τ = {tau_default}")
    ax.axhline(1.0, color=BLACK, lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Confidence threshold τ")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Exit and miss rate — 500 queries")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E07  WCET CI GEV
# ══════════════════════════════════════════════════════════════════════════════

def plot_wcet_ci_gev(results: dict, seq: int = 128) -> plt.Figure:
    """
    Grouped bars: Gumbel vs GEV pWCET(1e-6) per exit point (at given seq).
    Error bars = bootstrap 95% CI on Gumbel.
    """
    layer_keys   = ["5", "11", "16", "None"]
    layer_labels = ["L5", "L11", "L16", "Full"]
    seq_data     = results.get(str(seq), {})

    gumbel_v, gev_v, ci_lo, ci_hi = [], [], [], []
    for lk in layer_keys:
        cell = seq_data.get(lk, {})
        gumbel_v.append(cell.get("gumbel", {}).get("wcet_1e6", 0))
        gev_v.append(   cell.get("gev",    {}).get("wcet_1e6", 0))
        bci = cell.get("bootstrap_ci", {})
        ci_lo.append(bci.get("ci_lower", gumbel_v[-1]))
        ci_hi.append(bci.get("ci_upper", gumbel_v[-1]))

    x   = np.arange(len(layer_labels))
    fig, ax = _ax(H)

    bars_g = ax.bar(x - 0.18, gumbel_v, 0.32, color=BLUE,   label="Gumbel",
                    alpha=0.88, edgecolor="none")
    bars_e = ax.bar(x + 0.18, gev_v,    0.32, color=ORANGE, label="GEV",
                    alpha=0.88, edgecolor="none")

    err_lo = np.array(gumbel_v) - np.array(ci_lo)
    err_hi = np.array(ci_hi)    - np.array(gumbel_v)
    ax.errorbar(x - 0.18, gumbel_v,
                yerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=3, lw=1.0, capthick=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Exit point")
    ax.set_ylabel("pWCET(10⁻⁶) [ms]")
    ax.set_title(f"pWCET with bootstrap CI — seq={seq}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E08  Sample Independence
# ══════════════════════════════════════════════════════════════════════════════

def plot_acf(lags: list, acf_vals: list, conf_band: float,
             title: str = "ACF") -> plt.Figure:
    """ACF bar chart with ±95% confidence band."""
    fig, ax = _ax(H)
    ax.bar(lags, acf_vals, color=BLUE, alpha=0.75, width=0.6)
    ax.axhline( conf_band, color=VERMIL, lw=1.0, ls="--", label="±95% band")
    ax.axhline(-conf_band, color=VERMIL, lw=1.0, ls="--")
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_rolling_mean(indices, means, stds, title: str = "Rolling mean") -> plt.Figure:
    """Rolling latency mean ± std."""
    fig, ax = _ax_both(H)
    ax.plot(indices, means, color=BLUE, lw=1.4, label="Rolling mean")
    ax.fill_between(indices,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color=BLUE, label="±1 SD")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E09  Capacity Empirical
# ══════════════════════════════════════════════════════════════════════════════

def plot_capacity_miss(results: dict) -> plt.Figure:
    """Bar chart: miss rate vs N_concurrent. Green = schedulable, red = not."""
    Ns      = sorted(int(k) for k in results)
    misses  = [results[str(n)]["avg_miss_rate_pct"] for n in Ns]
    sched   = [results[str(n)]["schedulable"]       for n in Ns]
    colors  = [GREEN if s else VERMIL for s in sched]

    fig, ax = _ax(H_S)
    ax.bar(Ns, misses, color=colors, edgecolor="none", alpha=0.88, width=0.5)
    ax.axhline(1.0, color=BLACK, lw=0.9, ls="--", alpha=0.6,
               label="1% miss threshold")
    green_p = mpatches.Patch(color=GREEN,  label="Schedulable")
    red_p   = mpatches.Patch(color=VERMIL, label="Not schedulable")
    ax.legend(handles=[green_p, red_p, ax.lines[-1]],
              frameon=False, fontsize=7)
    ax.set_xlabel("Number of concurrent requests N")
    ax.set_ylabel("Average miss rate (%)")
    ax.set_title("Round-robin capacity — miss rate")
    ax.set_xticks(Ns)
    fig.tight_layout()
    return fig


def plot_capacity_throughput(results: dict) -> plt.Figure:
    """Line: throughput vs N_concurrent with N_max marker."""
    Ns      = sorted(int(k) for k in results)
    tputs   = [results[str(n)]["avg_throughput_tps"] for n in Ns]
    n_max   = max((n for n in Ns if results[str(n)]["schedulable"]), default=0)

    fig, ax = _ax(H_S)
    ax.plot(Ns, tputs, "-o", color=BLUE, lw=1.5)
    if n_max > 0:
        ax.axvline(n_max, color=GREEN, lw=1.0, ls="--",
                   label=f"N_max = {n_max}")
    ax.set_xlabel("Number of concurrent requests N")
    ax.set_ylabel("Throughput (tokens / s)")
    ax.set_title("Round-robin throughput scaling")
    ax.set_xticks(Ns)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E10  Tight Deadline
# ══════════════════════════════════════════════════════════════════════════════

def plot_tight_accuracy(rows: list) -> plt.Figure:
    """Accuracy + 95% CI band vs deadline in tight (14–30ms) regime."""
    Ds    = [r["deadline_ms"]                          for r in rows]
    accs  = [r.get("accuracy_pct", r.get("accuracy", 0)) for r in rows]
    ci_lo = [r.get("ci_lower", 0)                      for r in rows]
    ci_hi = [r.get("ci_upper", 0)                      for r in rows]

    fig, ax = _ax(H)
    ax.fill_between(Ds, ci_lo, ci_hi, alpha=0.2, color=BLUE, label="95% CI")
    ax.plot(Ds, accs, "-o", color=BLUE, lw=1.5, label="Accuracy")
    ax.axvline(20, color=ORANGE, lw=0.9, ls="--", label="D = 20 ms")
    ax.set_xlabel("Deadline D (ms)")
    ax.set_ylabel("PubMedQA accuracy (%)")
    ax.set_title("Accuracy in tight-deadline regime")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def plot_tight_miss(rows: list) -> plt.Figure:
    """Miss rate vs deadline with 1% threshold line."""
    Ds   = [r["deadline_ms"]                                   for r in rows]
    miss = [r.get("miss_rate_pct", r.get("miss_rate", 0))      for r in rows]

    fig, ax = _ax(H_S)
    ax.plot(Ds, miss, "-o", color=VERMIL, lw=1.5, label="Miss rate")
    ax.axhline(1.0, color=BLACK, lw=0.8, ls="--", alpha=0.6,
               label="1% threshold")
    ax.fill_between(Ds, 0, miss, alpha=0.12, color=VERMIL)
    ax.set_xlabel("Deadline D (ms)")
    ax.set_ylabel("Miss rate (%)")
    ax.set_title("Deadline miss rate — tight regime")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E11  Thermal Stability
# ══════════════════════════════════════════════════════════════════════════════

def plot_thermal_latency(latencies: list, rolling_p99: list) -> plt.Figure:
    """Latency time series (scatter) + rolling P99 overlay."""
    lats = np.array(latencies)
    rp_x = np.array([r["idx"]    for r in rolling_p99])
    rp_v = np.array([r["p99_ms"] for r in rolling_p99])

    fig, ax = _ax_both(H_T)
    ax.scatter(np.arange(len(lats)), lats,
               s=1.5, color=SKY, alpha=0.35, linewidths=0, label="Per-token latency")
    ax.plot(rp_x, rp_v, color=VERMIL, lw=1.5, label="Rolling P99 (w=100)")

    p99_f = float(np.percentile(lats[:200],  99))
    p99_l = float(np.percentile(lats[-200:], 99))
    drift  = (p99_l - p99_f) / p99_f * 100
    ax.text(0.97, 0.97, f"Drift: {drift:+.1f}%",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=7, color=VERMIL)

    ax.set_xlabel("Token index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("A100 latency over 1000 consecutive tokens")
    ax.legend(frameon=False, fontsize=7, markerscale=3)
    fig.tight_layout()
    return fig


def plot_thermal_temp(temp_log: list) -> plt.Figure:
    """GPU temperature over the sustained run."""
    if not temp_log or all(t.get("temp_C") is None for t in temp_log):
        fig, ax = _ax(H_S)
        ax.text(0.5, 0.5, "Temperature data unavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("GPU temperature during sustained run")
        fig.tight_layout()
        return fig

    xs = [t["token_idx"] for t in temp_log]
    ys = [t["temp_C"] or 0 for t in temp_log]

    fig, ax = _ax(H_S)
    ax.plot(xs, ys, "-o", color=ORANGE, lw=1.5, ms=5)
    ax.set_xlabel("Token index")
    ax.set_ylabel("GPU temperature (°C)")
    ax.set_title("GPU temperature during sustained run")
    ax.set_ylim(bottom=max(0, min(ys) - 5))
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E12  Exit-Head Training
# ══════════════════════════════════════════════════════════════════════════════

def plot_exit_head_accuracy(train_log: list) -> plt.Figure:
    """Train and validation accuracy curves."""
    epochs    = [e["epoch"]     for e in train_log]
    train_acc = [e["train_acc"] for e in train_log]
    val_acc   = [e["val_acc"]   for e in train_log]

    fig, ax = _ax(H)
    ax.plot(epochs, train_acc, "-",  color=BLUE,   lw=1.5, label="Train")
    ax.plot(epochs, val_acc,   "--", color=ORANGE, lw=1.5, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("ExitHead MLP — accuracy curves")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_exit_head_loss(train_log: list) -> plt.Figure:
    """Training cross-entropy loss."""
    epochs = [e["epoch"] for e in train_log]
    losses = [e["loss"]  for e in train_log]

    fig, ax = _ax(H_S)
    ax.plot(epochs, losses, "-", color=VERMIL, lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("ExitHead MLP — training loss")
    ax.grid(axis="both", linewidth=0.4, color="#cccccc")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E13  Dense Layer Ablation
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation_latency(rows: list) -> plt.Figure:
    """P99 latency vs number of transformer layers (connected dots)."""
    n_layers = [r.get("n_layers", r.get("exit_layer") or 22) for r in rows]
    p99s     = [r["p99_ms"]   for r in rows]
    deadline = rows[0].get("deadline_ms", 45.0) if rows else 45.0

    fig, ax = _ax(H)
    ax.plot(n_layers, p99s, "-o", color=BLUE, lw=1.5, ms=5)
    ax.axhline(deadline, color=VERMIL, lw=0.9, ls="--",
               label=f"D = {deadline} ms")
    ax.fill_between(n_layers, 0, p99s, alpha=0.08, color=BLUE)
    ax.set_xlabel("Number of transformer layers used")
    ax.set_ylabel("P99 latency (ms)")
    ax.set_title("Latency vs. exit depth (seq = 128)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_ablation_accuracy(rows: list) -> plt.Figure:
    """Accuracy + CI vs exit depth."""
    n_layers = [r.get("n_layers", r.get("exit_layer") or 22) for r in rows]
    accs     = [r.get("accuracy_pct", r.get("accuracy", 0)) for r in rows]
    ci_lo    = [r.get("ci_lower", 0) for r in rows]
    ci_hi    = [r.get("ci_upper", 0) for r in rows]

    fig, ax = _ax(H)
    ax.fill_between(n_layers, ci_lo, ci_hi, alpha=0.2, color=ORANGE)
    ax.plot(n_layers, accs, "-o", color=ORANGE, lw=1.5, ms=5)
    ax.set_xlabel("Number of transformer layers used")
    ax.set_ylabel("PubMedQA accuracy (%)")
    ax.set_title("Accuracy vs. exit depth (seq = 128)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


def plot_ablation_pareto(rows: list) -> plt.Figure:
    """Accuracy vs P99 latency scatter — Pareto frontier highlighted."""
    p99s  = [r["p99_ms"]                                     for r in rows]
    accs  = [r.get("accuracy_pct", r.get("accuracy", 0))     for r in rows]
    lbls  = [r.get("label", f"L{r.get('n_layers','')}") for r in rows]

    # Compute Pareto frontier (min latency for max accuracy)
    sorted_by_p99 = sorted(zip(p99s, accs, lbls))
    pareto_x, pareto_y = [], []
    best_acc = -1
    for px, py, _ in sorted_by_p99:
        if py > best_acc:
            best_acc = py
            pareto_x.append(px)
            pareto_y.append(py)

    fig, ax = _ax_both(H)
    ax.scatter(p99s, accs, color=BLUE, s=30, zorder=3, label="Exit points")
    ax.plot(pareto_x, pareto_y, "-", color=ORANGE, lw=1.4,
            label="Pareto frontier", zorder=4)

    for p99, acc, lbl in zip(p99s, accs, lbls):
        ax.annotate(lbl, (p99, acc),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=6.5, color="#444444")

    ax.set_xlabel("P99 latency (ms)")
    ax.set_ylabel("PubMedQA accuracy (%)")
    ax.set_title("Quality–latency trade-off (Pareto frontier)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig
