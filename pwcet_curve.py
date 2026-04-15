"""
pwcet_curve.py — Probabilistic WCET (pWCET) curve.

Uses the fitted Gumbel parameters from evt_wcet_results.json to compute pWCET
bounds across a log-spaced range of exceedance probabilities (10^-1 → 10^-7).

Key concept:
  pWCET(ε) = smallest bound C such that P(execution_time > C) ≤ ε
           = gumbel_r.ppf(1 - ε, loc, scale)

This gives a full probabilistic safety certificate: instead of claiming
"WCET = X ms" (which implies P=0), we claim "P(latency > X) ≤ 10^-6".

Also derives schedulability probability: P(meet deadline D) = CDF_gumbel(D)
and plots the miss-probability curve vs deadline for each sequence length.

Requires: evt_wcet_results.json (from evt_wcet_analysis.py)

Outputs:
  pwcet_curve_results.json
  pwcet_curve.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r

RESULTS_FILE  = "pwcet_curve_results.json"
FIGURE_FILE   = "pwcet_curve.png"
DEADLINE_MS   = 45.0
SEQ_LENGTHS   = [32, 64, 128, 256, 512, 1024]
FULL_KEY      = "None"

# Exceedance probabilities: 10^-1 down to 10^-7
EPS_VALUES    = np.logspace(-1, -7, 61)


def load_evt_params(evt_file="evt_wcet_results.json"):
    with open(evt_file) as f:
        data = json.load(f)
    return data["results"]


def compute_pwcet_curve(loc, scale):
    """pWCET at each exceedance probability."""
    return [float(gumbel_r.ppf(1.0 - eps, loc=loc, scale=scale)) for eps in EPS_VALUES]


def miss_probability(deadline_range, loc, scale):
    """P(latency > D) = 1 - CDF(D) for a range of deadlines."""
    cdfs = gumbel_r.cdf(deadline_range, loc=loc, scale=scale)
    return np.clip(1.0 - cdfs, 1e-12, 1.0)


def plot_pwcet(results_out):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # ── Panel 1: pWCET vs exceedance probability ──────────────────────────────
    ax1 = axes[0]
    for idx, sl in enumerate(SEQ_LENGTHS):
        key  = str(sl)
        cdat = results_out["full_pass"][key]
        ax1.plot(EPS_VALUES, cdat["pwcet_ms"], linewidth=2,
                 color=colours[idx], label=f"seq={sl}")

    ax1.axhline(DEADLINE_MS, color="black", linestyle="--", linewidth=1.2,
                label=f"Deadline ({DEADLINE_MS:.0f} ms)", zorder=5)

    # Mark current empirical×1.10 bounds as scatter points at ε=10^-3
    eps_marker = 1e-3
    for idx, sl in enumerate(SEQ_LENGTHS):
        emp_x110 = results_out["full_pass"][str(sl)]["wcet_empirical_x110"]
        ax1.scatter([eps_marker], [emp_x110], marker="x", s=60,
                    color=colours[idx], zorder=6)

    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("Exceedance Probability ε  (P[latency > pWCET])")
    ax1.set_ylabel("pWCET Bound (ms)")
    ax1.set_title("Probabilistic WCET Curve — Full(22) Pass")
    ax1.legend(loc="upper left", fontsize=8)

    # Annotate the ×1.10 scatter
    ax1.text(eps_marker * 1.8, DEADLINE_MS - 3.5, "× markers = empirical×1.10",
             fontsize=7.5, color="gray", ha="left")

    # ── Panel 2: P(miss deadline) vs deadline ─────────────────────────────────
    ax2 = axes[1]
    deadlines = np.linspace(15, 65, 300)

    for idx, sl in enumerate(SEQ_LENGTHS):
        key  = str(sl)
        loc  = results_out["full_pass"][key]["gumbel_loc"]
        scl  = results_out["full_pass"][key]["gumbel_scale"]
        miss = miss_probability(deadlines, loc, scl)
        ax2.plot(deadlines, miss, linewidth=2, color=colours[idx], label=f"seq={sl}")

    ax2.axvline(DEADLINE_MS, color="black", linestyle="--", linewidth=1.2,
                label=f"D = {DEADLINE_MS:.0f} ms")
    ax2.axhline(1e-4, color="gray", linestyle=":", linewidth=1.0, label="10⁻⁴ target")
    ax2.axhline(1e-6, color="silver", linestyle=":", linewidth=1.0, label="10⁻⁶ target")

    ax2.set_yscale("log")
    ax2.set_ylim(1e-8, 1.5)
    ax2.set_xlabel("Deadline D (ms)")
    ax2.set_ylabel("P(miss deadline)  [log scale]")
    ax2.set_title("Miss Probability vs Deadline — Full(22) Pass")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("pWCET Analysis — TinyLlama-1.1B  (RTX 4000 Ada, Gumbel EVT fit)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    print("=" * 60)
    print("pWCET CURVE ANALYSIS — TinyLlama-1.1B")
    print(f"Exceedance probs: {EPS_VALUES[0]:.0e} → {EPS_VALUES[-1]:.0e}  ({len(EPS_VALUES)} points)")
    print("=" * 60 + "\n")

    evt_data = load_evt_params()

    results_out = {"full_pass": {}, "deadline_ms": DEADLINE_MS}

    print(f"{'Seq':>6}  {'loc':>8}  {'scale':>8}  "
          f"{'pWCET(1e-4)':>12}  {'pWCET(1e-6)':>12}  {'P(miss@45ms)':>14}")
    print("-" * 68)

    for sl in SEQ_LENGTHS:
        cell     = evt_data[str(sl)][FULL_KEY]
        loc, scl = cell["gumbel_loc"], cell["gumbel_scale"]
        pwcet    = compute_pwcet_curve(loc, scl)
        miss_45  = float(1.0 - gumbel_r.cdf(DEADLINE_MS, loc=loc, scale=scl))
        miss_45  = max(miss_45, 1e-12)

        results_out["full_pass"][str(sl)] = {
            "gumbel_loc":         loc,
            "gumbel_scale":       scl,
            "wcet_empirical_x110": cell["wcet_empirical_x110"],
            "pwcet_ms":           [round(v, 4) for v in pwcet],
            "exceedance_probs":   [float(e) for e in EPS_VALUES],
            "miss_prob_at_45ms":  round(miss_45, 12),
            "pwcet_at_1e4":       round(float(gumbel_r.ppf(1 - 1e-4, loc=loc, scale=scl)), 4),
            "pwcet_at_1e6":       round(float(gumbel_r.ppf(1 - 1e-6, loc=loc, scale=scl)), 4),
        }

        print(f"{sl:>6}  {loc:>8.3f}  {scl:>8.4f}  "
              f"{results_out['full_pass'][str(sl)]['pwcet_at_1e4']:>12.3f}  "
              f"{results_out['full_pass'][str(sl)]['pwcet_at_1e6']:>12.3f}  "
              f"{miss_45:>14.3e}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    plot_pwcet(results_out)
