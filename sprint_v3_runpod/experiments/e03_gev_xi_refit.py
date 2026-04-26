"""
e03_gev_xi_refit.py — GEV xi analysis on spaced profiling data.

Uses E00 spaced data (200ms sleep, 50 warm-up) to refit GEV and check
whether spacing reduces xi to the Gumbel-valid range (xi < 0.15).

Reference: sprint_final E01 measured xi=1.36 for L16, seq=128.
Target: xi < 0.15 would restore Gumbel pWCET validity.
If xi >= 0.15, block maxima (E04) is the valid path forward.

Outputs:
  - GEV xi table for all cells
  - Anderson-Darling test results
  - pWCET comparison: Gumbel vs GEV
  - Figure: xi heatmap + AD result grid
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from evt_utils_v3 import (
    fit_gev, anderson_darling_gumbel, pot_tail,
    gev_pwcet, gumbel_pwcet, bootstrap_gev_pwcet_ci
)
from result_writer import write_results
from fig_style import apply_style, DOUBLE

apply_style()

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

POT_FRACTION    = 0.20
EXCEEDANCE_PROB = 1e-6
N_BOOTSTRAP     = 1000

# Reference xi from sprint_final E01 (without spacing)
REFERENCE = {
    "seq128_l16":  {"xi": 1.360218, "ad_stat": 17.5465, "gumbel_valid": False},
    "seq128_full": {"xi": 0.60,     "ad_stat": None,    "gumbel_valid": False},
}


def main():
    print("=" * 60)
    print("E03  GEV Shape Parameter Refit on Spaced Data")
    print(f"     POT={POT_FRACTION*100:.0f}%  exceedance=1e-6  "
          f"bootstrap={N_BOOTSTRAP}")
    print("=" * 60)

    e00_path = RESULTS_DIR / "e00_spaced_profiling.json"
    if not e00_path.exists():
        raise FileNotFoundError("E00 results not found — run e00 first.")

    with open(e00_path) as f:
        e00 = json.load(f)

    raw = e00["raw"]
    results = {}
    gumbel_valid_count = 0
    total_cells = len(raw)

    header = f"{'Cell':<22} {'xi':>7} {'AD':>8} {'Gumbel?':>8} {'pWCET_Gumbel':>14} {'pWCET_GEV':>12}"
    print(f"\n{header}")
    print("-" * 75)

    for key, lats in raw.items():
        arr   = np.array(lats)
        tail  = pot_tail(arr, POT_FRACTION)
        gev   = fit_gev(tail)
        ad    = anderson_darling_gumbel(tail)
        pw_g  = gumbel_pwcet(tail, EXCEEDANCE_PROB)
        pw_gev = gev_pwcet(gev, EXCEEDANCE_PROB)
        ci    = bootstrap_gev_pwcet_ci(tail, EXCEEDANCE_PROB, N_BOOTSTRAP)

        gumbel_ok = not ad["gumbel_rejected"]
        if gumbel_ok:
            gumbel_valid_count += 1

        pw_gev_str = (f"{pw_gev:.2f}ms" if pw_gev is not None and
                      np.isfinite(pw_gev) and pw_gev < 1e6 else "unbounded")

        print(f"  {key:<20} {gev['xi']:>7.4f} {ad['statistic']:>8.3f} "
              f"{'OK' if gumbel_ok else 'FAIL':>8} "
              f"{pw_g:>12.2f}ms {pw_gev_str:>12}")

        results[key] = {
            "gev": gev,
            "ad": ad,
            "gumbel_valid": gumbel_ok,
            "pwcet_gumbel_ms": round(float(pw_g), 3),
            "pwcet_gev_ms": round(float(pw_gev), 3) if (
                pw_gev is not None and np.isfinite(pw_gev) and pw_gev < 1e6
            ) else None,
            "pwcet_gev_ci": ci,
            "n_tail": len(tail),
            "mean_ms": round(float(arr.mean()), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
        }

    print(f"\n  Gumbel valid cells: {gumbel_valid_count}/{total_cells}")
    print(f"  Reference (no spacing): 0/{total_cells} (all rejected)")

    # ── Figure ────────────────────────────────────────────────────────────────
    keys = list(results.keys())
    xi_vals = [results[k]["gev"]["xi"] for k in keys]
    ad_vals = [results[k]["ad"]["statistic"] for k in keys]
    crit    = results[keys[0]]["ad"]["critical_value_5pct"]

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)

    colors_xi = ["tab:green" if results[k]["gumbel_valid"] else "tab:red" for k in keys]
    axes[0].bar(range(len(keys)), xi_vals, color=colors_xi, alpha=0.8,
                edgecolor="black", lw=0.5)
    axes[0].axhline(0.15, ls="--", color="black", lw=1.2, label="Gumbel limit (0.15)")
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels([k.replace("seq","s").replace("_"," ") for k in keys],
                             rotation=45, ha="right", fontsize=5)
    axes[0].set_ylabel("GEV shape ξ")
    axes[0].set_title("GEV ξ: spaced data (green=Gumbel OK)")
    axes[0].legend(fontsize=6)

    colors_ad = ["tab:green" if not results[k]["ad"]["gumbel_rejected"] else "tab:red"
                 for k in keys]
    axes[1].bar(range(len(keys)), ad_vals, color=colors_ad, alpha=0.8,
                edgecolor="black", lw=0.5)
    axes[1].axhline(crit, ls="--", color="black", lw=1.2,
                    label=f"Critical (5%): {crit:.3f}")
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels([k.replace("seq","s").replace("_"," ") for k in keys],
                             rotation=45, ha="right", fontsize=5)
    axes[1].set_ylabel("AD test statistic")
    axes[1].set_title("Anderson-Darling: spaced data")
    axes[1].legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e03_gev_xi_refit.png", dpi=150)
    plt.close()

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "e03_gev_xi_refit",
        "pot_fraction": POT_FRACTION,
        "exceedance_prob": EXCEEDANCE_PROB,
        "n_bootstrap": N_BOOTSTRAP,
        "gumbel_valid_count": gumbel_valid_count,
        "total_cells": total_cells,
        "ad_critical_5pct": crit,
        "reference": REFERENCE,
        "cells": results,
        "gumbel_fully_restored": gumbel_valid_count == total_cells,
    }
    write_results(output, RESULTS_DIR / "e03_gev_xi_refit.json")
    print("\nPASS: E03 complete\n")


if __name__ == "__main__":
    main()
