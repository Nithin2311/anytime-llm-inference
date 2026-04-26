"""
e01_iid_validation.py — IID validation on spaced vs unspaced profiling data.

Runs Ljung-Box test on E00 spaced data and compares against sprint_final
reference values (Ljung-Box p=0.0 for all cells without spacing).
Pass criterion: p > 0.05 at all lags for all cells.

Also generates ACF plots for visual confirmation.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from evt_utils_v3 import ljungbox_iid
from result_writer import write_results
from fig_style import apply_style, DOUBLE

apply_style()

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Reference: sprint_final E08 (all cells had lb_pvalue=0.0 without spacing)
REFERENCE_IID_PASS = False  # all cells failed without spacing
LAGS = [1, 5, 10, 20]


def main():
    print("=" * 60)
    print("E01  IID Validation: Ljung-Box on Spaced Profiling Data")
    print("=" * 60)

    e00_path = RESULTS_DIR / "e00_spaced_profiling.json"
    if not e00_path.exists():
        raise FileNotFoundError("E00 results not found — run e00 first.")

    with open(e00_path) as f:
        e00 = json.load(f)

    raw = e00["raw"]
    cells_results = {}
    n_pass = 0
    n_fail = 0

    print(f"\n{'Cell':<20} {'min_p':>8} {'IID?':>6}  lag detail")
    print("-" * 60)

    for key, lats in raw.items():
        arr = np.array(lats)
        result = ljungbox_iid(arr, lags=LAGS)
        cells_results[key] = result

        if result["iid_pass"]:
            n_pass += 1
            status = "PASS"
        else:
            n_fail += 1
            status = "FAIL"

        lag_summary = "  ".join(
            f"lag{lag}:p={result['lags'][f'lag_{lag}']['pvalue']:.3f}"
            for lag in LAGS
        )
        print(f"  {key:<18} {result['min_pvalue']:8.4f} {status:>6}  {lag_summary}")

    print(f"\n  Cells PASS: {n_pass}/{len(cells_results)}")
    print(f"  Cells FAIL: {n_fail}/{len(cells_results)}")
    print(f"  Reference (no spacing): ALL FAIL (p=0.0)")

    overall_pass = n_fail == 0

    # ── ACF plots ────────────────────────────────────────────────────────────
    keys_to_plot = [k for k in raw if "128" in k][:2]
    if keys_to_plot:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, axes = plt.subplots(1, len(keys_to_plot), figsize=DOUBLE)
        if len(keys_to_plot) == 1:
            axes = [axes]
        for ax, key in zip(axes, keys_to_plot):
            arr = np.array(raw[key])
            plot_acf(arr, lags=40, ax=ax, alpha=0.05, title=f"ACF: {key} (spaced)")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorrelation")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "e01_acf_spaced.png", dpi=150)
        plt.close()
        print("  ACF figure saved.")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "e01_iid_validation",
        "lags_tested": LAGS,
        "n_cells": len(cells_results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "overall_iid_pass": overall_pass,
        "reference_no_spacing_pass": REFERENCE_IID_PASS,
        "cells": cells_results,
    }
    write_results(output, RESULTS_DIR / "e01_iid_validation.json")
    print(f"\nPASS: E01 complete — IID overall: {'PASS' if overall_pass else 'PARTIAL'}\n")


if __name__ == "__main__":
    main()
