"""
e04_block_maxima_pwcet.py — Block maxima EVT: methodologically valid pWCET.

Block maxima is the standard EVT approach when:
  (1) individual samples are autocorrelated, or
  (2) the GEV tail is too heavy for Gumbel (xi >> 0.15)

For block size b, take the maximum from each non-overlapping block of b
consecutive samples. Maxima across blocks are approximately independent
(the autocorrelation structure decorrelates at the block level), so GEV
fits to block maxima are valid even when raw samples fail Ljung-Box.

Key decision logic for the report:
  - If E03 shows xi < 0.15 AND E01 shows IID pass → Gumbel on spaced data is valid
  - Else → block maxima GEV (this experiment) is the valid methodology

Block maxima pWCET interpretation:
  P(block max > t) = exceedance_prob
  To certify P(single token > t) < p_target with block of size b:
    use GEV quantile at (1 - p_target/b) if using union bound, or
    report directly as: "per-block exceedance prob = p"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from evt_utils_v3 import (
    analyze_block_maxima, ljungbox_iid, fit_gev,
    anderson_darling_gumbel, gev_pwcet, gumbel_pwcet
)
from result_writer import write_results
from fig_style import apply_style, DOUBLE, TRIPLE

apply_style()

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BLOCK_SIZES     = [5, 10, 20, 25, 50]
EXCEEDANCE_PROB = 1e-6
N_BOOTSTRAP     = 500
FOCUS_CELLS     = ["seq128_l16", "seq128_full", "seq512_l16", "seq512_full"]


def main():
    print("=" * 60)
    print("E04  Block Maxima EVT Analysis")
    print(f"     Block sizes: {BLOCK_SIZES}  exceedance=1e-6")
    print("=" * 60)

    e00_path = RESULTS_DIR / "e00_spaced_profiling.json"
    if not e00_path.exists():
        raise FileNotFoundError("E00 results not found — run e00 first.")

    with open(e00_path) as f:
        e00 = json.load(f)

    raw = e00["raw"]
    all_results = {}

    for key, lats in raw.items():
        arr = np.array(lats)
        cell_bm = {}

        print(f"\n  Cell: {key}  (n={len(arr)}, mean={arr.mean():.2f}ms, p99={np.percentile(arr,99):.2f}ms)")
        print(f"  {'BlockSz':>8} {'NBlocks':>8} {'IID?':>6} {'xi':>7} {'Gumbel?':>8} "
              f"{'pWCET_GEV':>12} {'pWCET_Gumbel':>14}")
        print(f"  {'-'*65}")

        for b in BLOCK_SIZES:
            if len(arr) // b < 20:
                continue

            bm = analyze_block_maxima(arr, b, EXCEEDANCE_PROB, N_BOOTSTRAP)
            cell_bm[f"b{b}"] = bm

            iid_ok  = bm["iid"]["iid_pass"]
            gev_xi  = bm["gev"]["xi"]
            gum_ok  = not bm["ad"]["gumbel_rejected"]
            pw_gev  = bm["pwcet_gev"]
            pw_gum  = bm["pwcet_gumbel"]

            pw_gev_str = f"{pw_gev:.2f}ms" if pw_gev is not None else "unbounded"
            print(f"  {b:>8} {bm['n_blocks']:>8} "
                  f"{'OK' if iid_ok else 'FAIL':>6} "
                  f"{gev_xi:>7.4f} "
                  f"{'OK' if gum_ok else 'FAIL':>8} "
                  f"{pw_gev_str:>12} {pw_gum:>12.2f}ms")

        all_results[key] = cell_bm

    # ── Find minimum valid block size (IID pass) for key cells ───────────────
    print("\n  Minimum block size achieving IID for focus cells:")
    min_valid = {}
    for key in FOCUS_CELLS:
        if key not in all_results:
            continue
        bm_res = all_results[key]
        for b in BLOCK_SIZES:
            k = f"b{b}"
            if k in bm_res and bm_res[k]["iid"]["iid_pass"]:
                min_valid[key] = b
                pw = bm_res[k]["pwcet_gev"]
                xi = bm_res[k]["gev"]["xi"]
                pw_str = f"{pw:.2f}ms" if pw is not None else "unbounded"
                print(f"    {key}: b={b}  xi={xi:.4f}  pWCET(1e-6)={pw_str}")
                break
        else:
            min_valid[key] = None
            print(f"    {key}: no block size achieves IID")

    # ── Figure: pWCET vs block size for focus cells ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)

    focus_plot = [k for k in FOCUS_CELLS if k in all_results][:4]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i, (cell_key, color) in enumerate(zip(focus_plot, colors)):
        bm_data = all_results[cell_key]
        valid_bs, pw_gevs, pw_gums, xis = [], [], [], []
        for b in BLOCK_SIZES:
            k = f"b{b}"
            if k not in bm_data:
                continue
            valid_bs.append(b)
            pw_gev = bm_data[k]["pwcet_gev"]
            pw_gums.append(bm_data[k]["pwcet_gumbel"])
            pw_gevs.append(pw_gev if pw_gev is not None else np.nan)
            xis.append(bm_data[k]["gev"]["xi"])

        axes[0].plot(valid_bs, pw_gums, "o-", color=color, lw=1.2,
                     label=cell_key.replace("seq", "s").replace("_", " "))
        valid_gev = [(b, v) for b, v in zip(valid_bs, pw_gevs) if np.isfinite(v) and v < 200]
        if valid_gev:
            bs_v, pw_v = zip(*valid_gev)
            axes[0].plot(bs_v, pw_v, "s--", color=color, lw=0.8, alpha=0.7)

        axes[1].plot(valid_bs, xis, "o-", color=color, lw=1.2,
                     label=cell_key.replace("seq", "s").replace("_", " "))

    axes[0].axhline(45, ls="--", color="gray", lw=1, label="D=45ms")
    axes[0].set_xlabel("Block size b")
    axes[0].set_ylabel("pWCET(1e-6) (ms)")
    axes[0].set_title("pWCET vs block size\n(solid=Gumbel, dashed=GEV)")
    axes[0].legend(fontsize=5)

    axes[1].axhline(0.15, ls="--", color="black", lw=1.2, label="Gumbel limit (0.15)")
    axes[1].set_xlabel("Block size b")
    axes[1].set_ylabel("GEV shape ξ of block maxima")
    axes[1].set_title("GEV ξ of block maxima vs block size")
    axes[1].legend(fontsize=5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e04_block_maxima.png", dpi=150)
    plt.close()

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "e04_block_maxima_pwcet",
        "block_sizes": BLOCK_SIZES,
        "exceedance_prob": EXCEEDANCE_PROB,
        "min_iid_valid_block": min_valid,
        "cells": all_results,
    }
    write_results(output, RESULTS_DIR / "e04_block_maxima.json")
    print("\nPASS: E04 complete\n")


if __name__ == "__main__":
    main()
