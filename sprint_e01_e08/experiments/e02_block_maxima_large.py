"""e02_block_maxima_large.py — Block-maxima GEV on 5000-sample E00 data."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib.pyplot as plt
from fig_style import apply_style, DOUBLE
from evt_utils import fit_gev_block_maxima
from result_writer import write_results
apply_style()
RESULTS_DIR = Path(__file__).parent.parent / "results"
BLOCK_SIZES = [10, 25, 50, 100]

def main():
    print("=" * 60)
    print("E02  Block-Maxima GEV  (requires E00)")
    print("=" * 60)
    e00_path = RESULTS_DIR / "e00_wcet_large_spaced.json"
    if not e00_path.exists():
        raise FileNotFoundError(f"Run e00 first: {e00_path}")
    with open(e00_path) as f:
        e00 = json.load(f)
    cells = e00["cells"]
    print(f"  Loaded {len(cells)} cells  (n={cells[0]['n_samples']} each)")

    all_results = []
    for cell in cells:
        cname = cell["cell"]
        samples = np.array(cell["samples"])
        print(f"\n  {cname}", flush=True)
        brs = []
        for b in BLOCK_SIZES:
            r = fit_gev_block_maxima(samples, block_size=b)
            if r is None:
                brs.append({"block_size": b, "skipped": True})
            elif "error" in r:
                brs.append(r)
            else:
                tag = "PASS" if r["gumbel_accepted"] else "FAIL"
                print(f"    b={b:3d}  n_b={r['n_blocks']:4d}  xi={r['xi']:+.3f}  "
                      f"pWCET={r['wcet_1e6']}ms  Gumbel={tag}")
                brs.append(r)
        all_results.append({"cell": cname, "seq_len": cell["seq_len"],
                            "exit": cell["exit"], "n": len(samples), "block_results": brs})

    # xi vs block-size grid
    ncols = 4; nrows = (len(all_results) + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, nrows * 1.9))
    axes = axes.flatten()
    for i, cr in enumerate(all_results):
        ax = axes[i]
        bs  = [r["block_size"] for r in cr["block_results"] if "xi" in r]
        xis = [r["xi"] for r in cr["block_results"] if "xi" in r]
        if bs: ax.plot(bs, xis, "o-", color="tab:blue", ms=4)
        ax.axhline(0,    ls="--", color="green", lw=0.8, label="xi=0")
        ax.axhline(0.15, ls=":",  color="red",   lw=0.8)
        ax.set_title(cr["cell"], fontsize=5); ax.set_xlabel("b", fontsize=5); ax.set_ylabel("xi", fontsize=5)
    for j in range(len(all_results), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("E02 — Block-Maxima GEV xi vs b (n=5000)", fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e02_block_maxima.png", dpi=150); plt.close()

    write_results({"experiment": "e02_block_maxima_large", "block_sizes": BLOCK_SIZES,
                   "cells": all_results}, RESULTS_DIR / "e02_block_maxima.json")

    tex = ("% E02 Block-Maxima GEV\n\\begin{tabular}{lccccc}\\toprule\n"
           "Cell & $b$ & $n_b$ & $\\xi$ & pWCET$_{10^{-6}}$ (ms) & Gumbel\\\\\n\\midrule\n")
    for cr in all_results:
        best = next((r for r in cr["block_results"] if "xi" in r and r.get("gumbel_accepted")), None)
        if best is None:
            best = next((r for r in cr["block_results"] if "xi" in r), None)
        if best:
            g = "\\checkmark" if best.get("gumbel_accepted") else "\\times"
            tex += (f"{cr['cell']} & {best['block_size']} & {best['n_blocks']} & "
                    f"{best['xi']:+.3f} & {best.get('wcet_1e6','--')} & {g}\\\\\n")
    tex += "\\bottomrule\\end{tabular}\n"
    with open(RESULTS_DIR / "table_block_maxima.tex", "w") as f: f.write(tex)
    print("\nPASS: E02 complete\n")

if __name__ == "__main__":
    main()
