"""e03_nonevt_wcet.py — Non-EVT empirical WCET bounds (primary camera-ready claim)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib.pyplot as plt
from fig_style import apply_style, DOUBLE
from evt_utils import nonevt_bounds
from result_writer import write_results
apply_style()
RESULTS_DIR = Path(__file__).parent.parent / "results"

def main():
    print("=" * 60)
    print("E03  Non-EVT Empirical WCET Bounds  (requires E00)")
    print("=" * 60)
    e00_path = RESULTS_DIR / "e00_wcet_large_spaced.json"
    if not e00_path.exists():
        raise FileNotFoundError(f"Run e00 first: {e00_path}")
    with open(e00_path) as f:
        e00 = json.load(f)

    all_bounds = []
    for cell in e00["cells"]:
        cname   = cell["cell"]
        samples = np.array(cell["samples"])
        b       = nonevt_bounds(samples, n_bootstrap=1000)
        b["cell"] = cname
        all_bounds.append(b)
        p999 = b["method_A"]["p99.9"]["point"]
        bB   = b["method_B"]["bound_ms"]
        bC   = b["method_C"]["bound_ms"]
        print(f"  {cname:22s}  P99.9={p999:7.2f}ms  P99+3s={bB:7.2f}ms  Hoeffding={bC:7.2f}ms")

    # Figure: bound comparison for two representative cells
    key = [b for b in all_bounds if "seq128_full" in b["cell"] or "seq512_full" in b["cell"]][:2]
    methods = ["P99", "P99.9", "P99.99", "P99+3σ", "Hoeffding"]
    colors  = ["#0072B2","#009E73","#D55E00","#CC79A7","#E69F00"]
    fig, axes = plt.subplots(1, max(len(key), 1), figsize=DOUBLE)
    if len(key) == 1: axes = [axes]
    for ax, bc in zip(axes, key):
        vals = [bc["method_A"]["p99.0"]["point"], bc["method_A"]["p99.9"]["point"],
                bc["method_A"]["p99.99"]["point"], bc["method_B"]["bound_ms"],
                bc["method_C"]["bound_ms"]]
        ci_lo = [bc["method_A"]["p99.0"]["ci_lower"], bc["method_A"]["p99.9"]["ci_lower"],
                 bc["method_A"]["p99.99"]["ci_lower"], vals[3], vals[4]]
        ci_hi = [bc["method_A"]["p99.0"]["ci_upper"], bc["method_A"]["p99.9"]["ci_upper"],
                 bc["method_A"]["p99.99"]["ci_upper"], vals[3], vals[4]]
        x = range(len(methods))
        ax.bar(x, vals, color=colors, alpha=0.8)
        ax.errorbar(x, vals, yerr=[[v-l for v,l in zip(vals,ci_lo)],[h-v for v,h in zip(vals,ci_hi)]],
                    fmt="none", color="black", capsize=4, lw=1.5)
        ax.set_xticks(list(x)); ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=6)
        ax.set_ylabel("WCET bound (ms)"); ax.set_title(f"E03 — {bc['cell']}", fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e03_nonevt_bounds.png", dpi=150); plt.close()

    write_results({"experiment": "e03_nonevt_wcet", "cells": all_bounds},
                  RESULTS_DIR / "e03_nonevt_bounds.json")

    tex = ("% E03 Non-EVT Bounds\n\\begin{tabular}{lcccc}\\toprule\n"
           "Cell & P99.9 (ms) & 95\\% CI & P99+3$\\sigma$ & Hoeffding\\\\\n\\midrule\n")
    for b in all_bounds:
        p = b["method_A"]["p99.9"]
        tex += (f"{b['cell']} & {p['point']:.1f} & [{p['ci_lower']:.1f},{p['ci_upper']:.1f}] & "
                f"{b['method_B']['bound_ms']:.1f} & {b['method_C']['bound_ms']:.1f}\\\\\n")
    tex += "\\bottomrule\\end{tabular}\n"
    with open(RESULTS_DIR / "table_nonevt_bounds.tex", "w") as f: f.write(tex)
    print("\nPASS: E03 complete\n")

if __name__ == "__main__":
    main()
