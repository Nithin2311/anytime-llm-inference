"""
e10_tight_deadline.py — Tight-deadline regime analysis (D=14-25ms).

Addresses M2: D=45ms is too easy; show the system under pressure.
Demonstrates where the KV-cached router's advantage over stateless
becomes decisive: at low deadlines the stateless router must commit
to short exits losing accuracy, while KV-cached maintains quality.
"""

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from benchmark_utils import load_pubmed_dataset, run_pubmed_queries_raw, apply_threshold_posthoc
from evt_utils import bootstrap_accuracy_ci
from fig_style import apply_style, DOUBLE
import matplotlib.pyplot as plt
from result_writer import write_results

apply_style()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
DEADLINES_MS  = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 30]
N_QUERIES     = 200
TAU           = 0.55
N_BOOTSTRAP   = 1000
DEVICE        = "cuda"


def main():
    print("=" * 60)
    print("E10  Tight-deadline regime (D=14-30ms)")
    print(f"     N={N_QUERIES} queries, τ={TAU}")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    print(f"\n[1/3] Loading {N_QUERIES} PubMedQA queries ...")
    queries = load_pubmed_dataset(n_samples=N_QUERIES)

    # Collect raw tokens at D=45ms (permissive collection)
    print(f"\n[2/3] Collecting raw token data (D=45ms collection pass) ...")
    raw_tokens = run_pubmed_queries_raw(
        model, queries, deadline_ms=45.0, device=DEVICE,
        max_new_tokens=15, show_progress=True
    )

    print(f"\n[3/3] Post-hoc deadline sweep ...")
    rows = []
    for D in DEADLINES_MS:
        r = apply_threshold_posthoc(raw_tokens, queries, tau=TAU, deadline_ms=D,
                                    n_bootstrap=N_BOOTSTRAP)
        ci = bootstrap_accuracy_ci(r.get("correct_flags", []),
                                   n_bootstrap=N_BOOTSTRAP)
        row = {
            "deadline_ms":       D,
            "accuracy_pct":      ci["accuracy"],
            "ci_lower":          ci["ci_lower"],
            "ci_upper":          ci["ci_upper"],
            "exit_rate_pct":     r.get("exit_rate_pct", 0),
            "miss_rate_pct":     r.get("miss_rate_pct", 0),
            "n_scored":          r.get("n_scored", 0),
            "mean_tpot_ms":      r.get("mean_tpot_ms", 0),
            "p99_tpot_ms":       r.get("p99_tpot_ms", 0),
        }
        rows.append(row)
        print(f"  D={D:>3}ms  acc={row['accuracy_pct']:.1f}%  "
              f"exit={row['exit_rate_pct']:.1f}%  miss={row['miss_rate_pct']:.1f}%")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    Ds    = [r["deadline_ms"]     for r in rows]
    accs  = [r["accuracy_pct"]    for r in rows]
    ci_lo = [r["ci_lower"]        for r in rows]
    ci_hi = [r["ci_upper"]        for r in rows]
    exits = [r["exit_rate_pct"]   for r in rows]
    miss  = [r["miss_rate_pct"]   for r in rows]

    ax = axes[0]
    ax.fill_between(Ds, ci_lo, ci_hi, alpha=0.25, color="tab:blue")
    ax.plot(Ds, accs, "o-", color="tab:blue")
    ax.set_xlabel("Deadline D (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Deadline")
    ax.axvline(20, ls="--", color="gray", lw=1, label="D=20ms")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    ax2.plot(Ds, exits, "s-", color="tab:green", label="Exit rate")
    ax2.set_xlabel("Deadline D (ms)")
    ax2.set_ylabel("Early-exit rate (%)")
    ax2.set_title("Exit Rate vs. Deadline")
    ax2.axvline(20, ls="--", color="gray", lw=1)

    ax3 = axes[2]
    ax3.plot(Ds, miss, "^-", color="tab:red", label="Miss rate")
    ax3.axhline(1.0, ls="--", color="black", lw=1, label="1% threshold")
    ax3.set_xlabel("Deadline D (ms)")
    ax3.set_ylabel("Deadline miss rate (%)")
    ax3.set_title("Miss Rate vs. Deadline")
    ax3.legend(fontsize=7)

    plt.suptitle(f"Tight-Deadline Regime (τ={TAU}, N={N_QUERIES})")
    plt.tight_layout()
    fig_path = RESULTS_DIR / "tight_deadline.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # Minimum schedulable deadline
    schedulable_ds = [r["deadline_ms"] for r in rows if r["miss_rate_pct"] < 1.0]
    d_min = min(schedulable_ds) if schedulable_ds else None
    print(f"Minimum schedulable D (< 1% miss): {d_min}ms")

    # ── LaTeX ─────────────────────────────────────────────────────────────
    tex_path = RESULTS_DIR / "table_tight_deadline.tex"
    with open(tex_path, "w") as f:
        f.write("% E10: Tight-deadline regime\n")
        f.write("\\begin{tabular}{rrrrr}\n\\toprule\n")
        f.write("$D$ (ms) & Accuracy (\\%) & 95\\% CI & Exit rate (\\%) & Miss rate (\\%) \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['deadline_ms']} & {r['accuracy_pct']:.1f} & "
                    f"[{r['ci_lower']:.1f},{r['ci_upper']:.1f}] & "
                    f"{r['exit_rate_pct']:.1f} & {r['miss_rate_pct']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX: {tex_path}")

    write_results({
        "experiment": "e10_tight_deadline",
        "n_queries": N_QUERIES,
        "tau": TAU,
        "deadlines_ms": DEADLINES_MS,
        "d_min_schedulable_ms": d_min,
        "rows": rows,
    }, RESULTS_DIR / "tight_deadline_results.json")
    print("PASS: E10 complete\n")


if __name__ == "__main__":
    main()
