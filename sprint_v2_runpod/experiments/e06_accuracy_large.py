"""
e06_accuracy_large.py — Large-scale PubMedQA accuracy evaluation (500 queries).

Addresses M1: 14-query accuracy CI [50%,93%] is clinically meaningless.
Target: ≥200 scoreable tokens → 95% CI width ≈ ±7%, resolving R3-W1.

Collects raw per-token data once via KV-cached forward pass,
applies τ=0.55 post-hoc, and computes bootstrap CI (2000 resamples).
"""

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sprint_v2_runpod" / "src"))

import numpy as np
import torch
from benchmark_utils import (
    load_pubmed_dataset,
    run_pubmed_queries_raw,
    apply_threshold_posthoc,
)
from evt_utils import bootstrap_accuracy_ci
from fig_style import apply_style, SINGLE, DOUBLE
import matplotlib.pyplot as plt
from result_writer import write_results

apply_style()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
N_QUERIES     = 500
DEADLINE_MS   = 45.0
TAU_DEFAULT   = 0.55
N_BOOTSTRAP   = 2000
DEVICE        = "cuda"


def main():
    print("=" * 60)
    print("E06  Large-scale PubMedQA accuracy evaluation")
    print(f"     N={N_QUERIES} queries, τ={TAU_DEFAULT}, D={DEADLINE_MS}ms")
    print("=" * 60)

    # ── Load model ──────────────────────────────────────────────────────────
    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading {N_QUERIES} PubMedQA queries ...")
    queries = load_pubmed_dataset(n_samples=N_QUERIES)
    print(f"      Loaded {len(queries)} queries")

    # ── Collect raw token data (one KV pass per query) ───────────────────
    print(f"\n[2/4] Collecting raw confidence data ({N_QUERIES} KV passes) ...")
    t0 = time.time()
    raw_tokens = run_pubmed_queries_raw(
        model, queries, deadline_ms=DEADLINE_MS, device=DEVICE,
        max_new_tokens=15, show_progress=True
    )
    elapsed_collect = time.time() - t0
    print(f"      Done in {elapsed_collect:.1f}s  ({len(raw_tokens)} token records)")

    # ── Post-hoc τ replay ────────────────────────────────────────────────
    print(f"\n[3/4] Post-hoc threshold replay τ={TAU_DEFAULT} ...")
    results_tau = apply_threshold_posthoc(raw_tokens, queries, tau=TAU_DEFAULT,
                                          deadline_ms=DEADLINE_MS,
                                          n_bootstrap=N_BOOTSTRAP)

    # Also sweep τ for supplemental figure
    tau_sweep = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep_rows = []
    for tau in tau_sweep:
        r = apply_threshold_posthoc(raw_tokens, queries, tau=tau,
                                    deadline_ms=DEADLINE_MS, n_bootstrap=N_BOOTSTRAP)
        sweep_rows.append({"tau": tau, **r})

    # ── Bootstrap CI ─────────────────────────────────────────────────────
    print(f"\n[4/4] Bootstrap CI (n_bootstrap={N_BOOTSTRAP}) ...")
    n_scoreable = results_tau.get("n_scored", 0)
    correct_flags = results_tau.get("correct_flags", [])
    ci = bootstrap_accuracy_ci(correct_flags, n_bootstrap=N_BOOTSTRAP)

    print(f"\n{'─'*50}")
    print(f"  Queries total        : {len(queries)}")
    print(f"  Scoreable (yes/no/maybe): {n_scoreable}")
    print(f"  Accuracy             : {ci['accuracy']:.1f}%")
    print(f"  95% CI               : [{ci['ci_lower']:.1f}%, {ci['ci_upper']:.1f}%]")
    print(f"  CI width             : {ci['ci_width']:.1f}pp")
    print(f"  Exit rate (τ={TAU_DEFAULT})  : {results_tau.get('exit_rate_pct', 0):.1f}%")
    print(f"  Deadline miss rate   : {results_tau.get('miss_rate_pct', 0):.1f}%")
    print(f"{'─'*50}\n")

    # ── Figures ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)

    # Left: accuracy vs τ with CI bands
    taus   = [r["tau"] for r in sweep_rows]
    accs   = [r.get("accuracy", 0) for r in sweep_rows]
    ci_lo  = [r.get("ci_lower", 0) for r in sweep_rows]
    ci_hi  = [r.get("ci_upper", 0) for r in sweep_rows]
    exits  = [r.get("exit_rate_pct", 0) for r in sweep_rows]

    ax = axes[0]
    ax.fill_between(taus, ci_lo, ci_hi, alpha=0.25, color="tab:blue", label="95% CI")
    ax.plot(taus, accs, "o-", color="tab:blue", label="Accuracy")
    ax.axvline(TAU_DEFAULT, ls="--", color="tab:orange", lw=1.2, label=f"τ={TAU_DEFAULT}")
    ax.set_xlabel("Confidence threshold τ")
    ax.set_ylabel("PubMedQA accuracy (%)")
    ax.set_title("Accuracy vs. τ (500-query)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # Right: exit rate vs τ
    ax2 = axes[1]
    ax2.plot(taus, exits, "s-", color="tab:green", label="Exit rate")
    ax2.axvline(TAU_DEFAULT, ls="--", color="tab:orange", lw=1.2)
    ax2.set_xlabel("Confidence threshold τ")
    ax2.set_ylabel("Early-exit rate (%)")
    ax2.set_title("Exit Rate vs. τ (500-query)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "accuracy_large.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    tex_path = RESULTS_DIR / "table_accuracy_large.tex"
    with open(tex_path, "w") as f:
        f.write("% E06: Large-scale PubMedQA accuracy (500 queries)\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("$\\tau$ & Accuracy (\\%) & 95\\% CI & CI width (pp) & Exit rate (\\%) & Miss rate (\\%) \\\\\n\\midrule\n")
        for r in sweep_rows:
            marker = " \\textbf{*}" if abs(r["tau"] - TAU_DEFAULT) < 1e-9 else ""
            f.write(f"{r['tau']:.2f}{marker} & {r.get('accuracy',0):.1f} & "
                    f"[{r.get('ci_lower',0):.1f}, {r.get('ci_upper',0):.1f}] & "
                    f"{r.get('ci_width',0):.1f} & {r.get('exit_rate_pct',0):.1f} & "
                    f"{r.get('miss_rate_pct',0):.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  LaTeX table: {tex_path}")

    # ── Save JSON ────────────────────────────────────────────────────────
    output = {
        "experiment": "e06_accuracy_large",
        "n_queries": len(queries),
        "n_scoreable": n_scoreable,
        "tau_default": TAU_DEFAULT,
        "deadline_ms": DEADLINE_MS,
        "n_bootstrap": N_BOOTSTRAP,
        "accuracy_pct": ci["accuracy"],
        "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"],
        "ci_width": ci["ci_width"],
        "exit_rate_pct": results_tau.get("exit_rate_pct", 0),
        "miss_rate_pct": results_tau.get("miss_rate_pct", 0),
        "tau_sweep": sweep_rows,
    }
    write_results(output, RESULTS_DIR / "accuracy_large_results.json")
    print("PASS: E06 complete\n")


if __name__ == "__main__":
    main()
