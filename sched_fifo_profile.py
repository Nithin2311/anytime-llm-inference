"""
sched_fifo_profile.py — WCET profiling under SCHED_OTHER vs SCHED_FIFO.

Runs the full WCET sweep twice on identical hardware state:
  1. Default Linux scheduler (SCHED_OTHER)
  2. SCHED_FIFO at priority 99 (real-time, requires sudo / CAP_SYS_NICE)

If SCHED_FIFO is unavailable, produces normal-scheduling results only and
notes fifo_available=false in the JSON.

Usage:
  python3 sched_fifo_profile.py          # SCHED_OTHER only
  sudo python3 sched_fifo_profile.py     # SCHED_OTHER + SCHED_FIFO

Outputs:
  sched_fifo_results.json
  sched_fifo_comparison.png
"""

import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from early_exit_model import EarlyExitTinyLlama
from profile_wcet import profile_gpu_execution, make_input

RESULTS_FILE = "sched_fifo_results.json"
FIGURE_FILE  = "sched_fifo_comparison.png"

# Mirror the sweep parameters from profile_wcet.py; 200 runs per cell for
# better tail statistics (vs 50 in the original profiler).
SEQ_LENGTHS = [32, 64, 128, 256, 512, 1024]
EXIT_LAYERS = [5, 11, 16, None]   # None = full 22-layer pass
NUM_WARMUP  = 5
NUM_RUNS    = 200


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(model, label=""):
    """
    Run the full (seq_len × exit_layer) WCET sweep.
    Returns a nested dict matching the wcet_results.json structure.
    """
    results = {}
    total   = len(SEQ_LENGTHS) * len(EXIT_LAYERS)
    cell    = 0

    for seq_len in SEQ_LENGTHS:
        input_ids = make_input(seq_len, device="cuda")
        results[seq_len] = {}

        for exit_layer in EXIT_LAYERS:
            cell += 1
            tag   = f"L{exit_layer}" if exit_layer is not None else "Full(22)"
            print(f"  {label} [{cell:>2}/{total}] seq={seq_len:>4}  exit={tag:<8}", end="  ", flush=True)

            mean_ms, p99_ms, wcet_ms = profile_gpu_execution(
                model, input_ids,
                exit_layer=exit_layer,
                use_cache=False,
                num_warmup=NUM_WARMUP,
                num_runs=NUM_RUNS,
            )
            results[seq_len][str(exit_layer)] = {
                "mean_ms": round(mean_ms, 3),
                "p99_ms":  round(p99_ms,  3),
                "wcet_ms": round(wcet_ms, 3),
            }
            print(f"mean={mean_ms:.2f}  p99={p99_ms:.2f}  wcet={wcet_ms:.2f}  (ms)")

    return results


# ── Figure — two-panel ─────────────────────────────────────────────────────────

def plot_comparison(normal_results, fifo_results, deadline_ms=45.0):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    full_key = "None"   # dict key for the full 22-layer pass

    normal_wcet = [normal_results[sl][full_key]["wcet_ms"] for sl in SEQ_LENGTHS]
    fifo_wcet   = [fifo_results[sl][full_key]["wcet_ms"]   for sl in SEQ_LENGTHS]

    # ── Panel 1: WCET vs sequence length ──────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(SEQ_LENGTHS, normal_wcet, marker="o", linewidth=2,
             color="#d62728", label="SCHED_OTHER (default)", zorder=3)
    ax1.plot(SEQ_LENGTHS, fifo_wcet,   marker="s", linewidth=2,
             color="#2ca02c", label="SCHED_FIFO (prio 99)",  zorder=3)
    ax1.axhline(y=deadline_ms, color="black", linestyle="--", linewidth=1.2,
                label=f"Deadline ({deadline_ms:.0f} ms)")

    ax1.set_xlabel("Input Sequence Length (tokens)")
    ax1.set_ylabel("WCET / Max Latency (ms)")
    ax1.set_title("WCET Comparison: Default vs SCHED_FIFO Scheduling")
    ax1.set_xticks(SEQ_LENGTHS)
    ax1.legend(loc="upper left")

    # Shade improvement region
    ax1.fill_between(SEQ_LENGTHS, normal_wcet, fifo_wcet,
                     alpha=0.12, color="#2ca02c", label="_nolegend_")

    # ── Panel 2: P99 latency reduction % per seq_len ──────────────────────────
    ax2 = axes[1]
    reductions = []
    for sl in SEQ_LENGTHS:
        n_p99 = normal_results[sl][full_key]["p99_ms"]
        f_p99 = fifo_results[sl][full_key]["p99_ms"]
        pct   = (n_p99 - f_p99) / n_p99 * 100 if n_p99 > 0 else 0.0
        reductions.append(round(pct, 2))

    x      = np.arange(len(SEQ_LENGTHS))
    colours = ["#2ca02c" if r >= 0 else "#d62728" for r in reductions]
    bars = ax2.bar(x, reductions, color=colours, alpha=0.82, width=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, reductions):
        va  = "bottom" if val >= 0 else "top"
        off = 0.1 if val >= 0 else -0.1
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + off,
                 f"{val:+.1f}%", ha="center", va=va, fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in SEQ_LENGTHS])
    ax2.set_xlabel("Sequence Length (tokens)")
    ax2.set_ylabel("P99 Latency Reduction (%)")
    ax2.set_title("P99 Latency Reduction with SCHED_FIFO")

    fig.suptitle("SCHED_FIFO Analysis — TinyLlama-1.1B (RTX 4000 Ada)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


def plot_normal_only(normal_results, deadline_ms=45.0):
    """Single-panel figure when SCHED_FIFO is not available."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, exit_layer in enumerate(EXIT_LAYERS):
        key    = str(exit_layer)
        label  = f"L{exit_layer}" if exit_layer is not None else "Full(22)"
        values = [normal_results[sl][key]["wcet_ms"] for sl in SEQ_LENGTHS]
        ax.plot(SEQ_LENGTHS, values, marker="o", linewidth=2,
                color=colours[idx % len(colours)], label=label)

    ax.axhline(y=deadline_ms, color="red", linestyle="--", linewidth=1.2,
               label=f"Deadline ({deadline_ms:.0f} ms)")
    ax.set_xlabel("Input Sequence Length (tokens)")
    ax.set_ylabel("WCET / Max Latency (ms)")
    ax.set_title("WCET Profile — SCHED_OTHER (SCHED_FIFO unavailable; run with sudo)")
    ax.set_xticks(SEQ_LENGTHS)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}' (single-panel — SCHED_FIFO not available)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SCHED_FIFO WCET COMPARISON — TinyLlama-1.1B")
    print(f"Seq lengths : {SEQ_LENGTHS}")
    print(f"Exit layers : {EXIT_LAYERS}  (None = full 22-layer pass)")
    print(f"Runs/cell   : {NUM_RUNS} (+ {NUM_WARMUP} warmup)")
    print("=" * 60 + "\n")

    # ── Phase 1: Normal scheduling (SCHED_OTHER) ──────────────────────────────
    print("── Phase 1: SCHED_OTHER (default Linux scheduler) ───────────")
    model = EarlyExitTinyLlama()
    normal_results = run_sweep(model, label="[SCHED_OTHER]")
    print()

    # ── Phase 2: Attempt SCHED_FIFO elevation ─────────────────────────────────
    fifo_available = False
    fifo_results   = None

    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
        fifo_available = True
        print("── Phase 2: SCHED_FIFO (priority 99) ────────────────────────")
        fifo_results = run_sweep(model, label="[SCHED_FIFO]")
        print()
        # Restore normal scheduler after profiling
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        print("Restored SCHED_OTHER.\n")
    except PermissionError:
        print("\n[WARNING] SCHED_FIFO requires elevated privileges.")
        print("Run with: sudo python3 sched_fifo_profile.py")
        print("Continuing with SCHED_OTHER results only.\n")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "hardware":       "RTX 4000 Ada",
        "model":          "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "num_runs":       NUM_RUNS,
        "seq_lengths":    SEQ_LENGTHS,
        "exit_layers":    [str(e) for e in EXIT_LAYERS],
        "fifo_available": fifo_available,
        "normal_sched": {"results": normal_results},
    }
    if fifo_available:
        output["fifo_sched"] = {"results": fifo_results}
    else:
        output["fifo_sched"] = {
            "results": None,
            "note": "SCHED_FIFO was not available. Re-run with sudo to collect this data.",
        }

    # Ensure all seq_len keys are strings for JSON serialisation
    def stringify_keys(d):
        if isinstance(d, dict):
            return {str(k): stringify_keys(v) for k, v in d.items()}
        return d

    output["normal_sched"]["results"] = stringify_keys(output["normal_sched"]["results"])
    if fifo_available:
        output["fifo_sched"]["results"] = stringify_keys(output["fifo_sched"]["results"])

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to '{RESULTS_FILE}'")

    # ── Figure ─────────────────────────────────────────────────────────────────
    if fifo_available:
        plot_comparison(normal_results, fifo_results)

        # Print comparison table
        print("\n" + "=" * 70)
        print(f"{'Seq Len':>8}  {'Normal WCET':>12}  {'FIFO WCET':>12}  "
              f"{'Normal P99':>12}  {'FIFO P99':>12}  {'P99 Red%':>8}")
        print("-" * 70)
        full_key = "None"
        for sl in SEQ_LENGTHS:
            n = normal_results[sl][full_key]
            f = fifo_results[sl][full_key]
            red = (n["p99_ms"] - f["p99_ms"]) / n["p99_ms"] * 100
            print(f"{sl:>8}  {n['wcet_ms']:>12.2f}  {f['wcet_ms']:>12.2f}  "
                  f"{n['p99_ms']:>12.2f}  {f['p99_ms']:>12.2f}  {red:>+7.1f}%")
        print("=" * 70)
    else:
        plot_normal_only(normal_results)

        print("\n" + "=" * 55)
        print(f"{'Seq Len':>8}  {'Mean (ms)':>10}  {'P99 (ms)':>10}  {'WCET (ms)':>10}")
        print("-" * 45)
        full_key = "None"
        for sl in SEQ_LENGTHS:
            d = normal_results[sl][full_key]
            print(f"{sl:>8}  {d['mean_ms']:>10.2f}  {d['p99_ms']:>10.2f}  {d['wcet_ms']:>10.2f}")
        print("=" * 55)
