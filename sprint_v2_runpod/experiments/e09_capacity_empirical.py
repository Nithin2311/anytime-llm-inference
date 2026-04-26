"""
e09_capacity_empirical.py — Empirical multi-request capacity measurement.

Validates the round-robin capacity model N_max = floor(D / P99).
Runs N=1,2,3,4 concurrent virtual requests under a round-robin token
schedule at D=45ms, measures per-token latency, miss rate, and throughput.
"""

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sprint_v2_runpod" / "src"))

import numpy as np
import torch
from fig_style import apply_style, SINGLE, DOUBLE
import matplotlib.pyplot as plt
from result_writer import write_results

apply_style()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
DEADLINE_MS    = 45.0
SEQ_LEN        = 128
N_TOKENS_EACH  = 30        # tokens per request
N_TRIALS       = 5         # repeats per N_concurrent
MAX_CONCURRENT = 4
DEVICE         = "cuda"


def run_round_robin(model, n_concurrent, seq_len, n_tokens_each, deadline_ms):
    """
    Simulate round-robin: n_concurrent requests, each getting n_tokens_each tokens.
    Returns per-token latency list, miss count, total time.
    """
    import warnings
    # Each request gets its own input sequence
    inputs = [
        torch.randint(100, 2000, (1, seq_len), device=DEVICE)
        for _ in range(n_concurrent)
    ]

    token_latencies = []
    misses = 0
    start_wall = time.time()

    # Round-robin: each "tick" serves one token per request in order
    for tick in range(n_tokens_each):
        for req_idx in range(n_concurrent):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with torch.inference_mode():
                    model.forward_cached(inputs[req_idx])
            end_evt.record()
            torch.cuda.synchronize()
            t_ms = start_evt.elapsed_time(end_evt)
            # Under round-robin, effective deadline per request = D/N_concurrent
            effective_deadline = deadline_ms / n_concurrent
            token_latencies.append(t_ms)
            if t_ms > effective_deadline:
                misses += 1

    total_time = time.time() - start_wall
    return token_latencies, misses, total_time


def main():
    print("=" * 60)
    print("E09  Empirical multi-request capacity")
    print(f"     D={DEADLINE_MS}ms, seq={SEQ_LEN}, {N_TOKENS_EACH} tokens/request")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    results = {}
    n_range = list(range(1, MAX_CONCURRENT + 1))

    for n in n_range:
        print(f"\n  N_concurrent={n}  (effective D/token = {DEADLINE_MS/n:.1f}ms)")
        trial_results = []
        for trial in range(N_TRIALS):
            lats, misses, wall = run_round_robin(
                model, n, SEQ_LEN, N_TOKENS_EACH, DEADLINE_MS
            )
            n_total = len(lats)
            trial_results.append({
                "trial": trial,
                "mean_ms":      round(float(np.mean(lats)), 3),
                "p99_ms":       round(float(np.percentile(lats, 99)), 3),
                "max_ms":       round(float(np.max(lats)), 3),
                "miss_count":   misses,
                "miss_rate_pct":round(100.0 * misses / n_total, 2),
                "throughput_tps":round(n_total / wall, 2),
                "wall_time_s":  round(wall, 2),
            })
            print(f"    trial {trial}: p99={trial_results[-1]['p99_ms']:.2f}ms "
                  f"miss={trial_results[-1]['miss_rate_pct']:.1f}% "
                  f"throughput={trial_results[-1]['throughput_tps']:.1f}t/s")

        all_lats = []
        for tr in trial_results:
            pass  # aggregated summary
        avg_miss  = np.mean([tr["miss_rate_pct"] for tr in trial_results])
        avg_p99   = np.mean([tr["p99_ms"]        for tr in trial_results])
        avg_tput  = np.mean([tr["throughput_tps"] for tr in trial_results])

        results[str(n)] = {
            "n_concurrent": n,
            "effective_deadline_ms": round(DEADLINE_MS / n, 2),
            "avg_p99_ms":   round(float(avg_p99), 3),
            "avg_miss_rate_pct": round(float(avg_miss), 2),
            "avg_throughput_tps": round(float(avg_tput), 2),
            "trials": trial_results,
            "schedulable": bool(avg_miss < 1.0),  # <1% miss = schedulable
        }
        print(f"    AVG: p99={avg_p99:.2f}ms  miss={avg_miss:.1f}%  "
              f"schedulable={'YES' if avg_miss < 1.0 else 'NO'}")

    # N_max: largest N with avg_miss < 1%
    n_max = max(
        (n for n in n_range if results[str(n)]["schedulable"]), default=0
    )
    # Theoretical prediction: floor(D / P99_single)
    p99_single = results["1"]["avg_p99_ms"]
    n_max_theory = int(DEADLINE_MS / p99_single)

    print(f"\nEmpirical N_max = {n_max}")
    print(f"Theory N_max    = floor({DEADLINE_MS}/{p99_single:.2f}) = {n_max_theory}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)

    miss_rates  = [results[str(n)]["avg_miss_rate_pct"] for n in n_range]
    throughputs = [results[str(n)]["avg_throughput_tps"]  for n in n_range]

    ax = axes[0]
    bars = ax.bar(n_range, miss_rates,
                  color=["tab:green" if r["schedulable"] else "tab:red"
                         for r in results.values()])
    ax.axhline(1.0, ls="--", color="black", lw=1.2, label="1% miss threshold")
    ax.set_xlabel("N concurrent requests")
    ax.set_ylabel("Average miss rate (%)")
    ax.set_title("Deadline miss rate vs. N")
    ax.legend(fontsize=8)
    ax.set_xticks(n_range)

    ax2 = axes[1]
    ax2.plot(n_range, throughputs, "s-", color="tab:blue")
    ax2.set_xlabel("N concurrent requests")
    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.set_title("Throughput vs. N")
    ax2.set_xticks(n_range)
    ax2.axvline(n_max, ls="--", color="tab:green", lw=1.2,
                label=f"N_max={n_max}")
    ax2.legend(fontsize=8)

    plt.suptitle(f"Round-Robin Capacity (D={DEADLINE_MS}ms, seq={SEQ_LEN})")
    plt.tight_layout()
    fig_path = RESULTS_DIR / "capacity_empirical.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved: {fig_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────
    tex_path = RESULTS_DIR / "table_capacity.tex"
    with open(tex_path, "w") as f:
        f.write("% E09: Empirical capacity\n")
        f.write("\\begin{tabular}{crrrrr}\n\\toprule\n")
        f.write("$N$ & Eff. deadline & Avg P99 & Miss rate & Throughput & Schedulable \\\\\n")
        f.write("& (ms) & (ms) & (\\%) & (t/s) & \\\\\n\\midrule\n")
        for n in n_range:
            r = results[str(n)]
            sched = "\\checkmark" if r["schedulable"] else "\\times"
            f.write(f"{n} & {r['effective_deadline_ms']:.1f} & "
                    f"{r['avg_p99_ms']:.2f} & {r['avg_miss_rate_pct']:.1f} & "
                    f"{r['avg_throughput_tps']:.1f} & {sched} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX: {tex_path}")

    write_results({
        "experiment": "e09_capacity_empirical",
        "deadline_ms": DEADLINE_MS,
        "seq_len": SEQ_LEN,
        "n_tokens_each": N_TOKENS_EACH,
        "n_trials": N_TRIALS,
        "n_max_empirical": n_max,
        "n_max_theory": n_max_theory,
        "p99_single_ms": round(float(p99_single), 3),
        "results": results,
    }, RESULTS_DIR / "capacity_empirical_results.json")
    print("PASS: E09 complete\n")


if __name__ == "__main__":
    main()
