"""
task_model_analysis.py — Formal real-time task model for LLM inference.

Frames token generation in the standard real-time scheduling formalism and
derives capacity/schedulability results for single and multi-request scenarios.

Task model
----------
Each token generation is a sporadic task τ:
  C  = WCET (measured P99 TPOT from benchmark) = 19.44 ms
  D  = hard deadline                            = 45 ms
  T  = minimum inter-token interval ≥ C         (self-imposed by serialised GPU)

Single-task utilisation:
  U = C / D = 0.432  < 1.0  →  SCHEDULABLE

Multi-request capacity (round-robin at token boundaries)
---------------------------------------------------------
With N concurrent inference requests, the GPU serves one token per request
per round under a token-level round-robin policy.  Each token of any given
request experiences an effective TPOT of N × C:

  N_max = ⌊ D / C ⌋     (schedulability ceiling for round-robin)

Liu & Layland utilisation bound for n=N periodic tasks with equal periods:
  U_LL(N) = N × (2^(1/N) − 1)

Since our task set has C_1 = C_2 = … = C_N (homogeneous), the utilisation of
the full task set is U_total = N × (C/T).  With T = D (tight binding):
  U_total = N × (C/D) = N × 0.432

Schedulable under RMS iff U_total ≤ U_LL(N).

Outputs:
  task_model_results.json
  task_model_analysis.png
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_FILE    = "task_model_results.json"
FIGURE_FILE     = "task_model_analysis.png"

# Measured system parameters (from benchmark + schedulability proof)
C_MS            = 19.44    # P99 TPOT from benchmark (KV-cached scheduler)
D_MS            = 45.0     # hard deadline
WCET_EMPIRICAL  = 19.73    # empirical max TPOT (from tail_latency_results)


def liu_layland_bound(n):
    """Liu & Layland utilisation bound for n equal-period tasks under RMS."""
    if n == 1:
        return 1.0
    return n * (2 ** (1.0 / n) - 1.0)


def effective_tpot(n_concurrent, c_ms=C_MS):
    """Effective per-token latency seen by each request under N-way round-robin."""
    return n_concurrent * c_ms


def response_time_bound(k_tokens, n_concurrent, c_ms=C_MS):
    """
    End-to-end response time for a K-token response with N concurrent requests.
    Under round-robin: every token waits for N−1 other requests' tokens.
    Total response time = K × N × C.
    """
    return k_tokens * n_concurrent * c_ms


def rms_schedulable(n, c_ms=C_MS, d_ms=D_MS):
    """True iff N identical tasks (C=c_ms, T=D=d_ms) are schedulable under RMS."""
    u_total = n * (c_ms / d_ms)
    return u_total <= liu_layland_bound(n)


def plot_task_model(results):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    N_range    = np.arange(1, 8)
    deadlines  = np.linspace(15, 80, 200)

    # ── Panel 1: Effective TPOT vs N concurrent requests ─────────────────────
    ax1 = axes[0]
    eff_tpot = [effective_tpot(n) for n in N_range]
    u_total  = [n * C_MS / D_MS   for n in N_range]

    colour = ["#2ca02c" if effective_tpot(n) <= D_MS else "#d62728" for n in N_range]
    bars = ax1.bar(N_range, eff_tpot, color=colour, alpha=0.82, width=0.6)
    ax1.axhline(D_MS, color="black", linestyle="--", linewidth=1.5, label=f"D = {D_MS:.0f} ms")

    for bar, val, n in zip(bars, eff_tpot, N_range):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                 f"{val:.1f}\n(U={n*C_MS/D_MS:.2f})",
                 ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(N_range)
    ax1.set_xlabel("Number of Concurrent Requests (N)")
    ax1.set_ylabel("Effective TPOT per Request (ms)")
    ax1.set_title("Round-Robin Capacity\n(green = schedulable, red = exceeds deadline)")
    ax1.legend(loc="upper left")

    admit_patch  = mpatches.Patch(color="#2ca02c", alpha=0.82, label="Schedulable (TPOT ≤ D)")
    reject_patch = mpatches.Patch(color="#d62728", alpha=0.82, label="Not schedulable")
    ax1.legend(handles=[admit_patch, reject_patch], loc="upper left", fontsize=8)

    # ── Panel 2: Liu & Layland bound vs actual utilisation ────────────────────
    ax2 = axes[1]
    ll_bounds = [liu_layland_bound(n) for n in N_range]
    u_actuals = [n * C_MS / D_MS      for n in N_range]

    ax2.plot(N_range, ll_bounds, marker="s", linewidth=2, color="#ff7f0e",
             label="Liu & Layland bound U_LL(N)")
    ax2.plot(N_range, u_actuals, marker="o", linewidth=2, color="#1f77b4",
             label=f"Actual U = N × (C/D)  (C={C_MS:.1f} ms)")
    ax2.axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="U = 1.0")

    # Shade schedulable region
    ax2.fill_between(N_range,
                     np.minimum(ll_bounds, 1.0),
                     np.minimum(u_actuals, 1.5),
                     where=[u <= ll for u, ll in zip(u_actuals, ll_bounds)],
                     alpha=0.12, color="#2ca02c")

    # Mark crossover
    for n in N_range:
        if u_actuals[n - 1] > ll_bounds[n - 1]:
            ax2.axvline(n, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.6)
            ax2.text(n + 0.05, 0.92, f"N={n}\nexceeds LL", fontsize=8, color="#d62728")
            break

    ax2.set_xticks(N_range)
    ax2.set_xlabel("Number of Concurrent Requests (N)")
    ax2.set_ylabel("Utilisation")
    ax2.set_title("Liu & Layland RMS Bound\nvs Actual Utilisation")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(0, 1.3)

    # ── Panel 3: Max N_concurrent vs deadline ─────────────────────────────────
    ax3 = axes[1].twinx()
    # Keep panel 2's axes clean; use axes[2] for Panel 3
    ax3 = axes[2]

    # For each deadline, find max N such that N × C ≤ D
    max_n = [int(d / C_MS) for d in deadlines]

    ax3.plot(deadlines, max_n, linewidth=2.5, color="#9467bd",
             label=f"N_max = ⌊D / C⌋  (C={C_MS:.1f} ms)")
    ax3.axvline(D_MS, color="black", linestyle="--", linewidth=1.2,
                label=f"Reference D = {D_MS:.0f} ms")

    # Shade integer steps
    for n_val in range(1, 6):
        d_lo = n_val * C_MS
        d_hi = (n_val + 1) * C_MS
        ax3.fill_between(deadlines,
                         n_val - 0.4, n_val + 0.4,
                         where=[(d_lo <= d <= d_hi) for d in deadlines],
                         alpha=0.12, color="#9467bd")

    ax3.set_xlabel("Deadline D (ms)")
    ax3.set_ylabel("Maximum Concurrent Requests N_max")
    ax3.set_title("Admission Capacity vs Deadline\n(round-robin, token-level)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_yticks(range(0, 7))
    ax3.set_ylim(0, 6)

    # Reference annotation at D=45ms
    n_at_45 = int(D_MS / C_MS)
    ax3.scatter([D_MS], [n_at_45], s=80, color="#d62728", zorder=5)
    ax3.annotate(f"N_max={n_at_45}\nat D={D_MS:.0f}ms",
                 xy=(D_MS, n_at_45), xytext=(D_MS + 3, n_at_45 - 0.4),
                 fontsize=9, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728"))

    fig.suptitle(
        f"Formal Task Model — TinyLlama-1.1B  "
        f"(τ: C={C_MS:.2f} ms, D={D_MS:.0f} ms, U={C_MS/D_MS:.3f})",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    print("=" * 65)
    print("FORMAL TASK MODEL ANALYSIS — TinyLlama-1.1B KV-Cached Scheduler")
    print("=" * 65 + "\n")

    U_single = C_MS / D_MS
    N_max    = int(D_MS / C_MS)

    print(f"Task parameters:")
    print(f"  C (P99 TPOT)   = {C_MS:.2f} ms   (measured, KV-cached benchmark)")
    print(f"  C (empirical)  = {WCET_EMPIRICAL:.2f} ms   (absolute max over 240 tokens)")
    print(f"  D (deadline)   = {D_MS:.1f} ms")
    print(f"  U = C/D        = {U_single:.4f}   ({'< 1.0 → SCHEDULABLE' if U_single < 1 else '≥ 1.0 → NOT SCHEDULABLE'})\n")

    print("Liu & Layland analysis (N equal-rate concurrent requests):")
    print(f"  {'N':>4}  {'U_actual':>10}  {'U_LL(N)':>10}  {'Sched?':>8}  {'N×C (ms)':>10}")
    print("  " + "-" * 48)
    rms_results = []
    for n in range(1, 8):
        u_act = n * U_single
        u_ll  = liu_layland_bound(n)
        sched = u_act <= u_ll
        nC    = n * C_MS
        print(f"  {n:>4}  {u_act:>10.4f}  {u_ll:>10.4f}  {'YES' if sched else 'NO':>8}  {nC:>10.2f}")
        rms_results.append({
            "n": n, "u_actual": round(u_act, 6), "u_ll": round(u_ll, 6),
            "rms_schedulable": sched, "effective_tpot_ms": round(nC, 4),
        })

    print(f"\nMaximum concurrent requests (round-robin, D={D_MS:.0f}ms): N_max = {N_max}")
    print(f"At N_max={N_max}: effective TPOT = {N_max * C_MS:.2f} ms  ≤ D={D_MS:.0f} ms")
    print(f"At N={N_max+1}:   effective TPOT = {(N_max+1) * C_MS:.2f} ms  > D={D_MS:.0f} ms  ← exceeds deadline\n")

    # Response time for full responses
    print("End-to-end response time (K tokens, N concurrent, round-robin):")
    print(f"  {'K tokens':>10}  {'N=1 (ms)':>10}  {'N=2 (ms)':>10}  {'N=3 (ms)':>10}")
    print("  " + "-" * 44)
    response_times = []
    for k in [5, 10, 15, 25, 50]:
        row = {"k_tokens": k}
        vals = []
        for n in [1, 2, 3]:
            rt = response_time_bound(k, n)
            row[f"n{n}_ms"] = round(rt, 2)
            vals.append(rt)
        print(f"  {k:>10}  {vals[0]:>10.1f}  {vals[1]:>10.1f}  {vals[2]:>10.1f}")
        response_times.append(row)

    results = {
        "task_params":     {"C_ms": C_MS, "D_ms": D_MS, "U_single": round(U_single, 6)},
        "N_max_round_robin": N_max,
        "rms_analysis":    rms_results,
        "response_times":  response_times,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    plot_task_model(results)
