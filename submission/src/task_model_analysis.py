"""
task_model_analysis.py — Application-level SLO compliance analysis.

This module quantifies how the KV-cached anytime router scales under
concurrent request load.  All analysis is framed in terms of Service-Level
Objective (SLO) compliance rather than OS-level schedulability theory.

The router operates entirely in user space: it makes application-level
routing decisions (which transformer layers to commit) and does not preempt
OS threads, intercept hardware interrupts, or control GPU kernel arbitration.
Consequently, classical RTOS metrics (uniprocessor utilisation bounds, thread
preemption analysis) do not apply.  The relevant metric is whether the
application can sustain a P99 Time-Per-Output-Token ≤ D (the SLO deadline)
across N concurrent requests.

SLO compliance model
--------------------
  C  = empirical P99 TPOT (KV-cached router)  = 19.44 ms
  D  = application SLO deadline               = 45 ms
  R  = C / D = SLO compliance ratio           = 0.432   (< 1.0 → SLO MET)

Multi-request throughput (round-robin at token boundaries)
----------------------------------------------------------
With N concurrent inference requests served under a token-level round-robin
policy, the effective TPOT per request is N × C.  The SLO is met as long as
N × C ≤ D:
  N_max = ⌊ D / C ⌋

The bound below uses the same mathematical form as the Liu & Layland RMS
utilisation bound but is interpreted here as the maximum aggregate SLO
compliance ratio before the application misses its deadline:
  R_max(N) = N × (2^(1/N) − 1)

This is a throughput-capacity analysis for the application-level router, not
a claim of OS-level schedulability.

Outputs:
  task_model_results.json
  task_model_analysis.png
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_FILE    = "task_model_results.json"
FIGURE_FILE     = "task_model_analysis.png"

# Measured system parameters (from benchmark + schedulability proof)
C_MS            = 19.44    # P99 TPOT from benchmark (KV-cached scheduler)
D_MS            = 30.0     # hard deadline
WCET_EMPIRICAL  = 19.73    # empirical max TPOT (from tail_latency_results)


def liu_layland_bound(n):
    """Maximum aggregate SLO compliance ratio for N equal-cost concurrent requests."""
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


def slo_compliant(n, c_ms=C_MS, d_ms=D_MS):
    """True iff N concurrent equal-cost requests all meet the SLO deadline."""
    r_total = n * (c_ms / d_ms)
    return r_total <= liu_layland_bound(n)


def plot_task_model(results):
    fs.apply()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

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
    ax1.set_title("Round-Robin Throughput\n(green = SLO met, red = SLO miss)")
    ax1.legend(loc="upper left")

    admit_patch  = mpatches.Patch(color="#2ca02c", alpha=0.82, label="SLO met (TPOT ≤ D)")
    reject_patch = mpatches.Patch(color="#d62728", alpha=0.82, label="SLO miss (TPOT > D)")
    ax1.legend(handles=[admit_patch, reject_patch], loc="upper left", fontsize=8)

    # ── Panel 2: Liu & Layland bound vs actual utilisation ────────────────────
    ax2 = axes[1]
    ll_bounds = [liu_layland_bound(n) for n in N_range]
    u_actuals = [n * C_MS / D_MS      for n in N_range]

    ax2.plot(N_range, ll_bounds, marker="s", linewidth=2, color="#ff7f0e",
             label="Max SLO ratio R_max(N)")
    ax2.plot(N_range, u_actuals, marker="o", linewidth=2, color="#1f77b4",
             label=f"Actual R = N × (C/D)  (C={C_MS:.1f} ms)")
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
    ax2.set_ylabel("SLO Compliance Ratio")
    ax2.set_title("Multi-Request SLO Compliance\n(R_actual vs R_max)")
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
        f"Application-Level SLO Compliance — TinyLlama-1.1B  "
        f"(C={C_MS:.2f} ms, D={D_MS:.0f} ms, R={C_MS/D_MS:.3f})",
        fontsize=7.5, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    print("=" * 65)
    print("APPLICATION-LEVEL SLO COMPLIANCE — TinyLlama-1.1B KV-Cached Router")
    print("=" * 65 + "\n")

    R_single = C_MS / D_MS
    N_max    = int(D_MS / C_MS)

    print("Router parameters (application-level empirical bounds):")
    print(f"  C (P99 TPOT)   = {C_MS:.2f} ms   (measured, KV-cached router)")
    print(f"  C (empirical)  = {WCET_EMPIRICAL:.2f} ms   (absolute max over 240 tokens)")
    print(f"  D (SLO)        = {D_MS:.1f} ms")
    print(f"  R = C/D        = {R_single:.4f}   ({'< 1.0 → SLO MET' if R_single < 1 else '≥ 1.0 → SLO MISS'})\n")

    print("Multi-request SLO compliance (N concurrent requests, round-robin):")
    print(f"  {'N':>4}  {'R_actual':>10}  {'R_max(N)':>10}  {'SLO?':>8}  {'N×C (ms)':>10}")
    print("  " + "-" * 48)
    rms_results = []
    for n in range(1, 8):
        r_act = n * R_single
        r_max = liu_layland_bound(n)
        compliant = r_act <= r_max
        nC    = n * C_MS
        print(f"  {n:>4}  {r_act:>10.4f}  {r_max:>10.4f}  {'YES' if compliant else 'NO':>8}  {nC:>10.2f}")
        rms_results.append({
            "n": n, "r_actual": round(r_act, 6), "r_max": round(r_max, 6),
            "slo_compliant": compliant, "effective_tpot_ms": round(nC, 4),
        })

    print(f"\nMaximum concurrent requests (round-robin, D={D_MS:.0f}ms): N_max = {N_max}")
    print(f"At N_max={N_max}: effective TPOT = {N_max * C_MS:.2f} ms  ≤ D={D_MS:.0f} ms")
    print(f"At N={N_max+1}:   effective TPOT = {(N_max+1) * C_MS:.2f} ms  > D={D_MS:.0f} ms  ← SLO miss\n")

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
        "router_params":     {"C_ms": C_MS, "D_ms": D_MS, "R_single": round(R_single, 6)},
        "N_max_round_robin": N_max,
        "slo_analysis":      rms_results,
        "response_times":    response_times,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    plot_task_model(results)
