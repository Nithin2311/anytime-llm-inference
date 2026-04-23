"""
jitter_analysis.py — Per-token latency jitter characterisation.

In real-time systems, jitter (temporal variability) is as critical as
worst-case latency.  A system with low mean but high jitter may still miss
deadlines unpredictably.  This script formally characterises the jitter
properties of the KV-cached anytime scheduler using the 30-query benchmark.

Metrics computed:
  - Jitter         : max_TPOT − min_TPOT  (absolute temporal spread)
  - Std deviation  : σ of TPOT distribution
  - CV             : σ / mean  (dimensionless, comparable across schedulers)
  - IQR            : P75 − P25  (robust spread)
  - Autocorrelation: Pearson r between TPOT[t] and TPOT[t+k]  (serial dependence)

If autocorrelation at lag-1 is high, successive tokens are correlated (e.g., due
to GPU thermal state or L2-cache effects), meaning the WCET guarantee for a
single token is insufficient — the system needs a response-time bound over a
window of tokens.

Requires: benchmark_results.json  (from benchmark.py)
          tail_latency_results.json (from evaluate_tail_latency.py)

Outputs:
  jitter_analysis_results.json
  jitter_analysis.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

RESULTS_FILE = "jitter_analysis_results.json"
FIGURE_FILE  = "jitter_analysis.png"
DEADLINE_MS  = 30.0


def load_tpot_series(bm_file="benchmark_results.json"):
    """
    Extract per-token TPOT series from benchmark_results.json.

    Returns:
        all_tpots   : flat list of all TPOT values (TTFT excluded)
        per_query   : list of per-query TPOT lists
        query_ids   : list of query IDs (1-based)
    """
    with open(bm_file) as f:
        data = json.load(f)

    per_query = []
    query_ids = []
    for qr in data["query_results"]:
        records = qr["token_records"]
        # Exclude TTFT (token index 1 = first token = full prompt encode)
        tpots = [r["time_ms"] for r in records if r["token_idx"] > 1]
        if tpots:
            per_query.append(tpots)
            query_ids.append(qr["query_id"])

    all_tpots = [t for q in per_query for t in q]
    return all_tpots, per_query, query_ids


def autocorrelation(series, max_lag=10):
    """
    Pearson autocorrelation of series at lags 1..max_lag.
    Returns (lags, acf_values).
    """
    arr  = np.array(series)
    mean = arr.mean()
    var  = arr.var()
    lags = list(range(1, max_lag + 1))
    acf  = []
    for lag in lags:
        if len(arr) > lag:
            r, _ = pearsonr(arr[:-lag], arr[lag:])
        else:
            r = 0.0
        acf.append(r)
    return lags, acf


def compute_jitter_stats(series):
    arr  = np.array(series)
    return {
        "n":        int(len(arr)),
        "mean_ms":  round(float(arr.mean()),  4),
        "std_ms":   round(float(arr.std()),   4),
        "cv":       round(float(arr.std() / arr.mean()), 4) if arr.mean() > 0 else 0.0,
        "p25_ms":   round(float(np.percentile(arr, 25)), 4),
        "p50_ms":   round(float(np.percentile(arr, 50)), 4),
        "p75_ms":   round(float(np.percentile(arr, 75)), 4),
        "p99_ms":   round(float(np.percentile(arr, 99)), 4),
        "max_ms":   round(float(arr.max()),   4),
        "min_ms":   round(float(arr.min()),   4),
        "jitter_ms":round(float(arr.max() - arr.min()), 4),
        "iqr_ms":   round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 4),
    }


def plot_jitter(all_tpots, per_query, query_ids, stats, acf_lags, acf_vals):
    fs.apply()

    arr  = np.array(all_tpots)
    mean = stats["mean_ms"]
    p50  = stats["p50_ms"]
    p95  = round(float(np.percentile(arr, 95)), 3)
    p99  = stats["p99_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel 1: TPOT over token sequence (temporal stability) ────────────────
    ax1 = axes[0]
    indices = np.arange(1, len(arr) + 1)
    ax1.scatter(indices, arr, s=6, alpha=0.40, color="#4878d0", label="Per-token TPOT")

    win = min(15, len(arr) // 5)
    if win > 1:
        roll = np.convolve(arr, np.ones(win) / win, mode="valid")
        ax1.plot(indices[win - 1:], roll, linewidth=2.0, color="#d62728",
                 label=f"Rolling mean (w={win})")

    ax1.axhline(DEADLINE_MS, color="black", linestyle="--", linewidth=1.2,
                label=f"D = {DEADLINE_MS:.0f} ms")
    ax1.axhline(mean, color="#ff7f0e", linestyle=":", linewidth=1.2,
                label=f"Mean = {mean:.1f} ms")

    ax1.set_xlabel("Token index (sequential)")
    ax1.set_ylabel("TPOT (ms)")
    ax1.set_title("TPOT Temporal Stability\n(flat rolling mean → no serial drift)")
    ax1.legend(loc="upper right", fontsize=8.5)

    # ── Panel 2: Histogram with percentile markers ────────────────────────────
    ax2 = axes[1]
    ax2.hist(arr, bins=40, color="#4878d0", alpha=0.75, density=True, label="TPOT samples")

    for val, lbl, col, ls in [
        (p50,        f"P50 = {p50:.1f} ms",    "#2ca02c", "-"),
        (p95,        f"P95 = {p95:.1f} ms",    "#ff7f0e", "--"),
        (p99,        f"P99 = {p99:.1f} ms",    "#9467bd", "--"),
        (DEADLINE_MS, f"D = {DEADLINE_MS:.0f} ms", "black",   ":"),
    ]:
        ax2.axvline(val, linewidth=1.8, linestyle=ls, color=col, label=lbl)

    ax2.set_xlabel("TPOT (ms)")
    ax2.set_ylabel("Density")
    ax2.set_title(
        f"TPOT Distribution\n"
        f"Jitter = {stats['jitter_ms']:.2f} ms  |  CV = {stats['cv']:.4f}"
    )
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Jitter Analysis — KV-Cached Anytime Scheduler (30-query PubMedQA Benchmark)",
        fontsize=8, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    print("=" * 60)
    print("JITTER ANALYSIS — KV-Cached Scheduler (benchmark data)")
    print("=" * 60 + "\n")

    all_tpots, per_query, query_ids = load_tpot_series()
    print(f"Loaded {len(all_tpots)} TPOT samples from {len(per_query)} queries.\n")

    stats     = compute_jitter_stats(all_tpots)
    acf_lags, acf_vals = autocorrelation(all_tpots, max_lag=10)

    print("Global TPOT statistics:")
    for k, v in stats.items():
        print(f"  {k:<15}: {v}")

    print("\nAutocorrelation at lag k:")
    n = len(all_tpots)
    ci = 1.96 / np.sqrt(n)
    for lag, r in zip(acf_lags, acf_vals):
        flag = "  *significant*" if abs(r) > ci else ""
        print(f"  lag={lag:>2}: r={r:>+.4f}{flag}")
    print(f"  (95% CI threshold: ±{ci:.4f}  n={n})")

    output = {
        "n_queries":     len(per_query),
        "n_tpot_samples": len(all_tpots),
        "deadline_ms":   DEADLINE_MS,
        "global_stats":  stats,
        "acf": {
            "lags":      acf_lags,
            "values":    [round(r, 6) for r in acf_vals],
            "ci_95":     round(ci, 6),
        },
        "per_query_stats": [
            {**{"query_id": qid}, **compute_jitter_stats(tpots)}
            for qid, tpots in zip(query_ids, per_query)
        ],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    plot_jitter(all_tpots, per_query, query_ids, stats, acf_lags, acf_vals)
