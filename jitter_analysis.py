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
DEADLINE_MS  = 45.0


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

    fig, axes = plt.subplots(1, 3, figsize=fs.TRIPLE)

    # ── Panel 1: Global TPOT distribution with jitter bands ───────────────────
    ax1 = axes[0]
    arr  = np.array(all_tpots)
    mean = stats["mean_ms"]
    std  = stats["std_ms"]
    p50  = stats["p50_ms"]
    p25  = stats["p25_ms"]
    p75  = stats["p75_ms"]
    p99  = stats["p99_ms"]

    ax1.hist(arr, bins=40, color="#4878d0", alpha=0.75, density=True, label="TPOT samples")

    # Jitter bands (mean ± 1σ, ± 2σ)
    for mult, alpha, label in [(1, 0.20, "μ ± 1σ"), (2, 0.10, "μ ± 2σ")]:
        ax1.axvspan(mean - mult * std, mean + mult * std, alpha=alpha,
                    color="#ee854a", label=label)

    ax1.axvline(mean, color="#d62728", linewidth=1.5, linestyle="-",  label=f"Mean {mean:.1f} ms")
    ax1.axvline(p99,  color="#9467bd", linewidth=1.5, linestyle="--", label=f"P99 {p99:.1f} ms")
    ax1.axvline(DEADLINE_MS, color="black", linewidth=1.5, linestyle=":", label=f"D={DEADLINE_MS:.0f} ms")

    ax1.set_xlabel("TPOT (ms)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"TPOT Distribution\nJitter = {stats['jitter_ms']:.2f} ms  |  CV = {stats['cv']:.4f}")
    ax1.legend(loc="upper right", fontsize=8)

    # Stats annotation
    ax1.text(0.03, 0.97,
             f"n={stats['n']}  σ={std:.2f} ms\nIQR={stats['iqr_ms']:.2f} ms",
             transform=ax1.transAxes, va="top", ha="left", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel 2: Autocorrelation function (ACF) ───────────────────────────────
    ax2 = axes[1]
    ax2.bar(acf_lags, acf_vals, color="#4878d0", alpha=0.8, width=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)

    # 95% confidence bands for white-noise hypothesis: ±1.96/√n
    n = len(all_tpots)
    ci = 1.96 / np.sqrt(n)
    ax2.axhline( ci, color="#d62728", linestyle="--", linewidth=1.2, label=f"95% CI (±{ci:.3f})")
    ax2.axhline(-ci, color="#d62728", linestyle="--", linewidth=1.2)

    ax2.set_xlabel("Lag (tokens)")
    ax2.set_ylabel("Autocorrelation  r")
    ax2.set_title("TPOT Autocorrelation Function (ACF)\n(serial dependence between consecutive tokens)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(-0.5, 1.0)
    ax2.set_xticks(acf_lags)

    # Interpretation note
    sig_lags = [l for l, r in zip(acf_lags, acf_vals) if abs(r) > ci]
    note = (f"Lag(s) outside CI: {sig_lags}" if sig_lags
            else "All lags within 95% CI\n→ no serial correlation")
    ax2.text(0.03, 0.03, note, transform=ax2.transAxes, va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel 3: Per-query TPOT box plot ──────────────────────────────────────
    ax3 = axes[2]
    bp = ax3.boxplot(
        per_query,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker=".", markersize=4, alpha=0.5),
        widths=0.55,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#4878d0")
        patch.set_alpha(0.70)

    ax3.axhline(DEADLINE_MS, color="black", linestyle=":", linewidth=1.2,
                label=f"D = {DEADLINE_MS:.0f} ms")
    ax3.set_xticks(range(1, len(query_ids) + 1))
    ax3.set_xticklabels([str(q) for q in query_ids], rotation=45, fontsize=7)
    ax3.set_xlabel("Query ID")
    ax3.set_ylabel("TPOT (ms)")
    ax3.set_title("Per-Query TPOT Distribution\n(box = IQR, whiskers = 1.5×IQR)")
    ax3.legend(loc="upper right", fontsize=8)

    fig.suptitle("Jitter Analysis — KV-Cached Anytime Scheduler (30-query PubMedQA Benchmark)",
                 fontsize=7.5, y=1.01)
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
