"""
visualize_metrics.py — IEEE-styled figures from benchmark_results.json.

Produces four figures:
  1. execution_timeline.png    — per-token latency bar chart (Query 1)
  2. tail_latency_cdf.png      — CDF of TPOT across all queries
  3. exit_distribution.png     — stacked bar of exit types per query
  4. accuracy_summary.png      — accuracy vs. deadline miss rate summary
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_FILE = "benchmark_results.json"

# ── IEEE Paper Styling ────────────────────────────────────────────────────────
fs.apply()

# ── Colour palette ────────────────────────────────────────────────────────────
C_FULL   = "#2b5b84"   # dark blue  — Full Pass
C_THRESH = "#2e8b57"   # sea green  — Early (Thresh)
C_FORCED = "#d9534f"   # red        — Early (Forced) / TTFT
C_DEAD   = "red"       # deadline line


def load_results(path=RESULTS_FILE):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] '{path}' not found. Run benchmark.py first.")
        sys.exit(1)


def exit_color(exit_type):
    if exit_type == "Full Pass":
        return C_FULL
    if exit_type.startswith("Early (Thresh"):
        return C_THRESH
    return C_FORCED


# ── Figure 1: Token Execution Timeline (Query 1) ─────────────────────────────
def plot_execution_timeline(data):
    q1      = data["query_results"][0]
    records = q1["token_records"]
    deadline = data["global_metrics"]["deadline_ms"]

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=fs.DOUBLE)
    colors = [exit_color(et) for et in df["exit_type"]]
    ax.bar(df["token_idx"], df["time_ms"], color=colors, edgecolor="black", linewidth=0.5)

    ax.axhline(y=deadline, color=C_DEAD, linestyle="--", linewidth=1.5,
               label=f"Hard Deadline ({deadline:.0f} ms)")

    ax.set_xticks(df["token_idx"])
    ax.set_xticklabels(df["token"], rotation=45, ha="right")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_xlabel("Generated Token")
    ax.set_title(f"Dynamic Scheduler — Token Execution Timeline (Query 1)")

    legend_elements = [
        mpatches.Patch(facecolor=C_FULL,   edgecolor="black", label="Full Pass (22 layers)"),
        mpatches.Patch(facecolor=C_THRESH, edgecolor="black", label="Early Exit (Thresh)"),
        mpatches.Patch(facecolor=C_FORCED, edgecolor="black", label="Early Exit (Forced) / TTFT"),
        ax.lines[0],
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig("execution_timeline.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved 'execution_timeline.png'")


# ── Figure 2: CDF of TPOT with bootstrap P99 CI ─────────────────────────────
def _bootstrap_p99_ci(samples, n_boot=2000, ci=95):
    """Return (p99_lo, p99_hi) bootstrap confidence interval for P99."""
    rng    = np.random.default_rng(42)
    p99s   = [np.percentile(rng.choice(samples, size=len(samples), replace=True), 99)
              for _ in range(n_boot)]
    lo = np.percentile(p99s, (100 - ci) / 2)
    hi = np.percentile(p99s, 100 - (100 - ci) / 2)
    return lo, hi


def plot_tail_latency_cdf(data):
    deadline  = data["global_metrics"]["deadline_ms"]
    all_tpot  = [
        r["time_ms"]
        for q in data["query_results"]
        for r in q["token_records"][1:]     # skip first token (prefill) per query
    ]
    tpot_arr    = np.array(all_tpot)
    tpot_sorted = np.sort(tpot_arr)
    cdf = np.arange(1, len(tpot_sorted) + 1) / len(tpot_sorted)

    p99         = np.percentile(tpot_arr, 99)
    p99_lo, p99_hi = _bootstrap_p99_ci(tpot_arr)

    fig, ax = plt.subplots(figsize=fs.SINGLE)
    ax.plot(tpot_sorted, cdf, marker="o", markersize=3, linestyle="-",
            color=C_FULL, linewidth=2, label="Anytime Scheduler")
    ax.axvline(x=deadline, color=C_DEAD, linestyle="--", linewidth=1.5,
               label=f"Hard Deadline ({deadline:.0f} ms)")
    ax.axhline(y=0.99, color="grey", linestyle=":", linewidth=1.0, label="P99")

    # Bootstrap CI band on P99
    ax.axvspan(p99_lo, p99_hi, alpha=0.18, color="#2b5b84",
               label=f"P99 95% CI [{p99_lo:.1f}, {p99_hi:.1f}] ms")
    ax.axvline(x=p99, color="#2b5b84", linestyle=":", linewidth=1.2)

    ax.set_xlabel("Execution Time (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"CDF of Token Generation Latency (TPOT) — {len(data['query_results'])} Queries")
    ax.legend(fontsize=8)

    ax.annotate(f"P99={p99:.1f} ms", xy=(p99, 0.99),
                xytext=(p99 + 1, 0.82), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="grey"))

    plt.tight_layout()
    plt.savefig("tail_latency_cdf.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved 'tail_latency_cdf.png'")


# ── Figure 3: Exit-Type Distribution per Query ───────────────────────────────
def plot_exit_distribution(data):
    queries  = data["query_results"]
    labels   = [f"Q{q['query_id']}" for q in queries]
    full_pct   = [q["metrics"]["full_pass_pct"]    for q in queries]
    thresh_pct = [q["metrics"]["early_thresh_pct"] for q in queries]
    forced_pct = [q["metrics"]["forced_exit_pct"]  for q in queries]

    x = np.arange(len(labels))
    n_queries = len(labels)
    fig, ax = plt.subplots(figsize=fs.DOUBLE)

    ax.bar(x, full_pct,   color=C_FULL,   edgecolor="black", linewidth=0.4, label="Full Pass")
    ax.bar(x, thresh_pct, color=C_THRESH, edgecolor="black", linewidth=0.4, label="Early (Thresh)",
           bottom=full_pct)
    bottom2 = [f + t for f, t in zip(full_pct, thresh_pct)]
    ax.bar(x, forced_pct, color=C_FORCED, edgecolor="black", linewidth=0.4, label="Early (Forced)",
           bottom=bottom2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Percentage of Tokens (%)")
    ax.set_xlabel("Clinical Query")
    ax.set_title(f"Exit-Type Distribution per Clinical Query  (n={n_queries})")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("exit_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved 'exit_distribution.png'")


# ── Figure 4: Comprehensive Metrics Dashboard ─────────────────────────────────
def plot_accuracy_summary(data):
    gm       = data["global_metrics"]
    deadline = gm["deadline_ms"]

    accuracy      = gm["accuracy"] if gm["accuracy"] is not None else 0.0
    compliance    = 100.0 - gm["deadline_miss_pct"]
    throughput    = gm.get("throughput_tps") or round(1000.0 / gm["global_mean_tpot_ms"], 2)
    util_ratio    = gm.get("util_ratio") or round(gm["global_p99_tpot_ms"] / deadline, 4)
    mean_tpot     = gm["global_mean_tpot_ms"]
    p99_tpot      = gm["global_p99_tpot_ms"]

    fig = plt.figure(figsize=fs.TRIPLE)
    fig.suptitle(
        f"Anytime Scheduler — Benchmark Dashboard  |  D={deadline:.0f} ms  |  n={gm['n_queries']} queries",
        fontsize=8,
    )

    # ── Panel A: Accuracy & Compliance bars ─────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    cats   = ["Accuracy\n(scored)", "Deadline\nCompliance"]
    vals   = [accuracy, compliance]
    colors = ["#5b8db8", "#2e8b57"]
    bars   = ax1.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.7, width=0.5)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax1.axhline(y=100, color="grey", linestyle=":", linewidth=1.0)
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Quality & Compliance")
    ax1.text(0.5, -0.18,
             f"Scored: {gm['n_scored']}/{gm['n_queries']}  |  Misses: {gm['deadline_miss_pct']}%",
             ha="center", transform=ax1.transAxes, fontsize=8, color="grey")

    # ── Panel B: TPOT latency summary ───────────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    tpot_cats  = ["Mean\nTPOT", "P99\nTPOT", "Deadline"]
    tpot_vals  = [mean_tpot, p99_tpot, deadline]
    tpot_colors= ["#5b8db8", "#2b5b84", "#d9534f"]
    bars2 = ax2.bar(tpot_cats, tpot_vals, color=tpot_colors, edgecolor="black", linewidth=0.7, width=0.5)
    for bar, val in zip(bars2, tpot_vals):
        offset = 1.5 if val > 5 else 0.3
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax2.set_ylim(0, max(tpot_vals) * 1.25)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Token Latency (TPOT)")
    sched_str = f"SLO MET  (R={util_ratio:.3f})" if util_ratio < 1.0 else f"SLO MISS  (R={util_ratio:.3f})"
    ax2.text(0.5, -0.18, sched_str,
             ha="center", transform=ax2.transAxes, fontsize=8,
             color="#2e8b57" if util_ratio < 1.0 else "#d9534f", fontweight="bold")

    # ── Panel C: Exit distribution + throughput ─────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    rows = [
        ["Exit: Full Pass",   f"{gm['full_pass_pct']}%"],
        ["Exit: Thresh",      f"{gm['early_thresh_pct']}%"],
        ["Exit: Forced",      f"{gm['forced_exit_pct']}%"],
        ["Throughput",        f"{throughput} tok/s"],
        ["Util (P99/D)",      f"{util_ratio:.4f}"],
        ["Schedulable?",      "YES" if util_ratio < 1.0 else "NO"],
    ]
    table = ax3.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="center", loc="center",
        bbox=[0.05, 0.0, 0.95, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for j in range(2):
        table[0, j].set_facecolor("#2b5b84")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Colour-code schedulable row
    sched_row = len(rows)
    sched_color = "#2e8b57" if util_ratio < 1.0 else "#d9534f"
    table[sched_row, 1].set_facecolor(sched_color)
    table[sched_row, 1].set_text_props(color="white", fontweight="bold")
    ax3.set_title("Scheduling Metrics", pad=10)

    plt.tight_layout()
    plt.savefig("accuracy_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved 'accuracy_summary.png'")


if __name__ == "__main__":
    data = load_results()
    gm   = data["global_metrics"]

    print(f"\nLoaded results: {gm['n_queries']} queries | deadline={gm['deadline_ms']} ms")
    print(f"Accuracy: {gm['accuracy']}% | P99 TPOT: {gm['global_p99_tpot_ms']} ms | "
          f"Deadline misses: {gm['deadline_miss_pct']}%\n")

    plot_execution_timeline(data)
    plot_tail_latency_cdf(data)
    plot_exit_distribution(data)
    plot_accuracy_summary(data)

    print("\nAll figures saved.")
