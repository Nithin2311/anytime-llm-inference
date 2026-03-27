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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_FILE = "benchmark_results.json"

# ── IEEE Paper Styling ────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "figure.titlesize": 14,
    "legend.fontsize":  10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

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

    fig, ax = plt.subplots(figsize=(9, 4))
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


# ── Figure 2: CDF of TPOT (all queries, skip TTFT per query) ─────────────────
def plot_tail_latency_cdf(data):
    deadline  = data["global_metrics"]["deadline_ms"]
    all_tpot  = [
        r["time_ms"]
        for q in data["query_results"]
        for r in q["token_records"][1:]     # skip first token (prefill) per query
    ]
    tpot_sorted = np.sort(all_tpot)
    cdf = np.arange(1, len(tpot_sorted) + 1) / len(tpot_sorted)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tpot_sorted, cdf, marker="o", markersize=3, linestyle="-",
            color=C_FULL, linewidth=2, label="Anytime Scheduler")
    ax.axvline(x=deadline, color=C_DEAD, linestyle="--", linewidth=1.5,
               label=f"Hard Deadline ({deadline:.0f} ms)")
    ax.axhline(y=0.99, color="grey", linestyle=":", linewidth=1.0, label="P99")

    ax.set_xlabel("Execution Time (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"CDF of Token Generation Latency (TPOT) — {len(data['query_results'])} Queries")
    ax.legend()

    p99 = np.percentile(tpot_sorted, 99)
    ax.annotate(f"P99={p99:.1f} ms", xy=(p99, 0.99),
                xytext=(p99 + 1, 0.85), fontsize=8,
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
    fig_w = max(8, n_queries * 0.35)   # scale width with query count
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    ax.bar(x, full_pct,   color=C_FULL,   edgecolor="black", linewidth=0.4, label="Full Pass")
    ax.bar(x, thresh_pct, color=C_THRESH, edgecolor="black", linewidth=0.4, label="Early (Thresh)",
           bottom=full_pct)
    bottom2 = [f + t for f, t in zip(full_pct, thresh_pct)]
    ax.bar(x, forced_pct, color=C_FORCED, edgecolor="black", linewidth=0.4, label="Early (Forced)",
           bottom=bottom2)

    ax.set_xticks(x)
    tick_fs = max(5, 9 - n_queries // 10)   # shrink font for many queries
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fs)
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

    fig = plt.figure(figsize=(12, 4.5))
    fig.suptitle(
        f"Anytime Scheduler — Benchmark Dashboard  |  D={deadline:.0f} ms  |  n={gm['n_queries']} queries",
        fontsize=13,
    )

    # ── Panel A: Accuracy & Compliance bars ─────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    cats   = ["Accuracy\n(scored)", "Deadline\nCompliance"]
    vals   = [accuracy, compliance]
    colors = ["#5b8db8", "#2e8b57"]
    bars   = ax1.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.7, width=0.5)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
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
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Token Latency (TPOT)")
    sched_str = f"SCHEDULABLE  (U={util_ratio:.3f})" if util_ratio < 1.0 else f"NOT SCHED.  (U={util_ratio:.3f})"
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
    table.set_fontsize(9.5)
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
