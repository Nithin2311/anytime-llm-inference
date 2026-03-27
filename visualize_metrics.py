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
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x, full_pct,   color=C_FULL,   edgecolor="black", linewidth=0.5, label="Full Pass")
    ax.bar(x, thresh_pct, color=C_THRESH, edgecolor="black", linewidth=0.5, label="Early (Thresh)",
           bottom=full_pct)
    bottom2 = [f + t for f, t in zip(full_pct, thresh_pct)]
    ax.bar(x, forced_pct, color=C_FORCED, edgecolor="black", linewidth=0.5, label="Early (Forced)",
           bottom=bottom2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage of Tokens (%)")
    ax.set_xlabel("Clinical Query")
    ax.set_title("Exit-Type Distribution per Clinical Query")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("exit_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved 'exit_distribution.png'")


# ── Figure 4: Accuracy & Schedulability Summary Bar ──────────────────────────
def plot_accuracy_summary(data):
    gm       = data["global_metrics"]
    deadline = gm["deadline_ms"]

    categories = ["Accuracy (%)", "Deadline\nCompliance (%)"]
    values = [
        gm["accuracy"] if gm["accuracy"] is not None else 0.0,
        100.0 - gm["deadline_miss_pct"],
    ]
    colors_bar = ["#5b8db8", "#2e8b57"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(categories, values, color=colors_bar, edgecolor="black", linewidth=0.7, width=0.5)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=100, color="grey", linestyle=":", linewidth=1.0)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Benchmark Summary  |  deadline={deadline:.0f} ms  |  n={gm['n_queries']} queries")

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
