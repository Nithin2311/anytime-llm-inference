"""
compare_schedulers.py — Static vs. Dynamic scheduler head-to-head comparison.

Runs both schedulers on the same set of PubMedQA prompts and produces:
  scheduler_comparison.json   — raw per-token records for both schedulers
  scheduler_comparison.png    — side-by-side CDF + exit-distribution figures
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset

from early_exit_model import EarlyExitTinyLlama
from static_scheduler import generate_with_deadline
from dynamic_scheduler import generate_stateless_anytime

RESULTS_FILE = "scheduler_comparison.json"
FIGURE_FILE  = "scheduler_comparison.png"

N_SAMPLES    = 5
DEADLINE_MS  = 45.0
MAX_TOKENS   = 15


def aggregate(all_records, deadline_ms):
    """Compute summary metrics from a flat list of token_records."""
    tpot = [r["time_ms"] for r in all_records[1::MAX_TOKENS + 1] or all_records]
    # Proper TPOT: skip index 0 (TTFT) within each query
    tpot_records = []
    for i, r in enumerate(all_records):
        if r["token_idx"] > 1:          # skip first token per query
            tpot_records.append(r["time_ms"])

    tpot_arr  = np.array(tpot_records) if tpot_records else np.array([0.0])
    mean_tpot = float(np.mean(tpot_arr))
    p99_tpot  = float(np.percentile(tpot_arr, 99))
    n = len(all_records)
    return {
        "n_tokens":          n,
        "full_pass_pct":     round(100 * sum(1 for r in all_records if r["exit_type"] == "Full Pass") / n, 1),
        "early_conf_pct":    round(100 * sum(1 for r in all_records
                                             if "High Conf" in r["exit_type"]
                                             or "Thresh" in r["exit_type"]) / n, 1),
        "forced_exit_pct":   round(100 * sum(1 for r in all_records
                                             if "Forced" in r["exit_type"]
                                             or "Deadline" in r["exit_type"]) / n, 1),
        "deadline_miss_pct": round(100 * sum(1 for r in all_records if r["time_ms"] > deadline_ms) / n, 1),
        "mean_tpot_ms":      round(mean_tpot, 3),
        "p99_tpot_ms":       round(p99_tpot, 3),
        "throughput_tps":    round(1000.0 / mean_tpot, 2) if mean_tpot > 0 else None,
        "util_ratio":        round(p99_tpot / deadline_ms, 4),
        "tpot_samples":      tpot_arr.tolist(),
    }


def run_comparison():
    print("Loading PubMedQA Dataset...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split=f"train[:{N_SAMPLES}]")

    print("Loading model (shared for both schedulers)...")
    model = EarlyExitTinyLlama()

    static_all_records  = []
    dynamic_all_records = []

    for i, item in enumerate(dataset):
        context  = item["context"]["contexts"][0]
        question = item["question"]
        prompt   = f"Context: {context}\nQuestion: {question}\nAnswer:"

        print(f"\n{'='*60}")
        print(f"Query {i+1}/{N_SAMPLES} | GT: {item['final_decision']}")
        print("=" * 60)

        print("\n--- Static Scheduler (Layer 5, fixed threshold=0.8) ---")
        static_records = generate_with_deadline(
            model, prompt,
            max_new_tokens=MAX_TOKENS,
            deadline_ms=DEADLINE_MS,
            conf_threshold=0.8,
        )
        static_all_records.extend(static_records)

        print("\n--- Dynamic Scheduler (Layer 16, threshold decay 0.8→0.3) ---")
        dynamic_records = generate_stateless_anytime(
            model, prompt,
            max_new_tokens=MAX_TOKENS,
            deadline_ms=DEADLINE_MS,
        )
        dynamic_all_records.extend(dynamic_records)

    static_metrics  = aggregate(static_all_records,  DEADLINE_MS)
    dynamic_metrics = aggregate(dynamic_all_records, DEADLINE_MS)

    output = {
        "n_samples":       N_SAMPLES,
        "deadline_ms":     DEADLINE_MS,
        "max_tokens":      MAX_TOKENS,
        "static_metrics":  static_metrics,
        "dynamic_metrics": dynamic_metrics,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    return static_metrics, dynamic_metrics, static_all_records, dynamic_all_records


def plot_comparison(static_metrics, dynamic_metrics):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    C_STATIC  = "#d62728"   # red
    C_DYNAMIC = "#2b5b84"   # dark blue
    DEAD_C    = "black"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ── Panel 1: TPOT CDF ─────────────────────────────────────────────────────
    ax = axes[0]
    for label, metrics, colour in [
        ("Static (L5, fixed)", static_metrics,  C_STATIC),
        ("Dynamic (L16, decay)", dynamic_metrics, C_DYNAMIC),
    ]:
        samples = np.sort(metrics["tpot_samples"])
        cdf     = np.arange(1, len(samples) + 1) / len(samples)
        ax.plot(samples, cdf, linewidth=2, color=colour, label=label)

    ax.axvline(x=DEADLINE_MS, color=DEAD_C, linestyle="--", linewidth=1.2,
               label=f"Deadline ({DEADLINE_MS:.0f} ms)")
    ax.axhline(y=0.99, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Token Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("TPOT CDF")
    ax.legend(fontsize=8)

    # ── Panel 2: Exit-Type Distribution ───────────────────────────────────────
    ax      = axes[1]
    labels  = ["Static\n(L5, fixed)", "Dynamic\n(L16, decay)"]
    metrics_list = [static_metrics, dynamic_metrics]
    x       = np.arange(len(labels))
    width   = 0.5
    colours_bar = {"Full Pass": "#2b5b84", "Early Conf/Thresh": "#2e8b57", "Forced/Deadline": "#d9534f"}

    full_vals   = [m["full_pass_pct"]   for m in metrics_list]
    conf_vals   = [m["early_conf_pct"]  for m in metrics_list]
    forced_vals = [m["forced_exit_pct"] for m in metrics_list]

    ax.bar(x, full_vals,   width, color="#2b5b84", edgecolor="black", linewidth=0.5, label="Full Pass")
    ax.bar(x, conf_vals,   width, color="#2e8b57", edgecolor="black", linewidth=0.5, label="Early (Conf/Thresh)",
           bottom=full_vals)
    bottom2 = [f + c for f, c in zip(full_vals, conf_vals)]
    ax.bar(x, forced_vals, width, color="#d9534f", edgecolor="black", linewidth=0.5, label="Forced/Deadline",
           bottom=bottom2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage of Tokens (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Exit-Type Distribution")
    ax.legend(fontsize=8)

    # ── Panel 3: Key Metric Table ──────────────────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    col_labels = ["Metric", "Static", "Dynamic"]
    rows = [
        ["Mean TPOT (ms)",    f"{static_metrics['mean_tpot_ms']:.1f}",  f"{dynamic_metrics['mean_tpot_ms']:.1f}"],
        ["P99 TPOT (ms)",     f"{static_metrics['p99_tpot_ms']:.1f}",   f"{dynamic_metrics['p99_tpot_ms']:.1f}"],
        ["Throughput (tok/s)",f"{static_metrics['throughput_tps']}",    f"{dynamic_metrics['throughput_tps']}"],
        ["Util (P99/D)",      f"{static_metrics['util_ratio']:.4f}",    f"{dynamic_metrics['util_ratio']:.4f}"],
        ["Deadline Miss (%)", f"{static_metrics['deadline_miss_pct']}", f"{dynamic_metrics['deadline_miss_pct']}"],
        ["Full Pass (%)",     f"{static_metrics['full_pass_pct']}",     f"{dynamic_metrics['full_pass_pct']}"],
        ["Early Exit (%)",    f"{static_metrics['early_conf_pct']}",    f"{dynamic_metrics['early_conf_pct']}"],
        ["Forced Exit (%)",   f"{static_metrics['forced_exit_pct']}",   f"{dynamic_metrics['forced_exit_pct']}"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.05, 1, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2b5b84")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary Metrics", pad=12)

    fig.suptitle(
        f"Static vs. Dynamic Scheduler  |  deadline={DEADLINE_MS:.0f} ms  |  n={N_SAMPLES} queries",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    static_metrics, dynamic_metrics, static_records, dynamic_records = run_comparison()

    print("\n" + "=" * 55)
    print("COMPARISON SUMMARY")
    print("=" * 55)
    for name, m in [("Static  (L5, fixed thresh)", static_metrics),
                    ("Dynamic (L16, decay thresh)", dynamic_metrics)]:
        print(f"\n{name}")
        print(f"  Mean TPOT:    {m['mean_tpot_ms']:.2f} ms  |  P99: {m['p99_tpot_ms']:.2f} ms")
        print(f"  Throughput:   {m['throughput_tps']} tok/s  |  Util (P99/D): {m['util_ratio']:.4f}")
        print(f"  Full Pass:    {m['full_pass_pct']}%  |  Early: {m['early_conf_pct']}%  "
              f"|  Forced: {m['forced_exit_pct']}%")
        print(f"  Missed deadlines: {m['deadline_miss_pct']}%")

    plot_comparison(static_metrics, dynamic_metrics)
