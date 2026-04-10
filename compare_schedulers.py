"""
compare_schedulers.py — Stateless vs. KV-Cached scheduler head-to-head comparison.

Runs both schedulers on the same set of PubMedQA prompts and produces:
  scheduler_comparison.json   — raw per-token records for both schedulers
  scheduler_comparison.png    — side-by-side CDF + exit-distribution figures
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import generate_stateless_anytime, generate_anytime_with_kv

RESULTS_FILE = "scheduler_comparison.json"
FIGURE_FILE  = "scheduler_comparison.png"

N_SAMPLES    = 5
DEADLINE_MS  = 45.0
MAX_TOKENS   = 15


def aggregate(all_records, deadline_ms):
    """Compute summary metrics from a flat list of token_records."""
    tpot_records = []
    for r in all_records:
        if r["token_idx"] > 1:          # skip first token per query (TTFT)
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


def _build_prompt(tokenizer, context, question):
    """Chat-template prompt matching benchmark.py for consistent evaluation."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical expert answering clinical questions. "
                "Answer each question with exactly one word: 'yes', 'no', or 'maybe'. "
                "Do not add any explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}",
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_comparison():
    print("Loading PubMedQA Dataset...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split=f"train[:{N_SAMPLES}]")

    print("Loading model (shared for both schedulers)...")
    model     = EarlyExitTinyLlama()
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    stateless_all_records = []
    kvcached_all_records  = []

    for i, item in enumerate(dataset):
        context  = item["context"]["contexts"][0]
        question = item["question"]
        prompt   = _build_prompt(tokenizer, context, question)

        print(f"\n{'='*60}")
        print(f"Query {i+1}/{N_SAMPLES} | GT: {item['final_decision']}")
        print("=" * 60)

        print("\n--- Stateless Scheduler (L16, threshold decay 0.8→0.3, WCET forced exit) ---")
        stateless_records = generate_stateless_anytime(
            model, prompt,
            max_new_tokens=MAX_TOKENS,
            deadline_ms=DEADLINE_MS,
            verbose=False,
        )
        stateless_all_records.extend(stateless_records)

        print("\n--- KV-Cached Scheduler (L16, fixed threshold 0.55, single-pass) ---")
        kvcached_records = generate_anytime_with_kv(
            model, prompt,
            max_new_tokens=MAX_TOKENS,
            deadline_ms=DEADLINE_MS,
            verbose=False,
        )
        kvcached_all_records.extend(kvcached_records)

    stateless_metrics = aggregate(stateless_all_records, DEADLINE_MS)
    kvcached_metrics  = aggregate(kvcached_all_records,  DEADLINE_MS)

    output = {
        "n_samples":          N_SAMPLES,
        "deadline_ms":        DEADLINE_MS,
        "max_tokens":         MAX_TOKENS,
        "stateless_metrics":  stateless_metrics,
        "kvcached_metrics":   kvcached_metrics,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    return stateless_metrics, kvcached_metrics, stateless_all_records, kvcached_all_records


def plot_comparison(stateless_metrics, kvcached_metrics):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    C_STATELESS = "#d62728"   # red
    C_KVCACHED  = "#2b5b84"   # dark blue
    DEAD_C      = "black"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ── Panel 1: TPOT CDF ─────────────────────────────────────────────────────
    ax = axes[0]
    for label, metrics, colour in [
        ("Stateless (L16, decay)",  stateless_metrics, C_STATELESS),
        ("KV-Cached (L16, fixed)",  kvcached_metrics,  C_KVCACHED),
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
    ax     = axes[1]
    labels = ["Stateless\n(L16, decay)", "KV-Cached\n(L16, fixed)"]
    metrics_list = [stateless_metrics, kvcached_metrics]
    x      = np.arange(len(labels))
    width  = 0.5

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
    col_labels = ["Metric", "Stateless", "KV-Cached"]
    rows = [
        ["Mean TPOT (ms)",    f"{stateless_metrics['mean_tpot_ms']:.1f}",  f"{kvcached_metrics['mean_tpot_ms']:.1f}"],
        ["P99 TPOT (ms)",     f"{stateless_metrics['p99_tpot_ms']:.1f}",   f"{kvcached_metrics['p99_tpot_ms']:.1f}"],
        ["Throughput (tok/s)",f"{stateless_metrics['throughput_tps']}",    f"{kvcached_metrics['throughput_tps']}"],
        ["Util (P99/D)",      f"{stateless_metrics['util_ratio']:.4f}",    f"{kvcached_metrics['util_ratio']:.4f}"],
        ["Deadline Miss (%)", f"{stateless_metrics['deadline_miss_pct']}", f"{kvcached_metrics['deadline_miss_pct']}"],
        ["Full Pass (%)",     f"{stateless_metrics['full_pass_pct']}",     f"{kvcached_metrics['full_pass_pct']}"],
        ["Early Exit (%)",    f"{stateless_metrics['early_conf_pct']}",    f"{kvcached_metrics['early_conf_pct']}"],
        ["Forced Exit (%)",   f"{stateless_metrics['forced_exit_pct']}",   f"{kvcached_metrics['forced_exit_pct']}"],
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
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2b5b84")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary Metrics", pad=12)

    fig.suptitle(
        f"Stateless (decay thresh) vs. KV-Cached (fixed thresh)  —  both L16  |  deadline={DEADLINE_MS:.0f} ms  |  n={N_SAMPLES} queries",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    stateless_metrics, kvcached_metrics, stateless_records, kvcached_records = run_comparison()

    print("\n" + "=" * 55)
    print("COMPARISON SUMMARY")
    print("=" * 55)
    for name, m in [("Stateless  (L16, decay thresh)", stateless_metrics),
                    ("KV-Cached  (L16, fixed thresh)", kvcached_metrics)]:
        print(f"\n{name}")
        print(f"  Mean TPOT:    {m['mean_tpot_ms']:.2f} ms  |  P99: {m['p99_tpot_ms']:.2f} ms")
        print(f"  Throughput:   {m['throughput_tps']} tok/s  |  Util (P99/D): {m['util_ratio']:.4f}")
        print(f"  Full Pass:    {m['full_pass_pct']}%  |  Early: {m['early_conf_pct']}%  "
              f"|  Forced: {m['forced_exit_pct']}%")
        print(f"  Missed deadlines: {m['deadline_miss_pct']}%")

    plot_comparison(stateless_metrics, kvcached_metrics)
