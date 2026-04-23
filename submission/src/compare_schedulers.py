"""
compare_schedulers.py — Three-way router comparison.

Compares all three application-level routing strategies on the same
PubMedQA prompts:
  1. Stateless two-pass router      (dynamic threshold decay 0.8→0.3, no KV cache)
  2. KV-cached single-pass router   (fixed threshold 0.55, post-hoc decision)
  3. Async-overlap KV-cached router (fixed threshold 0.55, CPU-GPU overlap pipeline)

Outputs:
  scheduler_comparison.json   — per-token records + aggregate metrics
  scheduler_comparison.png    — CDF + exit-distribution + summary table
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import (generate_stateless_anytime, generate_anytime_with_kv,
                               generate_anytime_async_overlap)

RESULTS_FILE = "scheduler_comparison.json"
FIGURE_FILE  = "scheduler_comparison.png"

N_SAMPLES    = 5
DEADLINE_MS  = 30.0
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
    async_all_records     = []

    for i, item in enumerate(dataset):
        context  = item["context"]["contexts"][0]
        question = item["question"]
        prompt   = _build_prompt(tokenizer, context, question)

        print(f"\n{'='*60}")
        print(f"Query {i+1}/{N_SAMPLES} | GT: {item['final_decision']}")
        print("=" * 60)

        print("\n--- Stateless Two-Pass Router (L16, threshold decay 0.8->0.3, no KV cache) ---")
        stateless_records = generate_stateless_anytime(
            model, prompt,
            max_new_tokens=MAX_TOKENS,
            deadline_ms=DEADLINE_MS,
            verbose=False,
        )
        stateless_all_records.extend(stateless_records)

        print("\n--- KV-Cached Single-Pass Router (L16, fixed threshold 0.55, post-hoc) ---")
        kvcached_records = generate_anytime_with_kv(
            model, prompt,
            max_new_tokens=MAX_TOKENS,
            deadline_ms=DEADLINE_MS,
            verbose=False,
        )
        kvcached_all_records.extend(kvcached_records)

        print("\n--- Async-Overlap KV-Cached Router (L16, fixed 0.55, CPU-GPU pipeline) ---")
        async_records = generate_anytime_async_overlap(
            model, prompt,
            max_new_tokens=MAX_TOKENS,
            deadline_ms=DEADLINE_MS,
            verbose=False,
        )
        async_all_records.extend(async_records)

    stateless_metrics = aggregate(stateless_all_records, DEADLINE_MS)
    kvcached_metrics  = aggregate(kvcached_all_records,  DEADLINE_MS)
    async_metrics     = aggregate(async_all_records,     DEADLINE_MS)

    output = {
        "n_samples":          N_SAMPLES,
        "deadline_ms":        DEADLINE_MS,
        "max_tokens":         MAX_TOKENS,
        "stateless_metrics":  stateless_metrics,
        "kvcached_metrics":   kvcached_metrics,
        "async_metrics":      async_metrics,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    return stateless_metrics, kvcached_metrics, async_metrics, \
           stateless_all_records, kvcached_all_records, async_all_records


def plot_comparison(stateless_metrics, kvcached_metrics, async_metrics=None):
    fs.apply()

    C_STATELESS = "#d62728"
    C_KVCACHED  = "#2b5b84"
    C_ASYNC     = "#2ca02c"

    metrics_list = [stateless_metrics, kvcached_metrics]
    router_labels = ["Stateless\n(two-pass)", "KV-Cached\n(single-pass)"]
    colours = [C_STATELESS, C_KVCACHED]
    linestyles = ["-", "--"]
    if async_metrics is not None:
        metrics_list.append(async_metrics)
        router_labels.append("Async-Overlap\n(pipelined)")
        colours.append(C_ASYNC)
        linestyles.append(":")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── Panel 1: TPOT CDF ─────────────────────────────────────────────────────
    ax = axes[0]
    for (label, m, col, ls) in zip(router_labels, metrics_list, colours, linestyles):
        samples = np.sort(m["tpot_samples"])
        cdf     = np.arange(1, len(samples) + 1) / len(samples)
        p99     = m["p99_tpot_ms"]
        ax.plot(samples, cdf, linewidth=2.5, color=col, linestyle=ls,
                label=f"{label.replace(chr(10), ' ')}  (P99={p99:.1f} ms)")

    ax.axvline(x=DEADLINE_MS, color="black", linestyle="--", linewidth=1.5,
               label=f"Deadline = {DEADLINE_MS:.0f} ms")
    ax.axhline(y=0.99, color="grey", linestyle=":", linewidth=1.0, label="P99 level")
    ax.set_xlabel("Token Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("TPOT Cumulative Distribution")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(left=0)

    # ── Panel 2: P99 TPOT + Miss Rate comparison ──────────────────────────────
    ax = axes[1]
    short_labels = [l.split("\n")[0] for l in router_labels]
    x     = np.arange(len(short_labels))
    width = 0.35

    p99_vals  = [m["p99_tpot_ms"]      for m in metrics_list]
    mean_vals = [m["mean_tpot_ms"]     for m in metrics_list]
    miss_vals = [m["deadline_miss_pct"] for m in metrics_list]

    bars1 = ax.bar(x - width/2, mean_vals, width, label="Mean TPOT",
                   color=[c + "aa" for c in colours], edgecolor="black", linewidth=0.7)
    bars2 = ax.bar(x + width/2, p99_vals,  width, label="P99 TPOT",
                   color=colours, edgecolor="black", linewidth=0.7)

    for bar, val in zip(bars1, mean_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)
    for bar, val, miss in zip(bars2, p99_vals, miss_vals):
        miss_str = f"\n{miss:.1f}% miss" if miss > 0 else "\n0% miss"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}{miss_str}", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(y=DEADLINE_MS, color="black", linestyle="--", linewidth=1.5,
               label=f"Deadline = {DEADLINE_MS:.0f} ms")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Mean & P99 TPOT per Router\n(with deadline miss rate)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, DEADLINE_MS * 1.05)

    n_routers = len(metrics_list)
    fig.suptitle(
        f"Router Comparison — {n_routers} routers | L16 exit | D={DEADLINE_MS:.0f} ms | n={N_SAMPLES} queries",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    stateless_m, kvcached_m, async_m, *_ = run_comparison()

    print("\n" + "=" * 60)
    print("ROUTER COMPARISON SUMMARY")
    print("=" * 60)
    for name, m in [("Stateless  (two-pass, decay thresh, no cache)", stateless_m),
                    ("KV-Cached  (single-pass, fixed thresh, sync)",   kvcached_m),
                    ("Async-Overlap (single-pass, fixed, CPU overlap)", async_m)]:
        print(f"\n{name}")
        print(f"  Mean TPOT : {m['mean_tpot_ms']:.2f} ms  |  P99: {m['p99_tpot_ms']:.2f} ms")
        print(f"  Throughput: {m['throughput_tps']} tok/s  |  SLO ratio (P99/D): {m['util_ratio']:.4f}")
        print(f"  Full Pass : {m['full_pass_pct']}%  |  Early: {m['early_conf_pct']}%  "
              f"|  Forced: {m['forced_exit_pct']}%")
        print(f"  SLO misses: {m['deadline_miss_pct']}%")

    plot_comparison(stateless_m, kvcached_m, async_m)
