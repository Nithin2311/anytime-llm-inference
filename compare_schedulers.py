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
import matplotlib.pyplot as plt
import fig_style as fs
from datasets import load_dataset
from transformers import AutoTokenizer

from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import (generate_stateless_anytime, generate_anytime_with_kv,
                               generate_anytime_async_overlap)

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

    C_STATELESS = "#d62728"   # red
    C_KVCACHED  = "#2b5b84"   # dark blue
    C_ASYNC     = "#2ca02c"   # green
    DEAD_C      = "black"

    all_series = [
        ("Stateless (decay, no cache)", stateless_metrics, C_STATELESS),
        ("KV-Cached (fixed, sync)",     kvcached_metrics,  C_KVCACHED),
    ]
    if async_metrics is not None:
        all_series.append(("Async-Overlap (fixed, overlap)", async_metrics, C_ASYNC))

    n_routers = len(all_series)
    fig, axes = plt.subplots(1, 3, figsize=fs.TRIPLE)

    # ── Panel 1: TPOT CDF ─────────────────────────────────────────────────────
    ax = axes[0]
    for label, metrics, colour in all_series:
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
    labels = ["Stateless\n(decay)", "KV-Cached\n(sync)", "Async\n(overlap)"] \
             if async_metrics else ["Stateless\n(L16, decay)", "KV-Cached\n(L16, fixed)"]
    metrics_list = [stateless_metrics, kvcached_metrics] + \
                   ([async_metrics] if async_metrics else [])
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
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=1, framealpha=0.9)

    # ── Panel 3: Key Metric Table ──────────────────────────────────────────────
    ax = axes[2]
    ax.axis("off")

    def _fmt(m):
        return [
            f"{m['mean_tpot_ms']:.1f}",
            f"{m['p99_tpot_ms']:.1f}",
            str(m['throughput_tps']),
            f"{m['util_ratio']:.4f}",
            f"{m['deadline_miss_pct']}",
            f"{m['full_pass_pct']}",
            f"{m['early_conf_pct']}",
            f"{m['forced_exit_pct']}",
        ]

    row_names = ["Mean TPOT (ms)", "P99 TPOT (ms)", "Throughput (t/s)",
                 "SLO ratio (P99/D)", "Miss Rate (%)", "Full Pass (%)",
                 "Early Exit (%)", "Forced Exit (%)"]

    if async_metrics:
        col_labels = ["Metric", "Stateless", "KV-Cached", "Async"]
        rows = [[rn] + [_fmt(m)[i] for m in [stateless_metrics, kvcached_metrics, async_metrics]]
                for i, rn in enumerate(row_names)]
        col_widths = [0.38, 0.2, 0.2, 0.2]
        header_colours = ["#2b5b84", "#2b5b84", "#2b5b84", "#2ca02c"]
    else:
        col_labels = ["Metric", "Stateless", "KV-Cached"]
        rows = [[rn] + [_fmt(m)[i] for m in [stateless_metrics, kvcached_metrics]]
                for i, rn in enumerate(row_names)]
        col_widths = [0.44, 0.28, 0.28]
        header_colours = ["#2b5b84"] * 3

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                     loc="center", bbox=[0, 0.02, 1, 0.96])
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    n_rows = len(rows) + 1
    for row_idx in range(n_rows):
        for col_idx, w in enumerate(col_widths):
            table[row_idx, col_idx].set_width(w)
        table[row_idx, 0].set_text_props(ha="left")
    for j, hc in enumerate(header_colours):
        table[0, j].set_facecolor(hc)
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary Metrics", pad=12)

    n_label = f"{n_routers} routers" if async_metrics else "2 routers"
    fig.suptitle(
        f"Application-Level Router Comparison — {n_label} | L16 exit | "
        f"deadline={DEADLINE_MS:.0f} ms | n={N_SAMPLES} queries",
        fontsize=7.5, y=1.02,
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
