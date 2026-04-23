"""
deadline_sweep.py — Utility-latency tradeoff across a range of deadlines.

Runs the dynamic scheduler at each deadline value on the same set of
PubMedQA prompts to characterize how tightening the time budget
shifts the exit-type distribution and affects schedulability.

Outputs:
  sweep_results.json     — per-deadline aggregated metrics
  deadline_tradeoff.png  — tradeoff figure (miss rate + exit dist vs deadline)
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
from dynamic_scheduler import generate_stateless_anytime
from benchmark import extract_label, _build_prompt

RESULTS_FILE = "sweep_results.json"
FIGURE_FILE  = "deadline_tradeoff.png"

# Deadline values to sweep (ms)
DEADLINES   = [20, 25, 30, 35, 40, 45, 50, 60]
N_SAMPLES   = 5
MAX_TOKENS  = 15


def run_one_deadline(model, prompts, ground_truths, deadline_ms):
    """Run all prompts at a single deadline and return aggregated metrics."""
    all_records = []
    n_correct   = 0
    n_scored    = 0

    for prompt, gt in zip(prompts, ground_truths):
        records = generate_stateless_anytime(
            model, prompt,
            max_new_tokens=MAX_TOKENS,
            deadline_ms=deadline_ms,
            verbose=False,
        )
        all_records.extend(records)

        generated_text = "".join(r["token"] for r in records)
        predicted = extract_label(generated_text)
        if predicted != "unknown":
            n_scored += 1
            if predicted == gt:
                n_correct += 1

    n = len(all_records)
    tpot_arr = np.array([r["time_ms"] for r in all_records if r["token_idx"] > 1])
    if len(tpot_arr) == 0:
        tpot_arr = np.array([0.0])

    return {
        "deadline_ms":       deadline_ms,
        "n_tokens":          n,
        "full_pass_pct":     round(100 * sum(1 for r in all_records if r["exit_type"] == "Full Pass") / n, 1),
        "early_thresh_pct":  round(100 * sum(1 for r in all_records if "Thresh" in r["exit_type"]) / n, 1),
        "forced_exit_pct":   round(100 * sum(1 for r in all_records if r["exit_type"] == "Early (Forced)") / n, 1),
        "deadline_miss_pct": round(100 * sum(1 for r in all_records if r["time_ms"] > deadline_ms) / n, 1),
        "mean_tpot_ms":      round(float(np.mean(tpot_arr)), 3),
        "p99_tpot_ms":       round(float(np.percentile(tpot_arr, 99)), 3),
        "accuracy":          round(100.0 * n_correct / n_scored, 1) if n_scored > 0 else None,
        "n_correct":         n_correct,
        "n_scored":          n_scored,
    }


def plot_tradeoff(sweep_results):
    fs.apply()

    deadlines    = [r["deadline_ms"]       for r in sweep_results]
    miss_pct     = [r["deadline_miss_pct"] for r in sweep_results]
    full_pct     = [r["full_pass_pct"]     for r in sweep_results]
    thresh_pct   = [r["early_thresh_pct"]  for r in sweep_results]
    forced_pct   = [r["forced_exit_pct"]   for r in sweep_results]
    mean_tpot    = [r["mean_tpot_ms"]      for r in sweep_results]
    p99_tpot     = [r["p99_tpot_ms"]       for r in sweep_results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # ── Panel 1: Exit-type distribution vs deadline (stacked area) ────────────
    ax = axes[0]
    ax.stackplot(
        deadlines,
        forced_pct, thresh_pct, full_pct,
        labels=["Early (Forced)", "Early (Thresh)", "Full Pass"],
        colors=["#d9534f", "#2e8b57", "#2b5b84"],
        alpha=0.85,
    )
    ax.set_xlabel("Deadline (ms)")
    ax.set_ylabel("Percentage of Tokens (%)")
    ax.set_title("Exit-Type Distribution vs. Deadline")
    ax.set_xticks(deadlines)
    ax.set_xlim(min(deadlines), max(deadlines))
    ax.legend(loc="center right", fontsize=8)

    # ── Panel 2: Schedulability + TPOT vs deadline ────────────────────────────
    ax1 = axes[1]
    colour_miss  = "#d9534f"
    colour_tpot  = "#2b5b84"

    ln1 = ax1.plot(deadlines, miss_pct,  color=colour_miss, linewidth=2.5,
                   marker="o", label="Miss Rate (%)")
    ax1.set_xlabel("Deadline (ms)")
    ax1.set_ylabel("Deadline Miss Rate (%)", color=colour_miss)
    ax1.tick_params(axis="y", labelcolor=colour_miss)
    ax1.set_ylim(-2, max(miss_pct) + 15 if max(miss_pct) > 0 else 20)
    ax1.set_xticks(deadlines)
    ax1.set_xlim(min(deadlines), max(deadlines))

    ax2 = ax1.twinx()
    ln2 = ax2.plot(deadlines, mean_tpot, color=colour_tpot, linewidth=2,
                   marker="s", linestyle="--", label="Mean TPOT (ms)")
    ln3 = ax2.plot(deadlines, p99_tpot,  color=colour_tpot, linewidth=1.5,
                   marker="^", linestyle=":",  label="P99 TPOT (ms)", alpha=0.8)
    ax2.set_ylabel("TPOT (ms)", color=colour_tpot)
    ax2.tick_params(axis="y", labelcolor=colour_tpot)

    lns    = ln1 + ln2 + ln3
    llabels = [l.get_label() for l in lns]
    ax1.legend(lns, llabels, loc="upper right", fontsize=8,
               bbox_to_anchor=(0.98, 0.97), framealpha=0.92)
    ax1.set_title("Miss Rate & Latency vs. Deadline")

    fig.suptitle(
        f"Dynamic Scheduler Deadline Sweep  |  n={N_SAMPLES} queries  |  {MAX_TOKENS} tokens/query",
        fontsize=8, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    print("Loading PubMedQA Dataset...")
    dataset   = load_dataset("pubmed_qa", "pqa_labeled", split=f"train[:{N_SAMPLES}]")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    prompts = [
        _build_prompt(tokenizer, item["context"]["contexts"][0], item["question"])
        for item in dataset
    ]
    ground_truths = [item["final_decision"] for item in dataset]

    print("Loading model...")
    model = EarlyExitTinyLlama()

    print(f"\n{'='*60}")
    print(f"DEADLINE SWEEP: {DEADLINES} ms  |  {N_SAMPLES} queries each")
    print("=" * 60)

    sweep_results = []
    for dl in DEADLINES:
        print(f"\n{'─'*55}")
        print(f">>> Deadline = {dl} ms")
        print("─" * 55)
        metrics = run_one_deadline(model, prompts, ground_truths, deadline_ms=dl)
        sweep_results.append(metrics)
        print(f"\n  Summary: Full={metrics['full_pass_pct']}%  "
              f"Thresh={metrics['early_thresh_pct']}%  "
              f"Forced={metrics['forced_exit_pct']}%  "
              f"Miss={metrics['deadline_miss_pct']}%  "
              f"meanTPOT={metrics['mean_tpot_ms']}ms")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"sweep": sweep_results}, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    print("\n" + "=" * 65)
    print(f"{'DL(ms)':>8}  {'Full%':>6}  {'Thresh%':>8}  {'Forced%':>8}  "
          f"{'Miss%':>6}  {'P99(ms)':>8}")
    print("-" * 65)
    for r in sweep_results:
        print(f"{r['deadline_ms']:>8}  {r['full_pass_pct']:>6}  "
              f"{r['early_thresh_pct']:>8}  {r['forced_exit_pct']:>8}  "
              f"{r['deadline_miss_pct']:>6}  {r['p99_tpot_ms']:>8}")
    print("=" * 65)

    plot_tradeoff(sweep_results)
