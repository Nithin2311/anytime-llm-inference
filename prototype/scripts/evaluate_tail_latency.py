"""
evaluate_tail_latency.py — Schedulability proof via tail-latency CDF.

Compares standard full-pass autoregressive baseline against the anytime
scheduler across 10 diverse clinical prompts, demonstrating hard-deadline
compliance through formal P99 analysis.

Prompts use the TinyLlama chat template (identical format to benchmark.py)
so that measured latencies reflect real deployment conditions, not artificially
short bare-text inputs.  Chat-template formatted prompts are typically 80–120
tokens, producing a context of 95–145 tokens by the end of generation — the
same range seen during the PubMedQA benchmark.

Outputs:
  schedulability_proof.png  — CDF comparison + summary table
  tail_latency_results.json — per-prompt and aggregate statistics
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style as fs
from transformers import AutoModelForCausalLM, AutoTokenizer

from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import generate_anytime_with_kv

FIGURE_FILE  = "schedulability_proof.png"
RESULTS_FILE = "tail_latency_results.json"
DEADLINE_MS  = 45.0
MAX_TOKENS   = 25   # tokens per prompt; 10 prompts × 24 TPOT ≈ 240 samples for P99

# Ten clinically diverse questions spanning cardiology, oncology, neurology,
# infectious disease, and metabolism.  Formatted with the TinyLlama chat
# template (same system prompt and structure as benchmark.py) so that prompt
# lengths match the real benchmark workload (~80–120 tokens after encoding).
_RAW_QUESTIONS = [
    "Does acute myocardial infarction cause elevated troponin levels within 6 hours of symptom onset?",
    "Do beta-lactam antibiotics inhibit gram-negative bacterial cell wall synthesis?",
    "Is insulin resistance the primary driver of hyperglycaemia in type 2 diabetes?",
    "Does early antibiotic administration reduce mortality in septic shock?",
    "Do amyloid plaques cause neuronal death in Alzheimer's disease?",
    "Is cisplatin-induced peripheral neuropathy caused by dorsal root ganglion damage?",
    "Does prolonged immobilisation increase the risk of deep vein thrombosis?",
    "Does angiotensin II cause vasoconstriction via AT1 receptor activation?",
    "Does uncontrolled hypertension accelerate chronic kidney disease progression?",
    "Do regulatory T cells suppress autoimmune responses via IL-10 secretion?",
]


def _build_prompt(tokenizer, question: str) -> str:
    """Format using TinyLlama chat template — identical to benchmark.py."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical expert answering clinical questions. "
                "Answer each question with exactly one word: 'yes', 'no', or 'maybe'. "
                "Do not add any explanation."
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─── Baseline: standard full-pass autoregressive ────────────────────────────

def get_baseline_latencies(prompts, tokenizer, max_tokens=MAX_TOKENS):
    """
    Run the standard HF model (no early exit) on each prompt and return
    all per-token TPOT measurements (first token / TTFT excluded).

    prompts   — list of already-formatted (chat-template) prompt strings
    tokenizer — shared tokenizer instance (passed in to avoid double-loading)
    """
    print("\nLoading standard HF model for baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    all_latencies = []

    with torch.inference_mode():
        for p_idx, prompt in enumerate(prompts):
            print(f"  Baseline prompt {p_idx+1}/{len(prompts)} ...", end=" ", flush=True)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            # warmup per prompt
            _ = model(input_ids)
            torch.cuda.synchronize()

            prompt_lats = []
            for _ in range(max_tokens):
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                outputs    = model(input_ids, use_cache=False)
                next_token = torch.argmax(outputs.logits[0, -1, :], dim=-1)
                end.record()
                torch.cuda.synchronize()
                prompt_lats.append(start.elapsed_time(end))
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

            tpot = prompt_lats[1:]   # skip TTFT
            all_latencies.extend(tpot)
            print(f"done ({len(tpot)} TPOT samples)")

    return all_latencies


# ─── Anytime: dynamic scheduler ─────────────────────────────────────────────

def get_anytime_latencies(anytime_model, prompts, deadline_ms=DEADLINE_MS):  # prompts already chat-template formatted
    """
    Run the anytime scheduler on each prompt and return all per-token
    TPOT measurements (first token / TTFT excluded).
    """
    all_latencies = []

    for p_idx, prompt in enumerate(prompts):
        print(f"  Anytime prompt {p_idx+1}/{len(prompts)} ...", end=" ", flush=True)
        records = generate_anytime_with_kv(
            anytime_model, prompt,
            max_new_tokens=MAX_TOKENS,
            deadline_ms=deadline_ms,
            verbose=False,
        )
        tpot = [r["time_ms"] for r in records[1:]]   # skip TTFT
        all_latencies.extend(tpot)
        print(f"done ({len(tpot)} TPOT samples)")

    return all_latencies


# ─── Formal schedulability analysis ─────────────────────────────────────────

def schedulability_analysis(latencies, deadline_ms, label):
    """
    Print formal schedulability verdict and return a stats dict.

    Schedulability criterion (EDF / deadline-monotonic):
      P99_TPOT ≤ D  →  SCHEDULABLE
    Utilization ratio: U = P99 / D  (target < 1.0)
    """
    arr  = np.array(latencies)
    p50  = float(np.percentile(arr, 50))
    p95  = float(np.percentile(arr, 95))
    p99  = float(np.percentile(arr, 99))
    pmax = float(np.max(arr))
    miss_pct = 100.0 * float(np.sum(arr > deadline_ms)) / len(arr)
    util     = p99 / deadline_ms
    schedulable = p99 <= deadline_ms

    print(f"\n{'─'*58}")
    print(f"  {label}")
    print(f"{'─'*58}")
    print(f"  Samples    : {len(arr)}")
    print(f"  P50 TPOT   : {p50:.3f} ms")
    print(f"  P95 TPOT   : {p95:.3f} ms")
    print(f"  P99 TPOT   : {p99:.3f} ms")
    print(f"  Max TPOT   : {pmax:.3f} ms")
    print(f"  Deadline   : {deadline_ms:.1f} ms")
    print(f"  Miss rate  : {miss_pct:.2f}%")
    print(f"  Util ratio : {util:.4f}  (P99/D — must be < 1.0)")
    verdict = "SCHEDULABLE" if schedulable else f"NOT SCHEDULABLE (+{p99-deadline_ms:.2f} ms over)"
    print(f"  VERDICT    : {verdict}")
    print(f"{'─'*58}")

    return {
        "label": label, "n": len(arr),
        "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3), "max_ms": round(pmax, 3),
        "miss_pct": round(miss_pct, 3),
        "util_ratio": round(util, 4),
        "schedulable": schedulable,
    }


# ─── Figure ──────────────────────────────────────────────────────────────────

def plot_schedulability(baseline_lats, anytime_lats, base_stats, any_stats, deadline_ms):
    fs.apply()

    C_BASE = "#d9534f"   # red
    C_ANY  = "#2b5b84"   # blue
    C_DEAD = "#2e8b57"   # green

    fig, axes = plt.subplots(1, 2, figsize=fs.DOUBLE)

    # ── Panel 1: CDF comparison ─────────────────────────────────────────────
    ax = axes[0]
    for lats, colour, label in [
        (baseline_lats, C_BASE, "Baseline (Full Pass)"),
        (anytime_lats,  C_ANY,  "Anytime Scheduler"),
    ]:
        arr = np.sort(np.array(lats))
        cdf = np.arange(1, len(arr) + 1) / len(arr) * 100
        ax.plot(arr, cdf, color=colour, linewidth=2.2, label=label)

    ax.axvline(x=deadline_ms, color=C_DEAD, linestyle="--", linewidth=1.8,
               label=f"Deadline = {deadline_ms:.0f} ms")
    # Mark P99 crossings
    for lats, colour in [(baseline_lats, C_BASE), (anytime_lats, C_ANY)]:
        p99 = np.percentile(lats, 99)
        ax.axvline(x=p99, color=colour, linestyle=":", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Per-Token Latency (ms)")
    ax.set_ylabel("CDF (%)")
    ax.set_title("TPOT CDF: Baseline vs. Anytime Scheduler")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.legend()

    # ── Panel 2: Schedulability summary table ───────────────────────────────
    ax = axes[1]
    ax.axis("off")

    metrics_order = ["n", "p50_ms", "p95_ms", "p99_ms", "miss_pct", "util_ratio", "schedulable"]
    labels_map    = {
        "n":           "Samples",
        "p50_ms":      "P50 (ms)",
        "p95_ms":      "P95 (ms)",
        "p99_ms":      "P99 (ms)",
        "miss_pct":    "Miss Rate (%)",
        "util_ratio":  "Util (P99/D)",
        "schedulable": "Schedulable?",
    }

    rows = [["Metric", "Baseline", "Anytime"]]
    for key in metrics_order:
        bval = base_stats[key]
        aval = any_stats[key]
        if key == "schedulable":
            bval = "YES" if bval else "NO"
            aval = "YES" if aval else "NO"
        rows.append([labels_map[key], str(bval), str(aval)])

    table = ax.table(
        cellText=rows[1:], colLabels=rows[0],
        cellLoc="center", loc="center",
        bbox=[0, 0.05, 1, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for j in range(3):
        table[0, j].set_facecolor("#2b5b84")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour-code the schedulable row
    sched_row = metrics_order.index("schedulable") + 1
    base_sched = base_stats["schedulable"]
    any_sched  = any_stats["schedulable"]
    table[sched_row, 1].set_facecolor("#2e8b57" if base_sched else "#d9534f")
    table[sched_row, 1].set_text_props(color="white", fontweight="bold")
    table[sched_row, 2].set_facecolor("#2e8b57" if any_sched  else "#d9534f")
    table[sched_row, 2].set_text_props(color="white", fontweight="bold")

    ax.set_title(f"Schedulability Analysis  (D = {deadline_ms:.0f} ms)", pad=12)

    fig.suptitle(
        f"Tail-Latency Schedulability Proof  |  10 prompts (chat-template)  "
        f"|  deadline = {deadline_ms:.0f} ms",
        fontsize=7.5, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved '{FIGURE_FILE}'")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build chat-template formatted prompts once (tokenizer needed for apply_chat_template)
    _tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    PROMPTS = [_build_prompt(_tok, q) for q in _RAW_QUESTIONS]

    print("=" * 60)
    print("SCHEDULABILITY EVALUATION — TAIL LATENCY BENCHMARK")
    print(f"Deadline: {DEADLINE_MS} ms  |  {len(PROMPTS)} prompts  |  {MAX_TOKENS} tokens each")
    print("=" * 60)

    # 1. Baseline (re-uses the tokenizer already loaded above)
    baseline_lats = get_baseline_latencies(PROMPTS, _tok)

    # 2. Anytime scheduler
    print("\nLoading anytime model...")
    anytime_model = EarlyExitTinyLlama()
    anytime_lats  = get_anytime_latencies(anytime_model, PROMPTS)

    # 3. Formal analysis
    print("\n" + "=" * 60)
    print("FORMAL SCHEDULABILITY ANALYSIS")
    print("=" * 60)
    base_stats = schedulability_analysis(baseline_lats, DEADLINE_MS, "Baseline (Standard Autoregressive)")
    any_stats  = schedulability_analysis(anytime_lats,  DEADLINE_MS, "Anytime Scheduler (Dynamic Threshold Decay)")

    # 4. Overall verdict
    print("\n" + "=" * 60)
    print("OVERALL VERDICT")
    print("=" * 60)
    if any_stats["schedulable"] and not base_stats["schedulable"]:
        print(f"  Anytime scheduler achieves schedulability that the baseline cannot.")
        print(f"  P99 reduction: {base_stats['p99_ms']:.2f} ms → {any_stats['p99_ms']:.2f} ms")
        print(f"  Utilization:   {base_stats['util_ratio']:.3f} → {any_stats['util_ratio']:.3f}")
    elif any_stats["schedulable"]:
        print(f"  Both systems meet the {DEADLINE_MS} ms deadline at P99.")
        print(f"  Anytime P99 = {any_stats['p99_ms']:.2f} ms  (util = {any_stats['util_ratio']:.3f})")
    else:
        print(f"  Anytime scheduler P99 = {any_stats['p99_ms']:.2f} ms exceeds deadline.")
        print(f"  Reduce deadline or adjust WCET threshold.")

    # 5. Save results
    results = {
        "deadline_ms": DEADLINE_MS,
        "n_prompts":   len(PROMPTS),
        "prompts":     _RAW_QUESTIONS,
        "baseline":    base_stats,
        "anytime":     any_stats,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to '{RESULTS_FILE}'")

    # 6. Figure
    plot_schedulability(baseline_lats, anytime_lats, base_stats, any_stats, DEADLINE_MS)
