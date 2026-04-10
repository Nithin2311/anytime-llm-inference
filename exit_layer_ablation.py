"""
exit_layer_ablation.py — Ablation study: exit layer vs quality/latency tradeoff.

Validates the core design decision to use Layer 16 as the early-exit point
by comparing L5, L11, and L16 across the same clinical prompts.

For each exit layer we measure:
  - Agreement rate with full-pass (L22) prediction
  - Confidence distribution (mean, P99)
  - Per-token latency (WCET, mean)
  - Early-exit rate when used inside the dynamic scheduler

Outputs:
  ablation_results.json   — raw per-token data for all layers
  exit_layer_ablation.png — three-panel comparison figure
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

from early_exit_model import EarlyExitTinyLlama
from benchmark import _build_prompt

RESULTS_FILE = "ablation_results.json"
FIGURE_FILE  = "exit_layer_ablation.png"
N_SAMPLES    = 10
MAX_TOKENS   = 15
EXIT_LAYERS  = [5, 11, 16, 17, 18, 19, 20]   # candidates; L22 is the full-pass oracle


def collect_ablation_data(model, prompts):
    """
    For each token position across all prompts, collect:
      - top-1 prediction and softmax confidence at L5, L11, L16
      - top-1 prediction at L22 (oracle)
      - per-layer wall-clock time

    Returns list of per-token dicts.
    """
    records = []

    with torch.inference_mode():
        for p_idx, prompt in enumerate(prompts):
            input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            # Warmup
            _ = model(input_ids, use_cache=False)
            torch.cuda.synchronize()

            print(f"  Prompt {p_idx+1}/{len(prompts)} "
                  f"(len={input_ids.shape[1]} tok)", end="", flush=True)

            for step in range(MAX_TOKENS):
                row = {"prompt_idx": p_idx, "step": step}

                # Measure each layer
                for layer in EXIT_LAYERS:
                    t_start = torch.cuda.Event(enable_timing=True)
                    t_end   = torch.cuda.Event(enable_timing=True)
                    t_start.record()
                    logits, _ = model(input_ids, exit_layer=layer, use_cache=False)
                    t_end.record()
                    torch.cuda.synchronize()

                    probs     = torch.softmax(logits[0, -1, :], dim=-1)
                    conf, tok = torch.max(probs, dim=-1)
                    row[f"conf_l{layer}"]  = round(conf.item(), 4)
                    row[f"pred_l{layer}"]  = tok.item()
                    row[f"time_l{layer}"]  = round(t_start.elapsed_time(t_end), 3)

                # Full pass oracle (L22)
                logits_full, _ = model(input_ids, use_cache=False)
                pred_l22 = torch.argmax(logits_full[0, -1, :], dim=-1).item()
                row["pred_l22"] = pred_l22

                # Agreement flags
                for layer in EXIT_LAYERS:
                    row[f"agree_l{layer}"] = bool(row[f"pred_l{layer}"] == pred_l22)

                records.append(row)

                # Advance with oracle token
                next_tok = torch.tensor([[pred_l22]], device=input_ids.device)
                input_ids = torch.cat([input_ids, next_tok], dim=-1)
                if pred_l22 == model.tokenizer.eos_token_id:
                    print(f" [EOS@{step+1}]", end="")
                    break

            print()

    return records


def summarise(records, layer):
    """Return summary stats for a given exit layer."""
    conf_key  = f"conf_l{layer}"
    agree_key = f"agree_l{layer}"
    time_key  = f"time_l{layer}"

    confs  = [r[conf_key]  for r in records]
    agrees = [r[agree_key] for r in records]
    times  = [r[time_key]  for r in records]

    hc = [r for r in records if r[conf_key] >= 0.5]
    hc_agree = (sum(1 for r in hc if r[agree_key]) / len(hc) * 100) if hc else 0.0

    return {
        "layer":         layer,
        "n":             len(records),
        "agree_pct":     round(100 * np.mean(agrees), 2),
        "agree_hc_pct":  round(hc_agree, 2),
        "mean_conf":     round(float(np.mean(confs)),  4),
        "p99_conf":      round(float(np.percentile(confs, 99)), 4),
        "mean_time_ms":  round(float(np.mean(times)),  3),
        "p99_time_ms":   round(float(np.percentile(times, 99)), 3),
        "wcet_ms":       round(float(np.max(times)), 3),
        "hc_sample_pct": round(100 * len(hc) / len(records), 2),
    }


def plot_ablation(records, summaries):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    colours = {5: "#d62728", 11: "#ff7f0e", 16: "#2b5b84",
               17: "#9467bd", 18: "#8c564b", 19: "#e377c2", 20: "#17becf"}
    layers  = EXIT_LAYERS

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # ── Panel 1: Agreement rate vs confidence threshold ──────────────────────
    ax = axes[0]
    thresholds = np.linspace(0.0, 1.0, 21)
    for layer in layers:
        rates = []
        for lo, hi in zip(thresholds[:-1], thresholds[1:]):
            bucket = [r for r in records if lo <= r[f"conf_l{layer}"] < hi]
            if bucket:
                rates.append((
                    (lo + hi) / 2,
                    100 * np.mean([r[f"agree_l{layer}"] for r in bucket])
                ))
        if rates:
            xs, ys = zip(*rates)
            ax.plot(xs, ys, color=colours[layer], linewidth=2, marker="o",
                    markersize=4, label=f"Layer {layer}")

    ax.axvspan(0.3, 0.8, alpha=0.08, color="green", label="Dynamic thresh range")
    ax.set_xlabel("Confidence at Exit Layer")
    ax.set_ylabel("Agreement with Full Pass (%)")
    ax.set_title("Calibration: Agreement vs. Confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # ── Panel 2: Latency distribution (box plot per layer) ───────────────────
    ax = axes[1]
    time_data = [[r[f"time_l{layer}"] for r in records] for layer in layers]
    bp = ax.boxplot(time_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, layer in zip(bp["boxes"], layers):
        patch.set_facecolor(colours[layer])
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(layers) + 1))
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_ylabel("Token Latency (ms)")
    ax.set_title("Per-Token Latency by Exit Layer")

    # Annotate WCET
    for i, (layer, s) in enumerate(zip(layers, summaries)):
        ax.text(i + 1, s["wcet_ms"] + 0.15, f"WCET={s['wcet_ms']:.1f}",
                ha="center", fontsize=7.5, color=colours[layer])

    # ── Panel 3: Summary table ────────────────────────────────────────────────
    ax = axes[2]
    ax.axis("off")

    col_labels = ["Metric"] + [f"L{l}" for l in layers]
    rows = [
        ["Agreement (%)",       *[f"{s['agree_pct']}"    for s in summaries]],
        ["Agree @ conf≥0.5",  *[f"{s['agree_hc_pct']}"  for s in summaries]],
        ["Mean conf",           *[f"{s['mean_conf']:.3f}" for s in summaries]],
        ["Mean TPOT (ms)",      *[f"{s['mean_time_ms']}"  for s in summaries]],
        ["P99 TPOT (ms)",       *[f"{s['p99_time_ms']}"   for s in summaries]],
        ["WCET (ms)",           *[f"{s['wcet_ms']}"        for s in summaries]],
        ["conf≥0.5 tok (%)",   *[f"{s['hc_sample_pct']}" for s in summaries]],
    ]
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0.05, 1, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2b5b84")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight the chosen layer (L16)
    if 16 in layers:
        chosen_col = layers.index(16) + 1
        for row_i in range(1, len(rows) + 1):
            table[row_i, chosen_col].set_facecolor("#dce9f5")
    ax.set_title("Exit Layer Comparison (L16 highlighted)", pad=10)

    fig.suptitle(
        f"Exit-Layer Ablation Study  |  {N_SAMPLES} prompts  |  {MAX_TOKENS} tokens each",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    print("Loading PubMedQA Dataset...")
    dataset   = load_dataset("pubmed_qa", "pqa_labeled", split=f"train[:{N_SAMPLES}]")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    prompts   = [
        _build_prompt(tokenizer, item["context"]["contexts"][0], item["question"])
        for item in dataset
    ]

    print("Loading model...")
    model = EarlyExitTinyLlama()

    print(f"\n{'='*60}")
    print(f"EXIT LAYER ABLATION  |  {N_SAMPLES} prompts × {MAX_TOKENS} tokens")
    print(f"Exit layers tested: {EXIT_LAYERS}  (oracle: L22)")
    print("=" * 60 + "\n")

    records = collect_ablation_data(model, prompts)

    summaries = [summarise(records, layer) for layer in EXIT_LAYERS]

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print("=" * 60)
    header = "  ".join(f"L{l:>2}" for l in EXIT_LAYERS)
    print(f"{'':30s}  {header}")
    print("-" * (32 + 10 * len(EXIT_LAYERS)))
    for key, label in [
        ("agree_pct",     "Overall agreement (%)"),
        ("agree_hc_pct",  "Agree @ conf>=0.5 (%)"),
        ("mean_conf",     "Mean confidence"),
        ("mean_time_ms",  "Mean TPOT (ms)"),
        ("wcet_ms",       "WCET (ms)"),
    ]:
        vals = "  ".join(f"{str(s[key]):>8}" for s in summaries)
        print(f"{label:30s}  {vals}")
    print("=" * 60)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"n_samples": N_SAMPLES, "max_tokens": MAX_TOKENS,
                   "exit_layers": EXIT_LAYERS, "summaries": summaries,
                   "records": records}, f, indent=2)
    print(f"\nSaved {len(records)} token records to '{RESULTS_FILE}'")

    plot_ablation(records, summaries)
