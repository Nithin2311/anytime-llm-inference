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
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

from early_exit_model import EarlyExitTinyLlama
from benchmark import _build_prompt

RESULTS_FILE = "ablation_results.json"
FIGURE_FILE  = "exit_layer_ablation.png"
N_SAMPLES    = 10
MAX_TOKENS   = 15
EXIT_LAYERS  = [5, 11, 16, 20]   # candidates; L22 is the full-pass oracle


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
    fs.apply()

    colours = {5: "#d62728", 11: "#ff7f0e", 16: "#2b5b84", 20: "#17becf"}
    layers  = EXIT_LAYERS

    fig, ax = plt.subplots(figsize=(7, 5))

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
            ax.plot(xs, ys, color=colours[layer], linewidth=2.5, marker="o",
                    markersize=5, label=f"Layer {layer}")

    ax.axvspan(0.3, 0.55, alpha=0.10, color="green", label="Routing threshold (0.3–0.55)")
    ax.axhline(50, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Softmax Confidence at Exit Layer")
    ax.set_ylabel("Agreement with Full-Pass Oracle (%)")
    ax.set_title(f"Exit-Layer Calibration  |  {N_SAMPLES} prompts × {MAX_TOKENS} tokens")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc="upper left")

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
