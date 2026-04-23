"""
calibration.py — Early-exit confidence calibration analysis.

Validates the foundational assumption of the anytime scheduler:
  "When the model is confident at an intermediate exit layer,
   its token prediction agrees with the full 22-layer prediction."

For each generated token we collect:
  - Top-1 token at L5 and L16 (early exits)
  - Top-1 token at L22 (full pass — ground truth oracle)
  - Softmax confidence at L5 and L16

Then we measure:
  1. Calibration curves: agreement rate vs. confidence threshold
     → Directly validates the exit threshold choice (0.3–0.8)
  2. Agreement rate histograms per exit layer
  3. Confidence distribution separated by agree/disagree

Outputs:
  calibration_results.json   — raw per-token data
  calibration.png            — three-panel figure
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

RESULTS_FILE = "calibration_results.json"
FIGURE_FILE  = "calibration.png"
N_SAMPLES    = 10
MAX_TOKENS   = 20   # slightly more than benchmark to get more calibration points


def collect_calibration_data(model, prompts):
    """
    For each token position, collect early-exit and full-pass predictions.

    Returns a list of dicts, one per generated token:
        conf_l5, pred_l5, conf_l16, pred_l16, pred_l22,
        agree_l5  (bool: l5 top-1 == l22 top-1),
        agree_l16 (bool: l16 top-1 == l22 top-1)
    """
    records = []

    with torch.inference_mode():
        for prompt_idx, prompt in enumerate(prompts):
            input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            # Warmup per prompt
            _ = model(input_ids, use_cache=False)
            torch.cuda.synchronize()

            print(f"  Prompt {prompt_idx + 1}/{len(prompts)}  "
                  f"(len={input_ids.shape[1]} tokens)", end="", flush=True)

            for step in range(MAX_TOKENS):
                # L5 early exit
                logits_l5, _  = model(input_ids, exit_layer=5,  use_cache=False)
                # L16 early exit
                logits_l16, _ = model(input_ids, exit_layer=16, use_cache=False)
                # Full pass (L22)
                logits_l22, _ = model(input_ids, use_cache=False)

                # Confidence and top-1 at each depth, evaluated at last token position
                probs_l5  = torch.softmax(logits_l5 [0, -1, :], dim=-1)
                probs_l16 = torch.softmax(logits_l16[0, -1, :], dim=-1)

                conf_l5,  pred_l5  = torch.max(probs_l5,  dim=-1)
                conf_l16, pred_l16 = torch.max(probs_l16, dim=-1)
                pred_l22 = torch.argmax(logits_l22[0, -1, :], dim=-1)

                conf_l5_val  = conf_l5.item()
                conf_l16_val = conf_l16.item()
                pred_l5_id   = pred_l5.item()
                pred_l16_id  = pred_l16.item()
                pred_l22_id  = pred_l22.item()

                records.append({
                    "prompt_idx": prompt_idx,
                    "step":       step,
                    "conf_l5":    round(conf_l5_val,  4),
                    "conf_l16":   round(conf_l16_val, 4),
                    "pred_l5":    pred_l5_id,
                    "pred_l16":   pred_l16_id,
                    "pred_l22":   pred_l22_id,
                    "agree_l5":   bool(pred_l5_id  == pred_l22_id),
                    "agree_l16":  bool(pred_l16_id == pred_l22_id),
                })

                # Advance using full-pass token (oracle continuation)
                input_ids = torch.cat(
                    [input_ids, pred_l22.unsqueeze(0).unsqueeze(0)], dim=-1
                )

                # Stop at EOS
                if pred_l22_id == model.tokenizer.eos_token_id:
                    print(f" [EOS@{step+1}]", end="")
                    break

            print()   # newline after each prompt's dot-progress

    return records


def compute_calibration_curve(records, conf_key, agree_key, n_bins=20):
    """
    Bin tokens by confidence, compute agreement rate per bin.
    Returns (bin_centres, agreement_rates, counts).
    """
    thresholds = np.linspace(0.0, 1.0, n_bins + 1)
    centres, rates, counts = [], [], []

    for lo, hi in zip(thresholds[:-1], thresholds[1:]):
        bucket = [r for r in records if lo <= r[conf_key] < hi]
        if bucket:
            centres.append((lo + hi) / 2)
            rates.append(np.mean([r[agree_key] for r in bucket]) * 100)
            counts.append(len(bucket))

    return np.array(centres), np.array(rates), np.array(counts)


def plot_calibration(records):
    fs.apply()

    C_L5  = "#d62728"   # red
    C_L16 = "#2b5b84"   # blue

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel 1: Calibration curves ──────────────────────────────────────────
    ax = axes[0]
    for conf_key, agree_key, colour, label in [
        ("conf_l5",  "agree_l5",  C_L5,  "Layer 5"),
        ("conf_l16", "agree_l16", C_L16, "Layer 16"),
    ]:
        centres, rates, counts = compute_calibration_curve(records, conf_key, agree_key)
        ax.plot(centres, rates, color=colour, linewidth=2, marker="o",
                markersize=4, label=label)

    # Mark the scheduler's threshold range
    ax.axvspan(0.3, 0.8, alpha=0.08, color="green", label="Dynamic thresh range (0.3–0.8)")
    ax.axhline(y=100, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Confidence at Exit Layer")
    ax.set_ylabel("Agreement with Full Pass (%)")
    ax.set_title("Calibration Curve\n(Early Exit vs. Full Pass)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # ── Panel 2: Confidence distributions (agree vs disagree) ────────────────
    ax = axes[1]
    for conf_key, agree_key, colour, label in [
        ("conf_l5",  "agree_l5",  C_L5,  "L5"),
        ("conf_l16", "agree_l16", C_L16, "L16"),
    ]:
        agree_confs    = [r[conf_key] for r in records if     r[agree_key]]
        disagree_confs = [r[conf_key] for r in records if not r[agree_key]]
        bins = np.linspace(0, 1, 25)
        ax.hist(agree_confs,    bins=bins, alpha=0.45, color=colour,
                label=f"{label} agree",    density=True, histtype="stepfilled")
        ax.hist(disagree_confs, bins=bins, alpha=0.45, color=colour,
                label=f"{label} disagree", density=True, histtype="step", linewidth=1.5)

    ax.set_xlabel("Confidence at Exit Layer")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution\n(Agree vs Disagree with Full Pass)")
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Early-Exit Confidence Calibration  |  {N_SAMPLES} prompts  |  {MAX_TOKENS} tokens each",
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

    print("Loading model...")
    model = EarlyExitTinyLlama()

    print(f"\n{'='*55}")
    print(f"CALIBRATION  |  {N_SAMPLES} prompts × {MAX_TOKENS} tokens")
    print("="*55 + "\n")

    records = collect_calibration_data(model, prompts)

    with open(RESULTS_FILE, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} token records to '{RESULTS_FILE}'")

    # Print summary
    total     = len(records)
    l5_agree  = sum(1 for r in records if r["agree_l5"])
    l16_agree = sum(1 for r in records if r["agree_l16"])
    l5_hc     = [r for r in records if r["conf_l5"]  >= 0.5]
    l16_hc    = [r for r in records if r["conf_l16"] >= 0.5]

    print(f"\n{'='*55}")
    print("CALIBRATION SUMMARY")
    print("="*55)
    print(f"{'':30s}  {'L5':>10}  {'L16':>10}")
    print(f"{'Overall agreement':30s}  {100*l5_agree/total:>9.1f}%  {100*l16_agree/total:>9.1f}%")
    mean_l5  = np.mean([r["conf_l5"]  for r in records])
    mean_l16 = np.mean([r["conf_l16"] for r in records])
    hc_l5_rate  = 100 * sum(1 for r in l5_hc  if r["agree_l5" ]) / len(l5_hc)  if l5_hc  else 0
    hc_l16_rate = 100 * sum(1 for r in l16_hc if r["agree_l16"]) / len(l16_hc) if l16_hc else 0
    print(f"{'Mean confidence':30s}  {mean_l5:>10.3f}  {mean_l16:>10.3f}")
    print(f"{'Agreement @ conf>=0.5':30s}  {hc_l5_rate:>9.1f}%  {hc_l16_rate:>9.1f}%")
    print(f"{'Tokens with conf≥0.8 (%)':30s}  "
          f"{100*sum(1 for r in records if r['conf_l5'] >=0.8)/total:>9.1f}%  "
          f"{100*sum(1 for r in records if r['conf_l16']>=0.8)/total:>9.1f}%")
    print("="*55)

    plot_calibration(records)
