"""
e00_wcet_large_spaced.py — 5000-sample TPOT profiling, 1 s inter-run spacing.

Addresses EVT sample-size critique (sprint_v3: only 500 samples, blocks too small)
and IID spacing failure (200 ms insufficient).  5000 samples at 1 s spacing gives
n/block_size = 100 usable blocks at b=50, meeting EVT block-maxima requirements.
"""

import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from fig_style import apply_style, DOUBLE
from result_writer import write_results

apply_style()
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLES  = 5000
N_WARMUP   = 50
SPACING_MS = 1000   # 1 second
DEVICE     = "cuda"
PROMPT     = "The pharmacokinetics of drug X suggest that the optimal dosing"

# 12 cells: (seq_len, label, exit_depth)
CELLS = [
    (32,   "L5",   5),
    (32,   "L16",  16),
    (32,   "Full", 22),
    (128,  "L5",   5),
    (128,  "L16",  16),
    (128,  "Full", 22),
    (256,  "L16",  16),
    (256,  "Full", 22),
    (512,  "L16",  16),
    (512,  "Full", 22),
    (1024, "L16",  16),
    (1024, "Full", 22),
]


def build_input(tokenizer, seq_len):
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
    while len(ids) < seq_len:
        ids = ids.repeat(2)
    return ids[:seq_len].unsqueeze(0).to(DEVICE)


def profile_cell(model, tokenizer, seq_len, exit_depth):
    ids      = build_input(tokenizer, seq_len)
    is_full  = (exit_depth == model.num_layers)

    with torch.inference_mode():
        # Pre-fill KV cache
        if is_full:
            _, _, pkv = model.forward_cached(ids)
        else:
            pkv = None

        # Warm-up (discarded)
        new_tok = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        for _ in range(N_WARMUP):
            if is_full:
                model.forward_cached(new_tok, past_key_values=pkv)
            else:
                model.forward_cached(new_tok, exit_layer=exit_depth)
        torch.cuda.synchronize()

        # Timed measurements
        times = []
        for i in range(N_SAMPLES):
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            if is_full:
                model.forward_cached(new_tok, past_key_values=pkv)
            else:
                model.forward_cached(new_tok, exit_layer=exit_depth)
            ev_e.record()
            torch.cuda.synchronize()
            times.append(ev_s.elapsed_time(ev_e))
            if SPACING_MS > 0 and i < N_SAMPLES - 1:
                time.sleep(SPACING_MS / 1000.0)

    return np.array(times)


def main():
    print("=" * 62)
    print(f"E00  Large Spaced WCET  N={N_SAMPLES}  spacing={SPACING_MS}ms  cells={len(CELLS)}")
    print("=" * 62)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    all_cells = []
    for idx, (seq_len, label, exit_depth) in enumerate(CELLS):
        cname = f"seq{seq_len}_{label.lower()}"
        print(f"\n[{idx+1}/{len(CELLS)}] {cname} ...", flush=True)
        t0 = time.time()

        s = profile_cell(model, model.tokenizer, seq_len, exit_depth)

        cell = {
            "cell": cname, "seq_len": seq_len, "exit": label,
            "exit_depth": exit_depth, "n_samples": len(s),
            "spacing_ms": SPACING_MS,
            "mean_ms":   round(float(np.mean(s)), 4),
            "std_ms":    round(float(np.std(s)), 4),
            "p90_ms":    round(float(np.percentile(s, 90)), 4),
            "p95_ms":    round(float(np.percentile(s, 95)), 4),
            "p99_ms":    round(float(np.percentile(s, 99)), 4),
            "p999_ms":   round(float(np.percentile(s, 99.9)), 4),
            "p9999_ms":  round(float(np.percentile(s, 99.99)), 4),
            "max_ms":    round(float(np.max(s)), 4),
            "min_ms":    round(float(np.min(s)), 4),
            "elapsed_s": round(time.time() - t0, 1),
            "samples":   [round(float(x), 4) for x in s],
        }
        all_cells.append(cell)
        print(f"  mean={cell['mean_ms']:.2f}  p99={cell['p99_ms']:.2f}  "
              f"p99.9={cell['p999_ms']:.2f}  max={cell['max_ms']:.2f} ms  "
              f"({cell['elapsed_s']:.0f}s)", flush=True)

        # Incremental save after every cell
        write_results({"experiment": "e00_wcet_large_spaced", "n_samples": N_SAMPLES,
                       "spacing_ms": SPACING_MS, "cells": all_cells},
                      RESULTS_DIR / "e00_wcet_large_spaced.json")

    # Summary figure
    fig, ax = plt.subplots(figsize=DOUBLE)
    names = [c["cell"] for c in all_cells]
    p99s  = [c["p99_ms"] for c in all_cells]
    means = [c["mean_ms"] for c in all_cells]
    x = range(len(names))
    ax.bar(x, p99s,  alpha=0.75, label="P99 TPOT")
    ax.bar(x, means, alpha=0.50, label="Mean TPOT")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(f"E00 — WCET Profile (N={N_SAMPLES}, spacing={SPACING_MS}ms)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e00_wcet_large_spaced.png", dpi=150)
    plt.close()
    print("\nPASS: E00 complete\n")


if __name__ == "__main__":
    main()
