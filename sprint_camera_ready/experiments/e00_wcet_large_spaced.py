"""
e00_wcet_large_spaced.py — 5000-sample TPOT profiling, round-robin interleaved,
1 s inter-run spacing per cell.

Round-robin design
------------------
Sequential per-cell sampling at 1 s spacing costs N_CELLS * N_SAMPLES seconds.
Round-robin interleaving costs ~N_SAMPLES seconds total: in each "round" we
sample every cell once back-to-back (≈12 ms aggregate GPU work), then sleep
to pad the round to 1 s wall-clock. Same-cell samples are therefore ≥1 s
apart (IID-safe), while cross-cell samples experience identical thermal /
load conditions (improves comparison fairness).

For TinyLlama-1.1B, holding 12 independent KV caches consumes <200 MB VRAM
out of 80 GB, so per-cell context isolation is essentially free.

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

N_SAMPLES   = 5000
N_WARMUP    = 50
SPACING_MS  = 1000           # 1 second between same-cell samples
CHECKPOINT_EVERY = 100       # save partial JSON every N rounds
DEVICE      = "cuda"
PROMPT      = "The pharmacokinetics of drug X suggest that the optimal dosing"

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


def setup_cell(model, tokenizer, seq_len, label, exit_depth):
    """Pre-fill KV cache (full-depth cells only) and return per-cell context dict."""
    ids = build_input(tokenizer, seq_len)
    is_full = (exit_depth == model.num_layers)

    with torch.inference_mode():
        if is_full:
            _, _, pkv = model.forward_cached(ids)
        else:
            pkv = None

        new_tok = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        for _ in range(N_WARMUP):
            if is_full:
                model.forward_cached(new_tok, past_key_values=pkv)
            else:
                model.forward_cached(new_tok, exit_layer=exit_depth)
        torch.cuda.synchronize()

    return {
        "name":       f"seq{seq_len}_{label.lower()}",
        "seq_len":    seq_len,
        "exit":       label,
        "exit_depth": exit_depth,
        "is_full":    is_full,
        "pkv":        pkv,
        "new_tok":    new_tok,
        "times":      [],
    }


def step_cell(model, ctx):
    """Run one timed forward for cell `ctx`, append elapsed_ms to ctx['times']."""
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    if ctx["is_full"]:
        model.forward_cached(ctx["new_tok"], past_key_values=ctx["pkv"])
    else:
        model.forward_cached(ctx["new_tok"], exit_layer=ctx["exit_depth"])
    ev_e.record()
    torch.cuda.synchronize()
    ctx["times"].append(ev_s.elapsed_time(ev_e))


def cell_to_record(ctx, elapsed_s):
    s = np.array(ctx["times"])
    return {
        "cell":      ctx["name"],
        "seq_len":   ctx["seq_len"],
        "exit":      ctx["exit"],
        "exit_depth": ctx["exit_depth"],
        "n_samples": int(s.size),
        "spacing_ms": SPACING_MS,
        "mean_ms":   round(float(np.mean(s)),   4),
        "std_ms":    round(float(np.std(s)),    4),
        "p90_ms":    round(float(np.percentile(s, 90)),   4),
        "p95_ms":    round(float(np.percentile(s, 95)),   4),
        "p99_ms":    round(float(np.percentile(s, 99)),   4),
        "p999_ms":   round(float(np.percentile(s, 99.9)), 4),
        "p9999_ms":  round(float(np.percentile(s, 99.99)), 4) if s.size >= 100 else None,
        "max_ms":    round(float(np.max(s)),    4),
        "min_ms":    round(float(np.min(s)),    4),
        "elapsed_s": round(elapsed_s, 1),
        "samples":   [round(float(x), 4) for x in s],
    }


def write_snapshot(contexts, sprint_start, n_done, total_n):
    cells = [cell_to_record(c, time.time() - sprint_start) for c in contexts]
    payload = {
        "experiment":   "e00_wcet_large_spaced",
        "mode":         "round_robin_interleaved",
        "n_samples":    total_n,
        "n_done":       n_done,
        "spacing_ms":   SPACING_MS,
        "cells":        cells,
    }
    write_results(payload, RESULTS_DIR / "e00_wcet_large_spaced.json")


def main():
    print("=" * 62)
    print(f"E00  Round-Robin WCET  N={N_SAMPLES}  spacing={SPACING_MS}ms  cells={len(CELLS)}")
    print("=" * 62)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    print(f"\nSetup: priming {len(CELLS)} cell contexts (warmup={N_WARMUP}/cell) ...", flush=True)
    setup_t0 = time.time()
    contexts = [setup_cell(model, model.tokenizer, sl, lbl, ed)
                for sl, lbl, ed in CELLS]
    print(f"  done in {time.time()-setup_t0:.1f}s", flush=True)

    sprint_start = time.time()
    next_log = 0
    print(f"\nTimed phase: {N_SAMPLES} rounds × {len(CELLS)} cells = "
          f"{N_SAMPLES*len(CELLS):,} samples, ETA ≈ {N_SAMPLES * SPACING_MS / 60_000:.1f} min", flush=True)

    with torch.inference_mode():
        for round_i in range(N_SAMPLES):
            round_start = time.time()
            for ctx in contexts:
                step_cell(model, ctx)

            # Pace round to SPACING_MS so same-cell samples are ≥SPACING_MS apart
            elapsed = time.time() - round_start
            target  = SPACING_MS / 1000.0
            if round_i < N_SAMPLES - 1 and elapsed < target:
                time.sleep(target - elapsed)

            # Periodic logging + checkpoint
            done = round_i + 1
            if done >= next_log:
                pct = 100.0 * done / N_SAMPLES
                wall = time.time() - sprint_start
                eta_s = (wall / done) * (N_SAMPLES - done)
                # Quick mid-run snapshot of cell 0's running p99 for liveness
                p99_so_far = np.percentile(contexts[0]["times"], 99) if done >= 100 else float('nan')
                print(f"  round {done:5d}/{N_SAMPLES}  ({pct:5.1f}%)  "
                      f"wall={wall/60:5.1f}min  eta={eta_s/60:5.1f}min  "
                      f"cell0_p99={p99_so_far:.3f}ms", flush=True)
                next_log = done + 100

            if done % CHECKPOINT_EVERY == 0:
                write_snapshot(contexts, sprint_start, done, N_SAMPLES)

    # Final write
    elapsed_total = time.time() - sprint_start
    write_snapshot(contexts, sprint_start, N_SAMPLES, N_SAMPLES)

    print("\nPer-cell summary:")
    for ctx in contexts:
        s = np.array(ctx["times"])
        print(f"  {ctx['name']:20s}  mean={np.mean(s):7.3f}  p99={np.percentile(s,99):7.3f}  "
              f"p99.9={np.percentile(s,99.9):7.3f}  max={np.max(s):7.3f} ms")

    # Summary figure
    fig, ax = plt.subplots(figsize=DOUBLE)
    names = [c["name"] for c in contexts]
    p99s  = [float(np.percentile(c["times"], 99)) for c in contexts]
    means = [float(np.mean(c["times"]))           for c in contexts]
    x = range(len(names))
    ax.bar(x, p99s,  alpha=0.75, label="P99 TPOT")
    ax.bar(x, means, alpha=0.50, label="Mean TPOT")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(f"E00 — WCET Profile (round-robin, N={N_SAMPLES}, spacing={SPACING_MS}ms)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e00_wcet_large_spaced.png", dpi=150)
    plt.close()

    print(f"\nPASS: E00 complete  ({elapsed_total/60:.1f} min)\n")


if __name__ == "__main__":
    main()
