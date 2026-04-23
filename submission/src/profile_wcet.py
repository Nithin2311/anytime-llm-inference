"""
profile_wcet.py — GPU Worst-Case Execution Time profiling.

Sweeps over sequence lengths × exit-layer configurations to characterize
the latency budget available to the real-time scheduler on this hardware.

Outputs:
  wcet_results.json   — raw timing data (mean, p99, WCET per cell)
  wcet_profile.png    — line plot of mean + WCET vs sequence length
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from early_exit_model import EarlyExitTinyLlama

RESULTS_FILE = "wcet_results.json"
FIGURE_FILE  = "wcet_profile.png"

# ── Profiler ──────────────────────────────────────────────────────────────────

def profile_gpu_execution(func, *args, num_warmup=5, num_runs=50, **kwargs):
    """
    Profile a GPU function with async CUDA events.

    Returns:
        mean_ms  (float)
        p99_ms   (float)
        wcet_ms  (float)  — max over all runs
    """
    with torch.inference_mode():
        for _ in range(num_warmup):
            func(*args, **kwargs)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    with torch.inference_mode():
        for i in range(num_runs):
            start_events[i].record()
            func(*args, **kwargs)
            end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return (
        float(np.mean(times_ms)),
        float(np.percentile(times_ms, 99)),
        float(np.max(times_ms)),
    )


def make_input(seq_len, device):
    """Create a random token sequence of exactly seq_len tokens."""
    return torch.randint(100, 30000, (1, seq_len), dtype=torch.long, device=device)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_wcet_sweep(model, seq_lengths, exit_layers, num_warmup=5, num_runs=50):
    """
    Profile every (seq_len, exit_layer) cell.

    exit_layer == None  →  full 22-layer pass.
    Returns a nested dict: results[seq_len][exit_layer] = {mean, p99, wcet}
    """
    results = {}
    total_cells = len(seq_lengths) * len(exit_layers)
    cell = 0

    for seq_len in seq_lengths:
        input_ids = make_input(seq_len, device="cuda")
        results[seq_len] = {}

        for exit_layer in exit_layers:
            cell += 1
            label = f"L{exit_layer}" if exit_layer is not None else "Full(22)"
            print(f"  [{cell:>2}/{total_cells}] seq={seq_len:>4}  exit={label:<8}", end="  ", flush=True)

            mean_ms, p99_ms, wcet_ms = profile_gpu_execution(
                model, input_ids,
                exit_layer=exit_layer,
                use_cache=False,
                num_warmup=num_warmup,
                num_runs=num_runs,
            )
            results[seq_len][str(exit_layer)] = {
                "mean_ms": round(mean_ms, 3),
                "p99_ms":  round(p99_ms,  3),
                "wcet_ms": round(wcet_ms, 3),
            }
            print(f"mean={mean_ms:.2f}  p99={p99_ms:.2f}  wcet={wcet_ms:.2f}  (ms)")

    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_wcet_profile(results, seq_lengths, exit_layers, deadline_ms=45.0):
    fs.apply()

    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for ax, metric, metric_label in [
        (axes[0], "mean_ms", "Mean Latency (ms)"),
        (axes[1], "wcet_ms", "WCET / Max Latency (ms)"),
    ]:
        for idx, exit_layer in enumerate(exit_layers):
            key    = str(exit_layer)
            label  = f"L{exit_layer}" if exit_layer is not None else "Full(22)"
            values = [results[sl][key][metric] for sl in seq_lengths]
            ax.plot(seq_lengths, values, marker="o", linewidth=2,
                    color=colours[idx % len(colours)], label=label)

        ax.axhline(y=deadline_ms, color="red", linestyle="--", linewidth=1.2,
                   label=f"Deadline ({deadline_ms:.0f} ms)")
        ax.set_xlabel("Input Sequence Length (tokens)")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label.replace(" (ms)", ""))
        ax.set_xticks(seq_lengths)
        ax.set_xticklabels(seq_lengths, rotation=45, ha='right')
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("TinyLlama-1.1B WCET Profile  —  RTX 6000 Ada", fontsize=8, y=1.01)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sequence lengths covering the realistic prompt+generation context range.
    # 512 and 1024 are added to cover PubMedQA clinical prompts (chat-template
    # formatted context+question is typically 300–700 tokens).
    SEQ_LENGTHS  = [32, 64, 128, 256, 512, 1024]
    # Exit layers: sparse probe (5), mid (11), pre-final (16), full pass (None=22)
    EXIT_LAYERS  = [5, 11, 16, None]

    print("=" * 55)
    print("GPU WCET PROFILING  —  TinyLlama-1.1B")
    print(f"Seq lengths : {SEQ_LENGTHS}")
    print(f"Exit layers : {EXIT_LAYERS}  (None = full 22-layer pass)")
    print(f"Runs/cell   : 50 (+ 5 warmup)")
    print("=" * 55 + "\n")

    model = EarlyExitTinyLlama()

    results = run_wcet_sweep(
        model,
        seq_lengths=SEQ_LENGTHS,
        exit_layers=EXIT_LAYERS,
        num_warmup=5,
        num_runs=50,
    )

    # Save raw data
    output = {
        "hardware":    "RTX 6000 Ada",
        "model":       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "num_runs":    50,
        "seq_lengths": SEQ_LENGTHS,
        "exit_layers": [str(e) for e in EXIT_LAYERS],
        "results":     results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Exit':>8}  " + "  ".join(f"seq={s:>4}" for s in SEQ_LENGTHS))
    print("          " + "  ".join(f"{'mean/p99':>9}" for _ in SEQ_LENGTHS))
    print("-" * 65)
    for exit_layer in EXIT_LAYERS:
        key   = str(exit_layer)
        label = f"L{exit_layer}" if exit_layer is not None else "Full(22)"
        row   = f"{label:>8}  "
        for sl in SEQ_LENGTHS:
            d = results[sl][key]
            row += f"{d['mean_ms']:>4.1f}/{d['p99_ms']:>4.1f}  "
        print(row)
    print("=" * 65)

    plot_wcet_profile(results, SEQ_LENGTHS, EXIT_LAYERS)
