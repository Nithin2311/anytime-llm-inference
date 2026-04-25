"""
E0: WCET re-profiling on A100 SXM4 — 50 runs/cell baseline.

Must run before any experiment that imports dynamic_scheduler. Sets the
WCET_TABLE_PATH env var so downstream imports use the fresh A100 table.

Outputs:
  results/wcet_results.json
  figures/wcet_profile.png
  latex/table_ii_wcet.tex   (Table II body rows for report_v2.tex)
"""

import argparse
import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import result_writer as rw
from early_exit_model import EarlyExitTinyLlama

EXPERIMENT_ID = "E0"
RESULTS_FILE  = "wcet_results.json"
HARDWARE      = "A100 SXM4"
SEQ_LENGTHS   = [32, 64, 128, 256, 512, 1024]
EXIT_LAYERS   = [5, 11, 16, None]
NUM_WARMUP    = 5
NUM_RUNS      = 50


def make_input(seq_len):
    return torch.randint(100, 30000, (1, seq_len), dtype=torch.long, device="cuda")


def profile_cell(model, input_ids, exit_layer):
    with torch.inference_mode():
        for _ in range(NUM_WARMUP):
            model(input_ids, exit_layer=exit_layer, use_cache=False)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    with torch.inference_mode():
        for i in range(NUM_RUNS):
            starts[i].record()
            model(input_ids, exit_layer=exit_layer, use_cache=False)
            ends[i].record()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return {
        "mean_ms": round(float(np.mean(times)), 3),
        "p99_ms":  round(float(np.percentile(times, 99)), 3),
        "wcet_ms": round(float(np.max(times)), 3),
    }


def build_latex(results):
    lines = [
        f"% E0: WCET Profile — {HARDWARE} (50 runs/cell)",
        "% Replaces Table II body rows in report_v2.tex",
        "% Columns: seq & L16 mean & L16 WCET & Full mean & Full WCET*1.10 \\\\",
        r"\midrule",
    ]
    for sl in SEQ_LENGTHS:
        l16  = results[str(sl)]["16"]
        full = results[str(sl)]["None"]
        lines.append(
            f"  {sl:>4} & {l16['mean_ms']:.2f} & {l16['wcet_ms']:.2f}"
            f" & {full['mean_ms']:.2f} & {full['wcet_ms'] * 1.10:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"


def main(dry_run=False):
    rw.configure(BASE_DIR)

    if dry_run:
        import transformers, scipy, matplotlib  # noqa: F401
        print(f"[{EXPERIMENT_ID}] dry-run OK | CUDA={torch.cuda.is_available()} | imports OK")
        sys.exit(0)

    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping — {RESULTS_FILE} exists.")
        os.environ["WCET_TABLE_PATH"] = rw.results_path(RESULTS_FILE)
        sys.exit(0)

    t0 = rw.log_start(EXPERIMENT_ID)
    try:
        print(f"\n{'='*60}\nE0: WCET PROFILE — {HARDWARE}\n"
              f"Seq: {SEQ_LENGTHS} | Layers: {EXIT_LAYERS} | {NUM_RUNS} runs/cell\n{'='*60}\n")

        model   = EarlyExitTinyLlama()
        results = {}
        total   = len(SEQ_LENGTHS) * len(EXIT_LAYERS)
        cell    = 0

        for sl in SEQ_LENGTHS:
            ids = make_input(sl)
            results[str(sl)] = {}
            for layer in EXIT_LAYERS:
                cell += 1
                tag = f"L{layer}" if layer is not None else "Full(22)"
                print(f"  [{cell:>2}/{total}] seq={sl:>4} exit={tag:<8}", end="  ", flush=True)
                r = profile_cell(model, ids, layer)
                results[str(sl)][str(layer)] = r
                print(f"mean={r['mean_ms']:.2f}  p99={r['p99_ms']:.2f}  wcet={r['wcet_ms']:.2f} ms")

        saved = rw.write_json(RESULTS_FILE, {
            "hardware":    HARDWARE,
            "model":       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "num_runs":    NUM_RUNS,
            "seq_lengths": SEQ_LENGTHS,
            "exit_layers": [str(e) for e in EXIT_LAYERS],
            "results":     results,
        })
        print(f"\n  Results → {saved}")
        os.environ["WCET_TABLE_PATH"] = saved

        _plot(results)
        rw.write_latex("table_ii_wcet.tex", build_latex(results))
        print(f"  LaTeX   → latex/table_ii_wcet.tex")
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc)
        raise


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as fs

    fs.apply()
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=fs.DOUBLE)

    for ax, metric, ylabel in [
        (axes[0], "mean_ms", "Mean Latency (ms)"),
        (axes[1], "wcet_ms", "WCET / Max Latency (ms)"),
    ]:
        for idx, layer in enumerate(EXIT_LAYERS):
            key    = str(layer)
            label  = f"L{layer}" if layer is not None else "Full(22)"
            values = [results[str(sl)][key][metric] for sl in SEQ_LENGTHS]
            ax.plot(SEQ_LENGTHS, values, marker="o", linewidth=2, color=colours[idx], label=label)
        ax.axhline(y=45.0, color="red", linestyle="--", linewidth=1.2, label="D=45 ms")
        ax.set_xlabel("Input Sequence Length (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.replace(" (ms)", ""))
        ax.set_xticks(SEQ_LENGTHS)
        ax.legend(fontsize=8)

    fig.suptitle(f"WCET Profile — TinyLlama-1.1B ({HARDWARE})", fontsize=8, y=1.01)
    plt.tight_layout()
    path = rw.figures_path("wcet_profile.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure  → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
