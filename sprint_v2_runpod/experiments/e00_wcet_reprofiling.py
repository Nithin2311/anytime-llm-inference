"""
E00: WCET Re-Profiling on A100 SXM4 — 100 runs/cell.
Unified hardware baseline for all latency tables in report_v2.
Addresses: C1 (hardware mismatch), sets WCET_TABLE_PATH for downstream experiments.
"""
import argparse, os, sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import result_writer as rw
from early_exit_model import EarlyExitTinyLlama

EXPERIMENT_ID = "E00"
RESULTS_FILE  = "wcet_results.json"
HARDWARE      = "A100 SXM4"
SEQ_LENGTHS   = [32, 64, 128, 256, 512, 1024]
EXIT_LAYERS   = [5, 11, 16, None]
NUM_WARMUP    = 10
NUM_RUNS      = 100


def make_input(sl):
    return torch.randint(100, 30000, (1, sl), dtype=torch.long, device="cuda")


def collect(model, ids, exit_layer):
    with torch.inference_mode():
        for _ in range(NUM_WARMUP):
            model(ids, exit_layer=exit_layer, use_cache=False)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    with torch.inference_mode():
        for i in range(NUM_RUNS):
            starts[i].record()
            model(ids, exit_layer=exit_layer, use_cache=False)
            ends[i].record()
    torch.cuda.synchronize()
    return np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])


def build_latex(results):
    lines = [
        f"% E00: WCET Profile — {HARDWARE} ({NUM_RUNS} runs/cell)",
        r"\midrule",
    ]
    for sl in SEQ_LENGTHS:
        r16  = results[str(sl)]["16"]
        rfull= results[str(sl)]["None"]
        lines.append(
            f"  {sl:>4} & {r16['mean_ms']:.2f} & {r16['wcet_ms']:.2f} & "
            f"{rfull['mean_ms']:.2f} & {rfull['wcet_ms']*1.10:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"


def main(dry_run=False):
    rw.configure(BASE_DIR)
    if dry_run:
        print(f"[{EXPERIMENT_ID}] dry-run OK | CUDA={torch.cuda.is_available()}")
        sys.exit(0)
    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping — {RESULTS_FILE} exists.")
        os.environ["WCET_TABLE_PATH"] = rw.results_path(RESULTS_FILE)
        sys.exit(0)

    t0 = rw.log_start(EXPERIMENT_ID)
    try:
        model   = EarlyExitTinyLlama()
        results = {}
        total   = len(SEQ_LENGTHS) * len(EXIT_LAYERS)
        cell    = 0
        for sl in SEQ_LENGTHS:
            ids = make_input(sl)
            results[str(sl)] = {}
            for layer in EXIT_LAYERS:
                cell += 1
                tag = f"L{layer}" if layer is not None else "Full"
                print(f"  [{cell:>2}/{total}] seq={sl:>4} exit={tag:<6}", end="  ", flush=True)
                samples = collect(model, ids, layer)
                results[str(sl)][str(layer)] = {
                    "mean_ms": round(float(np.mean(samples)), 4),
                    "p99_ms":  round(float(np.percentile(samples, 99)), 4),
                    "wcet_ms": round(float(np.max(samples)), 4),
                    "n_runs":  NUM_RUNS,
                }
                print(f"mean={np.mean(samples):.2f}  p99={np.percentile(samples,99):.2f}  wcet={np.max(samples):.2f}")

        saved = rw.write_json(RESULTS_FILE, {
            "hardware": HARDWARE, "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "num_runs": NUM_RUNS, "seq_lengths": SEQ_LENGTHS,
            "exit_layers": [str(e) for e in EXIT_LAYERS], "results": results,
        })
        os.environ["WCET_TABLE_PATH"] = saved
        print(f"  Results → {saved}")
        rw.write_latex("table_wcet.tex", build_latex(results))

        _plot(results)
        rw.log_success(EXPERIMENT_ID, t0)
    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc); raise


def _plot(results):
    import matplotlib.pyplot as plt
    import fig_style as fs
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=fs.DOUBLE)
    colors = {"5": "#2ca02c", "11": "#ff7f0e", "16": "#1f77b4", "None": "#d62728"}
    labels = {"5": "L5", "11": "L11", "16": "L16", "None": "Full(22)"}
    for key, ax_title, metric in [("mean_ms", "Mean TPOT (ms)", 0), ("wcet_ms", "WCET (ms)", 1)]:
        ax = axes[metric]
        for layer_k in ["5", "11", "16", "None"]:
            vals = [results[str(sl)][layer_k][key] for sl in SEQ_LENGTHS]
            ax.plot(SEQ_LENGTHS, vals, marker="o", color=colors[layer_k], label=labels[layer_k])
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel(ax_title)
        ax.set_title(ax_title)
        ax.legend()
        ax.set_xticks(SEQ_LENGTHS)
    fig.suptitle(f"WCET Profile — TinyLlama-1.1B ({HARDWARE})", fontsize=8)
    plt.tight_layout()
    path = rw.figures_path("wcet_profile.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Figure  → {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
