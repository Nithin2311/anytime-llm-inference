"""
e00_spaced_profiling.py — Re-profile with 200ms inter-run sleep.

Addresses IID violation from sprint_final E08 (Ljung-Box p=0.0 for all cells).
Spaced sampling lets GPU thermal, L2 cache, and CUDA stream state return to
baseline between measurements, breaking serial autocorrelation.

Protocol changes vs sprint_final E00:
  - sleep_ms=200 between every timed run
  - n_warmup=50 (up from 20)
  - n_runs=500 (unchanged)
  - Cells: seq=[32,64,128,256,512,1024] x layer=[l16, full]
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from profiling_utils import profile_layer_latency
from result_writer import write_results
from fig_style import apply_style, DOUBLE
import matplotlib.pyplot as plt

apply_style()

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LENS  = [32, 64, 128, 256, 512, 1024]
LAYERS    = ["l16", "full"]
N_WARMUP  = 50
N_RUNS    = 500
SLEEP_MS  = 200.0
DEVICE    = "cuda"


def main():
    print("=" * 60)
    print("E00  Spaced Profiling (sleep_ms=200, n_warmup=50)")
    print(f"     {len(SEQ_LENS)} seq_lens x {len(LAYERS)} layers = "
          f"{len(SEQ_LENS)*len(LAYERS)} cells")
    est_min = len(SEQ_LENS) * len(LAYERS) * N_RUNS * SLEEP_MS / 1000 / 60
    print(f"     Estimated time: ~{est_min:.0f} min (dominated by sleep)")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    cells = {}
    total_cells = len(SEQ_LENS) * len(LAYERS)
    cell_idx = 0

    for seq_len in SEQ_LENS:
        for layer in LAYERS:
            cell_idx += 1
            key = f"seq{seq_len}_{layer}"
            print(f"\n  [{cell_idx}/{total_cells}] Profiling {key} ...")
            t0 = time.time()

            latencies = profile_layer_latency(
                model,
                seq_len=seq_len,
                n_warmup=N_WARMUP,
                n_runs=N_RUNS,
                sleep_ms=SLEEP_MS,
                exit_layer=layer,
                device=DEVICE,
            )

            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.0f}s  "
                  f"mean={latencies.mean():.2f}ms  "
                  f"p99={np.percentile(latencies,99):.2f}ms  "
                  f"max={latencies.max():.2f}ms")

            cells[key] = latencies.tolist()

    # ── Summary stats ────────────────────────────────────────────────────────
    stats = {}
    for key, lats in cells.items():
        arr = np.array(lats)
        stats[key] = {
            "mean_ms": round(float(arr.mean()), 3),
            "p50_ms":  round(float(np.percentile(arr, 50)), 3),
            "p95_ms":  round(float(np.percentile(arr, 95)), 3),
            "p99_ms":  round(float(np.percentile(arr, 99)), 3),
            "max_ms":  round(float(arr.max()), 3),
            "std_ms":  round(float(arr.std()), 3),
        }
        print(f"  {key}: mean={stats[key]['mean_ms']}  "
              f"p99={stats[key]['p99_ms']}  max={stats[key]['max_ms']}")

    # ── Figures ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    colors = plt.cm.tab10.colors

    for i, seq_len in enumerate(SEQ_LENS):
        for j, layer in enumerate(LAYERS):
            key = f"seq{seq_len}_{layer}"
            ax = axes[j]
            arr = np.array(cells[key])
            ax.plot(arr[:100], alpha=0.6, lw=0.6, color=colors[i],
                    label=f"seq={seq_len}")

    for j, layer in enumerate(LAYERS):
        axes[j].set_title(f"{layer.upper()} — first 100 runs (spaced)")
        axes[j].set_xlabel("Run index")
        axes[j].set_ylabel("Latency (ms)")
        axes[j].legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e00_spaced_timeseries.png", dpi=150)
    plt.close()

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "e00_spaced_profiling",
        "n_warmup": N_WARMUP,
        "n_runs": N_RUNS,
        "sleep_ms": SLEEP_MS,
        "seq_lens": SEQ_LENS,
        "layers": LAYERS,
        "stats": stats,
        "raw": cells,
    }
    write_results(output, RESULTS_DIR / "e00_spaced_profiling.json")
    print("\nPASS: E00 complete\n")


if __name__ == "__main__":
    main()
