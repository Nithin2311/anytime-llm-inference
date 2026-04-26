"""
e11_thermal_a100.py — A100 thermal stability over sustained load.

Runs 1000 consecutive single-token forward passes and records
latency every 10 tokens. Checks for thermal throttling: if P99
in the last 200 tokens exceeds P99 in the first 200 by >5%,
thermal drift is flagged. Also reports GPU temperature via nvidia-smi.
"""

import json, os, sys, time, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from fig_style import apply_style, SINGLE, DOUBLE
import matplotlib.pyplot as plt
from result_writer import write_results

apply_style()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_TOKENS     = 1000
SEQ_LEN      = 128
WARMUP       = 20
DEVICE       = "cuda"
TEMP_INTERVAL = 100   # read GPU temp every N tokens


def get_gpu_temp():
    """Query GPU temperature via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True
        ).strip()
        return int(out.split("\n")[0])
    except Exception:
        return None


def main():
    print("=" * 60)
    print("E11  A100 thermal stability (1000 consecutive tokens)")
    print(f"     seq={SEQ_LEN}")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    import warnings
    dummy = torch.randint(100, 2000, (1, SEQ_LEN), device=DEVICE)

    # Warmup
    print(f"\n[1/2] Warmup ({WARMUP} runs) ...")
    for _ in range(WARMUP):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.inference_mode():
                model.forward_cached(dummy)
    torch.cuda.synchronize()

    # Sustained run
    print(f"[2/2] Sustained run ({N_TOKENS} tokens) ...")
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    latencies   = []
    temp_log    = []  # [(token_idx, temp_C)]
    wall_start  = time.time()

    for i in range(N_TOKENS):
        start_evt.record()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.inference_mode():
                model.forward_cached(dummy)
        end_evt.record()
        torch.cuda.synchronize()
        latencies.append(start_evt.elapsed_time(end_evt))

        if (i + 1) % TEMP_INTERVAL == 0:
            temp = get_gpu_temp()
            temp_log.append({"token_idx": i + 1, "temp_C": temp})
            p99_cur = np.percentile(latencies[-200:], 99) if len(latencies) >= 200 else None
            print(f"  token {i+1:>4}/{N_TOKENS}  "
                  f"p99(last200)={p99_cur:.2f}ms  "
                  f"temp={temp}°C")

    wall_elapsed = time.time() - wall_start
    latencies = np.array(latencies)

    # Thermal drift check
    first200_p99 = float(np.percentile(latencies[:200],  99))
    last200_p99  = float(np.percentile(latencies[-200:], 99))
    drift_pct    = (last200_p99 - first200_p99) / first200_p99 * 100.0
    thermal_flag = bool(drift_pct > 5.0)

    print(f"\nFirst-200 P99 : {first200_p99:.3f}ms")
    print(f"Last-200  P99 : {last200_p99:.3f}ms")
    print(f"Drift         : {drift_pct:.2f}%  {'FLAGGED' if thermal_flag else 'OK'}")

    # Rolling P99 (windows of 100)
    rolling_p99 = []
    window = 100
    for j in range(0, N_TOKENS - window + 1, 10):
        rolling_p99.append({
            "idx":     j + window // 2,
            "p99_ms":  round(float(np.percentile(latencies[j:j + window], 99)), 4),
            "mean_ms": round(float(np.mean(latencies[j:j + window])), 4),
        })

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)

    ax = axes[0]
    ax.plot(latencies, lw=0.5, color="tab:blue", alpha=0.6, label="Per-token latency")
    rp99_x   = [r["idx"]    for r in rolling_p99]
    rp99_val = [r["p99_ms"] for r in rolling_p99]
    ax.plot(rp99_x, rp99_val, color="tab:red", lw=1.5, label="Rolling P99 (w=100)")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency over 1000 tokens")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    if temp_log:
        temp_x = [t["token_idx"] for t in temp_log]
        temp_y = [t["temp_C"] or 0 for t in temp_log]
        ax2.plot(temp_x, temp_y, "o-", color="tab:orange")
        ax2.set_xlabel("Token index")
        ax2.set_ylabel("GPU Temperature (°C)")
        ax2.set_title("GPU Temperature During Run")
    else:
        ax2.text(0.5, 0.5, "Temperature data\nnot available",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.suptitle(f"A100 Thermal Stability (N={N_TOKENS}, seq={SEQ_LEN})")
    plt.tight_layout()
    fig_path = RESULTS_DIR / "thermal_stability.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved: {fig_path}")

    # Summary stats
    output = {
        "experiment":        "e11_thermal_a100",
        "n_tokens":          N_TOKENS,
        "seq_len":           SEQ_LEN,
        "total_wall_time_s": round(wall_elapsed, 2),
        "mean_ms":           round(float(np.mean(latencies)),            4),
        "std_ms":            round(float(np.std(latencies)),             4),
        "p50_ms":            round(float(np.percentile(latencies, 50)),  4),
        "p99_ms":            round(float(np.percentile(latencies, 99)),  4),
        "p999_ms":           round(float(np.percentile(latencies, 99.9)),4),
        "max_ms":            round(float(np.max(latencies)),             4),
        "first200_p99_ms":   round(first200_p99, 4),
        "last200_p99_ms":    round(last200_p99,  4),
        "drift_pct":         round(drift_pct, 3),
        "thermal_flagged":   thermal_flag,
        "temp_log":          temp_log,
        "rolling_p99":       rolling_p99,
    }
    write_results(output, RESULTS_DIR / "thermal_stability_results.json")
    print("PASS: E11 complete\n")


if __name__ == "__main__":
    main()
