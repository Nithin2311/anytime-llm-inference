"""e07_thermal_extended.py — 60-minute thermal stability soak on A100."""
import sys, time, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib.pyplot as plt
from fig_style import apply_style, DOUBLE
from result_writer import write_results
apply_style()
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SOAK_MINUTES = 60
SEQ_LEN      = 128
N_WARMUP     = 100
DEVICE       = "cuda"
RECORD_EVERY = 100   # record stats every N samples

def get_gpu_stats():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.current.sm,power.draw,utilization.gpu",
            "--format=csv,noheader"
        ], text=True).strip().split(",")
        return {
            "temp_c":    float(out[0].strip().replace(" C", "")),
            "clock_mhz": float(out[1].strip().replace(" MHz", "")),
            "power_w":   float(out[2].strip().replace(" W", "")),
            "util_pct":  float(out[3].strip().replace(" %", "")),
        }
    except Exception:
        return {"temp_c": None, "clock_mhz": None, "power_w": None, "util_pct": None}

def build_input(tokenizer):
    PROMPT = "The pharmacokinetics of drug X suggest that the optimal dosing interval"
    import torch
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
    while len(ids) < SEQ_LEN:
        ids = ids.repeat(2)
    return ids[:SEQ_LEN].unsqueeze(0).to(DEVICE)

def main():
    print("=" * 60)
    print(f"E07  60-Minute Thermal Soak  seq{SEQ_LEN}_full")
    print("=" * 60)
    import torch
    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    ids = build_input(model.tokenizer)
    with torch.inference_mode():
        _, _, pkv = model.forward_cached(ids)
        new_tok = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        print(f"\nWarm-up ({N_WARMUP} runs) ...")
        for _ in range(N_WARMUP):
            model.forward_cached(new_tok, past_key_values=pkv)
        torch.cuda.synchronize()

    soak_s   = SOAK_MINUTES * 60
    t_start  = time.time()
    t_end    = t_start + soak_s
    all_times  = []
    snapshots  = []
    sample_idx = 0

    print(f"\nSoaking for {SOAK_MINUTES} min ...")
    with torch.inference_mode():
        while time.time() < t_end:
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            model.forward_cached(new_tok, past_key_values=pkv)
            ev_e.record()
            torch.cuda.synchronize()
            t_ms = ev_s.elapsed_time(ev_e)
            elapsed = time.time() - t_start
            all_times.append({"elapsed_s": round(elapsed, 2), "tpot_ms": round(t_ms, 4)})

            if sample_idx % RECORD_EVERY == 0:
                stats = get_gpu_stats()
                snap  = {"elapsed_s": round(elapsed, 1), "tpot_ms": round(t_ms, 4), **stats}
                snapshots.append(snap)
                mins  = elapsed / 60
                print(f"  t={mins:5.1f}min  TPOT={t_ms:.2f}ms  "
                      f"temp={stats['temp_c']}C  clock={stats['clock_mhz']}MHz  "
                      f"power={stats['power_w']}W", flush=True)
            sample_idx += 1

    total_n    = len(all_times)
    times_arr  = np.array([r["tpot_ms"] for r in all_times])
    # First 10 min vs last 10 min drift
    n_10min    = sum(1 for r in all_times if r["elapsed_s"] <= 600)
    first_10   = times_arr[:n_10min]
    last_10    = times_arr[-n_10min:] if n_10min > 0 else times_arr[-100:]
    drift_ms   = float(np.mean(last_10) - np.mean(first_10))

    print(f"\n  Total samples : {total_n}")
    print(f"  Mean TPOT     : {np.mean(times_arr):.3f} ms")
    print(f"  Std TPOT      : {np.std(times_arr):.3f} ms")
    print(f"  Drift (L10-F10): {drift_ms:+.3f} ms  ({'+' if drift_ms>0 else ''}{100*drift_ms/np.mean(times_arr):.1f}%)")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=DOUBLE)
    t_min = [r["elapsed_s"]/60 for r in all_times]
    tpots = [r["tpot_ms"] for r in all_times]
    axes[0,0].plot(t_min, tpots, lw=0.3, color="tab:blue", alpha=0.6)
    axes[0,0].axhline(np.mean(times_arr), ls="--", color="red", lw=0.8, label="mean")
    axes[0,0].set_xlabel("Time (min)"); axes[0,0].set_ylabel("TPOT (ms)")
    axes[0,0].set_title("TPOT over 60 min"); axes[0,0].legend(fontsize=6)
    snap_t = [s["elapsed_s"]/60 for s in snapshots]
    for ax_idx, (key, label) in enumerate([("temp_c","GPU temp (C)"),
                                            ("clock_mhz","Clock (MHz)"),
                                            ("power_w","Power (W)")]):
        ax = axes.flatten()[ax_idx+1]
        vals = [s.get(key) for s in snapshots if s.get(key) is not None]
        t_v  = [s["elapsed_s"]/60 for s in snapshots if s.get(key) is not None]
        if vals:
            ax.plot(t_v, vals, color="tab:orange", lw=1)
        ax.set_xlabel("Time (min)"); ax.set_ylabel(label)
        ax.set_title(label)
    plt.suptitle(f"E07 — 60-min Thermal Soak (n={total_n})", fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e07_thermal_soak.png", dpi=150); plt.close()

    write_results({
        "experiment": "e07_thermal_extended", "soak_minutes": SOAK_MINUTES,
        "seq_len": SEQ_LEN, "n_samples": total_n,
        "mean_tpot_ms": round(float(np.mean(times_arr)),4),
        "std_tpot_ms":  round(float(np.std(times_arr)),4),
        "p99_tpot_ms":  round(float(np.percentile(times_arr,99)),4),
        "drift_ms":     round(drift_ms,4),
        "drift_pct":    round(100*drift_ms/np.mean(times_arr),2),
        "snapshots": snapshots,
    }, RESULTS_DIR / "e07_thermal_soak.json")
    print("PASS: E07 complete\n")

if __name__ == "__main__":
    main()
