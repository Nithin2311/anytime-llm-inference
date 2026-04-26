"""
e02_outlier_warmup_study.py — Identify whether extreme outliers are warm-up artifacts.

Profiles WITHOUT discarding any early runs, then compares the distribution of
the first 50 runs (cold-start) vs runs 51-500 (warm) to determine if the
outliers driving xi=1.36 are concentrated in the warm-up phase.

Also profiles with extended warm-up (100 passes) to check if P99 stabilizes.
This determines whether the heavy GEV tail is:
  (A) a warm-up artifact removable by better protocol, or
  (B) an intrinsic property of the GPU execution time distribution.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from profiling_utils import profile_warmup_artifact, profile_layer_latency
from evt_utils_v3 import fit_gev, anderson_darling_gumbel, pot_tail
from result_writer import write_results
from fig_style import apply_style, DOUBLE, TRIPLE

apply_style()

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE  = "cuda"
SEQ_LEN = 128
N_COLD  = 600   # no warm-up discard — first 50 are "cold"
SPLIT   = 50    # cold/warm boundary


def main():
    print("=" * 60)
    print("E02  Outlier & Warm-Up Artifact Study (seq=128, Full Pass)")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    # ── Run 1: no warm-up discard (expose cold-start) ────────────────────────
    print(f"\n[1/3] Cold-start profiling ({N_COLD} runs, no warm-up discard) ...")
    cold_data = profile_warmup_artifact(model, seq_len=SEQ_LEN, n_runs=N_COLD, device=DEVICE)

    cold_phase = cold_data[:SPLIT]
    warm_phase = cold_data[SPLIT:]

    print(f"      Cold ({SPLIT} runs): mean={cold_phase.mean():.2f}ms  "
          f"p99={np.percentile(cold_phase,99):.2f}ms  max={cold_phase.max():.2f}ms")
    print(f"      Warm ({len(warm_phase)} runs): mean={warm_phase.mean():.2f}ms  "
          f"p99={np.percentile(warm_phase,99):.2f}ms  max={warm_phase.max():.2f}ms")

    outlier_threshold = warm_phase.mean() + 3 * warm_phase.std()
    cold_outliers = int((cold_phase > outlier_threshold).sum())
    warm_outliers = int((warm_phase > outlier_threshold).sum())
    print(f"      Outliers > {outlier_threshold:.1f}ms: cold={cold_outliers}/{SPLIT}  "
          f"warm={warm_outliers}/{len(warm_phase)}")

    # ── GEV fit: full cold run vs warm-only ───────────────────────────────────
    print(f"\n[2/3] GEV fit comparison ...")
    tail_cold_all = pot_tail(cold_data, 0.20)
    tail_warm     = pot_tail(warm_phase, 0.20)

    gev_cold_all  = fit_gev(tail_cold_all)
    gev_warm      = fit_gev(tail_warm)
    ad_cold_all   = anderson_darling_gumbel(tail_cold_all)
    ad_warm       = anderson_darling_gumbel(tail_warm)

    print(f"      GEV xi (all 600 runs): {gev_cold_all['xi']:.4f}  "
          f"AD stat={ad_cold_all['statistic']:.3f}  "
          f"Gumbel {'REJECTED' if ad_cold_all['gumbel_rejected'] else 'OK'}")
    print(f"      GEV xi (warm 550 runs): {gev_warm['xi']:.4f}  "
          f"AD stat={ad_warm['statistic']:.3f}  "
          f"Gumbel {'REJECTED' if ad_warm['gumbel_rejected'] else 'OK'}")
    print(f"      Reference (sprint_final E01): xi=1.36, AD=17.55, REJECTED")

    # ── Run 2: extended warm-up (100 passes), then 500 timed, no sleep ───────
    print(f"\n[3/3] Extended warm-up profiling (100 warm-up, 500 timed, no sleep) ...")
    extended_data = profile_layer_latency(
        model, seq_len=SEQ_LEN, n_warmup=100, n_runs=500,
        sleep_ms=0.0, exit_layer="full", device=DEVICE
    )
    tail_extended = pot_tail(extended_data, 0.20)
    gev_extended  = fit_gev(tail_extended)
    ad_extended   = anderson_darling_gumbel(tail_extended)

    print(f"      GEV xi (100 warm-up, no sleep): {gev_extended['xi']:.4f}  "
          f"AD stat={ad_extended['statistic']:.3f}  "
          f"Gumbel {'REJECTED' if ad_extended['gumbel_rejected'] else 'OK'}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=TRIPLE)

    # Left: time series showing cold vs warm
    axes[0].plot(range(SPLIT), cold_phase, "o-", ms=2, lw=0.5,
                 color="tab:red", label="Cold (runs 1–50)")
    axes[0].plot(range(SPLIT, N_COLD), warm_phase, "o-", ms=1, lw=0.4,
                 color="tab:blue", alpha=0.5, label="Warm (51–600)")
    axes[0].axvline(SPLIT, ls="--", color="gray", lw=1)
    axes[0].axhline(outlier_threshold, ls=":", color="tab:orange",
                    lw=1, label=f"μ+3σ={outlier_threshold:.1f}ms")
    axes[0].set_title("Cold-start time series")
    axes[0].set_xlabel("Run index")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(fontsize=6)

    # Middle: histogram comparison
    bins = np.linspace(min(cold_data.min(), extended_data.min()),
                       min(max(cold_data.max(), extended_data.max()), 50), 60)
    axes[1].hist(cold_phase,   bins=bins, alpha=0.6, color="tab:red",
                 density=True, label=f"Cold (xi={gev_cold_all['xi']:.2f})")
    axes[1].hist(warm_phase,   bins=bins, alpha=0.6, color="tab:blue",
                 density=True, label=f"Warm (xi={gev_warm['xi']:.2f})")
    axes[1].hist(extended_data, bins=bins, alpha=0.5, color="tab:green",
                 density=True, label=f"100-warmup (xi={gev_extended['xi']:.2f})")
    axes[1].set_title("Latency distributions")
    axes[1].set_xlabel("Latency (ms)")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=6)

    # Right: GEV xi summary
    conditions = ["All 600\n(no wu)", "Warm 550\n(no wu)", "100 wu\nno sleep", "Ref\n(20wu,nosleep)"]
    xi_vals    = [gev_cold_all["xi"], gev_warm["xi"], gev_extended["xi"], 1.36]
    colors     = ["tab:red", "tab:blue", "tab:green", "tab:gray"]
    bars = axes[2].bar(conditions, xi_vals, color=colors, alpha=0.8, edgecolor="black", lw=0.5)
    axes[2].axhline(0.15, ls="--", color="black", lw=1, label="Gumbel threshold (xi<0.15)")
    axes[2].set_title("GEV shape ξ comparison")
    axes[2].set_ylabel("GEV shape parameter ξ")
    axes[2].legend(fontsize=6)
    for bar, val in zip(bars, xi_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e02_warmup_study.png", dpi=150)
    plt.close()

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "e02_outlier_warmup_study",
        "seq_len": SEQ_LEN,
        "cold_split": SPLIT,
        "cold_phase": {
            "n": SPLIT,
            "mean_ms": round(float(cold_phase.mean()), 3),
            "p99_ms":  round(float(np.percentile(cold_phase, 99)), 3),
            "max_ms":  round(float(cold_phase.max()), 3),
            "outliers_above_threshold": cold_outliers,
        },
        "warm_phase": {
            "n": len(warm_phase),
            "mean_ms": round(float(warm_phase.mean()), 3),
            "p99_ms":  round(float(np.percentile(warm_phase, 99)), 3),
            "max_ms":  round(float(warm_phase.max()), 3),
            "outliers_above_threshold": warm_outliers,
        },
        "outlier_threshold_ms": round(float(outlier_threshold), 3),
        "gev_cold_all":  gev_cold_all,
        "gev_warm":      gev_warm,
        "gev_extended":  gev_extended,
        "ad_cold_all":   ad_cold_all,
        "ad_warm":       ad_warm,
        "ad_extended":   ad_extended,
        "reference_xi":  1.36,
        "reference_ad":  17.5465,
        "conclusion": (
            "warm_up_artifact"
            if gev_warm["xi"] < gev_cold_all["xi"] * 0.6
            else "intrinsic_heavy_tail"
        ),
    }
    write_results(output, RESULTS_DIR / "e02_outlier_warmup.json")
    print("\nPASS: E02 complete\n")


if __name__ == "__main__":
    main()
