"""
e07_wcet_ci_gev.py — pWCET bootstrap CI comparison: Gumbel vs GEV.

Addresses M3: missing bootstrap CI on pWCET, and R2-M1: missing GEV comparison.
Runs 1000 timing samples per cell, fits both Gumbel and GEV,
computes parametric bootstrap CI (1000 resamples) for pWCET(1e-6),
reports AD GOF test result per cell.
"""

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from evt_utils import (
    fit_gumbel, fit_gev, anderson_darling_gumbel,
    bootstrap_wcet_ci, ljung_box_test
)
from fig_style import apply_style, DOUBLE, QUAD
import matplotlib.pyplot as plt
from result_writer import write_results

apply_style()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
N_SAMPLES     = 1000
N_WARMUP      = 20
N_BOOTSTRAP   = 1000
POT_FRACTION  = 0.20
SEQ_LENGTHS   = [64, 128, 256, 512]
EXIT_LAYERS   = [5, 11, 16, None]  # None = full pass (22 layers)
DEVICE        = "cuda"


def profile_cell(model, seq_len, exit_layer, n_samples, n_warmup):
    """Collect n_samples CUDA-event timings for one (seq_len, exit_layer) cell."""
    import warnings
    times = []
    dummy = torch.randint(100, 2000, (1, seq_len), device=DEVICE)

    # Warmup
    for _ in range(n_warmup):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.inference_mode():
                model.forward_cached(dummy)
        torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    for _ in range(n_samples):
        start_evt.record()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.inference_mode():
                if exit_layer is None:
                    model.forward_cached(dummy)
                else:
                    model.forward_cached(dummy, exit_layer=exit_layer)
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))

    return np.array(times, dtype=np.float64)


def main():
    print("=" * 60)
    print("E07  WCET bootstrap CI: Gumbel vs GEV comparison")
    print(f"     N={N_SAMPLES} samples/cell, bootstrap={N_BOOTSTRAP}")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    results = {}
    total_cells = len(SEQ_LENGTHS) * len(EXIT_LAYERS)
    cell_idx = 0

    for seq in SEQ_LENGTHS:
        results[str(seq)] = {}
        for layer in EXIT_LAYERS:
            cell_idx += 1
            layer_key = str(layer)
            print(f"\n  [{cell_idx}/{total_cells}] seq={seq}, layer={layer} ...")

            t0 = time.time()
            samples = profile_cell(model, seq, layer, N_SAMPLES, N_WARMUP)
            elapsed = time.time() - t0
            print(f"    Collected {N_SAMPLES} samples in {elapsed:.1f}s")

            g   = fit_gumbel(samples, fraction=POT_FRACTION)
            gev = fit_gev(samples, fraction=POT_FRACTION)
            ad  = anderson_darling_gumbel(samples, fraction=POT_FRACTION)
            bci = bootstrap_wcet_ci(samples, fraction=POT_FRACTION,
                                     n_bootstrap=N_BOOTSTRAP)
            lb  = ljung_box_test(samples)

            cell_result = {
                "seq": seq, "layer": layer,
                "n_samples": N_SAMPLES,
                "mean_ms":   round(float(np.mean(samples)), 4),
                "p99_ms":    round(float(np.percentile(samples, 99)), 4),
                "emp_max_ms":round(float(np.max(samples)), 4),
                "gumbel":    g,
                "gev":       gev,
                "ad_test":   ad,
                "bootstrap_ci": bci,
                "ljung_box": lb,
            }
            results[str(seq)][layer_key] = cell_result

            xi = gev["xi"]
            print(f"    Gumbel pWCET(1e-6)={g['wcet_1e6']:.3f}ms  "
                  f"GEV xi={xi:.4f} ({'valid' if gev['gumbel_valid'] else 'INVALID'})  "
                  f"AD={'pass' if ad.get('fit_not_rejected_at_5pct') else 'FAIL'}  "
                  f"CI=[{bci['ci_lower']:.3f},{bci['ci_upper']:.3f}]")

    # ── Figures ───────────────────────────────────────────────────────────
    # 2×2 panel: pWCET Gumbel vs GEV for seq=[64,128,256,512] (full pass)
    fig, axes = plt.subplots(2, 2, figsize=QUAD)
    for i, seq in enumerate(SEQ_LENGTHS):
        ax = axes[i // 2][i % 2]
        cell = results[str(seq)]["None"]
        layers_plot = [5, 11, 16, None]
        gumbel_wcets = []
        gev_wcets    = []
        ci_lo, ci_hi = [], []
        for L in layers_plot:
            c = results[str(seq)][str(L)]
            gumbel_wcets.append(c["gumbel"]["wcet_1e6"])
            gev_wcets.append(c["gev"]["wcet_1e6"])
            ci_lo.append(c["bootstrap_ci"]["ci_lower"])
            ci_hi.append(c["bootstrap_ci"]["ci_upper"])

        x = np.arange(len(layers_plot))
        labels = ["L5", "L11", "L16", "Full"]
        ax.bar(x - 0.2, gumbel_wcets, 0.35, label="Gumbel", color="tab:blue", alpha=0.8)
        ax.bar(x + 0.2, gev_wcets, 0.35, label="GEV", color="tab:orange", alpha=0.8)
        ax.errorbar(x - 0.2, gumbel_wcets,
                    yerr=[np.array(gumbel_wcets) - np.array(ci_lo),
                          np.array(ci_hi) - np.array(gumbel_wcets)],
                    fmt="none", color="black", capsize=3, lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"seq={seq}")
        ax.set_ylabel("pWCET(1e-6) [ms]")
        if i == 0:
            ax.legend(fontsize=7)

    plt.suptitle("pWCET(1e-6): Gumbel vs GEV with Bootstrap CI")
    plt.tight_layout()
    fig_path = RESULTS_DIR / "wcet_ci_gev.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    tex_path = RESULTS_DIR / "table_wcet_ci_gev.tex"
    with open(tex_path, "w") as f:
        f.write("% E07: WCET CI Gumbel vs GEV\n")
        f.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\n")
        f.write("Seq & Layer & Gumbel pWCET & GEV pWCET & $\\xi$ & CI lo & CI hi & AD & LB \\\\\n")
        f.write("& & (ms) & (ms) & & (ms) & (ms) & 5\\% & $p$ \\\\\n\\midrule\n")
        for seq in SEQ_LENGTHS:
            for layer in EXIT_LAYERS:
                c  = results[str(seq)][str(layer)]
                lbl = "Full" if layer is None else f"L{layer}"
                ad_pass = "\\checkmark" if c["ad_test"].get("fit_not_rejected_at_5pct") else "\\times"
                lb_p    = c["ljung_box"].get("lb_pvalue", "N/A")
                lb_str  = f"{lb_p:.3f}" if isinstance(lb_p, float) else "N/A"
                f.write(f"{seq} & {lbl} & "
                        f"{c['gumbel']['wcet_1e6']:.3f} & "
                        f"{c['gev']['wcet_1e6']:.3f} & "
                        f"{c['gev']['xi']:.4f} & "
                        f"{c['bootstrap_ci']['ci_lower']:.3f} & "
                        f"{c['bootstrap_ci']['ci_upper']:.3f} & "
                        f"{ad_pass} & {lb_str} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table: {tex_path}")

    write_results({
        "experiment": "e07_wcet_ci_gev",
        "n_samples": N_SAMPLES,
        "n_bootstrap": N_BOOTSTRAP,
        "pot_fraction": POT_FRACTION,
        "seq_lengths": SEQ_LENGTHS,
        "exit_layers": [str(l) for l in EXIT_LAYERS],
        "results": results,
    }, RESULTS_DIR / "wcet_ci_gev_results.json")
    print("PASS: E07 complete\n")


if __name__ == "__main__":
    main()
