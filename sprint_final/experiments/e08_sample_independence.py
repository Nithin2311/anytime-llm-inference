"""
e08_sample_independence.py — Ljung-Box and ACF for WCET sample independence.

Addresses reviewer concern that EVT requires i.i.d. samples.
Runs Ljung-Box test (H0: no autocorrelation) and computes sample ACF
for WCET timing measurements. Also checks for stationarity via
rolling mean/std over the time series.
"""

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from evt_utils import ljung_box_test, sample_acf
from fig_style import apply_style, DOUBLE, QUAD
import matplotlib.pyplot as plt
from result_writer import write_results

apply_style()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLES    = 1000
N_WARMUP     = 20
MAX_LAG      = 40
SEQ_LENGTHS  = [64, 128, 512]
EXIT_LAYERS  = [11, 16, None]
DEVICE       = "cuda"


def profile_sequence(model, seq_len, exit_layer, n_samples, n_warmup):
    """Return ordered timing sequence (preserving serial order)."""
    import warnings
    times = []
    dummy = torch.randint(100, 2000, (1, seq_len), device=DEVICE)

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


def rolling_stats(samples, window=50):
    """Compute rolling mean and std over the time series."""
    n = len(samples)
    means, stds, idx = [], [], []
    for i in range(0, n - window + 1, window // 2):
        chunk = samples[i:i + window]
        means.append(np.mean(chunk))
        stds.append(np.std(chunk))
        idx.append(i + window // 2)
    return np.array(idx), np.array(means), np.array(stds)


def main():
    print("=" * 60)
    print("E08  WCET sample independence: Ljung-Box + ACF")
    print(f"     N={N_SAMPLES}, max_lag={MAX_LAG}")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    results = {}
    cells = [(s, l) for s in SEQ_LENGTHS for l in EXIT_LAYERS]
    total = len(cells)

    # Figure: 2 representative cells — ACF + rolling mean
    fig_acf, axes = plt.subplots(len(SEQ_LENGTHS), 2, figsize=DOUBLE)

    for cell_i, (seq, layer) in enumerate(cells):
        key = f"seq{seq}_L{layer}"
        print(f"\n  [{cell_i+1}/{total}] seq={seq}, layer={layer} ...")

        samples = profile_sequence(model, seq, layer, N_SAMPLES, N_WARMUP)

        lb  = ljung_box_test(samples, max_lag=MAX_LAG)
        acf = sample_acf(samples, n_lags=MAX_LAG)
        rolling_i, rolling_m, rolling_s = rolling_stats(samples, window=50)

        # Confidence band for ACF: ±1.96/sqrt(N)
        conf_band = 1.96 / np.sqrt(N_SAMPLES)

        results[key] = {
            "seq": seq, "layer": str(layer),
            "n_samples": N_SAMPLES,
            "mean_ms": round(float(np.mean(samples)), 4),
            "std_ms":  round(float(np.std(samples)), 4),
            "ljung_box": lb,
            "acf_lags": [int(lag) for lag, _ in acf],
            "acf_vals": [float(v) for _, v in acf],
            "conf_band_95pct": round(float(conf_band), 6),
            "n_lags_outside_band": int(sum(
                1 for _, v in acf if abs(v) > conf_band
            )),
            "rolling_mean_range_ms": round(float(
                np.max(rolling_m) - np.min(rolling_m)
            ), 4),
        }

        p_val = lb.get("lb_pvalue", None)
        indep = lb.get("independent_at_5pct", None)
        print(f"    LB p-value={p_val}  independent={indep}  "
              f"ACF outside band: {results[key]['n_lags_outside_band']}/{MAX_LAG}")

        # Plot for first 3 unique seq lengths (one layer each: None=full)
        plot_row = SEQ_LENGTHS.index(seq) if layer is None else -1
        if plot_row >= 0 and plot_row < len(SEQ_LENGTHS):
            lags = [lag for lag, _ in acf]
            vals = [v for _, v in acf]

            ax_a = axes[plot_row][0]
            ax_a.bar(lags, vals, color="tab:blue", alpha=0.7)
            ax_a.axhline(conf_band,  ls="--", color="red", lw=1, label="±95% band")
            ax_a.axhline(-conf_band, ls="--", color="red", lw=1)
            ax_a.axhline(0, color="black", lw=0.5)
            ax_a.set_title(f"ACF  seq={seq} Full", fontsize=9)
            ax_a.set_xlabel("Lag")
            ax_a.set_ylabel("ACF")
            if plot_row == 0:
                ax_a.legend(fontsize=7)

            ax_r = axes[plot_row][1]
            ax_r.plot(rolling_i, rolling_m, color="tab:blue", label="Rolling mean")
            ax_r.fill_between(rolling_i,
                               rolling_m - rolling_s,
                               rolling_m + rolling_s,
                               alpha=0.3, color="tab:blue")
            ax_r.set_title(f"Rolling mean  seq={seq} Full", fontsize=9)
            ax_r.set_xlabel("Sample index")
            ax_r.set_ylabel("Latency [ms]")

    plt.suptitle("WCET Sample Independence Analysis", y=1.01)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "sample_independence.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    tex_path = RESULTS_DIR / "table_independence.tex"
    with open(tex_path, "w") as f:
        f.write("% E08: Sample independence results\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Seq & Layer & LB stat & LB $p$ & Indep? & ACF outside band \\\\\n\\midrule\n")
        for key, r in results.items():
            lbl = "Full" if r["layer"] == "None" else f"L{r['layer']}"
            lb  = r["ljung_box"]
            stat = lb.get("lb_stat", "N/A")
            pval = lb.get("lb_pvalue", "N/A")
            indep = "\\checkmark" if lb.get("independent_at_5pct") else "\\times"
            stat_str = f"{stat:.2f}" if isinstance(stat, float) else "N/A"
            pval_str = f"{pval:.4f}" if isinstance(pval, float) else "N/A"
            f.write(f"{r['seq']} & {lbl} & {stat_str} & {pval_str} & {indep} & "
                    f"{r['n_lags_outside_band']}/{MAX_LAG} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table: {tex_path}")

    write_results({
        "experiment": "e08_sample_independence",
        "n_samples": N_SAMPLES,
        "max_lag": MAX_LAG,
        "conf_band_formula": "1.96/sqrt(N)",
        "results": results,
    }, RESULTS_DIR / "sample_independence_results.json")
    print("PASS: E08 complete\n")


if __name__ == "__main__":
    main()
