"""
E4: POT threshold sensitivity — how robust are pWCET bounds to the choice of
peak-fraction used for Gumbel fitting?

Re-profiles GPU latency at 500 runs/cell and fits Gumbel_r with POT fractions
in {10%, 15%, 20%, 25%}. Justifies the paper's choice of 20%.

Outputs:
  results/pot_sensitivity_results.json
  figures/pot_sensitivity.png
  latex/table_pot_sensitivity.tex
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import gumbel_r

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import result_writer as rw
from early_exit_model import EarlyExitTinyLlama

EXPERIMENT_ID = "E4"
RESULTS_FILE  = "pot_sensitivity_results.json"
POT_FRACTIONS = [0.10, 0.15, 0.20, 0.25]
SEQ_LENGTHS   = [32, 64, 128, 256, 512, 1024]
EXIT_LAYERS   = [5, 11, 16, None]
NUM_WARMUP    = 5
NUM_RUNS      = 500
DEADLINE_MS   = 45.0


def make_input(seq_len):
    return torch.randint(100, 30000, (1, seq_len), dtype=torch.long, device="cuda")


def collect_samples(model, input_ids, exit_layer):
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
    return np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])


def fit_pot(samples, fraction):
    n_tail     = max(int(fraction * len(samples)), 10)
    tail       = np.sort(samples)[-n_tail:]
    loc, scale = gumbel_r.fit(tail)
    return {
        "pot_fraction":  fraction,
        "n_tail":        n_tail,
        "gumbel_loc":    round(float(loc),   6),
        "gumbel_scale":  round(float(scale), 6),
        "wcet_evt_1e4":  round(float(gumbel_r.ppf(1.0 - 1e-4, loc=loc, scale=scale)), 4),
        "wcet_evt_1e6":  round(float(gumbel_r.ppf(1.0 - 1e-6, loc=loc, scale=scale)), 4),
    }


def _frac_key(f):
    return str(round(f, 2))


def build_latex(all_res):
    lines = [
        "% E4: POT Threshold Sensitivity -- A100 SXM4, seq=[32-1024], Full(22) pass",
        "% Justifies choice of 20% POT fraction used in E1/Table XI",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"POT\% & $n_{\text{tail}}$ & $\mu$ & $\sigma$ & "
        r"pWCET($10^{-4}$) & pWCET($10^{-6}$) & $\Delta_{20\%}$ (ms) \\",
        r"\midrule",
    ]
    ref20 = all_res["128"]["None"][_frac_key(0.20)]["wcet_evt_1e6"]
    for frac in POT_FRACTIONS:
        r     = all_res["128"]["None"][_frac_key(frac)]
        delta = r["wcet_evt_1e6"] - ref20
        mark  = r"~$\leftarrow$ paper" if abs(frac - 0.20) < 1e-9 else ""
        lines.append(
            f"  {int(frac*100)}\\% & {r['n_tail']} & {r['gumbel_loc']:.3f}"
            f" & {r['gumbel_scale']:.3f} & {r['wcet_evt_1e4']:.2f}"
            f" & {r['wcet_evt_1e6']:.2f} & {delta:+.2f}{mark} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(dry_run=False):
    rw.configure(BASE_DIR)

    if dry_run:
        import scipy  # noqa: F401
        print(f"[{EXPERIMENT_ID}] dry-run OK | CUDA={torch.cuda.is_available()} | imports OK")
        sys.exit(0)

    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping -- {RESULTS_FILE} exists.")
        sys.exit(0)

    t0 = rw.log_start(EXPERIMENT_ID)
    try:
        print(f"\n{'='*65}\nE4: POT THRESHOLD SENSITIVITY\n"
              f"POT fractions: {[int(f*100) for f in POT_FRACTIONS]}%\n"
              f"Seq: {SEQ_LENGTHS} | {NUM_RUNS} runs/cell\n"
              f"Expected runtime: ~2-3 hours\n{'='*65}\n")

        model   = EarlyExitTinyLlama()
        all_res = {}
        total   = len(SEQ_LENGTHS) * len(EXIT_LAYERS)
        cell    = 0

        for sl in SEQ_LENGTHS:
            ids = make_input(sl)
            all_res[str(sl)] = {}
            for layer in EXIT_LAYERS:
                cell += 1
                tag = f"L{layer}" if layer is not None else "Full(22)"
                print(f"  [{cell:>2}/{total}] seq={sl:>4} exit={tag:<8}", end="  ", flush=True)
                samples  = collect_samples(model, ids, layer)
                lkey     = str(layer)
                all_res[str(sl)][lkey] = {
                    _frac_key(f): fit_pot(samples, f) for f in POT_FRACTIONS
                }
                ref = all_res[str(sl)][lkey][_frac_key(0.20)]
                print(f"pWCET(1e-6)@20%={ref['wcet_evt_1e6']:.2f} ms")

        saved = rw.write_json(RESULTS_FILE, {
            "hardware":      "A100 SXM4",
            "model":         "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "n_samples":     NUM_RUNS,
            "pot_fractions": POT_FRACTIONS,
            "seq_lengths":   SEQ_LENGTHS,
            "deadline_ms":   DEADLINE_MS,
            "results":       all_res,
        })
        print(f"\n  Results --> {saved}")

        _plot(all_res)
        rw.write_latex("table_pot_sensitivity.tex", build_latex(all_res))
        print(f"  LaTeX   --> latex/table_pot_sensitivity.tex")
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc)
        raise


def _plot(all_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as fs

    fs.apply()
    full_key  = "None"
    fracs_pct = [int(f * 100) for f in POT_FRACTIONS]

    fig, axes = plt.subplots(1, 2, figsize=fs.DOUBLE)

    for sl in SEQ_LENGTHS:
        e6 = [all_res[str(sl)][full_key][_frac_key(f)]["wcet_evt_1e6"] for f in POT_FRACTIONS]
        axes[0].plot(fracs_pct, e6, marker="o", linewidth=1.5, label=f"seq={sl}")
    axes[0].axhline(y=DEADLINE_MS, color="red", linestyle="--", linewidth=1.2,
                    label=f"D={DEADLINE_MS:.0f} ms")
    axes[0].axvline(x=20, color="gray", linestyle=":", linewidth=1.0, label="20% (paper)")
    axes[0].set_xlabel("POT Fraction (%)")
    axes[0].set_ylabel(r"pWCET (ms) at $p=10^{-6}$")
    axes[0].set_title("pWCET vs POT Fraction (Full Pass)")
    axes[0].legend(fontsize=7, ncol=2)

    for sl in SEQ_LENGTHS:
        ref20  = all_res[str(sl)][full_key][_frac_key(0.20)]["wcet_evt_1e6"]
        deltas = [all_res[str(sl)][full_key][_frac_key(f)]["wcet_evt_1e6"] - ref20
                  for f in POT_FRACTIONS]
        axes[1].plot(fracs_pct, deltas, marker="o", linewidth=1.5, label=f"seq={sl}")
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    axes[1].axvline(x=20, color="gray", linestyle=":", linewidth=1.0, label="20% reference")
    axes[1].set_xlabel("POT Fraction (%)")
    axes[1].set_ylabel(r"$\Delta$ pWCET vs 20\% baseline (ms)")
    axes[1].set_title("pWCET Sensitivity (relative to 20%)")
    axes[1].legend(fontsize=7, ncol=2)

    fig.suptitle("POT Threshold Sensitivity -- Full Pass, A100 SXM4", fontsize=8, y=1.01)
    plt.tight_layout()
    path = rw.figures_path("pot_sensitivity.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure  --> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
