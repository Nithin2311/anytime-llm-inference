"""
E04: POT Fraction Sensitivity — justify 20% tail threshold choice.
1000 samples/cell, fractions {5,10,15,20,25,30}%.
Addresses: M3 (POT threshold not justified in paper).
"""
import argparse, os, sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import result_writer as rw
from early_exit_model import EarlyExitTinyLlama
import evt_utils as eu

EXPERIMENT_ID = "E04"
RESULTS_FILE  = "pot_sensitivity_results.json"
HARDWARE      = "A100 SXM4"
SEQ_LENGTHS   = [64, 128, 256, 512]       # subset — full pass only
NUM_WARMUP    = 10
NUM_RUNS      = 1000
FRACTIONS     = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def make_input(sl):
    return torch.randint(100, 30000, (1, sl), dtype=torch.long, device="cuda")


def collect(model, ids):
    with torch.inference_mode():
        for _ in range(NUM_WARMUP):
            model(ids, exit_layer=None, use_cache=False)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    with torch.inference_mode():
        for i in range(NUM_RUNS):
            starts[i].record()
            model(ids, exit_layer=None, use_cache=False)
            ends[i].record()
    torch.cuda.synchronize()
    return np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])


def _frac_key(f): return str(round(f, 2))


def main(dry_run=False):
    rw.configure(BASE_DIR)
    if dry_run:
        print(f"[{EXPERIMENT_ID}] dry-run OK"); sys.exit(0)
    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping."); sys.exit(0)

    t0    = rw.log_start(EXPERIMENT_ID)
    model = EarlyExitTinyLlama()
    all_results = {}

    try:
        for sl in SEQ_LENGTHS:
            ids = make_input(sl)
            print(f"  seq={sl}  collecting {NUM_RUNS} samples...", flush=True)
            samples = collect(model, ids)
            all_results[str(sl)] = {}

            for frac in FRACTIONS:
                g   = eu.fit_gumbel(samples, frac)
                gev = eu.fit_gev(samples, frac)
                ad  = eu.anderson_darling_gumbel(samples, frac)
                all_results[str(sl)][_frac_key(frac)] = {
                    "pot_fraction":    frac,
                    "n_tail":          g["n_tail"],
                    "gumbel_loc":      round(g["loc"], 4),
                    "gumbel_scale":    round(g["scale"], 4),
                    "wcet_evt_1e4":    round(g["wcet_1e4"], 4),
                    "wcet_evt_1e6":    round(g["wcet_1e6"], 4),
                    "gev_xi":          round(gev["xi"], 4),
                    "ad_stat":         round(ad["ad_stat"], 4),
                    "gumbel_valid":    ad["fit_not_rejected_at_5pct"],
                }
                print(f"    frac={frac:.0%}  n_tail={g['n_tail']:>3}  "
                      f"wcet_1e6={g['wcet_1e6']:.3f}  xi={gev['xi']:+.3f}  "
                      f"AD={'OK' if ad['fit_not_rejected_at_5pct'] else 'FAIL'}")

        # Compute stability: max pWCET variation across fractions (per seq)
        stability = {}
        for sl in SEQ_LENGTHS:
            vals = [all_results[str(sl)][_frac_key(f)]["wcet_evt_1e6"] for f in FRACTIONS]
            stability[str(sl)] = {
                "wcet_1e6_range_ms": round(max(vals) - min(vals), 4),
                "wcet_1e6_cv":       round(float(np.std(vals) / np.mean(vals)), 4),
            }

        rw.write_json(RESULTS_FILE, {
            "hardware": HARDWARE, "n_samples": NUM_RUNS,
            "seq_lengths": SEQ_LENGTHS, "fractions": FRACTIONS,
            "results": all_results, "stability": stability,
        })
        _build_latex(all_results, stability)
        _plot(all_results)
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc); raise


def _build_latex(all_results, stability):
    lines = [r"% E04: POT Sensitivity — Full Pass, pWCET(10^{-6}) vs fraction", r"\midrule"]
    for sl in SEQ_LENGTHS:
        for frac in FRACTIONS:
            c = all_results[str(sl)][_frac_key(frac)]
            bold = r"\textbf{" if frac == 0.20 else ""
            bende = r"}" if frac == 0.20 else ""
            lines.append(
                f"  {sl} & {bold}{int(frac*100)}\\%{bende} & {c['n_tail']} & "
                f"{c['wcet_evt_1e4']:.2f} & {bold}{c['wcet_evt_1e6']:.2f}{bende} & "
                f"{c['gev_xi']:+.3f} \\\\"
            )
        lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    rw.write_latex("table_pot_sensitivity.tex", "\n".join(lines) + "\n")


def _plot(all_results):
    import matplotlib.pyplot as plt
    import fig_style as fs
    fs.apply()
    n_seq = len(SEQ_LENGTHS)
    fig, axes = plt.subplots(1, n_seq, figsize=(7.0, 3.2))
    if n_seq == 1: axes = [axes]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for ax, sl in zip(axes, SEQ_LENGTHS):
        e6_vals = [all_results[str(sl)][_frac_key(f)]["wcet_evt_1e6"] for f in FRACTIONS]
        ax.plot([int(f*100) for f in FRACTIONS], e6_vals, marker="o", color="#1f77b4")
        ax.axvline(20, color="gray", ls="--", lw=0.8, label="20% (paper)")
        ax.set_xlabel("POT Fraction (%)"); ax.set_ylabel("pWCET(10⁻⁶) (ms)")
        ax.set_title(f"seq={sl}"); ax.legend()
    fig.suptitle(f"POT Sensitivity — Full Pass ({HARDWARE})", fontsize=8)
    plt.tight_layout()
    path = rw.figures_path("pot_sensitivity.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Figure → {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
