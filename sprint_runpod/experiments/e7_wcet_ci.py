"""
E7: Parametric bootstrap 95% CI on pWCET(1e-6) from Gumbel fit.

Academic reviewers asked for uncertainty bounds on the EVT-derived WCET
estimates. This experiment re-profiles 500 latency samples per cell (same
as E1), then bootstraps the Gumbel fit 1000 times by resampling the tail
to propagate fitting uncertainty into pWCET(1e-6) bounds.

Outputs:
  results/wcet_ci_results.json
  figures/wcet_ci.png
  latex/table_wcet_ci.tex
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

EXPERIMENT_ID = "E7"
RESULTS_FILE  = "wcet_ci_results.json"
SEQ_LENGTHS   = [32, 64, 128, 256, 512, 1024]
EXIT_LAYERS   = [5, 11, 16, None]
NUM_WARMUP    = 5
NUM_RUNS      = 500
TOP_FRACTION  = 0.20
N_BOOTSTRAP   = 1000
ALPHA         = 0.05
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


def fit_and_ci(samples, top_frac=TOP_FRACTION, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    n_tail     = max(int(top_frac * len(samples)), 10)
    tail       = np.sort(samples)[-n_tail:]
    loc, scale = gumbel_r.fit(tail)
    point_e6   = float(gumbel_r.ppf(1.0 - 1e-6, loc=loc, scale=scale))

    rng     = np.random.default_rng(42)
    boot_e6 = []
    for _ in range(n_boot):
        resamp    = rng.choice(tail, size=len(tail), replace=True)
        bl, bs    = gumbel_r.fit(resamp)
        boot_e6.append(float(gumbel_r.ppf(1.0 - 1e-6, loc=bl, scale=bs)))

    ci_lo = float(np.percentile(boot_e6, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_e6, 100 * (1 - alpha / 2)))
    return {
        "gumbel_loc":         round(float(loc),        6),
        "gumbel_scale":       round(float(scale),       6),
        "wcet_evt_1e6_point": round(point_e6,           4),
        "ci_lower":           round(ci_lo,               4),
        "ci_upper":           round(ci_hi,               4),
        "ci_width":           round(ci_hi - ci_lo,       4),
        "n_tail":             n_tail,
        "n_bootstrap":        n_boot,
    }


def build_latex(all_res):
    lines = [
        "% E7: pWCET(1e-6) Parametric Bootstrap CI -- A100 SXM4, 500 runs, B=1000",
        "% Full(22) pass only; primary WCET for scheduler admission",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Seq & pWCET (ms) & CI$_{95}$ lower & CI$_{95}$ upper & Width \\",
        r"\midrule",
    ]
    full_key = "None"
    for sl in SEQ_LENGTHS:
        r = all_res[str(sl)][full_key]
        lines.append(
            f"  {sl} & {r['wcet_evt_1e6_point']:.2f} & {r['ci_lower']:.2f}"
            f" & {r['ci_upper']:.2f} & {r['ci_width']:.2f} \\\\"
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
        print(f"\n{'='*65}\nE7: WCET PARAMETRIC BOOTSTRAP CI\n"
              f"Seq: {SEQ_LENGTHS} | {NUM_RUNS} runs/cell | B={N_BOOTSTRAP} bootstraps\n"
              f"Expected runtime: ~3-4 hours\n{'='*65}\n")

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
                samples = collect_samples(model, ids, layer)
                r       = fit_and_ci(samples)
                all_res[str(sl)][str(layer)] = r
                print(f"pWCET(1e-6)={r['wcet_evt_1e6_point']:.2f}  "
                      f"95%CI=[{r['ci_lower']:.2f},{r['ci_upper']:.2f}]  "
                      f"width={r['ci_width']:.2f} ms")

        saved = rw.write_json(RESULTS_FILE, {
            "hardware":     "A100 SXM4",
            "model":        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "n_samples":    NUM_RUNS,
            "top_fraction": TOP_FRACTION,
            "n_bootstrap":  N_BOOTSTRAP,
            "alpha":        ALPHA,
            "deadline_ms":  DEADLINE_MS,
            "seq_lengths":  SEQ_LENGTHS,
            "results":      all_res,
        })
        print(f"\n  Results --> {saved}")
        _plot(all_res)
        rw.write_latex("table_wcet_ci.tex", build_latex(all_res))
        print(f"  LaTeX   --> latex/table_wcet_ci.tex")
        _print_summary(all_res)
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc)
        raise


def _print_summary(all_res):
    full_key = "None"
    print(f"\n{'='*70}\nWCET CI SUMMARY -- Full(22) Pass\n"
          f"{'Seq':>6}  {'Point':>8}  {'CI_lo':>8}  {'CI_hi':>8}  "
          f"{'Width':>8}  {'<D=45?':>8}")
    print("-" * 70)
    for sl in SEQ_LENGTHS:
        r  = all_res[str(sl)][full_key]
        ok = "YES" if r["wcet_evt_1e6_point"] < DEADLINE_MS else "NO ⚠"
        print(f"{sl:>6}  {r['wcet_evt_1e6_point']:>8.3f}  {r['ci_lower']:>8.3f}"
              f"  {r['ci_upper']:>8.3f}  {r['ci_width']:>8.3f}  {ok:>8}")
    print("=" * 70)


def _plot(all_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as fs

    fs.apply()
    full_key = "None"
    points   = [all_res[str(sl)][full_key]["wcet_evt_1e6_point"] for sl in SEQ_LENGTHS]
    ci_lo    = [all_res[str(sl)][full_key]["ci_lower"]           for sl in SEQ_LENGTHS]
    ci_hi    = [all_res[str(sl)][full_key]["ci_upper"]           for sl in SEQ_LENGTHS]
    yerr     = [
        [p - lo for p, lo in zip(points, ci_lo)],
        [hi - p for p, hi in zip(points, ci_hi)],
    ]

    fig, ax = plt.subplots(figsize=fs.SINGLE)
    ax.errorbar(SEQ_LENGTHS, points, yerr=yerr, fmt="o-", linewidth=2,
                capsize=5, color="#2b5b84", label=r"pWCET($10^{-6}$) ± 95% CI")
    ax.axhline(y=DEADLINE_MS, color="red", linestyle="--", linewidth=1.2,
               label=f"D={DEADLINE_MS:.0f} ms")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("pWCET (ms)")
    ax.set_title(r"pWCET($10^{-6}$) Bootstrap CI -- Full Pass, A100 SXM4")
    ax.set_xticks(SEQ_LENGTHS)
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = rw.figures_path("wcet_ci.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure  --> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
