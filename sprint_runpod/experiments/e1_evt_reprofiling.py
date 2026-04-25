"""
E1: EVT WCET re-profiling on A100 SXM4 — 500 runs/cell.

Critical academic fix: prior EVT results used RTX 4000 Ada (hardware mismatch
identified during peer review). This re-runs the full EVT sweep on A100 SXM4
to produce unified-hardware WCET bounds for Table XI in report_v2.tex.

Method: Peaks-Over-Threshold (POT) — fit Gumbel_r to top 20% of samples.
Outputs:
  results/evt_wcet_results.json
  figures/evt_wcet_analysis.png
  latex/table_xi_evt.tex          (Table XI body rows for report_v2.tex)
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import gumbel_r, probplot

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import result_writer as rw
from early_exit_model import EarlyExitTinyLlama

EXPERIMENT_ID = "E1"
RESULTS_FILE  = "evt_wcet_results.json"
HARDWARE      = "A100 SXM4"
SEQ_LENGTHS   = [32, 64, 128, 256, 512, 1024]
EXIT_LAYERS   = [5, 11, 16, None]
NUM_WARMUP    = 5
NUM_RUNS      = 500
TOP_FRACTION  = 0.20
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


def fit_gumbel_evt(samples):
    n_tail     = max(int(TOP_FRACTION * len(samples)), 10)
    tail       = np.sort(samples)[-n_tail:]
    loc, scale = gumbel_r.fit(tail)
    return {
        "gumbel_loc":   round(float(loc),   6),
        "gumbel_scale": round(float(scale), 6),
        "wcet_evt_1e4": round(float(gumbel_r.ppf(1.0 - 1e-4, loc=loc, scale=scale)), 4),
        "wcet_evt_1e6": round(float(gumbel_r.ppf(1.0 - 1e-6, loc=loc, scale=scale)), 4),
        "n_tail_used":  int(n_tail),
    }


def build_latex(all_results):
    lines = [
        f"% E1: EVT WCET Bounds — {HARDWARE} (500 runs/cell, POT top 20\\%)",
        "% Replaces Table XI body rows in report_v2.tex",
        r"% Columns: seq & Emp\times1.10 & pWCET($10^{-4}$) & pWCET($10^{-6}$) & $P(\text{miss}|D)$ \\",
        r"\midrule",
    ]
    full_key = "None"
    for sl in SEQ_LENGTHS:
        c    = all_results[str(sl)][full_key]
        x11  = c["wcet_empirical_x110"]
        e4   = c["wcet_evt_1e4"]
        e6   = c["wcet_evt_1e6"]
        miss = r"$<10^{-6}$" if e6 < DEADLINE_MS else r"$>10^{-6}${\dag}"
        lines.append(f"  {sl:>4} & {x11:.1f} & {e4:.1f} & {e6:.1f} & {miss} \\\\")
    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"


def main(dry_run=False):
    rw.configure(BASE_DIR)

    if dry_run:
        import scipy, matplotlib  # noqa: F401
        print(f"[{EXPERIMENT_ID}] dry-run OK | CUDA={torch.cuda.is_available()} | imports OK")
        sys.exit(0)

    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping — {RESULTS_FILE} exists.")
        sys.exit(0)

    t0 = rw.log_start(EXPERIMENT_ID)
    try:
        print(f"\n{'='*65}\nE1: EVT WCET ANALYSIS — {HARDWARE}\n"
              f"Seq: {SEQ_LENGTHS} | {NUM_RUNS} runs/cell | POT top {int(TOP_FRACTION*100)}%\n"
              f"Expected runtime: ~2-3 hours\n{'='*65}\n")

        model       = EarlyExitTinyLlama()
        all_results = {}
        total       = len(SEQ_LENGTHS) * len(EXIT_LAYERS)
        cell        = 0

        for sl in SEQ_LENGTHS:
            ids = make_input(sl)
            all_results[str(sl)] = {}
            for layer in EXIT_LAYERS:
                cell += 1
                tag = f"L{layer}" if layer is not None else "Full(22)"
                print(f"  [{cell:>2}/{total}] seq={sl:>4} exit={tag:<8}", end="  ", flush=True)

                samples = collect_samples(model, ids, layer)
                evt     = fit_gumbel_evt(samples)
                emp_max = float(np.max(samples))

                all_results[str(sl)][str(layer)] = {
                    "mean_ms":             round(float(np.mean(samples)), 4),
                    "empirical_max_ms":    round(emp_max, 4),
                    "wcet_empirical_x110": round(emp_max * 1.10, 4),
                    "wcet_evt_1e4":        evt["wcet_evt_1e4"],
                    "wcet_evt_1e6":        evt["wcet_evt_1e6"],
                    "gumbel_loc":          evt["gumbel_loc"],
                    "gumbel_scale":        evt["gumbel_scale"],
                    "n_samples":           NUM_RUNS,
                }
                flag = "UNDER ⚠" if evt["wcet_evt_1e6"] > emp_max * 1.10 else "ok"
                print(f"mean={np.mean(samples):.2f}  EVT(1e-4)={evt['wcet_evt_1e4']:.2f}"
                      f"  EVT(1e-6)={evt['wcet_evt_1e6']:.2f}  {flag}")

        saved = rw.write_json(RESULTS_FILE, {
            "hardware":     HARDWARE,
            "model":        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "n_samples":    NUM_RUNS,
            "top_fraction": TOP_FRACTION,
            "deadline_ms":  DEADLINE_MS,
            "seq_lengths":  SEQ_LENGTHS,
            "exit_layers":  [str(e) for e in EXIT_LAYERS],
            "results":      all_results,
        })
        print(f"\n  Results → {saved}")

        _plot(all_results, model)
        rw.write_latex("table_xi_evt.tex", build_latex(all_results))
        print(f"  LaTeX   → latex/table_xi_evt.tex")

        _print_summary(all_results)
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc)
        raise


def _print_summary(all_results):
    full_key = "None"
    print(f"\n{'='*75}\nEVT SUMMARY — Full(22) Pass\n"
          f"{'Seq':>6}  {'Mean':>8}  {'EmpMax':>8}  {'×1.10':>8}  "
          f"{'EVT(1e-4)':>10}  {'EVT(1e-6)':>10}  {'<D=45?':>8}")
    print("-" * 75)
    for sl in SEQ_LENGTHS:
        c  = all_results[str(sl)][full_key]
        ok = "YES" if c["wcet_evt_1e6"] < DEADLINE_MS else "NO ⚠"
        print(f"{sl:>6}  {c['mean_ms']:>8.3f}  {c['empirical_max_ms']:>8.3f}"
              f"  {c['wcet_empirical_x110']:>8.3f}  {c['wcet_evt_1e4']:>10.3f}"
              f"  {c['wcet_evt_1e6']:>10.3f}  {ok:>8}")
    print("=" * 75)


def _plot(all_results, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import fig_style as fs

    fs.apply()
    full_key = "None"
    fig = plt.figure(figsize=fs.TRIPLE)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    emp_x110 = [all_results[str(sl)][full_key]["wcet_empirical_x110"] for sl in SEQ_LENGTHS]
    evt_1e4  = [all_results[str(sl)][full_key]["wcet_evt_1e4"]        for sl in SEQ_LENGTHS]
    evt_1e6  = [all_results[str(sl)][full_key]["wcet_evt_1e6"]        for sl in SEQ_LENGTHS]

    ax1.plot(SEQ_LENGTHS, emp_x110, marker="o", linewidth=2, color="#1f77b4", label="Emp max × 1.10")
    ax1.plot(SEQ_LENGTHS, evt_1e4,  marker="s", linewidth=2, color="#ff7f0e", label="EVT (p=10⁻⁴)")
    ax1.plot(SEQ_LENGTHS, evt_1e6,  marker="^", linewidth=2, color="#d62728", label="EVT (p=10⁻⁶)")
    ax1.axhline(y=DEADLINE_MS, color="black", linestyle="--", linewidth=1.1,
                label=f"D={DEADLINE_MS:.0f} ms")
    ax1.set_xlabel("Sequence Length (tokens)")
    ax1.set_ylabel("WCET Bound (ms)")
    ax1.set_title("WCET Bounds: Empirical vs EVT")
    ax1.set_xticks(SEQ_LENGTHS)
    ax1.legend(fontsize=8)

    # Gumbel QQ at seq=128, full pass
    c128 = all_results["128"][full_key]
    ids128 = make_input(128)
    qq_samp = collect_samples(model, ids128, None)
    (osm, osr), (slope, intercept, r) = probplot(
        qq_samp, dist=gumbel_r, sparams=(c128["gumbel_loc"], c128["gumbel_scale"])
    )
    ax2.scatter(osm, osr, s=10, alpha=0.6, color="#1f77b4", label="Sample quantiles")
    ax2.plot([min(osm), max(osm)],
             [slope * min(osm) + intercept, slope * max(osm) + intercept],
             color="#d62728", linewidth=1.5, label=f"Gumbel (R²={r**2:.4f})")
    ax2.set_xlabel("Theoretical Gumbel quantiles (ms)")
    ax2.set_ylabel("Observed latency quantiles (ms)")
    ax2.set_title("Gumbel Fit Quality (seq=128)")
    ax2.legend(fontsize=8)

    ax3.axis("off")
    col_labels = ["Seq", "×1.10", "10⁻⁴", "10⁻⁶", "<D?"]
    table_data = []
    for sl in SEQ_LENGTHS:
        c  = all_results[str(sl)][full_key]
        ok = "yes" if c["wcet_evt_1e6"] < DEADLINE_MS else "NO"
        table_data.append([str(sl), f"{c['wcet_empirical_x110']:.1f}",
                           f"{c['wcet_evt_1e4']:.1f}", f"{c['wcet_evt_1e6']:.1f}", ok])
    tbl = ax3.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.6)
    for col_idx in range(len(col_labels)):
        tbl[0, col_idx].set_facecolor("#d0e8f0")
        tbl[0, col_idx].set_text_props(weight="bold")
    ax3.set_title("EVT Summary — Full Pass", pad=12)

    fig.suptitle(f"EVT WCET Analysis — TinyLlama-1.1B ({HARDWARE})", fontsize=7.5, y=1.01)
    path = rw.figures_path("evt_wcet_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure  → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
