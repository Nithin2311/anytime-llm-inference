"""
evt_wcet_analysis.py — Extreme Value Theory (EVT) WCET bounds.

Derives statistically rigorous worst-case execution time estimates from 500
timing samples per (seq_len, exit_layer) cell by fitting a Gumbel (Type-I
Extreme Value) distribution to the upper tail of the observed latency data.

Why EVT?
  The empirical WCET (max × 1.10) is a single-sample bound — it grows slowly
  with n and gives no probabilistic guarantee. EVT extrapolates the tail to
  exceedance probabilities far below 1/n (one-in-a-million, one-in-ten-thousand),
  giving a defensible WCET bound for hard real-time claims.

Method:
  Peaks-Over-Threshold (POT): fit Gumbel_r to the top 20% of samples.
  (Using the shared LM head logit space; only latency values are analysed.)

Outputs:
  evt_wcet_results.json
  evt_wcet_analysis.png
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gumbel_r, genextreme, anderson, probplot

from early_exit_model import EarlyExitTinyLlama
from profile_wcet import make_input

RESULTS_FILE  = "evt_wcet_results.json"
FIGURE_FILE   = "evt_wcet_analysis.png"
SEQ_LENGTHS   = [32, 64, 128, 256, 512, 1024]
EXIT_LAYERS   = [5, 11, 16, None]          # None = full 22-layer pass
NUM_WARMUP    = 5
NUM_RUNS      = 500                        # richer tail than the 50-run profile
TOP_FRACTION  = 0.20                       # fit Gumbel to the top 20% of samples
DEADLINE_MS  = 30.0


# ── GPU timing (raw samples) ───────────────────────────────────────────────────

def collect_samples(model, input_ids, exit_layer, n_warmup, n_runs):
    """
    Collect n_runs GPU-timed forward-pass durations via CUDA events.

    Returns a numpy array of shape (n_runs,) with times in ms.
    """
    with torch.inference_mode():
        for _ in range(n_warmup):
            model(input_ids, exit_layer=exit_layer, use_cache=False)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]

    with torch.inference_mode():
        for i in range(n_runs):
            start_events[i].record()
            model(input_ids, exit_layer=exit_layer, use_cache=False)
            end_events[i].record()

    torch.cuda.synchronize()
    return np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])


# ── EVT fitting ────────────────────────────────────────────────────────────────

def fit_gumbel_evt(samples):
    """
    Fit Gumbel_r to the top TOP_FRACTION of samples (POT method) and
    extrapolate to exceedance probabilities 1e-4 and 1e-6.

    Returns a dict with:
      loc, scale           : Gumbel location and scale parameters
      wcet_evt_1e4         : latency at exceedance probability 10^-4
      wcet_evt_1e6         : latency at exceedance probability 10^-6
    """
    n_tail  = max(int(TOP_FRACTION * len(samples)), 10)
    tail    = np.sort(samples)[-n_tail:]               # top 20%

    loc, scale = gumbel_r.fit(tail)

    # ppf(p) = value x such that P(X <= x) = p
    # exceedance probability q → cumulative probability 1 - q
    wcet_1e4 = float(gumbel_r.ppf(1.0 - 1e-4, loc=loc, scale=scale))
    wcet_1e6 = float(gumbel_r.ppf(1.0 - 1e-6, loc=loc, scale=scale))

    return {
        "gumbel_loc":   round(float(loc),   6),
        "gumbel_scale": round(float(scale), 6),
        "wcet_evt_1e4": round(wcet_1e4, 4),
        "wcet_evt_1e6": round(wcet_1e6, 4),
        "n_tail_used":  int(n_tail),
    }


def fit_gev_evt(samples):
    """
    Fit GEV (genextreme) to the top TOP_FRACTION of samples and run
    Anderson-Darling test for Gumbel (xi=0).  scipy sign convention: c = -xi.
    """
    n_tail = max(int(TOP_FRACTION * len(samples)), 10)
    tail   = np.sort(samples)[-n_tail:]
    c_fit, loc, scale = genextreme.fit(tail)
    xi = -c_fit  # positive xi = Fréchet (heavy tail)
    wcet_1e4 = float(genextreme.ppf(1.0 - 1e-4, c_fit, loc=loc, scale=scale))
    wcet_1e6 = float(genextreme.ppf(1.0 - 1e-6, c_fit, loc=loc, scale=scale))
    ad       = anderson(tail, dist="gumbel_r")
    crit5    = float(ad.critical_values[2]) if len(ad.critical_values) >= 3 else None
    return {
        "gev_xi":          round(float(xi), 4),
        "gev_loc":         round(float(loc), 6),
        "gev_scale":       round(float(scale), 6),
        "gev_wcet_1e4":    round(wcet_1e4, 4) if np.isfinite(wcet_1e4) else None,
        "gev_wcet_1e6":    round(wcet_1e6, 4) if np.isfinite(wcet_1e6) else None,
        "ad_stat":         round(float(ad.statistic), 4),
        "ad_crit_5pct":    crit5,
        "gumbel_rejected": bool(ad.statistic > crit5) if crit5 is not None else None,
        "n_tail_used":     int(n_tail),
    }


# ── Main sweep ─────────────────────────────────────────────────────────────────

def run_evt_sweep(model):
    """
    Collect 500 timing samples per (seq_len, exit_layer) cell and fit EVT.
    Returns the nested results dict ready for JSON serialisation.
    """
    all_results = {}
    total = len(SEQ_LENGTHS) * len(EXIT_LAYERS)
    cell  = 0

    for seq_len in SEQ_LENGTHS:
        input_ids = make_input(seq_len, device="cuda")
        all_results[str(seq_len)] = {}

        for exit_layer in EXIT_LAYERS:
            cell += 1
            tag   = f"L{exit_layer}" if exit_layer is not None else "Full(22)"
            print(f"  [{cell:>2}/{total}] seq={seq_len:>4}  exit={tag:<8}", end="  ", flush=True)

            samples = collect_samples(model, input_ids, exit_layer, NUM_WARMUP, NUM_RUNS)
            evt     = fit_gumbel_evt(samples)
            gev     = fit_gev_evt(samples)

            emp_max = float(np.max(samples))
            mean_ms = float(np.mean(samples))

            cell_result = {
                "mean_ms":            round(mean_ms, 4),
                "empirical_max_ms":   round(emp_max, 4),
                "wcet_empirical_x110":round(emp_max * 1.10, 4),
                "wcet_evt_1e4":       evt["wcet_evt_1e4"],
                "wcet_evt_1e6":       evt["wcet_evt_1e6"],
                "gumbel_loc":         evt["gumbel_loc"],
                "gumbel_scale":       evt["gumbel_scale"],
                "gev_xi":             gev["gev_xi"],
                "gev_loc":            gev["gev_loc"],
                "gev_scale":          gev["gev_scale"],
                "gev_wcet_1e4":       gev["gev_wcet_1e4"],
                "gev_wcet_1e6":       gev["gev_wcet_1e6"],
                "ad_stat":            gev["ad_stat"],
                "ad_crit_5pct":       gev["ad_crit_5pct"],
                "gumbel_rejected":    gev["gumbel_rejected"],
                "n_samples":          NUM_RUNS,
            }
            all_results[str(seq_len)][str(exit_layer)] = cell_result

            underest = "⚠ UNDER" if evt["wcet_evt_1e6"] > emp_max * 1.10 else "  ok  "
            print(
                f"mean={mean_ms:.2f}  empMax={emp_max:.2f}  "
                f"EVT(1e-4)={evt['wcet_evt_1e4']:.2f}  "
                f"EVT(1e-6)={evt['wcet_evt_1e6']:.2f}  {underest}"
            )

    return all_results


# ── Figure ─────────────────────────────────────────────────────────────────────

def plot_evt(all_results, qq_model=None):
    fs.apply()

    full_key = "None"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel 1: WCET bounds vs sequence length (Full pass) ───────────────────
    ax1 = axes[0]
    emp_x110 = [all_results[str(sl)][full_key]["wcet_empirical_x110"] for sl in SEQ_LENGTHS]
    evt_1e4  = [all_results[str(sl)][full_key]["wcet_evt_1e4"]        for sl in SEQ_LENGTHS]
    evt_1e6  = [all_results[str(sl)][full_key]["wcet_evt_1e6"]        for sl in SEQ_LENGTHS]

    ax1.plot(SEQ_LENGTHS, emp_x110, marker="o", linewidth=2, color="#1f77b4",
             label="Empirical max × 1.10")
    ax1.plot(SEQ_LENGTHS, evt_1e4,  marker="s", linewidth=2, color="#ff7f0e",
             label="EVT bound (p=10⁻⁴)")
    ax1.plot(SEQ_LENGTHS, evt_1e6,  marker="^", linewidth=2, color="#d62728",
             label="EVT bound (p=10⁻⁶)")
    ax1.axhline(y=DEADLINE_MS, color="black", linestyle="--", linewidth=1.1,
                label=f"Deadline ({DEADLINE_MS:.0f} ms)")

    ax1.set_xlabel("Input Sequence Length (tokens)")
    ax1.set_ylabel("WCET Bound (ms)")
    ax1.set_title("WCET Bounds: Empirical vs EVT\n(Full 22-layer pass)")
    ax1.set_xticks(SEQ_LENGTHS)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(loc="upper left", fontsize=8)

    for sl, e_x, v in zip(SEQ_LENGTHS, emp_x110, evt_1e6):
        if v > e_x:
            ax1.annotate("!", xy=(sl, v), xytext=(sl, v + 0.5),
                         ha="center", fontsize=9, fontweight="bold", color="#d62728")

    # ── Panel 2: Exceedance P(T > deadline) vs deadline — per exit layer ──────
    ax2 = axes[1]
    layer_colours = {5: "#d62728", 11: "#ff7f0e", 16: "#2b5b84", "None": "#2ca02c"}
    layer_labels  = {5: "L5", 11: "L11", 16: "L16", "None": "Full (L22)"}

    deadline_range = np.linspace(8, 70, 400)
    seq_key = "128"

    for layer_key in [5, 11, 16, "None"]:
        key = str(layer_key)
        if seq_key not in all_results or key not in all_results[seq_key]:
            continue
        cell  = all_results[seq_key][key]
        loc   = cell["gumbel_loc"]
        scale = cell["gumbel_scale"]
        exceedance = np.clip(1.0 - gumbel_r.cdf(deadline_range, loc=loc, scale=scale), 1e-9, 1.0)
        ax2.semilogy(deadline_range, exceedance, linewidth=2.2,
                     color=layer_colours[layer_key], label=layer_labels[layer_key])

    ax2.axvline(DEADLINE_MS, color="black", linestyle="--", linewidth=1.2,
                label=f"D = {DEADLINE_MS:.0f} ms")
    ax2.axhline(1e-4, color="grey", linestyle=":", linewidth=1.0, alpha=0.7, label="p = 10⁻⁴")
    ax2.set_xlabel("Deadline (ms)")
    ax2.set_ylabel("P(token latency > deadline)")
    ax2.set_title("Deadline Miss Probability (EVT)\n(seq = 128, Gumbel-fitted tail)")
    ax2.set_xlim(8, 70)
    ax2.set_ylim(1e-8, 1.1)
    ax2.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Extreme Value Theory WCET Analysis — TinyLlama-1.1B (RTX 6000 Ada)",
                 fontsize=8, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # scipy is required for Gumbel fitting
    try:
        from scipy.stats import gumbel_r  # noqa: F401
    except ImportError:
        raise SystemExit("scipy not found. Install with: pip install scipy")

    print("=" * 65)
    print("EVT WCET ANALYSIS — TinyLlama-1.1B")
    print(f"Seq lengths   : {SEQ_LENGTHS}")
    print(f"Exit layers   : {EXIT_LAYERS}  (None = full 22-layer pass)")
    print(f"Runs/cell     : {NUM_RUNS} (+ {NUM_WARMUP} warmup)")
    print(f"EVT tail frac : top {int(TOP_FRACTION*100)}% of samples")
    print("=" * 65 + "\n")

    model       = EarlyExitTinyLlama()
    all_results = run_evt_sweep(model)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "hardware":     "RTX 6000 Ada",
        "model":        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "n_samples":    NUM_RUNS,
        "top_fraction": TOP_FRACTION,
        "seq_lengths":  SEQ_LENGTHS,
        "exit_layers":  [str(e) for e in EXIT_LAYERS],
        "results":      all_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    # ── Figure ─────────────────────────────────────────────────────────────────
    plot_evt(all_results, qq_model=model)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("EVT WCET SUMMARY — Full(22) Pass")
    print(f"{'Seq':>6}  {'Mean':>8}  {'EmpMax':>8}  {'×1.10':>8}  "
          f"{'EVT(1e-4)':>10}  {'EVT(1e-6)':>10}  {'Status':>10}")
    print("-" * 75)
    full_key = "None"
    for sl in SEQ_LENGTHS:
        c   = all_results[str(sl)][full_key]
        e4  = c["wcet_evt_1e4"]
        e6  = c["wcet_evt_1e6"]
        x11 = c["wcet_empirical_x110"]
        status = "UNDER-EST ⚠" if e6 > x11 else "ok"
        print(f"{sl:>6}  {c['mean_ms']:>8.3f}  {c['empirical_max_ms']:>8.3f}  "
              f"{x11:>8.3f}  {e4:>10.3f}  {e6:>10.3f}  {status:>10}")
    print("=" * 75)
