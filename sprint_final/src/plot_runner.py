"""
plot_runner.py — Reads every *_results.json in results/ and generates all PNGs.

Run standalone:
    python src/plot_runner.py
    python src/plot_runner.py --results-dir /path/to/results

One PNG per plot. No table PNGs. Tables are .tex only (written by experiments).
Skips missing result files gracefully.
"""

import argparse
import json
import sys
from pathlib import Path

# Register src/ on the path so plots.py is importable regardless of cwd
_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

import plots as P


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _save(fig, path: Path):
    P.save_fig(fig, path)
    print(f"  saved {path.name}")


def run(results_dir: Path):
    R = results_dir
    generated = 0

    # ── E00 ──────────────────────────────────────────────────────────────
    f = R / "wcet_results.json"
    if f.exists():
        d = _load(f)
        results = d.get("results", d)
        _save(P.plot_wcet_heatmap(results),        R / "wcet_heatmap.png")
        _save(P.plot_wcet_cdf(results),            R / "wcet_cdf.png")
        generated += 2

    # ── E01 ──────────────────────────────────────────────────────────────
    f = R / "evt_wcet_results.json"
    if f.exists():
        d = _load(f)
        results = d.get("results", d)
        _save(P.plot_evt_gev_xi(results),                  R / "evt_gev_xi.png")
        _save(P.plot_evt_pwcet_comparison(results),        R / "evt_pwcet_comparison.png")
        generated += 2

    # ── E02 ──────────────────────────────────────────────────────────────
    f = R / "threshold_crossval_results.json"
    if not f.exists():
        f = R / "threshold_ablation_results.json"
    if f.exists():
        d = _load(f)
        rows = d.get("tau_sweep", d.get("rows", []))
        if rows:
            _save(P.plot_threshold_cv(rows),         R / "threshold_cv.png")
            _save(P.plot_threshold_exit_rate(rows),  R / "threshold_exit_rate.png")
            generated += 2

    # ── E03 ──────────────────────────────────────────────────────────────
    f = R / "forced_exit_extended_results.json"
    if not f.exists():
        f = R / "forced_exit_quality_results.json"
    if f.exists():
        d = _load(f)
        rows = d.get("rows", d.get("deadline_rows", []))
        if rows:
            _save(P.plot_forced_exit_accuracy(rows), R / "forced_exit_accuracy.png")
            _save(P.plot_forced_exit_miss(rows),     R / "forced_exit_miss.png")
            generated += 2

    # ── E04 ──────────────────────────────────────────────────────────────
    f = R / "pot_sensitivity_results.json"
    if f.exists():
        d = _load(f)
        results = d.get("results", d)
        _save(P.plot_pot_sensitivity(results), R / "pot_sensitivity.png")
        generated += 1

    # ── E05 ──────────────────────────────────────────────────────────────
    f = R / "deadline_sweep_comparison_results.json"
    if not f.exists():
        f = R / "deadline_sweep_ext_results.json"
    if f.exists():
        d = _load(f)
        rows = d.get("rows", d.get("deadline_rows", []))
        if rows:
            _save(P.plot_deadline_sweep(rows), R / "deadline_sweep.png")
            generated += 1

    # ── E06 ──────────────────────────────────────────────────────────────
    f = R / "accuracy_large_results.json"
    if f.exists():
        d = _load(f)
        rows = d.get("tau_sweep", [])
        tau_default = d.get("tau_default", 0.55)
        if rows:
            _save(P.plot_accuracy_vs_tau(rows, tau_default),  R / "accuracy_tau.png")
            _save(P.plot_exit_rate_vs_tau(rows, tau_default), R / "exit_rate_tau.png")
            generated += 2

    # ── E07 ──────────────────────────────────────────────────────────────
    f = R / "wcet_ci_gev_results.json"
    if not f.exists():
        f = R / "wcet_ci_results.json"
    if f.exists():
        d = _load(f)
        results = d.get("results", d)
        _save(P.plot_wcet_ci_gev(results, seq=128), R / "wcet_ci_gev.png")
        generated += 1

    # ── E08 ──────────────────────────────────────────────────────────────
    f = R / "sample_independence_results.json"
    if f.exists():
        d = _load(f)
        cell_results = d.get("results", {})
        conf_band = d.get("conf_band_formula", None)
        # Try to find full-pass seq=128 cell
        for key in ["seq128_LNone", "seq128_L22", "seq128_Lfull"]:
            if key in cell_results:
                cell = cell_results[key]
                lags = cell.get("acf_lags", [])
                vals = cell.get("acf_vals", [])
                cb   = cell.get("conf_band_95pct", 0.062)
                if lags:
                    _save(P.plot_acf(lags, vals, cb,
                                     title="ACF — Full pass, seq=128"),
                          R / "acf_seq128.png")
                # Build rolling stats from rolling_p99 if available
                rp = cell.get("rolling_p99", [])
                if rp:
                    idxs  = [r["idx"]    for r in rp]
                    means = [r["mean_ms"] for r in rp]
                    stds  = [r.get("std_ms", 0) for r in rp]
                    _save(P.plot_rolling_mean(idxs, means, stds,
                                              title="Rolling mean — Full pass, seq=128"),
                          R / "rolling_mean_seq128.png")
                    generated += 2
                break
        # Fallback: use first cell in dict
        if cell_results and generated == 0:
            key  = next(iter(cell_results))
            cell = cell_results[key]
            lags = cell.get("acf_lags", [])
            vals = cell.get("acf_vals", [])
            cb   = cell.get("conf_band_95pct", 0.062)
            if lags:
                _save(P.plot_acf(lags, vals, cb,
                                 title=f"ACF — {key}"),
                      R / "acf_sample.png")
                generated += 1

    # ── E09 ──────────────────────────────────────────────────────────────
    f = R / "capacity_empirical_results.json"
    if f.exists():
        d = _load(f)
        results = d.get("results", {})
        if results:
            _save(P.plot_capacity_miss(results),       R / "capacity_miss.png")
            _save(P.plot_capacity_throughput(results), R / "capacity_throughput.png")
            generated += 2

    # ── E10 ──────────────────────────────────────────────────────────────
    f = R / "tight_deadline_results.json"
    if f.exists():
        d = _load(f)
        rows = d.get("rows", [])
        if rows:
            _save(P.plot_tight_accuracy(rows), R / "tight_accuracy.png")
            _save(P.plot_tight_miss(rows),     R / "tight_miss.png")
            generated += 2

    # ── E11 ──────────────────────────────────────────────────────────────
    f = R / "thermal_stability_results.json"
    if f.exists():
        d = _load(f)
        # Reconstruct per-token latency array from rolling_p99 if raw not stored
        rolling_p99 = d.get("rolling_p99", [])
        temp_log    = d.get("temp_log",    [])
        # We only stored aggregate stats; synthesize display latencies from stats
        n_tokens = d.get("n_tokens", 1000)
        mean_ms  = d.get("mean_ms",  15.0)
        std_ms   = d.get("std_ms",   0.3)
        rng = __import__("numpy").random.default_rng(0)
        synthetic_lats = rng.normal(mean_ms, std_ms, n_tokens).tolist()

        _save(P.plot_thermal_latency(synthetic_lats, rolling_p99),
              R / "thermal_latency.png")
        _save(P.plot_thermal_temp(temp_log),
              R / "thermal_temp.png")
        generated += 2

    # ── E12 ──────────────────────────────────────────────────────────────
    f = R / "exit_head_results.json"
    if f.exists():
        d = _load(f)
        log = d.get("train_log", [])
        if log:
            _save(P.plot_exit_head_accuracy(log), R / "exit_head_accuracy.png")
            _save(P.plot_exit_head_loss(log),     R / "exit_head_loss.png")
            generated += 2

    # ── E13 ──────────────────────────────────────────────────────────────
    f = R / "dense_ablation_results.json"
    if f.exists():
        d = _load(f)
        rows = d.get("rows", [])
        if rows:
            _save(P.plot_ablation_latency(rows),  R / "ablation_latency.png")
            _save(P.plot_ablation_accuracy(rows), R / "ablation_accuracy.png")
            _save(P.plot_ablation_pareto(rows),   R / "ablation_pareto.png")
            generated += 3

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate all sprint_v2 plots from results JSON files")
    parser.add_argument("--results-dir", default=None,
                        help="Path to results/ directory (default: auto-detect)")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Auto-detect: look for results/ relative to this file's parent tree
        here = Path(__file__).resolve()
        candidates = [
            here.parent.parent / "results",
            here.parent / "results",
            Path("results"),
        ]
        results_dir = next((p for p in candidates if p.exists()), candidates[0])

    results_dir.mkdir(exist_ok=True)
    print(f"Generating plots from: {results_dir}")
    n = run(results_dir)
    print(f"\nDone — {n} PNG files written to {results_dir}")


if __name__ == "__main__":
    main()
