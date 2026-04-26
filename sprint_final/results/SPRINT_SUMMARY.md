# Sprint Final — Experiment Summary

**Status:** ALL EXPERIMENTS PASSED
**Completed:** Sun Apr 26 19:09:20 UTC 2026
**GPU:** NVIDIA A100-SXM4-80GB (80 GB)
**Results:** PASS=14  FAIL=0

## Experiment Results

| #   | Experiment                       | Status | Review Issue Addressed |
|-----|----------------------------------|--------|------------------------|
| E00 | e00_wcet_reprofiling             | PASS   | C1: EVT re-baseline (500 samples) |
| E01 | e01_evt_gev_ad                   | PASS   | C1: GEV ξ validation + Anderson-Darling |
| E02 | e02_threshold_crossval           | PASS   | C2: τ generalization cal/holdout |
| E03 | e03_forced_exit_extended         | PASS   | R1: forced exit quality L5–L22 |
| E04 | e04_pot_sensitivity              | PASS   | C1: POT fraction robustness |
| E05 | e05_deadline_sweep_comparison    | PASS   | R2: router comparison, exit-rate confound |
| E06 | e06_accuracy_large               | PASS   | M1: 500-query CI ≈ ±7pp + conf histogram |
| E07 | e07_wcet_ci_gev                  | PASS   | M2: pWCET bootstrap CI Gumbel/GEV |
| E08 | e08_sample_independence          | PASS   | C1: Ljung-Box IID confirmation |
| E09 | e09_capacity_empirical           | PASS   | R2: empirical capacity N=1..4 |
| E10 | e10_tight_deadline               | PASS   | R3: tight deadline 14–30 ms |
| E11 | e11_thermal_a100                 | PASS   | R3: thermal stability soak |
| E12 | e12_exit_head_training           | PASS   | C2: exit-head MLP, exit rate ↑ |
| E13 | e13_dense_ablation               | PASS   | C2: dense layer sweep L12–L20 |

## Patches Applied During Run

The following compatibility patches were applied to make the experiment harness
match the calling conventions used in the experiment scripts:

1. `src/fig_style.py`        — added `apply_style` alias for `apply()`
2. `src/result_writer.py`    — added `write_results(data, path)` (atomic JSON to absolute path)
3. `src/early_exit_model.py` — added `EarlyExitModel(device=...)` alias subclass over `EarlyExitTinyLlama`
4. `src/early_exit_model.py` — added `exit_layer=` kwarg to `forward_cached()` (delegates to `forward()`)
5. `experiments/e11_thermal_a100.py` — guarded f-string against `None` from rolling-p99 before window fills
6. `src/benchmark_utils.py`  — `run_pubmed_queries_raw()` flexible signature
   accepting `(model, dataset, deadline_ms=, show_progress=, forced_exit_layer=)`
7. `src/benchmark_utils.py`  — `apply_threshold_posthoc()` flexible signature returning
   `correct_flags`, `exit_rate_pct`, `miss_rate_pct`, `mean_tpot_ms`, `p99_tpot_ms`
   alongside the legacy `early_exit_pct`/`deadline_miss_pct`/`global_*` fields

## Output Files

```
results/ablation_accuracy.png
results/ablation_latency.png
results/ablation_pareto.png
results/accuracy_large.png
results/accuracy_large_results.json
results/accuracy_tau.png
results/acf_seq128.png
results/capacity_empirical.png
results/capacity_empirical_results.json
results/capacity_miss.png
results/capacity_throughput.png
results/confidence_distribution.png
results/deadline_sweep_comparison_results.json
results/dense_ablation.png
results/dense_ablation_results.json
results/evt_results.json
results/exit_head_accuracy.png
results/exit_head_loss.png
results/exit_head_results.json
results/exit_head_training.png
results/exit_rate_tau.png
results/forced_exit_extended_results.json
results/pot_sensitivity.png
results/pot_sensitivity_results.json
results/sample_independence.png
results/sample_independence_results.json
results/table_accuracy_large.tex
results/table_capacity.tex
results/table_dense_ablation.tex
results/table_independence.tex
results/table_tight_deadline.tex
results/table_wcet_ci_gev.tex
results/thermal_latency.png
results/thermal_stability.png
results/thermal_stability_results.json
results/thermal_temp.png
results/threshold_crossval_results.json
results/tight_accuracy.png
results/tight_deadline.png
results/tight_deadline_results.json
results/tight_miss.png
results/wcet_cdf.png
results/wcet_ci_gev.png
results/wcet_ci_gev_results.json
results/wcet_heatmap.png
results/wcet_results.json
```
