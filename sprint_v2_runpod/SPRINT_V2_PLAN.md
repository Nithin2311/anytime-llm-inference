# Sprint V2 — Experiment Plan

## Overview

Sprint V2 runs 14 experiments on a **single A100 SXM4 80GB** instance to address every
Critical, Major, and Minor issue identified in the academic peer review of the
anytime-llm-inference paper. All experiments operate on **TinyLlama-1.1B-Chat** and the
**PubMedQA** (pqa_labeled, train split) benchmark.

---

## Issue → Experiment Cross-Reference

| Review Issue | Severity | Addressed By |
|---|---|---|
| Hardware mismatch (body RTX 6000 Ada, table RTX 4000 Ada) | **Critical** | E00 re-profiles everything on A100 SXM4 |
| Missing GEV/AD goodness-of-fit test | **Critical** | E01 adds GEV ξ, AD stat, AD p-value per cell |
| Hard-RT language unjustified | **Critical** | E00+E07+E08 together: pWCET, CI, IID test |
| Accuracy CI [50%,93%] too wide (14 queries) | **Major** | E06: 500 queries → CI ≈ ±7% |
| D=45ms not justified, no tight-deadline analysis | **Major** | E10: D=14–30ms sweep; E09: N_max empirical |
| POT fraction 20% unjustified | **Major** | E04: fraction sensitivity [5%,30%]; E08: IID |
| Missing bootstrap CI on pWCET | **Major** | E07: 1000 bootstrap resamples, Gumbel+GEV |
| Missing related work on LLM serving latency | **Major** | addressed in paper revision (not an experiment) |
| ROUGE-L inappropriate for binary QA | **Moderate** | E03 replaces ROUGE-L with parseable_pct |
| Threshold τ selected on eval set (data leakage) | **Moderate** | E02: cal/holdout split, post-hoc replay |
| No exit-head quality analysis | **Moderate** | E12: train+eval dedicated MLP classifier |
| Only L5/L11/L16/Full tested | **Moderate** | E13: dense L12–L20 ablation |
| No sample IID validation for EVT | **Moderate** | E08: Ljung-Box, ACF, rolling statistics |
| No capacity analysis | **Minor** | E09: empirical N_max round-robin |
| A100 latency paradox unexplained | **Minor** | E00 re-profiles; E11 thermal stability check |

---

## Experiment Descriptions

### E00 — WCET Reprofiling
**File:** `experiments/e00_wcet_reprofiling.py`  
**Purpose:** Establish ground-truth latency on A100 SXM4 (100 runs/cell, 10 warmup).  
**Config:** seq ∈ {32,64,128,256,512,1024}, layers ∈ {5,11,16,Full}  
**Outputs:** `wcet_results.json`, `wcet_heatmap.png`, `wcet_cdf.png`, `table_wcet.tex`

### E01 — EVT GEV + Anderson-Darling
**File:** `experiments/e01_evt_gev_ad.py`  
**Purpose:** Fit Gumbel AND GEV to POT tail; report ξ and AD GOF test per cell.  
**Config:** 1000 samples/cell, POT 20%  
**Outputs:** `evt_wcet_results.json`, `evt_gev_xi.png`, `evt_pwcet_comparison.png`, `table_evt.tex`, `table_evt_ad.tex`

### E02 — Threshold Cross-Validation
**File:** `experiments/e02_threshold_crossval.py`  
**Purpose:** Select τ on calibration half; evaluate on held-out half (post-hoc replay).  
**Config:** 300 queries, τ ∈ [0.30, 0.90], 150/150 cal/holdout split  
**Outputs:** `threshold_crossval_results.json`, `threshold_cv.png`, `threshold_exit_rate.png`

### E03 — Forced-Exit Extended
**File:** `experiments/e03_forced_exit_extended.py`  
**Purpose:** Replace ROUGE-L with parseable_pct; both routers at D ∈ {14…30}ms.  
**Config:** D ∈ {14,16,18,20,22,25,30}ms, 50 queries  
**Outputs:** `forced_exit_extended_results.json`, `forced_exit_accuracy.png`, `forced_exit_miss.png`

### E04 — POT Sensitivity
**File:** `experiments/e04_pot_sensitivity.py`  
**Purpose:** Justify 20% POT fraction by showing pWCET stability across fractions.  
**Config:** 1000 samples, fractions ∈ {5%,10%,15%,20%,25%,30%}, seq ∈ {64,128,256,512}  
**Outputs:** `pot_sensitivity_results.json`, `pot_sensitivity.png`, `table_pot.tex`

### E05 — Deadline Sweep Comparison
**File:** `experiments/e05_deadline_sweep_comparison.py`  
**Purpose:** Side-by-side stateless vs KV-cached miss rate across a wide deadline range.  
**Config:** D ∈ {14…100}ms, 50 queries each  
**Outputs:** `deadline_sweep_comparison_results.json`, `deadline_sweep.png`, `table_deadline_sweep.tex`

### E06 — Large-Scale Accuracy (500 queries)
**File:** `experiments/e06_accuracy_large.py`  
**Purpose:** Reduce 95% CI from ±21pp to ≈±7pp with ≥200 scoreable tokens.  
**Config:** 500 queries, τ sweep [0.30–0.90], 2000 bootstrap resamples  
**Outputs:** `accuracy_large_results.json`, `accuracy_tau.png`, `exit_rate_tau.png`, `table_accuracy_large.tex`

### E07 — WCET Bootstrap CI + GEV Comparison
**File:** `experiments/e07_wcet_ci_gev.py`  
**Purpose:** Parametric bootstrap CI on pWCET(1e-6); Gumbel vs GEV comparison; Ljung-Box.  
**Config:** 1000 samples/cell, 1000 bootstrap resamples  
**Outputs:** `wcet_ci_gev_results.json`, `wcet_ci_gev.png`, `table_wcet_ci_gev.tex`

### E08 — Sample Independence
**File:** `experiments/e08_sample_independence.py`  
**Purpose:** Validate EVT i.i.d. assumption with Ljung-Box test and ACF analysis.  
**Config:** 1000 samples, max_lag=40, rolling window=50  
**Outputs:** `sample_independence_results.json`, `acf_seq128.png`, `rolling_mean_seq128.png`, `table_independence.tex`

### E09 — Empirical Capacity
**File:** `experiments/e09_capacity_empirical.py`  
**Purpose:** Empirically confirm N_max = floor(D / P99) round-robin model.  
**Config:** D=45ms, seq=128, N ∈ {1,2,3,4}, 5 trials each  
**Outputs:** `capacity_empirical_results.json`, `capacity_miss.png`, `capacity_throughput.png`, `table_capacity.tex`

### E10 — Tight-Deadline Regime
**File:** `experiments/e10_tight_deadline.py`  
**Purpose:** Demonstrate system under real pressure; find minimum schedulable D.  
**Config:** D ∈ {14…30}ms, 200 queries, post-hoc replay  
**Outputs:** `tight_deadline_results.json`, `tight_accuracy.png`, `tight_miss.png`, `table_tight_deadline.tex`

### E11 — A100 Thermal Stability
**File:** `experiments/e11_thermal_a100.py`  
**Purpose:** Verify no thermal drift over sustained 1000-token run (A100 latency paradox).  
**Config:** 1000 consecutive tokens, seq=128, GPU temp sampled every 100 tokens  
**Outputs:** `thermal_stability_results.json`, `thermal_latency.png`, `thermal_temp.png`

### E12 — Exit-Head MLP Training
**File:** `experiments/e12_exit_head_training.py`  
**Purpose:** Train dedicated 2-layer MLP on L16 hidden states; evaluate oracle agreement.  
**Config:** 300 train / 100 eval queries, 30 epochs, τ=0.55  
**Outputs:** `exit_head_results.json`, `exit_head_accuracy.png`, `exit_head_loss.png`, checkpoint `exit_head.pt`

### E13 — Dense Layer Ablation L5–L22
**File:** `experiments/e13_dense_ablation.py`  
**Purpose:** Fill quality-latency curve gaps by testing every integer layer 12–20.  
**Config:** seq=128, 100 queries, 500 timing samples per layer  
**Outputs:** `dense_ablation_results.json`, `ablation_latency.png`, `ablation_accuracy.png`, `ablation_pareto.png`, `table_dense_ablation.tex`

---

## Acceptance Gates

All gates must pass before the sprint is considered complete.

| Gate | Criterion |
|---|---|
| G1 | All cells profiled on A100 SXM4 (verify `wcet_results.json` hardware field) |
| G2 | GEV ξ reported for all cells; `gev.xi` present in `evt_wcet_results.json` |
| G3 | AD test present per cell; `ad_test.fit_not_rejected_at_5pct` in results |
| G4 | Bootstrap CI width < 3ms for all cells at N=1000 (from `wcet_ci_gev_results.json`) |
| G5 | Ljung-Box p > 0.05 for majority of cells (from `sample_independence_results.json`) |
| G6 | 500-query accuracy CI width < 20pp (from `accuracy_large_results.json`) |
| G7 | D_min schedulable identified (from `tight_deadline_results.json`) |
| G8 | ExitHead val accuracy > LM-head baseline 32% (from `exit_head_results.json`) |
| G9 | Dense ablation identifies Pareto-optimal exit point (from `dense_ablation_results.json`) |
| G10 | Thermal drift < 5% (from `thermal_stability_results.json`) |

---

## Directory Layout

```
sprint_v2_runpod/
├── experiments/
│   ├── e00_wcet_reprofiling.py
│   ├── e01_evt_gev_ad.py
│   ├── e02_threshold_crossval.py
│   ├── e03_forced_exit_extended.py
│   ├── e04_pot_sensitivity.py
│   ├── e05_deadline_sweep_comparison.py
│   ├── e06_accuracy_large.py
│   ├── e07_wcet_ci_gev.py
│   ├── e08_sample_independence.py
│   ├── e09_capacity_empirical.py
│   ├── e10_tight_deadline.py
│   ├── e11_thermal_a100.py
│   ├── e12_exit_head_training.py
│   └── e13_dense_ablation.py
├── src/
│   ├── early_exit_model.py
│   ├── dynamic_scheduler.py
│   ├── benchmark_utils.py
│   ├── evt_utils.py
│   ├── exit_head_trainer.py
│   ├── plots.py           ← fresh single-column plot library
│   ├── plot_runner.py     ← standalone plot regeneration
│   └── result_writer.py
├── results/               ← all JSON + PNG + TEX outputs land here
├── logs/                  ← per-experiment .log files
├── run_sprint.sh          ← master orchestrator
├── setup.sh               ← env setup + dry-run
├── tmux_launch.sh         ← 3-window tmux session
├── monitor.sh             ← re-attach / show status
├── requirements.txt
├── SPRINT_V2_PLAN.md      ← this file
└── README.md
```

---

## Hardware Recommendation

**Primary:** RunPod A100 SXM4 80GB (1× GPU)

The A100 SXM4 is chosen over RTX variants for reproducibility and memory headroom.
TinyLlama-1.1B fits easily in 80GB; the A100's HBM2e bandwidth and NVLink make
sustained 1000-token thermal runs more representative of server-class deployment.

> **Note on the "A100 is slower than RTX 6000 Ada" finding:** Single-batch,
> single-token TinyLlama inference is memory-bandwidth-bound per CUDA core.
> RTX Ada Lovelace's architecture is better tuned for this workload. The A100
> is the correct choice for the paper because it is the target deployment class;
> the Ada finding should be discussed as a workload-size caveat, not hidden.

**vCPU:** 8–16 vCPUs are sufficient. The bottleneck is always the GPU for these
experiments. Use whatever is cheapest alongside the A100 SXM4.

**Storage:** 50 GB disk (model cache ~4GB, dataset ~1GB, results ~200MB).

**Estimated wall time on A100 SXM4:** 4–6 hours for E00–E13 end-to-end.
