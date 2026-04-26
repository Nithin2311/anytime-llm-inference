# Sprint Final — Experiment Pipeline

## Overview

14 experiments addressing every reviewer concern from the Round 2 peer review of:
*"Predictive Early-Exit Routing for SLO-Compliant LLM Inference"*

**Hardware target:** NVIDIA A100 SXM4 80 GB  
**Estimated total runtime:** 8–14 hours on A100 SXM4  
**Model:** TinyLlama-1.1B-Chat-v1.0 (1.1B parameters, fp16)

---

## GPU / Hardware Recommendation

| Component | Recommended | Notes |
|---|---|---|
| GPU | **A100 SXM4 80 GB** | NVLink bandwidth ~2 TB/s; best for memory-bandwidth-bound inference |
| GPU alt | A100 PCIe 80 GB | ~15% slower than SXM4; fine for all experiments |
| CPU | 16+ vCPUs (any) | TinyLlama is GPU memory-bandwidth-bound; CPU is idle during inference |
| RAM | 80 GB | Model fp16 ~2.2 GB VRAM; system RAM for dataset caching |
| Disk | 200 GB | Model ~4 GB, PubMedQA ~1 GB, results ~2 GB, safety margin |
| RunPod template | PyTorch 2.1 (CUDA 12.1) | Pre-installed torch, transformers |

**Why A100 SXM4 over H100?** TinyLlama 1.1B in fp16 saturates A100 memory bandwidth at batch=1 (~17 ms/token). H100 would not reduce latency meaningfully at this scale while costing 3× more per hour.

**CPU note:** RunPod bundles CPU automatically with the GPU pod. You do not select CPU separately. The default (AMD EPYC 7532 / Intel Xeon Platinum) at 16 vCPUs is sufficient.

---

## Reviewer Issue → Experiment Mapping

### C1 — EVT Certification Without Prerequisites
> *"pWCET estimates presented without GEV tail-shape validation or IID confirmation"*

| Experiment | What It Does | Expected Output |
|---|---|---|
| **E00** `e00_wcet_reprofiling` | Re-baseline with 500 samples; correct warm-up protocol (20 warm-up passes per sequence-length change, 5 discarded timed runs after each transition) | `wcet_reprof_results.json`: P99, P99.9, Gumbel fit parameters |
| **E01** `e01_evt_gev_ad` | Fit GEV to POT tail; compute shape parameter ξ; run Anderson-Darling GoF test | `evt_gev_results.json`: ξ value, AD statistic, `gumbel_valid` flag |
| **E04** `e04_pot_sensitivity` | Sweep POT fraction 10%–40%; test robustness of pWCET estimate | `pot_sensitivity_results.json`: pWCET vs fraction table |
| **E07** `e07_wcet_ci_gev` | Non-parametric bootstrap CI on pWCET (1000 resamples of POT tail); compare Gumbel vs GEV bounds | `wcet_ci_results.json`: `ci_lower`, `ci_upper`, `ci_width` |
| **E08** `e08_sample_independence` | Ljung-Box test on latency sequence (H0: no autocorrelation); ACF at lags 1–20 | `independence_results.json`: `lb_pvalue`, `independent_at_5pct`, ACF values |

**Paper update from these experiments:** Replace "preliminary Gumbel EVT estimates" with confirmed GEV ξ, Anderson-Darling pass/fail, and bootstrap CI bounds in Table II.

---

### C2 — Anytime Framing vs 0.7% Voluntary Exit Rate
> *"0.7% exit rate at τ=0.55 contradicts the 'anytime routing' framing"*

| Experiment | What It Does | Expected Output |
|---|---|---|
| **E12** `e12_exit_head_training` | Train 2-layer MLP exit head on L16 hidden states (PubMedQA); compare exit rate before/after | `exit_head_results.json`: exit rate ↑ from 0.7%, oracle agreement %, accuracy |
| **E13** `e13_dense_ablation` | Sweep exit points L12–L20; show accuracy/latency tradeoff curve | `dense_ablation_results.json`: accuracy × latency per layer |
| **E02** `e02_threshold_crossval` | Split τ tuning to calibration set; validate on holdout | `crossval_results.json`: holdout accuracy, overfitting gap |

**Paper update:** Section II.A bridging sentence and contribution 1 are already softened ("infrastructure enabling anytime-style routing"). E12 results will replace "0.7% voluntary exit" with trained-head exit rate.

---

### M1 — Accuracy CI Too Wide (14-query baseline)
> *"14-query accuracy CI [50%,93%] is clinically meaningless"*

| Experiment | What It Does | Expected Output |
|---|---|---|
| **E06** `e06_accuracy_large` | 500-query PubMedQA accuracy; post-hoc τ sweep; bootstrap CI (2000 resamples); **L16 confidence distribution histogram** | `accuracy_large_results.json`: accuracy, `ci_lower`, `ci_upper`, CI width ≈ ±7pp; `confidence_distribution.png` |

**Paper update:** Replace [50%,93%] CI with narrow CI from 500 queries. Add confidence histogram figure as supplementary.

---

### M2 — Single-Sample EVT, No Uncertainty Quantification
> *"single-run pWCET point estimate presented without CI"*

Addressed by **E07** above (bootstrap CI on pWCET).

---

### R1 — Exit Quality Across Layers
> *"forced exit quality only shown for L16; L5/L11/L22 behavior unclear"*

| Experiment | What It Does | Expected Output |
|---|---|---|
| **E03** `e03_forced_exit_extended` | Forced-exit ROUGE/accuracy at L5, L8, L11, L14, L16, L19, L22 | `forced_exit_results.json`: quality vs layer table |

---

### R2 — Router Comparison and Exit-Rate Confound
> *"comparison table confounded by different exit rates across routers"*

| Experiment | What It Does | Expected Output |
|---|---|---|
| **E05** `e05_deadline_sweep_comparison` | Deadline sweep D=14–60ms for stateless, KV-cached, async routers; fixed τ; reports exit rate per router | `deadline_sweep_results.json`: per-router exit rate, miss rate, latency |
| **E09** `e09_capacity_empirical` | Empirical throughput for N=1,2,3,4 concurrent requests under round-robin | `capacity_results.json`: throughput vs N table |

**Paper update:** Table caption already notes exit-rate confound (stateless 5.3% vs KV-cached 0.7%). E05 data fills in the fair-comparison column.

---

### R3 — Statistical Rigor
> *"deadline miss claims rest on 0/150 observations; tight deadlines untested"*

| Experiment | What It Does | Expected Output |
|---|---|---|
| **E10** `e10_tight_deadline` | Stress test D=14–30ms with 500 samples; report miss rate and Clopper-Pearson UCB | `tight_deadline_results.json`: miss rate per deadline, CP 95% UCB |
| **E11** `e11_thermal_a100` | 30-minute sustained load on A100; track latency drift vs temperature | `thermal_results.json`: latency over time, temperature trace |

---

## Experiment Dependency Graph

```
E00 (re-baseline)
  └─> E01 (GEV fit) ──> E07 (pWCET CI) ──> E04 (POT sensitivity)
  └─> E08 (IID test)

E06 (500-query acc)   # independent

E12 (exit head) ──> E02 (cross-val)
E13 (dense ablation)  # independent

E03, E05, E09, E10, E11  # independent
```

All experiments are **independent** at the Python level — no experiment imports another's output at runtime. The dependency graph above shows logical ordering for paper claims only.

---

## Expected Run Times (A100 SXM4)

| Experiment | Est. Time | Notes |
|---|---|---|
| E00 | 15 min | 500 × 50 CUDA timed runs |
| E01 | 5 min | Pure scipy/numpy after E00 data |
| E02 | 20 min | Cal/holdout split, 200 queries each |
| E03 | 25 min | 7 exit layers × 100 queries |
| E04 | 10 min | Post-hoc on E00 data |
| E05 | 30 min | 3 routers × 10 deadlines × 150 queries |
| E06 | 45 min | 500 KV-pass queries + bootstrap |
| E07 | 10 min | Bootstrap 1000 resamples |
| E08 | 5 min | Ljung-Box on E00 latency data |
| E09 | 40 min | N=1..4 × 150 queries |
| E10 | 20 min | 500 × 5 deadline values |
| E11 | 35 min | 30-min sustained load |
| E12 | 60 min | Hidden state collection + MLP training |
| E13 | 90 min | 9 layers × 150 queries × 3 metrics |
| **Total** | **~7h** | With 3× retry overhead ~10h |

---

## File Structure

```
sprint_final/
├── PIPELINE.md           ← this file
├── README.md             ← quick-start + GPU config
├── requirements.txt      ← pip dependencies
├── setup.sh              ← environment bootstrap (run once)
├── tmux_launch.sh        ← 5-window session launcher (WiFi-resilient)
├── run_all.sh            ← master orchestrator (retry + resume)
├── watchdog.sh           ← auto-restart on unexpected exit
├── monitor.sh            ← re-attach helper
├── collect_results.sh    ← archive results to tarball
├── src/                  ← shared library modules
│   ├── early_exit_model.py
│   ├── benchmark_utils.py
│   ├── dynamic_scheduler.py
│   ├── evt_utils.py
│   ├── exit_head_trainer.py
│   ├── fig_style.py
│   ├── plot_runner.py
│   ├── plots.py
│   └── result_writer.py
├── experiments/          ← E00–E13 scripts
├── results/              ← JSON + PNG + LaTeX outputs (created at runtime)
├── logs/                 ← per-experiment stdout/stderr (created at runtime)
└── figures/              ← consolidated figures (created at runtime)
```

---

## Quick Start (on RunPod)

```bash
# 1. Upload or git-clone the sprint_final folder
git clone <your-repo> && cd anytime-llm-inference/sprint_final
# OR: scp -r sprint_final/ user@runpod-ip:~/

# 2. Run setup (once per instance)
bash setup.sh

# 3. Launch tmux session (WiFi-resilient)
bash tmux_launch.sh

# 4. If connection drops, re-attach from anywhere
bash monitor.sh

# 5. After completion, collect results
bash collect_results.sh
```

---

## Resume / Error Recovery

| Scenario | Command |
|---|---|
| WiFi dropped, reconnect | `bash monitor.sh` |
| Experiment failed, retry that one | `bash run_all.sh --only E06` |
| Session killed, restart from checkpoint | `bash tmux_launch.sh --resume` |
| OOM on one experiment | Check `logs/e12_exit_head_training.log`; reduce batch size in that script |
| Full restart from scratch | `rm -f results/.*.done && bash tmux_launch.sh` |
