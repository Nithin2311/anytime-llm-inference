# Sprint V2 — RunPod Experiment Package

Self-contained package for reproducing all experiments in the revised
**Anytime LLM Inference with EVT-Bounded WCET** paper on a single A100 SXM4 instance.

Addresses every Critical, Major, and Minor issue from the peer review.
See `SPRINT_V2_PLAN.md` for the full issue-to-experiment map.

---

## Quick Start

```bash
# 1. SSH into your RunPod A100 SXM4 instance and upload this directory
scp -r sprint_v2_runpod/ root@<runpod-ip>:/workspace/

# 2. On the instance — set up environment (≈5 min)
cd /workspace/sprint_v2_runpod
bash setup.sh

# 3. Launch the experiment session in tmux (survives WiFi disconnects)
bash tmux_launch.sh

# 4. Detach anytime with Ctrl-B D — experiments keep running
# 5. Re-attach from any connection:
bash monitor.sh
```

After completion, all results are in `results/`:
- `*.json` — raw numbers
- `*.png`  — single-column publication figures
- `*.tex`  — LaTeX table fragments

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | A100 SXM4 40GB | A100 SXM4 **80GB** |
| vCPU | 4 cores | 8–16 cores |
| RAM | 32 GB | 64 GB |
| Disk | 30 GB | 50 GB |
| CUDA | 11.8+ | 12.x |
| Python | 3.10+ | 3.11 |

**Estimated wall time:** 4–6 hours end-to-end on A100 SXM4 80GB.

> The GPU is the only bottleneck. Any number of vCPUs works.

---

## Resuming After Interruption

```bash
# The --resume flag skips experiments that already wrote a .done flag
bash run_sprint.sh --resume

# Or relaunch tmux with --resume:
bash tmux_launch.sh --resume
```

Completed experiments leave a `.results/.<exp_name>.done` marker.
Delete a marker to force re-run of a specific experiment.

---

## Running a Single Experiment

```bash
# From the sprint_v2_runpod/ directory:
PYTHONPATH=src python3 experiments/e06_accuracy_large.py

# Or via the orchestrator:
bash run_sprint.sh --only e06_accuracy_large
```

---

## Regenerating Plots Only

```bash
# After experiments have run, regenerate all PNGs without re-running experiments:
PYTHONPATH=src python3 src/plot_runner.py --results-dir results/
```

All plots are single-column IEEE width (3.5 in), one PNG per figure.

---

## Directory Layout

```
sprint_v2_runpod/
├── experiments/        E00–E13 experiment scripts
├── src/
│   ├── early_exit_model.py    TinyLlama with early-exit hooks
│   ├── dynamic_scheduler.py   Stateless + KV-cached routers
│   ├── benchmark_utils.py     Raw token collection + post-hoc replay
│   ├── evt_utils.py           Gumbel, GEV, AD test, bootstrap CI
│   ├── exit_head_trainer.py   ExitHead MLP: collect, train, eval
│   ├── plots.py               Single-column plot functions (fresh)
│   ├── plot_runner.py         Reads all result JSONs → PNGs
│   └── result_writer.py       Safe JSON write helper
├── results/            All outputs (JSON + PNG + TEX)
├── logs/               Per-experiment log files
├── run_sprint.sh       Master orchestrator
├── setup.sh            Env setup + dry-run
├── tmux_launch.sh      3-window tmux session
├── monitor.sh          Re-attach / show last status
├── requirements.txt
├── SPRINT_V2_PLAN.md   Architecture + acceptance gates
└── README.md           This file
```

---

## Tmux Window Layout

| Window | Name | Contents |
|---|---|---|
| 0 | `orchestrator` | `run_sprint.sh` output — main job |
| 1 | `gpu-watch` | `nvidia-smi` refresh every 5s |
| 2 | `log-tail` | Live tail of current experiment log |

Switch windows: `Ctrl-B 0`, `Ctrl-B 1`, `Ctrl-B 2`  
Detach: `Ctrl-B D`  
Re-attach: `tmux attach -t sprint_v2` or `bash monitor.sh`

---

## Key Files Written by Each Experiment

| Experiment | JSON | PNGs | TEX |
|---|---|---|---|
| E00 | wcet_results.json | wcet_heatmap, wcet_cdf | table_wcet |
| E01 | evt_wcet_results.json | evt_gev_xi, evt_pwcet_comparison | table_evt, table_evt_ad |
| E02 | threshold_crossval_results.json | threshold_cv, threshold_exit_rate | — |
| E03 | forced_exit_extended_results.json | forced_exit_accuracy, forced_exit_miss | — |
| E04 | pot_sensitivity_results.json | pot_sensitivity | table_pot |
| E05 | deadline_sweep_comparison_results.json | deadline_sweep | table_deadline_sweep |
| E06 | accuracy_large_results.json | accuracy_tau, exit_rate_tau | table_accuracy_large |
| E07 | wcet_ci_gev_results.json | wcet_ci_gev | table_wcet_ci_gev |
| E08 | sample_independence_results.json | acf_seq128, rolling_mean_seq128 | table_independence |
| E09 | capacity_empirical_results.json | capacity_miss, capacity_throughput | table_capacity |
| E10 | tight_deadline_results.json | tight_accuracy, tight_miss | table_tight_deadline |
| E11 | thermal_stability_results.json | thermal_latency, thermal_temp | — |
| E12 | exit_head_results.json | exit_head_accuracy, exit_head_loss | — |
| E13 | dense_ablation_results.json | ablation_latency, ablation_accuracy, ablation_pareto | table_dense_ablation |
