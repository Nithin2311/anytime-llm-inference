# Sprint Camera-Ready — RunPod Quick Start

Camera-ready experiments addressing all peer-review recommendations.
See SPRINT_PLAN.md for full architecture, experiment designs, and GPU selection rationale.

---

## Step 1 — Provision RunPod Pod

```
GPU:    1x A100 SXM4 80 GB   (NOT dual-core CPU — see SPRINT_PLAN.md §CPU)
vCPU:   12
RAM:    64 GB
Volume: 50 GB network volume at /workspace/volume
Image:  runpod/pytorch:2.3.0-py3.11-cuda12.1.1-devel-ubuntu22.04
```

## Step 2 — Upload This Folder

```bash
rsync -avz sprint_camera_ready/ root@<pod-ip>:/workspace/sprint_camera_ready/
```

## Step 3 — Setup (one time)

```bash
cd /workspace/sprint_camera_ready
bash setup.sh
```

Installs all deps, downloads TinyLlama (~2 GB), detects network volume, verifies GPU.

## Step 4 — Launch (WiFi-resilient)

```bash
bash tmux_launch.sh
```

Creates tmux session `sprint_cr`. Survives SSH disconnects.

## Step 5 — Monitor

```bash
bash monitor.sh          # re-attach from anywhere
tail -f results/run.log  # live log without attaching
```

Detach without killing: `Ctrl-B` then `D`.

## Step 6 — Watchdog (recommended)

Open a second terminal window:
```bash
bash watchdog.sh &
```

Polls every 2 min. Auto-restarts on OOM/crash with `--resume`.

---

## Partial Runs

```bash
bash run_all.sh --resume        # skip completed experiments
bash run_all.sh --only e04      # single experiment
bash run_all.sh --dry           # print plan, no execution
```

---

## Output Files

All results land in `results/` (symlinked to network volume if mounted):

| File | Contents |
|------|----------|
| `e00_wcet_large_spaced.json` | 5000-sample TPOT + stats per cell |
| `e01_iid_spacing.json` | Ljung-Box at 4 spacing levels |
| `e02_block_maxima.json` | GEV fits per (cell, block size) |
| `e03_nonevt_bounds.json` | P99/P99.9/P99.99 + safety-factor bounds |
| `e04_accuracy_1000.json` | 1000-query accuracy, 95% CI |
| `e05_tau_crossval.json` | 5-fold CV results |
| `e06_router_comparison.json` | Three-router accuracy/latency table |
| `e07_thermal_soak.json` | 60-min latency + GPU thermal data |
| `e08_capacity.json` | Capacity N=1..8 |
| `table_*.tex` | Camera-ready LaTeX fragments |
| `CAMERA_READY_SUMMARY.md` | Key numbers for paper update |
| `SPRINT_SUMMARY.md` | Automated pass/fail report |

---

## Troubleshooting

**OOM during E00:**
```bash
python3 -c "import torch; torch.cuda.empty_cache()"
bash run_all.sh --resume --only e00
```

**HuggingFace slow/fails:**
```bash
export HF_TOKEN=<your_token>   # TinyLlama is public; usually not needed
bash setup.sh
```

**tmux session missing:**
```bash
tmux list-sessions
bash tmux_launch.sh
```
