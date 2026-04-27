# Sprint Camera-Ready — Full Experiment Plan

**Target GPU:** NVIDIA A100 SXM4 80 GB  
**Target CPU:** 12–16 vCPU (RunPod standard allocation; see §GPU Recommendation)  
**Estimated GPU-hours:** 24–28 h  
**Wall-clock window:** 5 days  
**Resilience:** tmux + watchdog auto-restart; `--resume` skips completed experiments

---

## GPU / Instance Recommendation

### GPU: A100 SXM4 80 GB

Use **A100 SXM4 80 GB** — the same tier used in sprint_v3.

- TinyLlama-1.1B uses < 2 GB VRAM. The 80 GB headroom eliminates any
  risk of fragmentation during 5000-sample profiling runs.
- SXM4's 2 TB/s HBM2e bandwidth removes the memory-wall bottleneck in
  single-token forward passes (the TPOT micro-benchmark is bandwidth-bound).
- **Do NOT use H100.** Zero throughput benefit for a 1.1 B model; costs 2–3×
  more per hour on RunPod.
- **Do NOT use A6000/A5000** (PCIe, consumer cooling, thermal throttle risk
  during the 60-min soak in E07).

### CPU: 12–16 vCPU (NOT dual-core)

Do not use a 2-vCPU pod. Reasons:

- The Python orchestrator, Matplotlib rendering, scipy EVT fits, and the
  HuggingFace data loader all contend for CPU concurrently.
- With 2 vCPU, GIL + OS scheduling makes inter-run `time.sleep()` calls
  non-deterministic — the IID spacing study (E01) becomes unreliable.
- 12 vCPU costs negligibly more relative to the A100 hourly price.

**Recommended RunPod pod spec:**
```
GPU:              1× A100 SXM4 80 GB
vCPU:             12  (or 16)
RAM:              64 GB
Network volume:   50 GB   ← results survive pod restarts
Container disk:   20 GB
Image: runpod/pytorch:2.3.0-py3.11-cuda12.1.1-devel-ubuntu22.04
```

### Network Volume

Mount a RunPod network volume at `/workspace/volume` before launch.
`setup.sh` detects it and symlinks `results/` to the volume so all outputs
survive pod termination. The `--resume` flag in `run_all.sh` reads `.done`
flags from the volume and skips completed experiments on restart.

---

## Review Issues Being Addressed

### EVT / WCET Methodology

| Issue | Sprint Response |
|-------|----------------|
| IID violated at 200 ms spacing (Ljung-Box p=0.0 at all lags) | E01: test 0/200 ms/1 s/5 s; find minimum IID-passing spacing |
| 500–1000 samples; block-maxima blocks too small | E00: 5000 samples/cell + E02: blocks b=25/50/100 |
| GEV xi 0.21–1.18; Gumbel rejected by AD in all cells | E02: report honest GEV with wide CI; E03: non-EVT empirical bounds |
| pWCET CI spans 28 ms to 47100 ms — indefensible | E03: P99 + safety factor as primary claim; EVT as supplementary transparency |

### Accuracy / Threshold

| Issue | Sprint Response |
|-------|----------------|
| 14-query CI [50%, 93%] is clinically meaningless | E04: 1000-query accuracy → 95% CI ≈ ±3.1 pp |
| tau tuned on same 30 queries used for evaluation | E05: 5-fold cross-validation; tau stability across folds |

### System Evaluation

| Issue | Sprint Response |
|-------|----------------|
| Router comparison confounded by exit rate | E06: 500 queries per configuration; equal-latency comparison |
| Capacity only N=1..4 | E08: N=1..8 with queue-depth analysis |

### Thermal / Stability

| Issue | Sprint Response |
|-------|----------------|
| 30-min soak may be insufficient for camera-ready | E07: 60-min soak; latency drift first 10 vs last 10 min |

---

## Experiment Designs

### E00 — Large Spaced WCET Profiling (5000 samples, 1 s spacing)

**Goal:** High-sample dataset for block-maxima EVT and IID study.

**Protocol:**
- 12 cells: seq_len in {32, 128, 256, 512, 1024} x exit in {L5, L16, Full}
- 50 warm-up runs per cell, discarded
- N_SAMPLES = 5000 TPOT measurements per cell
- SPACING_MS = 1000 ms between each measurement
- Measure: single-token generation (KV-cache pre-filled from prompt)
- Output: raw time-series JSON per cell + summary stats

**Estimated time:** 5000 x 12 x 1.0 s + 20 ms measure ≈ 17 h (run overnight)

### E01 — IID Spacing Study

**Goal:** Find minimum spacing for IID samples (addresses core EVT validity issue).

**Protocol:**
- Cell: seq_len=128, Full pass
- N_SAMPLES = 1000 per spacing level
- Spacings: 0 ms, 200 ms, 1000 ms, 5000 ms
- Test: Ljung-Box lags 1–20; lag-1 ACF
- Output: table of (spacing, LB_pvalue, lag1_ACF, IID_pass)

**Estimated time:** ~2.5 h

### E02 — Block-Maxima GEV (n=5000, large blocks)

**Goal:** Determine whether block-maxima EVT yields defensible pWCET.

**Protocol:**
- Use E00 data (5000 samples/cell)
- Block sizes: b = 10, 25, 50, 100
- Per cell x block size: fit GEV to block maxima, AD test, compute pWCET(1e-6)
- Output: ξ, pWCET, 95% CI width, AD pass/fail for all (cell, b) pairs

**Estimated time:** CPU-only, ~10 min (depends on E00)

### E03 — Non-EVT Empirical WCET Bounds

**Goal:** Provide a WCET bound defensible without EVT assumptions.

**Protocol:**
- Use E00 data
- Method A: Empirical P99 / P99.9 / P99.99 with 95% bootstrap CI
- Method B: Safety-factor = P99 + 3*sigma
- Method C: Hoeffding (distribution-free; requires range = max - min)
- Output: unified LaTeX comparison table; Method A is the primary claim

**Estimated time:** CPU-only, ~5 min

### E04 — 1000-Query Accuracy

**Goal:** 95% CI width ≈ ±3.1 pp (vs ±6.25 pp at 500 queries).

**Protocol:**
- 1000 PubMedQA queries, tau=0.55, D=45 ms
- 2000-resample bootstrap CI
- Compare to E06 (500-query baseline for consistency)

**Estimated time:** ~5 min GPU

### E05 — tau Cross-Validation (5-Fold)

**Goal:** Demonstrate tau=0.55 generalises; address overfitting critique.

**Protocol:**
- 5-fold CV on 1000 queries (800 train / 200 test)
- Train: tau* = argmax accuracy over {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}
- Test: evaluate at tau*
- Report: per-fold tau*, accuracy; mean ± std

**Estimated time:** ~10 min GPU

### E06 — Full Router Comparison (500 queries each)

**Goal:** Remove exit-rate confound; equal-latency comparison of all routers.

**Protocol:**
- Router 1 (Full-pass): always use L22 logits; no early exit
- Router 2 (L16 threshold): exit at L16 if conf >= 0.55
- Router 3 (Oracle): exit at L16 if L16 token == L22 token
- 500 queries each; all three run the same 22-layer forward pass post-hoc
- Primary metric: accuracy at D=45 ms; secondary: exit rate, mean TPOT

**Estimated time:** ~8 min GPU

### E07 — 60-Minute Thermal Soak

**Goal:** Confirm latency stability over a sustained window.

**Protocol:**
- Continuous profiling: seq_len=128, Full pass, no spacing between measurements
- Record TPOT every 100 measurements
- Record GPU temp/clock/power via nvidia-smi every 60 s
- Analyse: drift = mean(last 10 min) - mean(first 10 min)

**Estimated time:** 60 min wall clock

### E08 — Multi-Request Capacity (N=1..8)

**Goal:** Extend capacity analysis; add throughput saturation point.

**Protocol:**
- Sequential batch: N requests back-to-back; 100 repetitions per N
- N in {1, 2, 4, 8}
- Metrics: per-request P99 TPOT, SLO compliance rate, throughput (queries/s)

**Estimated time:** ~20 min GPU

### E09 — Final Report Generation

**Goal:** Produce all camera-ready LaTeX tables and figures.

**Protocol:**
- Load all E00–E08 result JSONs from results/
- Generate LaTeX tables: EVT summary, non-EVT bounds, accuracy CI, tau CV,
  router comparison, capacity
- Generate figures: IID spacing comparison, block-maxima vs block size,
  empirical bounds, router accuracy-vs-latency
- Write CAMERA_READY_SUMMARY.md with key numbers for paper update

**Estimated time:** ~5 min

---

## 5-Day Schedule

| Day | Experiments | Est. Wall Clock |
|-----|------------|----------------|
| 1 (overnight) | E00 — 5000-sample profiling | 17–20 h |
| 2 | E01, E02, E03 — IID study + EVT post-processing | 3 h |
| 2 | E04, E05, E06 — accuracy + router | 30 min |
| 3 | E07 — 60-min thermal soak | 1.5 h |
| 3 | E08 — capacity | 30 min |
| 4 | E09 — report generation | 15 min |
| 4–5 | Paper update from CAMERA_READY_SUMMARY.md | manual |

---

## File Structure

```
sprint_camera_ready/
├── SPRINT_PLAN.md          <- this file
├── README.md               <- quick-start for RunPod
├── setup.sh                <- one-shot environment setup
├── run_all.sh              <- master orchestrator
├── tmux_launch.sh          <- WiFi-resilient launcher
├── monitor.sh              <- re-attach helper
├── watchdog.sh             <- crash-recovery daemon
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── early_exit_model.py
│   ├── evt_utils.py        <- extended: block-maxima, non-EVT bounds
│   ├── benchmark_utils.py
│   ├── fig_style.py
│   └── result_writer.py
├── experiments/
│   ├── e00_wcet_large_spaced.py
│   ├── e01_iid_spacing_study.py
│   ├── e02_block_maxima_large.py
│   ├── e03_nonevt_wcet.py
│   ├── e04_accuracy_1000.py
│   ├── e05_tau_crossval.py
│   ├── e06_router_comparison.py
│   ├── e07_thermal_extended.py
│   ├── e08_capacity_full.py
│   └── e09_final_report.py
└── results/                <- all JSON/PNG/TEX outputs
```

---

## Autonomous Operation Contract

1. `tmux_launch.sh` creates session `sprint_cr`.
2. `run_all.sh` runs E00→E09 sequentially; up to 3 retry attempts per
   experiment (30 s backoff + GPU cache clear).
3. Each successful experiment writes a `.done` flag to `results/`.
4. `watchdog.sh` polls every 2 min; if session dies it clears GPU cache,
   waits 60 s, restarts via `tmux_launch.sh --resume`.
5. Re-attach after WiFi drop: `bash monitor.sh`.
6. `run_all.sh --dry` prints the plan without executing.
