# Sprint V3 — IID & GEV Resolution Pipeline

## Problem Statement

The sprint_final EVT analysis (E01, E08) produced two invalidating findings:

| Finding | Result | Implication |
|---------|--------|-------------|
| GEV shape parameter ξ | 1.36 (Fréchet) | Gumbel pWCET anti-conservative; actual tail much heavier |
| Anderson-Darling | stat=17.55 >> crit=0.746 | Gumbel formally rejected at 5% significance |
| Ljung-Box (IID) | p=0.0 for all 9 cells | EVT IID prerequisite violated |

The Gumbel pWCET claim (`P(TPOT>45ms) < 1e-6`) in the original report is therefore invalid.
This sprint resolves both issues and produces a formally defensible pWCET table.

---

## Resolution Strategy

### Path 1: Fix IID (E00 → E01)

**Cause:** GPU thermal state, L2 cache occupancy, and CUDA stream state carry over
between consecutive measurements, creating positive autocorrelation.

**Fix:** Insert 200ms sleep between each timed run. This allows:
- GPU die temperature to stabilize (~100ms thermal time constant)
- L2 cache to partially evict stale KV-cache tensors
- CUDA stream queue to drain completely

**Validation:** E01 runs Ljung-Box on spaced data; pass criterion is p > 0.05 at all lags.

### Path 2: Fix GEV tail (E02 → E03 → E04)

**Cause (hypothesis A):** Extreme outliers are warm-up artifacts (JIT compilation, cold cache
at session start). If true, extended warm-up (50+ passes, up from 20) plus spacing removes them,
reducing ξ below 0.15 where Gumbel is appropriate.

**Cause (hypothesis B):** The heavy tail is intrinsic — GPU scheduling jitter, thermal variation,
or memory bandwidth contention produces genuine polynomial-tail latency distributions. In this
case, Gumbel is permanently inappropriate.

**E02** diagnoses which hypothesis holds by profiling without warm-up discard and comparing
cold-run vs warm-run distributions.

**E03** refits GEV to spaced data to measure the actual ξ after fixing IID. If ξ < 0.15,
Gumbel is restored. If ξ ≥ 0.15, Path 2B applies.

**E04** implements block maxima EVT — the methodologically correct approach regardless of
which hypothesis holds:
- Block maxima are approximately independent even when raw samples are autocorrelated
- GEV fit to block maxima is valid for any ξ value
- Block size is tuned to achieve Ljung-Box IID pass on the maxima series

### Decision Logic (E05)

```
       ┌── E01: IID pass on spaced data? ──┐
       YES                               NO (residual correlation)
       │                                   │
  E03: ξ < 0.15?                      Use block maxima (E04)
  YES           NO                         │
  │             │                     E04: find min block size
  Gumbel OK    GEV (POT)              where Ljung-Box passes
  (spaced)     (spaced)               then fit GEV to maxima
```

E05 reads E01/E03/E04 results, applies this logic per cell, and outputs
the final pWCET table with the appropriate methodology noted per row.

---

## Experiments

| Exp | Script | Inputs | Outputs | Est. Time |
|-----|--------|--------|---------|-----------|
| E00 | e00_spaced_profiling.py | GPU model | 12 cells × 500 samples, 200ms sleep | ~35 min |
| E01 | e01_iid_validation.py | E00 | Ljung-Box per cell, ACF plots | ~2 min |
| E02 | e02_outlier_warmup.py | GPU model | Cold vs warm distribution, ξ comparison | ~5 min |
| E03 | e03_gev_xi_refit.py | E00 | GEV ξ table, AD test, pWCET comparison | ~3 min |
| E04 | e04_block_maxima_pwcet.py | E00 | Block maxima analysis, block sizes 5–50 | ~5 min |
| E05 | e05_final_pwcet_report.py | E01+E03+E04 | Final LaTeX table + JSON | ~3 min |

**Total estimated: ~55 min on A100 SXM4**

---

## Hardware

**Recommended: A100 SXM4 80 GB**
- Same hardware as sprint_final for direct comparability
- 80 GB HBM2e: holds TinyLlama (2.2 GB fp16) with ample headroom
- NVLink/SXM4 bandwidth: ensures profiling reflects production-grade A100 behaviour
- CPU: 16 vCPUs (auto-bundled by RunPod) — sufficient for Python orchestration
- Disk: 50 GB minimum (model cache ~4 GB, results ~1 GB)
- RAM: 30 GB minimum (RunPod defaults to 60 GB with A100)

**Why not H100:**
TinyLlama batch=1 is memory-bandwidth-bound, not compute-bound.
H100 costs ~3× more but provides no measurable speedup at this batch size.
The timing reproducibility (comparison to sprint_final) is more important
than raw throughput for this sprint's objectives.

---

## Expected Outcomes

| Scenario | Probability | Report Implication |
|----------|-------------|-------------------|
| IID pass + ξ < 0.15 | ~25% | Gumbel pWCET restored; original claim partially valid |
| IID pass + ξ still high | ~45% | GEV (POT) on spaced data is valid; pWCET changes significantly |
| IID fail + ξ still high | ~30% | Block maxima GEV is the methodology; report as finding |

In all three scenarios, the empirical SLO compliance result (P99=18.26ms, 0% misses) 
is unaffected — that is the paper's central contribution. The EVT analysis is reframed
as an investigation of GPU timing statistics, with honest methodology.

---

## Running on RunPod

```bash
# 1. Upload this directory to RunPod (rsync or git push)
git add sprint_v3_runpod/ && git commit -m "feat: sprint_v3 IID+GEV resolution package"
git push

# 2. On RunPod instance
cd /workspace
git clone <your-repo>
cd anytime-llm-inference/sprint_v3_runpod

# 3. One-time setup
bash setup.sh

# 4. Launch and detach (WiFi-resilient)
bash tmux_launch.sh     # creates session, attaches
# Ctrl-B D to detach (session keeps running)

# 5. Re-attach from anywhere
bash monitor.sh

# 6. Resume after interruption
bash tmux_launch.sh --resume
```

---

## Output Files

After E05 completes, pull these for the report:

```
results/
  e00_spaced_profiling.json    # raw latency arrays, stats
  e01_iid_validation.json      # Ljung-Box results per cell
  e02_outlier_warmup.json      # warm-up artifact diagnosis
  e03_gev_xi_refit.json        # GEV xi per cell after spacing
  e04_block_maxima.json        # block maxima EVT results
  e05_final_pwcet.json         # final pWCET per cell + methodology
  table_final_pwcet.tex        # drop-in LaTeX table for report
  e00_spaced_timeseries.png    # latency time series (spaced)
  e01_acf_spaced.png           # ACF plots
  e02_warmup_study.png         # cold vs warm distribution
  e03_gev_xi_refit.png         # xi heatmap
  e04_block_maxima.png         # pWCET vs block size
  e05_final_pwcet.png          # final pWCET bar chart
  SPRINT_V3_SUMMARY.md         # pass/fail table
```
