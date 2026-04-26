# Sprint Final — RunPod Self-Sustaining Experiment Package

Addresses all peer-review recommendations for:
*"Predictive Early-Exit Routing for SLO-Compliant LLM Inference:
Probabilistic WCET Bounds via Post-Hoc KV-Cached Forward Hooks"*

## GPU Recommendation

**Use NVIDIA A100 SXM4 80 GB.**

TinyLlama-1.1B in fp16 at batch=1 is memory-bandwidth-bound (~17 ms/token). The A100 SXM4's 2 TB/s NVLink bandwidth delivers near-optimal latency for this workload. An H100 would cost 3× more per hour without meaningful speedup at batch=1.

| Spec | Value |
|---|---|
| GPU | A100 SXM4 80 GB |
| CPU | 16 vCPUs (auto-bundled by RunPod — any is fine) |
| RAM | 80 GB |
| Disk | 200 GB |
| RunPod template | PyTorch 2.1 / CUDA 12.1 |

## Quick Start

```bash
# 1. Setup (once per fresh instance — ~10 min)
bash setup.sh

# 2. Launch WiFi-resilient tmux session
bash tmux_launch.sh

# 3. Detach (job keeps running): Ctrl-B D
# 4. Re-attach after WiFi drop:
bash monitor.sh

# 5. After ~10h, collect results
bash collect_results.sh
```

## What Runs

14 experiments (E00–E13) in sequence, ~7–10 hours total on A100 SXM4:

| # | Experiment | Reviewer Issue |
|---|---|---|
| E00 | WCET re-profiling (500 samples) | C1 re-baseline |
| E01 | GEV fit + Anderson-Darling | C1 tail validation |
| E02 | τ cross-validation | C2 generalization |
| E03 | Forced-exit quality L5–L22 | R1 exit quality |
| E04 | POT fraction sensitivity | C1 robustness |
| E05 | Deadline sweep, 3 routers | R2 fair comparison |
| E06 | 500-query accuracy + CI + histogram | M1 narrow CI |
| E07 | pWCET bootstrap CI | M2 uncertainty |
| E08 | Ljung-Box IID test | C1 IID confirmation |
| E09 | Empirical capacity N=1..4 | R2 capacity table |
| E10 | Tight deadline stress D=14–30 ms | R3 stress test |
| E11 | A100 thermal stability 30 min | R3 thermal |
| E12 | Exit-head MLP training | C2 exit rate ↑ |
| E13 | Dense ablation L12–L20 | C2 layer sweep |

See `PIPELINE.md` for full details.

## Error Recovery

```bash
# Resume from last checkpoint (skip completed experiments)
bash tmux_launch.sh --resume

# Re-run one experiment
bash run_all.sh --only E06

# Check for errors across all logs
grep -rn "ERROR\|Traceback" logs/

# Watchdog (auto-restart on unexpected exit)
bash watchdog.sh &
```
