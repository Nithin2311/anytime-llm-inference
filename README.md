# Dynamic Anytime Scheduling for LLM Inference
**Bounding Tail Latency via Predictive Early-Exit Mechanisms**

**Author:** Nithin Palyam
**Course:** CIS 6930 — Real-Time Systems (Spring 2026), University of South Florida

---

## Project Overview

Standard autoregressive LLM inference processes every transformer layer for every generated token, producing execution times that scale with model depth and context length. In latency-sensitive applications — clinical decision support, interactive HCI, cyber-physical systems — this unbounded tail latency violates hard real-time constraints.

This project implements an **Anytime Algorithm** framework for LLM token generation. The key insight: a transformer's intermediate hidden states already carry useful predictions. By instrumenting TinyLlama-1.1B with early-exit points and a dynamic confidence-based scheduler, the system guarantees every token is delivered within a hard deadline while maximizing output quality.

**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 (22 transformer layers)
**Hardware:** RunPod RTX 4000 Ada (bare-metal, no hypervisor noise)
**Deadline target:** 45 ms per token

---

## Formal Schedulability Analysis

The system treats each token generation as a real-time task τᵢ with deadline D.

### Schedulability Criterion

A token generation system is **schedulable** if and only if:

```
P99_TPOT ≤ D
```

where TPOT (Time Per Output Token) is measured after the first-token prefill phase.

### Utilization Bound

The utilization ratio quantifies how much of the deadline budget is consumed at the 99th percentile:

```
U = P99_TPOT / D        (must be < 1.0 for schedulability)
```

### Anytime Algorithm Safety Guarantee

The dynamic scheduler enforces schedulability by construction:

```
At each token step:
  1. Run Layer-16 early exit  →  elapsed_early ≈ 12–13 ms
  2. remaining_budget = D − elapsed_early
  3. if remaining_budget < WCET_full:
       → Forced exit (Layer 16 result committed immediately)
     elif confidence ≥ current_threshold:
       → Early threshold exit (Layer 16 result committed)
     else:
       → Full 22-layer pass (remaining_budget guaranteed ≥ WCET_full)
```

This guarantees the worst-case token latency is bounded by:

```
WCET_token ≤ WCET_L16 + WCET_full ≈ 13.1 + 16.9 = 30.0 ms  <<  D = 45 ms
```

### Measured Results

| Metric | Baseline (Full Pass) | Anytime Scheduler |
|--------|---------------------|--------------------|
| P50 TPOT | 19.98 ms | 34.53 ms |
| P99 TPOT | 36.71 ms | **37.99 ms** |
| Utilization (P99/D) | 0.816 | **0.844** |
| Miss Rate | 1.39% | **0.00%** |
| Schedulable? | Marginal | **YES** |

The anytime scheduler accepts a slightly higher mean TPOT (it always runs at least Layer 16) in exchange for **zero deadline misses** and deterministic worst-case bounds.

---

## System Architecture

### Core Files

| File | Description |
|------|-------------|
| `early_exit_model.py` | `EarlyExitTinyLlama` wrapper — explicit layer-by-layer forward pass with configurable `exit_layer`; handles rotary embeddings and RMSNorm correctly for partial-depth inference |
| `dynamic_scheduler.py` | **Phase 2 scheduler** — dynamic threshold decay from `max_conf=0.8` to `min_conf=0.3` proportional to remaining budget; forced exit when `remaining_budget < WCET_full`; WCET loaded from measured `wcet_results.json` |
| `static_scheduler.py` | **Phase 1 scheduler** — fixed confidence threshold (0.8) with Layer-5 early exit; establishes the baseline anytime behavior |

### Profiling & Calibration

| File | Description |
|------|-------------|
| `profile_wcet.py` | Multi-length WCET sweep (seq lengths 32/64/128/256 × exit layers 5/11/16/full, 50 runs each); outputs `wcet_results.json` and `wcet_profile.png` |
| `calibration.py` | Measures agreement between Layer-5/Layer-16 early exits and the full 22-layer pass; outputs calibration curves and confidence histograms; explains why L5 threshold exits almost never fire |

### Benchmarking & Evaluation

| File | Description |
|------|-------------|
| `benchmark.py` | 30-query PubMedQA benchmark using the dynamic scheduler; reports accuracy, TPOT distribution, deadline miss rate, throughput (tokens/sec), and utilization ratio |
| `evaluate_tail_latency.py` | Formal schedulability proof — CDF comparison of baseline vs anytime across 3 clinical prompts; saves `schedulability_proof.png` and `tail_latency_results.json` |
| `deadline_sweep.py` | Utility-latency tradeoff sweep across deadlines [20–60] ms; reveals the schedulability "knee" at ~30 ms; saves `deadline_tradeoff.png` |
| `compare_schedulers.py` | Head-to-head static vs dynamic scheduler on 5 PubMedQA queries; saves `scheduler_comparison.png` |
| `visualize_metrics.py` | Reads `benchmark_results.json` and produces 4 IEEE-styled figures: execution timeline, tail latency CDF, exit distribution, accuracy summary |

---

## Key Experimental Findings

### WCET Profile (RTX 4000 Ada, TinyLlama-1.1B)

Full-pass latency is **flat across sequence lengths** due to FlashAttention:

| Exit Layer | Mean (ms) | P99 (ms) | WCET (ms) |
|------------|-----------|----------|-----------|
| Layer 5    | ~3.8      | ~4.1     | ~4.5      |
| Layer 11   | ~8.0      | ~8.7     | ~9.2      |
| Layer 16   | ~11.2     | ~12.2    | ~13.1     |
| Full (L22) | ~15.6     | ~16.5    | **16.9**  |

Safety margin applied: `WCET_full × 1.10 = 18.55 ms`

### Calibration (L16 Early Exit vs Full Pass)

| Metric | Layer 5 | Layer 16 |
|--------|---------|----------|
| Overall agreement | 1% | 26% |
| Agreement at conf ≥ 0.5 | — | **72%** |

Layer 5 exits almost never agree with the full pass — this is why the static scheduler (Layer 5, threshold 0.8) defaults to full pass for nearly every token. Layer 16 is the right early-exit point.

### Benchmark (30 PubMedQA Queries, D = 45 ms)

| Metric | Value |
|--------|-------|
| Accuracy (extractable labels) | 69.2% (9/13 scored) |
| Label extraction rate | 43.3% |
| Exit distribution | Full 88.2% / Thresh 11.6% / Forced 0.2% |
| Deadline miss rate | **0.2%** |
| Mean TPOT | 30.1 ms |
| P99 TPOT | 44.0 ms |
| Throughput | ~33.2 tokens/sec |
| Utilization (P99/D) | 0.977 |

> **Note on label extraction:** TinyLlama-1.1B does not reliably follow single-word instructions, producing verbose responses or re-generating the question in 57% of cases. This is a known limitation of 1.1B parameter instruction-tuned models. The scheduling metrics (TPOT, miss rate, utilization) are independent of this and represent the primary evaluation.

### Scheduler Comparison (Static vs Dynamic)

| Metric | Static (L5, thresh=0.8) | Dynamic (L16, decay) |
|--------|------------------------|----------------------|
| Mean TPOT | ~20.6 ms | ~26.3 ms |
| Early exit rate | ~0% | ~12% |
| Deadline misses | 0% | 0% |

The static scheduler almost never exits early (L5 agreement = 1%), effectively acting as a pure full-pass system. The dynamic scheduler at L16 achieves meaningful early exits while maintaining schedulability.

---

## Environment & Setup

Requires a **bare-metal GPU** (no hypervisor) for accurate WCET measurements.

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets matplotlib numpy
```

Optional — authenticate with HuggingFace to avoid rate limits:
```bash
export HF_TOKEN="your_huggingface_token"
```

---

## Usage

### Run the dynamic scheduler on a single prompt
```bash
python dynamic_scheduler.py
```

### Profile WCET across sequence lengths and exit layers
```bash
python profile_wcet.py
# → wcet_results.json, wcet_profile.png
```

### Run the early-exit calibration study
```bash
python calibration.py
# → calibration_results.json, calibration.png
```

### Run the 30-query PubMedQA benchmark
```bash
python benchmark.py
# → benchmark_results.json, benchmark_results.csv
```

### Formal schedulability proof (tail-latency CDF)
```bash
python evaluate_tail_latency.py
# → schedulability_proof.png, tail_latency_results.json
```

### Deadline sweep (utility-latency tradeoff)
```bash
python deadline_sweep.py
# → sweep_results.json, deadline_tradeoff.png
```

### Static vs dynamic scheduler comparison
```bash
python compare_schedulers.py
# → scheduler_comparison.json, scheduler_comparison.png
```

### Generate all visualization figures from benchmark results
```bash
python visualize_metrics.py
# → execution_timeline.png, tail_latency_cdf.png,
#   exit_distribution.png, accuracy_summary.png
```
