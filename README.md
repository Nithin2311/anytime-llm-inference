# Dynamic Anytime Scheduling for LLM Inference
**Bounding Tail Latency via Predictive Early-Exit Mechanisms**

**Author:** Nithin Palyam
**Course:** CIS 6930 — Real-Time Systems (Spring 2026), University of South Florida

---

## Overview

Standard autoregressive LLM inference processes every transformer layer for every generated token, producing execution times that are unbounded in the worst case. In latency-sensitive applications — clinical decision support, interactive HCI, cyber-physical systems — this tail latency violates hard real-time constraints.

This project implements an **Anytime Algorithm** framework for LLM token generation on TinyLlama-1.1B-Chat. The core insight: a transformer's intermediate hidden states at Layer 16 already carry predictive signal (32.0% overall agreement with the full 22-layer oracle, rising to 64.7% when confidence ≥ 0.5). A KV-cached scheduler exploits this by committing early-exit tokens when confidence exceeds a fixed threshold, guaranteeing every token is delivered within a hard deadline **D = 45 ms**.

**Hardware:** RunPod RTX 4000 Ada (bare-metal, 21 GB VRAM)
**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 (22 transformer layers, hidden dim 2048, vocab 32000)
**Framework:** PyTorch 2.4.1 + CUDA 12.4, transformers 5.5.3

---

## Repository Structure

```
anytime-llm-inference/
├── Core Implementation
│   ├── early_exit_model.py       # TinyLlama with configurable early exit
│   ├── static_scheduler.py       # Phase 1: fixed-threshold anytime scheduler
│   └── dynamic_scheduler.py      # Phase 2: dynamic threshold decay scheduler
│
├── Profiling & Calibration
│   ├── profile_wcet.py           # GPU worst-case execution time profiling
│   ├── calibration.py            # Early-exit confidence calibration study
│   └── exit_layer_ablation.py    # Ablation: L5/L11/L16/L17/L18/L19/L20 exit layers
│
├── Evaluation
│   ├── benchmark.py              # 30-query PubMedQA clinical benchmark
│   ├── evaluate_tail_latency.py  # Schedulability proof via tail-latency CDF
│   ├── compare_schedulers.py     # Stateless vs KV-cached head-to-head comparison
│   └── deadline_sweep.py         # Utility-latency tradeoff across deadlines
│
├── Visualisation & Infrastructure
│   ├── visualize_metrics.py      # IEEE-style figures from benchmark data
│   ├── verify_env.py             # 17-check pre-flight environment validation
│   └── run_all.sh                # Full reproduction pipeline
│
├── Data Outputs (generated)
│   ├── wcet_results.json         # Measured WCET per exit layer and sequence length
│   ├── calibration_results.json  # Per-token confidence and agreement data
│   ├── ablation_results.json     # Exit-layer ablation per-token records
│   ├── benchmark_results.json    # 30-query benchmark results
│   ├── benchmark_results.csv     # Flat per-query table (LaTeX/spreadsheet)
│   ├── tail_latency_results.json # Schedulability proof statistics
│   ├── scheduler_comparison.json # Static vs dynamic comparison metrics
│   └── sweep_results.json        # Deadline sweep aggregated metrics
│
└── Figures (generated)
    ├── wcet_profile.png           # WCET vs sequence length per exit layer
    ├── calibration.png            # Calibration curves and confidence distributions
    ├── exit_layer_ablation.png    # L5/L11/L16/L17/L18/L19/L20 comparison figure
    ├── schedulability_proof.png   # Baseline vs anytime CDF with deadline line
    ├── scheduler_comparison.png   # Stateless vs KV-cached three-panel comparison
    ├── deadline_tradeoff.png      # Exit distribution and miss rate vs deadline
    ├── execution_timeline.png     # Per-token latency bar chart (Query 1)
    ├── tail_latency_cdf.png       # TPOT CDF with bootstrap P99 CI
    ├── exit_distribution.png      # Exit-type stacked bar per query
    └── accuracy_summary.png       # Benchmark dashboard (3 panels)
```

---

## Formal Schedulability Analysis

Each token generation is treated as a real-time task τᵢ with hard deadline **D**.

### Schedulability Criterion

```
P99_TPOT ≤ D      →  SCHEDULABLE
```

### Utilization Bound

```
U = P99_TPOT / D        (must be < 1.0)
```

### Anytime Algorithm Safety Guarantee

The KV-cached scheduler enforces schedulability **by construction**:

```
Per-token decision (KV-cached single-pass):
  1. Run all 22 layers in one forward_cached() call — O(n_cache) attention
     Simultaneously capture L16 hidden state via forward hook
  2. Compute L16 logits + full logits from the same pass
  3. If L16 confidence ≥ threshold (0.55) → commit L16 token
     Else                                 → commit full-pass token

WCET bound (KV-cached, single-token pass):
  All tokens complete in 18–21 ms regardless of context length
  (O(n_cache) attention vs O(n²) full recompute)
```

The stateless two-pass variant enforces schedulability differently:

```
Stateless decision at each step:
  1. Run Layer-16 probe     →  elapsed_early
  2. remaining_budget = D − elapsed_early

  3a. remaining_budget < WCET_full(seq_len) →  Forced early exit
  3b. confidence ≥ current_threshold         →  Threshold early exit
  3c. else                                   →  Full 22-layer pass

Dynamic threshold decay: min_conf + (max_conf − min_conf) × time_ratio
```

At seq_len ≤ 256 the stateless scheduler fits within D. At seq_len ≥ ~500 tokens the L16 probe alone takes ≥ 18 ms and the full pass adds another 24+ ms — exceeding D = 45 ms. The KV-cached approach eliminates this problem entirely.

### Measured Schedulability (KV-Cached Anytime Scheduler)

10 diverse clinical prompts × 25 tokens = 240 TPOT samples each (D = 45 ms):

| Metric | Baseline (Full Pass) | Anytime Scheduler (KV-Cached) |
|--------|---------------------|-------------------------------|
| Samples | 240 | 240 |
| P50 TPOT | 18.80 ms | **19.05 ms** |
| P95 TPOT | 21.65 ms | **19.29 ms** |
| P99 TPOT | 22.96 ms | **19.44 ms** |
| Max TPOT | 23.43 ms | **19.73 ms** |
| Utilization (P99/D) | 0.5102 | **0.4321** |
| Miss Rate | 0.00% | **0.00%** |
| Schedulable? | YES | **YES** |

The KV-cached anytime scheduler achieves a **lower P99** than the baseline (19.4 ms vs 23.0 ms) and dramatically tighter variance: the max observed latency is 19.7 ms vs 23.4 ms for the baseline. This is the key advantage of the KV-cached design — the single-pass approach with O(n_cache) attention keeps latency nearly flat regardless of context length, while the baseline's full recompute grows with context.

---

## Implementation Details

### `early_exit_model.py` — EarlyExitTinyLlama

Wraps TinyLlama-1.1B with explicit layer-by-layer forward pass. The standard HuggingFace `forward()` is bypassed to allow halting at any layer.

```python
model = EarlyExitTinyLlama()
logits, _ = model(input_ids, exit_layer=16, use_cache=False)  # exit at L16
logits, _ = model(input_ids)                                  # full 22-layer pass
```

Key correctness invariants:
- **RMSNorm always applied** at the exit point regardless of depth
- **Rotary embeddings computed once** and shared across all layers (matches standard LlamaModel exactly)
- **No in-place model state mutation** — safe for repeated calls with different `exit_layer` values

The model also exposes `forward_cached(input_ids, past_key_values)` for KV-cached inference. This uses a forward hook on layer 15 (0-indexed = "Layer 16") to capture the intermediate hidden state while running all 22 layers in a single pass. Both L16 and full-pass logits are returned from the same call:

```python
l16_logits, full_logits, past_kv = model.forward_cached(input_ids)
# next call: single new token, O(n_cache) attention
l16_logits, full_logits, past_kv = model.forward_cached(new_token, past_key_values=past_kv)
```

This avoids the KV-cache desynchronization problem of the two-pass stateless approach (where L16-exit tokens never update layers 16–21's KV states, corrupting subsequent full-pass attention).

### `static_scheduler.py` — Phase 1 Scheduler

Fixed confidence threshold (0.8) at Layer 16. Two-stage per-token decision:
1. Run Layer-16 early exit; measure confidence
2. If `conf >= 0.8` → commit early; else if `remaining < WCET_full(seq_len)` → forced exit; else → full pass

```
Exit types: "Early (High Conf)" | "Early (Deadline)" | "Full Pass"
```

### `dynamic_scheduler.py` — Phase 2 Scheduler

Two generation modes:

**`generate_stateless_anytime`** — dynamic threshold decay at Layer 16. Three-stage per-token decision:
1. Run Layer-16 probe; measure elapsed time and confidence
2. Compute `current_threshold` via linear decay proportional to remaining budget
3. Commit decision: forced exit / threshold exit / full pass

```
Exit types: "Full Pass" | "Early (Thresh: X.XX)" | "Early (Forced)"
```

`_WCET_TABLE` is loaded at import time from `wcet_results.json`. Each token call looks up the correct WCET bin for the current `input_ids` length via `_wcet_for_seq_len()` (ceiling lookup with 1.10× safety factor). This is stateless (no KV cache) — **not schedulable at D = 45 ms for prompts > 256 tokens**.

**`generate_anytime_with_kv`** — KV-cached single-pass inference. One `forward_cached()` call per token provides both L16 and full-pass logits. Fixed threshold at `(max_conf + min_conf) / 2 = 0.55`. No forced-exit logic needed — the single-pass is always fast (18–21 ms).

```
Exit types: "Full Pass" | "Early (Thresh: 0.55)"
```

---

## Profiling & Calibration Results

### WCET Profile (RTX 4000 Ada, TinyLlama-1.1B)

Full-pass latency is **approximately flat** across short sequence lengths (32–256 tokens) due to FlashAttention, but grows noticeably at 512+ tokens as attention cost becomes the bottleneck:

| Seq Len | L16 mean | L16 WCET | Full mean | Full WCET | Full WCET ×1.10 |
|---------|----------|----------|-----------|-----------|-----------------|
| 32      | 13.6 ms  | 14.5 ms  | 18.8 ms   | 19.1 ms   | 21.0 ms         |
| 64      | 13.7 ms  | 14.3 ms  | 18.1 ms   | 18.8 ms   | 20.7 ms         |
| 128     | 13.4 ms  | 13.8 ms  | 18.2 ms   | 18.8 ms   | 20.7 ms         |
| 256     | 13.4 ms  | 14.1 ms  | 18.1 ms   | 19.2 ms   | 21.1 ms         |
| **512** | **17.4 ms** | **18.0 ms** | **23.5 ms** | **24.3 ms** | **26.7 ms** |
| **1024**| **33.7 ms** | **33.8 ms** | **45.6 ms** | **45.8 ms** | **50.3 ms** |

> **Critical finding — 1024-token context limit:** At seq_len = 1024 the full-pass WCET (45.75 ms) equals D = 45 ms, and the two-pass total (L16 ≈ 34 ms + full ≈ 46 ms) far exceeds it. After the L16 probe (≈ 34 ms), only ≈ 11 ms of budget remains, which is less than any profiled full-pass WCET — so the scheduler **always takes a forced early exit** at these lengths. This is the correct anytime behavior: given a tight budget, commit the best available answer (L16 token) rather than miss the deadline. The practical implication is that PubMedQA prompts longer than ~900 tokens will always be served from L16 rather than the full 22-layer pass.

Both schedulers load the full per-length WCET table at import time and perform a ceiling lookup at each token via `_wcet_for_seq_len(input_ids.shape[1], _WCET_TABLE)`.

### Exit-Layer Ablation Study

Validates the design choice of Layer 16 as the early-exit point, with extended analysis up to Layer 20:

| Metric | L5 | L11 | **L16** | L17 | L18 | L19 | L20 |
|--------|----|-----|---------|-----|-----|-----|-----|
| Agreement with oracle | 0.7% | 3.3% | **32.0%** | 40.7% | 44.0% | 62.0% | 70.0% |
| Agreement @ conf ≥ 0.5 | 25.0% | 33.3% | **64.7%** | 65.9% | 73.0% | 74.0% | 87.3% |
| Mean confidence | 0.065 | 0.068 | **0.185** | 0.317 | 0.453 | 0.647 | 0.723 |
| Mean TPOT | 5.0 ms | 9.7 ms | **13.6 ms** | 14.4 ms | 15.1 ms | 15.9 ms | 16.7 ms |
| WCET | 6.0 ms | 10.4 ms | **15.1 ms** | 16.1 ms | 16.7 ms | 17.7 ms | 19.0 ms |

**L5 and L11 have near-zero agreement with the full-pass oracle** (0.7% and 3.3%) — they carry negligible predictive signal. Agreement and mean confidence both rise monotonically from L16 toward L22. Since the KV-cached scheduler always runs all 22 layers (capturing L16 via a forward hook), the marginal latency cost of choosing a deeper exit layer is only 1–4 ms. **L16 remains the design choice** as the earliest layer with substantial signal (32% agreement, 64.7% at conf ≥ 0.5). L19–L20 could serve as alternatives for applications that prioritize prediction quality over exit speed.

### Confidence Calibration (L16 vs L22)

| Metric | Layer 5 | Layer 16 |
|--------|---------|----------|
| Overall agreement | 2.0% | **31.0%** |
| Mean confidence | 0.063 | **0.166** |
| Agreement @ conf ≥ 0.5 | 25.0% | **66.7%** |
| Tokens with conf ≥ 0.8 | 0.5% | **3.5%** |

When the L16 softmax is confident (conf ≥ 0.5), it agrees with the full 22-layer pass **66.7%** of the time — directly validating the confidence-threshold approach. Only 3.5% of tokens reach the high-confidence regime (conf ≥ 0.8), confirming that the fixed threshold at 0.55 (midpoint of the decay range) correctly captures the majority of early-exit opportunities while avoiding the over-conservative behavior of a 0.8 fixed threshold.

---

## Benchmark Results

### 30-Query PubMedQA Clinical Benchmark (KV-Cached, D = 45 ms)

| Metric | Value |
|--------|-------|
| Accuracy (extractable labels) | **71.4%** (10/14 scored) |
| Label extraction rate | **46.7%** |
| Exit distribution | Full 95.3% / Thresh 4.7% / Forced 0.0% |
| Deadline miss rate | **0.0%** |
| Mean TPOT | 19.5 ms |
| P99 TPOT | **20.7 ms** |
| Throughput | **51.2 tokens/sec** |
| Utilization (P99/D) | **0.46** |
| Schedulable? | **YES** |

> **Note on label extraction:** TinyLlama-1.1B-Chat does not reliably follow single-word instructions, generating verbose responses in ~53% of cases. Reducing `max_new_tokens` from 15 → 5 improved extraction from 43.3% → 46.7%. The scheduling metrics (TPOT, miss rate, utilization ratio) are independent of label quality and represent the primary system evaluation.

### Stateless vs. KV-Cached Comparison (5 queries, D = 45 ms)

Both schedulers use **Layer 16** as the early-exit point. This comparison isolates the effect of KV caching on schedulability.

| Metric | Stateless (L16, decay 0.8→0.3) | KV-Cached (L16, fixed 0.55) |
|--------|--------------------------------|------------------------------|
| Mean TPOT | 39.2 ms | **20.0 ms** |
| Throughput | 25.5 tok/s | **49.9 tok/s** |
| P99 TPOT | 48.3 ms | **22.0 ms** |
| Utilization (P99/D) | 1.073 | **0.488** |
| Early exit rate | 10.7% | 9.3% |
| Forced exit rate | 0.0% | 0.0% |
| Deadline misses | **16.0%** | **0.0%** |
| Schedulable? | **NO** | **YES** |

The stateless scheduler is **not schedulable** at D = 45 ms for these PubMedQA prompts (typical length 150–200 tokens in the chat template). Each token requires two sequential forward passes (L16 probe ≈ 14–20 ms, then full pass ≈ 19 ms), totalling 33–39 ms mean TPOT and a P99 of 48 ms that exceeds the deadline.

The KV-cached scheduler runs a **single forward pass** per token with O(n_cache) attention, keeping all tokens in the 18–22 ms range regardless of context length — half the latency and zero misses. **KV caching is essential for schedulability** at this deadline.

### Deadline Sweep (5 queries, stateless scheduler)

The deadline sweep uses the **stateless** dynamic scheduler (`generate_stateless_anytime`) because its explicit forced-exit mechanism makes deadline-adaptation behavior visible across deadlines. The KV-cached scheduler has no forced exits (single-pass always completes in ~20 ms), so its exit distribution is flat across all deadlines — less informative for this analysis.

| Deadline (ms) | Full% | Thresh% | Forced% | Miss% | Mean TPOT |
|---------------|-------|---------|---------|-------|-----------|
| 20 | 0.0 | 0.0 | 100.0 | 37.3 | 19.2 ms |
| 25 | 0.0 | 0.0 | 100.0 | 0.0 | 15.4 ms |
| 30 | 64.0 | 9.3 | 26.7 | 1.3 | 21.2 ms |
| **35** | **90.7** | **9.3** | **0.0** | **0.0** | **26.3 ms** |
| 40 | 92.0 | 8.0 | 0.0 | 0.0 | 26.6 ms |
| 45 | 90.7 | 9.3 | 0.0 | 0.0 | 26.9 ms |
| 50–60 | 90.7 | 9.3 | 0.0 | 0.0 | ~25–27 ms |

Schedulability **knee** at D = 30 ms. At D = 25 ms the L16 probe (~14 ms) leaves only ~11 ms of budget, which is less than the full-pass WCET (~21 ms at seq_len ≤ 256), so the scheduler correctly forces 100% early exits with zero misses. From D = 35 ms onward the system is fully schedulable with no forced exits. The optimal operating point for the stateless scheduler is D = 35–45 ms; the KV-cached scheduler operates comfortably at D = 22 ms and above.

---

## Environment & Setup

Requires a **bare-metal GPU** (no VM/hypervisor) for valid WCET measurements.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets matplotlib numpy pandas

# Optional: authenticate with HuggingFace to avoid rate limits
export HF_TOKEN="your_huggingface_token"
```

Validate the environment before running experiments:

```bash
python verify_env.py
```

This checks Python version, all package imports, CUDA availability, GPU memory, event timer precision, model loadability, early-exit forward pass correctness, KV-cache path correctness, WCET data integrity, dataset accessibility, and chat-template formatting. All 17 checks must pass.

---

## Usage

### Reproduce everything (full pipeline)

```bash
source venv/bin/activate
bash run_all.sh               # ~45–60 min on RTX 4000 Ada
bash run_all.sh --skip-slow   # ~15 min, skips 30-query benchmark and deadline sweep
```

### Run individual components

**Profile WCET across sequence lengths and exit layers:**
```bash
python profile_wcet.py
# → wcet_results.json, wcet_profile.png
```

**Early-exit confidence calibration:**
```bash
python calibration.py
# → calibration_results.json, calibration.png
```

**Exit-layer ablation study (L5 / L11 / L16 / L17 / L18 / L19 / L20 vs oracle L22):**
```bash
python exit_layer_ablation.py
# → ablation_results.json, exit_layer_ablation.png
```

**Formal schedulability proof (tail-latency CDF, 10 prompts):**
```bash
python evaluate_tail_latency.py
# → schedulability_proof.png, tail_latency_results.json
```

**Stateless vs KV-cached scheduler comparison (5 queries):**
```bash
python compare_schedulers.py
# → scheduler_comparison.json, scheduler_comparison.png
```

**Deadline sweep — utility-latency tradeoff:**
```bash
python deadline_sweep.py
# → sweep_results.json, deadline_tradeoff.png
```

**Full 30-query PubMedQA benchmark:**
```bash
python benchmark.py
# → benchmark_results.json, benchmark_results.csv
```

**Regenerate all benchmark figures:**
```bash
python visualize_metrics.py
# → execution_timeline.png, tail_latency_cdf.png,
#   exit_distribution.png, accuracy_summary.png
```

**Run a single prompt through the dynamic scheduler:**
```bash
python dynamic_scheduler.py
```

**Run a single prompt through the static scheduler:**
```bash
python static_scheduler.py
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Early-exit layer | Layer 16 (of 22) | Earliest layer with substantial oracle agreement (32.0%); L5=0.7%, L11=3.3% |
| KV-cache strategy | Single `forward_cached()` pass, L16 via hook | Avoids KV desync from two-pass approach; O(n_cache) attention keeps latency flat (18–21 ms) vs O(n²) stateless (33–48 ms) |
| Exit threshold (KV-cached) | Fixed 0.55 = midpoint of 0.8→0.3 decay range | Post-hoc decision after both logits are in hand; fixed threshold is sufficient since latency is already bounded by single-pass |
| Threshold decay (stateless) | Linear from 0.8 → 0.3 | Proportional to remaining budget; allows full pass early when budget is ample |
| WCET safety margin | max_observed × 1.10 | 10% headroom over measured WCET; loaded from `wcet_results.json` at import time (stateless mode only) |
| Benchmark dataset | PubMedQA (`pqa_labeled`) | Clinical domain; yes/no/maybe labels; challenges instruction-following |
| Prompt format | TinyLlama chat template | Consistent formatting across all experiments; system instruction constrains output to one word |
