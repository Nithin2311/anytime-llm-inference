"""
pipeline_latency_model.py — Heterogeneous CPU-PCIe-GPU pipeline latency model.

Models the true end-to-end token generation pipeline as a three-stage
heterogeneous workload spanning host CPU, PCIe bus, and discrete GPU:

    T_total_sync  = T_cpu + T_pcie + T_gpu      (synchronous sequential loop)
    T_total_async = max(T_cpu, T_gpu) + T_sync   (async-overlap pipeline)

where:
    T_cpu    CPU overhead: argmax, softmax, tokenizer decode, state update
    T_pcie   Host-to-device PCIe DMA latency for a single token ID tensor
    T_gpu    CUDA kernel execution time: full 22-layer transformer pass
    T_sync   torch.cuda.synchronize() round-trip cost

The GPU-only schedulability model (T_gpu ≤ D) understates the true latency
burden.  This module measures each component in isolation, derives the
updated SLO compliance ratio under both execution paradigms, and plots the
pipeline breakdown.

Outputs:
    pipeline_latency_results.json
    pipeline_latency_model.png
"""

import json
import time
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from early_exit_model import EarlyExitTinyLlama

RESULTS_FILE = "pipeline_latency_results.json"
FIGURE_FILE  = "pipeline_latency_model.png"
DEADLINE_MS  = 30.0
N_SAMPLES    = 200
N_WARMUP     = 20


# ── Component measurements ──────────────────────────────────────────────────────

def measure_t_gpu(model, input_ids, n_warmup=N_WARMUP, n_runs=N_SAMPLES):
    """
    Measure pure GPU kernel time for the 22-layer KV-cached forward pass.

    Uses CUDA events (hardware timers) bracketing only the GPU kernels.
    The synchronise-once-after-all-runs pattern keeps measurement overhead
    outside the timed window.
    """
    with torch.inference_mode():
        for _ in range(n_warmup):
            model.forward_cached(input_ids)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]

    # Prime KV cache so subsequent calls measure decode-phase latency only
    _, _, past_kv = model.forward_cached(input_ids)
    single_tok = torch.zeros((1, 1), dtype=torch.long, device="cuda")

    with torch.inference_mode():
        for i in range(n_runs):
            starts[i].record()
            model.forward_cached(single_tok, past_key_values=past_kv)
            ends[i].record()

    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def measure_t_pcie(n_warmup=N_WARMUP, n_runs=N_SAMPLES):
    """
    Measure PCIe host-to-device DMA latency for a single token ID tensor.

    Uses a pinned (page-locked) source buffer and non_blocking=True to
    exercise the DMA engine without CPU involvement, matching the async
    overlap router's transfer path.
    """
    pinned = torch.zeros((1, 1), dtype=torch.long).pin_memory()

    # Warmup
    for _ in range(n_warmup):
        pinned[0, 0] = 42
        t = pinned.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_runs):
        pinned[0, 0] = 1
        t0 = time.perf_counter_ns()
        t = pinned.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)

    return times_ms


def measure_t_cpu(model, n_warmup=N_WARMUP, n_runs=N_SAMPLES):
    """
    Measure CPU-side post-processing overhead per token:
        softmax → max-confidence → argmax → tokenizer.decode()

    This is the work the async router hides behind the GPU forward pass.
    """
    # Realistic logit tensor (vocab size of TinyLlama = 32 000)
    vocab_size = model.base_model.config.vocab_size
    dummy_logits = torch.randn(1, 1, vocab_size, device="cuda")

    # Warmup
    for _ in range(n_warmup):
        probs = torch.softmax(dummy_logits[0, -1, :], dim=-1)
        _, _ = torch.max(probs, dim=-1)
        torch.argmax(dummy_logits[0, -1, :], dim=-1).item()
        model.tokenizer.decode([1])

    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        probs = torch.softmax(dummy_logits[0, -1, :], dim=-1)
        conf, _ = torch.max(probs, dim=-1)
        _ = conf.item()                              # D2H for 1 scalar
        next_tok = torch.argmax(dummy_logits[0, -1, :], dim=-1).item()
        model.tokenizer.decode([next_tok])           # CPU string operation
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)

    return times_ms


def measure_t_sync(n_warmup=N_WARMUP, n_runs=N_SAMPLES):
    """Measure torch.cuda.synchronize() call overhead."""
    for _ in range(n_warmup):
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)

    return times_ms


# ── Summary helper ──────────────────────────────────────────────────────────────

def summarise(samples, name):
    a = np.array(samples)
    s = {
        "name":    name,
        "mean_ms": round(float(np.mean(a)),         4),
        "p50_ms":  round(float(np.median(a)),       4),
        "p99_ms":  round(float(np.percentile(a, 99)), 4),
        "max_ms":  round(float(np.max(a)),          4),
        "std_ms":  round(float(np.std(a)),          4),
    }
    print(f"  {name:<12}  mean={s['mean_ms']:6.3f}  p99={s['p99_ms']:6.3f}  "
          f"max={s['max_ms']:6.3f}  std={s['std_ms']:5.3f}  (ms)")
    return s


# ── Pipeline model computation ──────────────────────────────────────────────────

def compute_pipeline_models(gpu, pcie, cpu, sync, deadline_ms=DEADLINE_MS):
    """
    Derive synchronous and async-overlap end-to-end pipeline latency estimates.

    Synchronous model (naive loop):
        T_sync = T_cpu + T_pcie + T_gpu + T_synchronize

    Async-overlap model (generate_anytime_async_overlap):
        T_async = max(T_cpu, T_gpu) + T_pcie + T_synchronize
        (T_cpu is fully hidden when T_cpu << T_gpu, which holds for TinyLlama)
    """
    results = {}
    for stat in ("mean_ms", "p99_ms"):
        g  = gpu[stat]
        pc = pcie[stat]
        c  = cpu[stat]
        sy = sync[stat]

        t_sync  = g + pc + c + sy
        t_async = max(g, c) + pc + sy

        results[stat] = {
            "T_gpu":         g,
            "T_pcie":        pc,
            "T_cpu":         c,
            "T_sync_call":   sy,
            "T_total_sync":  round(t_sync,  4),
            "T_total_async": round(t_async, 4),
            "R_sync":        round(t_sync  / deadline_ms, 4),
            "R_async":       round(t_async / deadline_ms, 4),
            "slo_met_sync":  t_sync  <= deadline_ms,
            "slo_met_async": t_async <= deadline_ms,
        }

    return results


# ── Figure ──────────────────────────────────────────────────────────────────────

def plot_pipeline(gpu_s, pcie_s, cpu_s, sync_s, pipeline, deadline_ms=DEADLINE_MS):
    fs.apply()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    # ── Panel 1: Pie chart of pipeline component proportions ──────────────────
    ax1 = axes[0]
    p = pipeline["mean_ms"]
    t_gpu   = p["T_gpu"]
    t_pcie  = p["T_pcie"]
    t_cpu   = p["T_cpu"]
    t_sync  = p["T_sync_call"]
    total   = t_gpu + t_pcie + t_cpu + t_sync

    pie_vals   = [t_gpu, t_cpu + t_sync, t_pcie]
    pie_labels = [
        f"T_gpu\n{t_gpu:.2f} ms\n({100*t_gpu/total:.1f}%)",
        f"T_cpu+sync\n{t_cpu+t_sync:.3f} ms\n({100*(t_cpu+t_sync)/total:.1f}%)",
        f"T_pcie\n{t_pcie:.3f} ms\n({100*t_pcie/total:.1f}%)",
    ]
    pie_colours = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    wedge_props = dict(width=0.55, edgecolor="white", linewidth=1.5)
    ax1.pie(pie_vals, labels=pie_labels, colors=pie_colours,
            wedgeprops=wedge_props, startangle=90,
            textprops=dict(fontsize=8.5))
    ax1.set_title(f"Pipeline Component Share\n(T_total = {total:.2f} ms)")

    # ── Panel 2: Stacked pipeline bar (mean, sync vs async) ───────────────────
    ax2 = axes[1]
    p = pipeline["mean_ms"]

    categories = ["Synchronous\nLoop", "Async-Overlap\nRouter"]
    t_gpu  = [p["T_gpu"],  p["T_gpu"]]
    t_pcie = [p["T_pcie"], p["T_pcie"]]
    t_cpu  = [p["T_cpu"],  0.0]          # hidden in async
    t_sync = [p["T_sync_call"], p["T_sync_call"]]

    x = np.arange(len(categories))
    width = 0.5

    b1 = ax2.bar(x, t_gpu,  width, label="T_gpu",  color="#1f77b4", alpha=0.82)
    b2 = ax2.bar(x, t_pcie, width, bottom=t_gpu, label="T_pcie", color="#ff7f0e", alpha=0.82)
    b3 = ax2.bar(x, t_cpu,  width, bottom=[t_gpu[i]+t_pcie[i] for i in range(2)],
                 label="T_cpu (hidden in async)", color="#2ca02c", alpha=0.82)
    b4 = ax2.bar(x, t_sync, width,
                 bottom=[t_gpu[i]+t_pcie[i]+t_cpu[i] for i in range(2)],
                 label="T_sync", color="#9467bd", alpha=0.82)

    totals = [p["T_total_sync"], p["T_total_async"]]
    for xi, tot in enumerate(totals):
        ax2.text(xi, tot + 0.3, f"{tot:.2f} ms\n(R={tot/deadline_ms:.3f})",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.axhline(deadline_ms, color="black", linestyle="--", linewidth=1.5,
                label=f"SLO D = {deadline_ms:.0f} ms")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("End-to-End TPOT (ms)")
    ax2.set_title("Pipeline Breakdown (mean latency)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, deadline_ms * 1.4)

    # ── Panel 3: SLO compliance ratio across deadline values ──────────────────
    ax3 = axes[2]
    deadlines = np.linspace(15, 80, 200)

    r_sync  = [p["T_total_sync"]  / d for d in deadlines]
    r_async = [p["T_total_async"] / d for d in deadlines]
    r_gpu   = [p["T_gpu"]         / d for d in deadlines]

    ax3.plot(deadlines, r_gpu,   linewidth=1.5, linestyle=":",  color="#1f77b4",
             label="GPU-only model R=T_gpu/D")
    ax3.plot(deadlines, r_sync,  linewidth=2.0, color="#d62728",
             label="Sync model R=T_total_sync/D")
    ax3.plot(deadlines, r_async, linewidth=2.0, color="#2ca02c",
             label="Async model R=T_total_async/D")
    ax3.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
                label="R = 1.0 (SLO limit)")
    ax3.axvline(deadline_ms, color="grey", linestyle=":", linewidth=1.0,
                label=f"Reference D = {deadline_ms:.0f} ms")

    ax3.fill_between(deadlines, r_sync, r_gpu, alpha=0.08, color="#d62728",
                     label="Underestimate gap")

    ax3.set_xlabel("SLO Deadline D (ms)")
    ax3.set_ylabel("SLO Compliance Ratio R")
    ax3.set_title("SLO Compliance vs Deadline\n(GPU-only vs full pipeline)")
    ax3.legend(loc="upper right", fontsize=7.5)
    ax3.set_ylim(0, 2.2)

    fig.suptitle(
        "Heterogeneous Pipeline Latency Model — TinyLlama-1.1B (RTX 6000 Ada)",
        fontsize=7.5, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("HETEROGENEOUS PIPELINE LATENCY MODEL — TinyLlama-1.1B")
    print("=" * 65 + "\n")

    model     = EarlyExitTinyLlama()
    input_ids = model.tokenizer(
        "Does statin therapy reduce mortality in patients with heart failure?",
        return_tensors="pt"
    ).input_ids.to("cuda")

    print("Measuring T_gpu  (22-layer KV-cached single-token forward pass) …")
    gpu_raw = measure_t_gpu(model, input_ids)
    gpu_s   = summarise(gpu_raw, "T_gpu")
    gpu_s["raw"] = gpu_raw

    print("Measuring T_pcie (pinned H2D transfer, 1 token ID) …")
    pcie_raw = measure_t_pcie()
    pcie_s   = summarise(pcie_raw, "T_pcie")
    pcie_s["raw"] = pcie_raw

    print("Measuring T_cpu  (softmax + argmax + tokenizer.decode) …")
    cpu_raw = measure_t_cpu(model)
    cpu_s   = summarise(cpu_raw, "T_cpu")
    cpu_s["raw"] = cpu_raw

    print("Measuring T_sync (torch.cuda.synchronize overhead) …")
    sync_raw = measure_t_sync()
    sync_s   = summarise(sync_raw, "T_sync")
    sync_s["raw"] = sync_raw

    print()
    pipeline = compute_pipeline_models(gpu_s, pcie_s, cpu_s, sync_s)

    for stat_key, label in [("mean_ms", "MEAN"), ("p99_ms", "P99")]:
        p = pipeline[stat_key]
        print(f"── {label} pipeline latency ──────────────────────────────────────")
        print(f"  Synchronous loop  : T_total = {p['T_total_sync']:.3f} ms  "
              f"(R={p['R_sync']:.4f}, SLO {'MET' if p['slo_met_sync'] else 'MISS'})")
        print(f"  Async-overlap     : T_total = {p['T_total_async']:.3f} ms  "
              f"(R={p['R_async']:.4f}, SLO {'MET' if p['slo_met_async'] else 'MISS'})")
        print(f"  GPU-only model    : T_gpu   = {p['T_gpu']:.3f} ms  "
              f"(R={p['T_gpu']/DEADLINE_MS:.4f})")
        print(f"  Underestimate gap : {p['T_total_sync'] - p['T_gpu']:.3f} ms "
              f"({(p['T_total_sync']-p['T_gpu'])/p['T_total_sync']*100:.1f}% of true TPOT)\n")

    # Serialise results (strip raw arrays for compact JSON)
    def _strip_raw(s):
        return {k: v for k, v in s.items() if k != "raw"}

    output = {
        "deadline_ms":  DEADLINE_MS,
        "n_samples":    N_SAMPLES,
        "components": {
            "T_gpu":   _strip_raw(gpu_s),
            "T_pcie":  _strip_raw(pcie_s),
            "T_cpu":   _strip_raw(cpu_s),
            "T_sync":  _strip_raw(sync_s),
        },
        "pipeline_models": pipeline,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to '{RESULTS_FILE}'")

    plot_pipeline(gpu_s, pcie_s, cpu_s, sync_s, pipeline)
