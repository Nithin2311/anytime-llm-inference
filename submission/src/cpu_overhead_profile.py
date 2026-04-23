"""
cpu_overhead_profile.py — CPU overhead measurement for KV-cached and stateless schedulers.

For each token generation step, records:
  gpu_ms           : torch.cuda.Event elapsed time (pure GPU kernel time)
  wall_ms          : perf_counter_ns wall-clock time wrapping the same block
  cpu_overhead_ms  : wall_ms - gpu_ms  (Python/CPU overhead)

Loads first 3 prompts from benchmark_results.json, generates ~200 tokens
per scheduler, saves statistics and a 2-panel comparison figure.

Outputs:
  cpu_overhead_results.json
  cpu_overhead.png
"""

import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt

from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import _WCET_TABLE, _wcet_for_seq_len

RESULTS_FILE = "cpu_overhead_results.json"
FIGURE_FILE  = "cpu_overhead.png"
DEADLINE_MS  = 30.0
N_TARGET     = 200   # token samples per scheduler


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(tokenizer, question: str) -> str:
    """Format a question using TinyLlama chat template (no context — timing only)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical expert answering clinical questions. "
                "Answer each question with exactly one word: 'yes', 'no', or 'maybe'. "
                "Do not add any explanation."
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── KV-cached timing ───────────────────────────────────────────────────────────

def collect_kvcached(model, prompts, n_target, deadline_ms):
    """
    Time each forward_cached() call using both CUDA events and perf_counter_ns.

    Timed block:
      cpu_start → start_event.record() → forward_cached() → end_event.record()
      → synchronize() → cpu_end

    gpu_ms  = start_event.elapsed_time(end_event)   (pure GPU kernel time)
    wall_ms = (cpu_end - cpu_start) / 1e6           (total wall-clock time)
    """
    print("  [KV-cached] Warming up …")
    with torch.inference_mode():
        for prompt in prompts:
            ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            _, _, wkv = model.forward_cached(ids)
            dummy = torch.tensor([[1]], dtype=torch.long, device="cuda")
            model.forward_cached(dummy, past_key_values=wkv)
            model.forward_cached(dummy, past_key_values=wkv)
    torch.cuda.synchronize()
    print("  [KV-cached] Warm-up done. Collecting samples …")

    gpu_list  = []
    wall_list = []
    kv_threshold = 0.55
    prompt_idx   = 0

    with torch.inference_mode():
        while len(gpu_list) < n_target:
            prompt    = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            past_kv   = None
            generated = []

            for step in range(70):
                if len(gpu_list) >= n_target:
                    break

                cur_input = (
                    input_ids if step == 0
                    else torch.tensor([[generated[-1]]], dtype=torch.long, device="cuda")
                )

                # ── timed block ──────────────────────────────────────────────
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)

                cpu_start = time.perf_counter_ns()
                s_ev.record()
                l16_logits, full_logits, past_kv = model.forward_cached(
                    cur_input, past_key_values=past_kv
                )
                e_ev.record()
                torch.cuda.synchronize()
                cpu_end = time.perf_counter_ns()
                # ── end timed block ──────────────────────────────────────────

                gpu_ms  = s_ev.elapsed_time(e_ev)
                wall_ms = (cpu_end - cpu_start) / 1e6
                gpu_list.append(gpu_ms)
                wall_list.append(wall_ms)

                # Token selection (outside the timed block)
                probs = torch.softmax(l16_logits[0, -1, :], dim=-1)
                conf, next_l16 = torch.max(probs, dim=-1)
                next_tok = next_l16 if conf.item() >= kv_threshold else torch.argmax(full_logits[0, -1, :], dim=-1)
                tok_id = next_tok.item()
                generated.append(tok_id)
                if tok_id == model.tokenizer.eos_token_id:
                    break

    return gpu_list[:n_target], wall_list[:n_target]


# ── Stateless timing ───────────────────────────────────────────────────────────

def collect_stateless(model, prompts, n_target, deadline_ms):
    """
    Time the full two-pass decision block (L16 probe + optional full pass).

    Timed block:
      cpu_start → start_event.record()
        → L16 forward → mid_event.record() → synchronize()   [budget check]
        → [optional full pass]
        → end_event.record() → synchronize()
      → cpu_end

    gpu_ms  = start_event.elapsed_time(end_event)   (GPU kernel time, includes
              any inter-kernel idle while CPU computes the budget decision)
    wall_ms = (cpu_end - cpu_start) / 1e6
    """
    print("  [Stateless] Warming up …")
    with torch.inference_mode():
        for prompt in prompts:
            ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            model(ids, use_cache=False)
            model(ids, exit_layer=16, use_cache=False)
            model(ids, exit_layer=16, use_cache=False)
    torch.cuda.synchronize()
    print("  [Stateless] Warm-up done. Collecting samples …")

    gpu_list   = []
    wall_list  = []
    max_conf   = 0.8
    min_conf   = 0.3
    prompt_idx = 0

    with torch.inference_mode():
        while len(gpu_list) < n_target:
            prompt    = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            generated = []

            for step in range(70):
                if len(gpu_list) >= n_target:
                    break

                full_pass_wcet = _wcet_for_seq_len(input_ids.shape[1], _WCET_TABLE)

                # ── timed block ──────────────────────────────────────────────
                s_ev   = torch.cuda.Event(enable_timing=True)
                mid_ev = torch.cuda.Event(enable_timing=True)
                e_ev   = torch.cuda.Event(enable_timing=True)

                cpu_start = time.perf_counter_ns()
                s_ev.record()

                # Stage 1: L16 probe
                logits_early, _ = model(input_ids, exit_layer=16, use_cache=False)
                mid_ev.record()
                torch.cuda.synchronize()                       # needed to read budget
                elapsed_early = s_ev.elapsed_time(mid_ev)

                # CPU budget decision (this gap contributes to cpu_overhead)
                probs = torch.softmax(logits_early[0, -1, :], dim=-1)
                conf, next_early = torch.max(probs, dim=-1)
                conf_val  = conf.item()
                remaining = deadline_ms - elapsed_early

                if remaining >= full_pass_wcet:
                    ratio     = (remaining - full_pass_wcet) / max(deadline_ms - full_pass_wcet, 1e-6)
                    threshold = min_conf + (max_conf - min_conf) * ratio
                else:
                    threshold = 0.0

                # Stage 2: commit decision
                if remaining < full_pass_wcet:
                    next_tok = next_early
                    e_ev.record()
                elif conf_val >= threshold:
                    next_tok = next_early
                    e_ev.record()
                else:
                    logits_full, _ = model(input_ids, use_cache=False)
                    next_tok = torch.argmax(logits_full[0, -1, :], dim=-1)
                    e_ev.record()

                torch.cuda.synchronize()
                cpu_end = time.perf_counter_ns()
                # ── end timed block ──────────────────────────────────────────

                gpu_ms  = s_ev.elapsed_time(e_ev)
                wall_ms = (cpu_end - cpu_start) / 1e6
                gpu_list.append(gpu_ms)
                wall_list.append(wall_ms)

                tok_id = next_tok.item()
                generated.append(tok_id)
                input_ids = torch.cat(
                    [input_ids, next_tok.unsqueeze(0).unsqueeze(0)], dim=-1
                )
                if tok_id == model.tokenizer.eos_token_id:
                    break

    return gpu_list[:n_target], wall_list[:n_target]


# ── Statistics helper ──────────────────────────────────────────────────────────

def compute_stats(gpu_list, wall_list):
    gpu  = np.array(gpu_list,  dtype=float)
    wall = np.array(wall_list, dtype=float)
    oh   = wall - gpu

    mean_gpu  = float(np.mean(gpu))
    mean_wall = float(np.mean(wall))
    mean_oh   = float(np.mean(oh))

    return {
        "gpu_ms":               [round(v, 4) for v in gpu.tolist()],
        "wall_ms":              [round(v, 4) for v in wall.tolist()],
        "cpu_overhead_ms":      [round(v, 4) for v in oh.tolist()],
        "mean_gpu_ms":          round(mean_gpu,  4),
        "mean_wall_ms":         round(mean_wall, 4),
        "mean_cpu_overhead_ms": round(mean_oh,   4),
        "p99_gpu_ms":           round(float(np.percentile(gpu,  99)), 4),
        "p99_wall_ms":          round(float(np.percentile(wall, 99)), 4),
        "overhead_pct":         round(mean_oh / mean_wall * 100, 2),
    }


# ── Figure ─────────────────────────────────────────────────────────────────────

def plot_overhead(kv_stats, sl_stats):
    fs.apply()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel 1: box plot gpu_ms vs wall_ms, both schedulers ──────────────────
    ax1 = axes[0]

    kv_gpu   = kv_stats["gpu_ms"]
    kv_wall  = kv_stats["wall_ms"]
    sl_gpu   = sl_stats["gpu_ms"]
    sl_wall  = sl_stats["wall_ms"]

    labels   = ["KV-Cached\nGPU", "KV-Cached\nWall", "Stateless\nGPU", "Stateless\nWall"]
    data     = [kv_gpu, kv_wall, sl_gpu, sl_wall]
    colours  = ["#4878d0", "#ee854a", "#4878d0", "#ee854a"]
    hatches  = ["", "", "//", "//"]

    bp = ax1.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        widths=0.55,
    )
    for patch, colour, hatch in zip(bp["boxes"], colours, hatches):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
        patch.set_hatch(hatch)

    ax1.set_xticks(range(1, 5))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("GPU vs Wall-Clock Latency per Token")

    # Legend patches
    import matplotlib.patches as mpatches
    gpu_patch  = mpatches.Patch(facecolor="#4878d0", alpha=0.75, label="GPU time (CUDA events)")
    wall_patch = mpatches.Patch(facecolor="#ee854a", alpha=0.75, label="Wall-clock time")
    ax1.legend(handles=[gpu_patch, wall_patch], loc="upper right")

    # Annotate medians
    for i, (bx, d) in enumerate(zip(bp["boxes"], data)):
        med = float(np.median(d))
        ax1.text(i + 1, med + 0.15, f"{med:.1f}", ha="center", va="bottom",
                 fontsize=7.5, color="black")

    # ── Panel 2: CPU overhead breakdown ───────────────────────────────────────
    ax2  = axes[1]
    ax2b = ax2.twinx()

    schedulers = ["KV-Cached", "Stateless"]
    oh_ms  = [kv_stats["mean_cpu_overhead_ms"], sl_stats["mean_cpu_overhead_ms"]]
    oh_pct = [kv_stats["overhead_pct"],          sl_stats["overhead_pct"]]
    x      = np.arange(len(schedulers))
    width  = 0.38

    bars_ms = ax2.bar(x - width / 2, oh_ms, width, color=["#4878d0", "#6acc65"],
                      alpha=0.82, label="Mean CPU overhead (ms)")
    bars_pct = ax2b.bar(x + width / 2, oh_pct, width, color=["#d65f5f", "#ee854a"],
                        alpha=0.72, label="Overhead %", hatch="//")

    ax2.set_ylabel("Mean CPU Overhead (ms)")
    ax2b.set_ylabel("Overhead (% of wall time)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(schedulers)
    ax2.set_title("CPU Overhead Breakdown")

    for bar, val in zip(bars_ms, oh_ms):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f} ms", ha="center", va="bottom", fontsize=8.5)
    for bar, val in zip(bars_pct, oh_pct):
        ax2b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                  f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.suptitle("CPU Overhead Analysis — TinyLlama-1.1B (RTX 6000 Ada)", fontsize=7.5, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CPU OVERHEAD PROFILING — TinyLlama-1.1B")
    print(f"Target samples per scheduler: {N_TARGET}")
    print("=" * 60 + "\n")

    # Load prompts from benchmark results
    with open("benchmark_results.json") as f:
        bm = json.load(f)

    model = EarlyExitTinyLlama()

    prompts = [
        build_prompt(model.tokenizer, q["question"])
        for q in bm["query_results"][:3]
    ]
    print(f"Loaded {len(prompts)} prompts from benchmark_results.json\n")
    for i, p in enumerate(prompts):
        tok_len = model.tokenizer(p, return_tensors="pt").input_ids.shape[1]
        print(f"  Prompt {i+1}: {tok_len} tokens")
    print()

    # ── KV-cached ──────────────────────────────────────────────────────────────
    print("── KV-Cached Scheduler ──────────────────────────────────────")
    kv_gpu, kv_wall = collect_kvcached(model, prompts, N_TARGET, DEADLINE_MS)
    kv_stats = compute_stats(kv_gpu, kv_wall)
    print(f"  mean GPU    : {kv_stats['mean_gpu_ms']:.3f} ms")
    print(f"  mean wall   : {kv_stats['mean_wall_ms']:.3f} ms")
    print(f"  mean OH     : {kv_stats['mean_cpu_overhead_ms']:.3f} ms  "
          f"({kv_stats['overhead_pct']:.1f}% of wall)")
    print(f"  P99 GPU     : {kv_stats['p99_gpu_ms']:.3f} ms")
    print(f"  P99 wall    : {kv_stats['p99_wall_ms']:.3f} ms\n")

    # ── Stateless ──────────────────────────────────────────────────────────────
    print("── Stateless Scheduler ──────────────────────────────────────")
    sl_gpu, sl_wall = collect_stateless(model, prompts, N_TARGET, DEADLINE_MS)
    sl_stats = compute_stats(sl_gpu, sl_wall)
    print(f"  mean GPU    : {sl_stats['mean_gpu_ms']:.3f} ms")
    print(f"  mean wall   : {sl_stats['mean_wall_ms']:.3f} ms")
    print(f"  mean OH     : {sl_stats['mean_cpu_overhead_ms']:.3f} ms  "
          f"({sl_stats['overhead_pct']:.1f}% of wall)")
    print(f"  P99 GPU     : {sl_stats['p99_gpu_ms']:.3f} ms")
    print(f"  P99 wall    : {sl_stats['p99_wall_ms']:.3f} ms\n")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "n_samples":  N_TARGET,
        "deadline_ms": DEADLINE_MS,
        "kvcached":   kv_stats,
        "stateless":  sl_stats,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to '{RESULTS_FILE}'")

    # ── Figure ─────────────────────────────────────────────────────────────────
    plot_overhead(kv_stats, sl_stats)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'KV-Cached':>12} {'Stateless':>12}")
    print("-" * 56)
    print(f"{'Mean GPU (ms)':<30} {kv_stats['mean_gpu_ms']:>12.3f} {sl_stats['mean_gpu_ms']:>12.3f}")
    print(f"{'Mean Wall (ms)':<30} {kv_stats['mean_wall_ms']:>12.3f} {sl_stats['mean_wall_ms']:>12.3f}")
    print(f"{'Mean CPU Overhead (ms)':<30} {kv_stats['mean_cpu_overhead_ms']:>12.3f} {sl_stats['mean_cpu_overhead_ms']:>12.3f}")
    print(f"{'Overhead %':<30} {kv_stats['overhead_pct']:>11.1f}% {sl_stats['overhead_pct']:>11.1f}%")
    print(f"{'P99 Wall (ms)':<30} {kv_stats['p99_wall_ms']:>12.3f} {sl_stats['p99_wall_ms']:>12.3f}")
    print("=" * 60)
