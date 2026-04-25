"""
thermal_profile.py — Sustained load and thermal stability profiling.

A critical question for real-time deployment: does WCET drift upward under
sustained load due to GPU thermal throttling?

Method:
  Generate N_TOKENS consecutive tokens from a fixed prompt using
  forward_cached() and record per-token TPOT.  Divide into N_WINDOWS windows
  and compute mean/P99/WCET per window.  Any upward drift flags a violation
  of the static WCET assumption underpinning the scheduler.

Hardware context:
  RTX 4000 Ada (21 GB GDDR6, 130 W TDP, Boost 2610 MHz).  Thermal throttle
  typically kicks in around 87–90°C junction temperature.  This test checks
  whether the GPU reaches a steady-state clock before the WCET degrades.

Outputs:
  thermal_profile_results.json
  thermal_profile.png
"""

import json
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt

from early_exit_model import EarlyExitTinyLlama

RESULTS_FILE = "thermal_profile_results.json"
FIGURE_FILE  = "thermal_profile.png"
N_TOKENS     = 500          # consecutive tokens to generate
N_WINDOWS    = 10           # analysis windows (50 tokens each)
WINDOW_SIZE  = N_TOKENS // N_WINDOWS
DEADLINE_MS  = 45.0

# A long, information-dense medical prompt chosen to produce a realistic
# context length (~120 tokens after chat-template formatting).
PROMPT_TEXT  = (
    "Explain the molecular mechanisms of oxidative phosphorylation in mitochondria, "
    "including the role of the electron transport chain complexes I through IV, "
    "ATP synthase, and the proton gradient across the inner mitochondrial membrane."
)


def format_prompt(tokenizer, text):
    messages = [
        {"role": "system", "content": "You are a biomedical expert. Be concise."},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_sustained_generation(model, prompt, n_tokens):
    """
    Generate n_tokens consecutive tokens using forward_cached().
    Returns a list of per-token TPOT measurements (ms), including TTFT.
    """
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    print(f"  Prompt length: {input_ids.shape[1]} tokens")

    # Warm-up: 10 tokens to let GPU clock stabilise before timing starts
    print("  Running warm-up (10 tokens) …")
    with torch.inference_mode():
        _, _, wkv = model.forward_cached(input_ids)
        dummy = torch.tensor([[1]], dtype=torch.long, device="cuda")
        for _ in range(9):
            model.forward_cached(dummy, past_key_values=wkv)
    torch.cuda.synchronize()
    print("  Warm-up done. Starting sustained timing …\n")

    tpot_ms   = []
    past_kv   = None
    generated = []

    with torch.inference_mode():
        for i in range(n_tokens):
            if i == 0:
                cur_input = input_ids
            else:
                cur_input = torch.tensor(
                    [[generated[-1]]], dtype=torch.long, device="cuda"
                )

            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                l16, full, past_kv = model.forward_cached(cur_input, past_key_values=past_kv)
            e_ev.record()
            torch.cuda.synchronize()

            tpot_ms.append(s_ev.elapsed_time(e_ev))

            probs = torch.softmax(l16[0, -1, :], dim=-1)
            conf, next_l16 = torch.max(probs, dim=-1)
            next_tok = (next_l16 if conf.item() >= 0.55
                        else torch.argmax(full[0, -1, :], dim=-1))
            generated.append(next_tok.item())

            if (i + 1) % 50 == 0:
                w = (i + 1) // WINDOW_SIZE
                window_ms = tpot_ms[-WINDOW_SIZE:]
                print(f"  Window {w:>2}/{N_WINDOWS}: "
                      f"tokens {i+1-WINDOW_SIZE+1}–{i+1}  "
                      f"mean={np.mean(window_ms):.2f}  "
                      f"P99={np.percentile(window_ms,99):.2f}  "
                      f"max={np.max(window_ms):.2f}  (ms)")

    return tpot_ms


def analyse_windows(tpot_ms):
    windows = []
    for w in range(N_WINDOWS):
        start = w * WINDOW_SIZE
        end   = start + WINDOW_SIZE
        seg   = tpot_ms[start:end]
        windows.append({
            "window":   w + 1,
            "start_tok": start + 1,
            "end_tok":   end,
            "mean_ms":  round(float(np.mean(seg)),          4),
            "p99_ms":   round(float(np.percentile(seg, 99)), 4),
            "max_ms":   round(float(np.max(seg)),            4),
            "std_ms":   round(float(np.std(seg)),            4),
        })
    return windows


def plot_thermal(tpot_ms, windows):
    fs.apply()

    fig, axes = plt.subplots(1, 2, figsize=fs.DOUBLE)

    # ── Panel 1: Raw TPOT time series + rolling mean ──────────────────────────
    ax1 = axes[0]
    tokens = np.arange(1, N_TOKENS + 1)
    arr    = np.array(tpot_ms)
    roll   = np.convolve(arr, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode="valid")

    ax1.plot(tokens, arr, alpha=0.35, linewidth=0.7, color="#4878d0", label="Per-token TPOT")
    ax1.plot(tokens[WINDOW_SIZE - 1:], roll, linewidth=2.0, color="#d62728",
             label=f"Rolling mean ({WINDOW_SIZE}-token window)")
    ax1.axhline(DEADLINE_MS, color="black", linestyle="--", linewidth=1.2,
                label=f"Deadline ({DEADLINE_MS:.0f} ms)")

    ax1.set_xlabel("Token index")
    ax1.set_ylabel("TPOT (ms)")
    ax1.set_title(f"Sustained Load: Per-Token Latency\n({N_TOKENS} consecutive tokens, KV-cached)")
    ax1.legend(loc="upper right", fontsize=8)

    # ── Panel 2: Window-level P99 / WCET — thermal drift ─────────────────────
    ax2 = axes[1]
    win_nums  = [w["window"]  for w in windows]
    win_means = [w["mean_ms"] for w in windows]
    win_p99   = [w["p99_ms"]  for w in windows]
    win_max   = [w["max_ms"]  for w in windows]

    ax2.plot(win_nums, win_means, marker="o", linewidth=2, color="#4878d0",  label="Window mean")
    ax2.plot(win_nums, win_p99,   marker="s", linewidth=2, color="#ff7f0e",  label="Window P99")
    ax2.plot(win_nums, win_max,   marker="^", linewidth=2, color="#d62728",  label="Window max (WCET)")
    ax2.axhline(DEADLINE_MS, color="black", linestyle="--", linewidth=1.2,
                label=f"Deadline ({DEADLINE_MS:.0f} ms)")

    # Shade region between first-window max and deadline — the safety margin
    ax2.fill_between(win_nums, win_max, DEADLINE_MS,
                     where=[m < DEADLINE_MS for m in win_max],
                     alpha=0.10, color="#2ca02c", label="Safety margin")

    # Drift annotation
    drift = win_max[-1] - win_max[0]
    ax2.annotate(f"Max drift: {drift:+.2f} ms",
                 xy=(win_nums[-1], win_max[-1]),
                 xytext=(win_nums[-1] - 4.0, win_max[-1] + 2.5),
                 fontsize=8, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728"))

    ax2.set_xticks(win_nums)
    ax2.set_xticklabels([f"W{w}\n({windows[i]['start_tok']}–{windows[i]['end_tok']})"
                         for i, w in enumerate(win_nums)], fontsize=6, rotation=30, ha='right')
    ax2.set_xlabel(f"Window (each = {WINDOW_SIZE} tokens)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Thermal Stability: WCET Drift Over Time\n(upward drift → thermal throttle warning)")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Sustained Load Profile — TinyLlama-1.1B  (RTX 4000 Ada, {N_TOKENS} tokens)",
                 fontsize=7.5, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


if __name__ == "__main__":
    print("=" * 60)
    print("THERMAL / SUSTAINED LOAD PROFILE — TinyLlama-1.1B")
    print(f"Tokens: {N_TOKENS}  |  Windows: {N_WINDOWS}  |  "
          f"Window size: {WINDOW_SIZE} tokens")
    print("=" * 60 + "\n")

    model  = EarlyExitTinyLlama()
    prompt = format_prompt(model.tokenizer, PROMPT_TEXT)

    tpot_ms = run_sustained_generation(model, prompt, N_TOKENS)
    windows  = analyse_windows(tpot_ms)

    global_arr = np.array(tpot_ms)
    baseline_max = windows[0]["max_ms"]
    final_max    = windows[-1]["max_ms"]
    drift_pct    = (final_max - baseline_max) / baseline_max * 100

    print(f"\n{'─'*50}")
    print(f"Global mean : {global_arr.mean():.3f} ms")
    print(f"Global P99  : {np.percentile(global_arr, 99):.3f} ms")
    print(f"Global max  : {global_arr.max():.3f} ms")
    print(f"Baseline max (W1): {baseline_max:.3f} ms")
    print(f"Final max    (W{N_WINDOWS}): {final_max:.3f} ms")
    print(f"Max drift   : {final_max - baseline_max:+.3f} ms  ({drift_pct:+.2f}%)")
    if abs(drift_pct) < 5:
        print("→ STABLE: No significant thermal drift detected (< 5%).")
    elif drift_pct > 0:
        print("→ WARNING: WCET increased — possible thermal throttling.")
    else:
        print("→ NOTE: WCET decreased — GPU clock-up after warm-up.")

    n_miss = sum(1 for t in tpot_ms if t > DEADLINE_MS)
    print(f"Deadline misses: {n_miss}/{N_TOKENS} ({100*n_miss/N_TOKENS:.2f}%)")

    output = {
        "n_tokens":       N_TOKENS,
        "n_windows":      N_WINDOWS,
        "window_size":    WINDOW_SIZE,
        "deadline_ms":    DEADLINE_MS,
        "global": {
            "mean_ms":    round(float(global_arr.mean()),              4),
            "p99_ms":     round(float(np.percentile(global_arr, 99)), 4),
            "max_ms":     round(float(global_arr.max()),               4),
            "std_ms":     round(float(global_arr.std()),               4),
            "miss_count": int(n_miss),
            "miss_rate":  round(n_miss / N_TOKENS, 6),
        },
        "drift": {
            "baseline_max_ms": round(baseline_max, 4),
            "final_max_ms":    round(final_max,    4),
            "drift_ms":        round(final_max - baseline_max, 4),
            "drift_pct":       round(drift_pct, 4),
        },
        "windows":  windows,
        "tpot_ms":  [round(t, 4) for t in tpot_ms],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to '{RESULTS_FILE}'")

    plot_thermal(tpot_ms, windows)
