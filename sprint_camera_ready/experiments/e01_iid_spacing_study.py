"""
e01_iid_spacing_study.py — IID validation at multiple inter-run spacings.

Tests whether increasing inter-run spacing yields IID latency samples.
Spacings: 0ms, 200ms, 1000ms, 5000ms.  N=1000 per spacing level.
Cell: seq_len=128, Full pass (representative of the main result cell).
"""

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from fig_style import apply_style, DOUBLE
from evt_utils import ljung_box_test, sample_acf
from result_writer import write_results

apply_style()
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLES  = 1000
N_WARMUP   = 50
SEQ_LEN    = 128
EXIT_DEPTH = 22   # Full pass
DEVICE     = "cuda"
SPACINGS   = [0, 200, 1000, 5000]   # milliseconds


def build_input(tokenizer, seq_len):
    PROMPT = "The pharmacokinetics of drug X suggest that the optimal dosing"
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
    while len(ids) < seq_len:
        ids = ids.repeat(2)
    return ids[:seq_len].unsqueeze(0).to(DEVICE)


def collect_samples(model, tokenizer, n_samples, n_warmup, spacing_ms):
    ids = build_input(tokenizer, SEQ_LEN)
    with torch.inference_mode():
        _, _, pkv = model.forward_cached(ids)
        new_tok = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        for _ in range(n_warmup):
            model.forward_cached(new_tok, past_key_values=pkv)
        torch.cuda.synchronize()
        times = []
        for i in range(n_samples):
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            model.forward_cached(new_tok, past_key_values=pkv)
            ev_e.record()
            torch.cuda.synchronize()
            times.append(ev_s.elapsed_time(ev_e))
            if spacing_ms > 0 and i < n_samples - 1:
                time.sleep(spacing_ms / 1000.0)
    return np.array(times)


def main():
    print("=" * 60)
    print(f"E01  IID Spacing Study")
    print(f"     N={N_SAMPLES}  cell=seq{SEQ_LEN}_full  spacings={SPACINGS}ms")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    rows = []
    all_samples = {}

    for spacing_ms in SPACINGS:
        label = f"{spacing_ms}ms"
        print(f"\n  Spacing={label} ...", flush=True)
        t0 = time.time()
        s  = collect_samples(model, model.tokenizer, N_SAMPLES, N_WARMUP, spacing_ms)
        elapsed = time.time() - t0

        lb  = ljung_box_test(s, max_lag=20)
        acf = sample_acf(s, n_lags=5)
        lag1_acf = acf[0][1] if acf else None

        row = {
            "spacing_ms":          spacing_ms,
            "n_samples":           len(s),
            "mean_ms":             round(float(np.mean(s)), 4),
            "std_ms":              round(float(np.std(s)), 4),
            "p99_ms":              round(float(np.percentile(s, 99)), 4),
            "ljung_box":           lb,
            "lag1_acf":            round(float(lag1_acf), 4) if lag1_acf is not None else None,
            "iid_pass":            lb["independent_at_5pct"],
            "elapsed_s":           round(elapsed, 1),
        }
        rows.append(row)
        all_samples[label] = [round(float(x), 4) for x in s]

        status = "PASS" if lb["independent_at_5pct"] else "FAIL"
        print(f"  LB p={lb['lb_pvalue']}  lag1_ACF={lag1_acf:.3f}  IID={status}  ({elapsed:.0f}s)")

    # Figure: ACF plots for each spacing
    fig, axes = plt.subplots(2, 2, figsize=DOUBLE)
    axes = axes.flatten()
    for i, (spacing_ms, row) in enumerate(zip(SPACINGS, rows)):
        ax = axes[i]
        s  = np.array(all_samples[f"{spacing_ms}ms"])
        acf_vals = sample_acf(s, n_lags=20)
        lags = [a[0] for a in acf_vals]
        vals = [a[1] for a in acf_vals]
        ax.bar(lags, vals, color="tab:blue", alpha=0.7, width=0.6)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(1.96 / np.sqrt(len(s)),  color="red", ls="--", lw=0.8, label="95% CI")
        ax.axhline(-1.96 / np.sqrt(len(s)), color="red", ls="--", lw=0.8)
        lb_p = row["ljung_box"]["lb_pvalue"]
        ax.set_title(f"spacing={spacing_ms}ms  LB p={lb_p:.4f}", fontsize=7)
        ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
        ax.legend(fontsize=6)
    plt.suptitle(f"E01 — IID ACF Study (N={N_SAMPLES}, seq{SEQ_LEN}_full)", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e01_iid_spacing.png", dpi=150)
    plt.close()

    write_results({
        "experiment": "e01_iid_spacing_study",
        "cell": f"seq{SEQ_LEN}_full",
        "n_samples": N_SAMPLES,
        "spacings": rows,
        "samples": all_samples,
    }, RESULTS_DIR / "e01_iid_spacing.json")

    print("\n  Summary:")
    for r in rows:
        status = "IID PASS" if r["iid_pass"] else "IID FAIL"
        print(f"    {r['spacing_ms']:5d}ms  LB_p={r['ljung_box']['lb_pvalue']:.4f}  "
              f"lag1={r['lag1_acf']:.3f}  {status}")
    print("\nPASS: E01 complete\n")


if __name__ == "__main__":
    main()
