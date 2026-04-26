"""
e13_dense_ablation.py — Dense layer-wise ablation L12-L20.

Addresses R1-M3: only L5/L11/L16/Full tested; gaps in quality-latency
trade-off curve. Runs all integer exit layers from 12 to 20 (plus 5,
11, 22 for anchors) at a fixed seq=128 with 100 queries each.
Reports: WCET P99, mean TPOT, accuracy, exit rate.
"""

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sprint_v2_runpod" / "src"))

import numpy as np
import torch
from benchmark_utils import load_pubmed_dataset, run_pubmed_queries_raw, apply_threshold_posthoc
from evt_utils import bootstrap_accuracy_ci
from fig_style import apply_style, DOUBLE
import matplotlib.pyplot as plt
from result_writer import write_results

apply_style()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
SEQ_LEN       = 128
N_QUERIES     = 100
N_TIMING      = 500     # timing samples per exit layer
TAU           = 0.55
DEADLINE_MS   = 45.0
N_BOOTSTRAP   = 1000
DEVICE        = "cuda"

EXIT_LAYERS   = [5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, None]  # None=22


def profile_layer(model, exit_layer, n_samples, seq_len, n_warmup=10):
    """Time exit_layer for n_samples passes at seq_len."""
    import warnings
    dummy = torch.randint(100, 2000, (1, seq_len), device=DEVICE)
    for _ in range(n_warmup):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.inference_mode():
                model.forward_cached(dummy)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_samples):
        s.record()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.inference_mode():
                if exit_layer is None:
                    model.forward_cached(dummy)
                else:
                    model.forward_cached(dummy, exit_layer=exit_layer)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.array(times)


def main():
    print("=" * 60)
    print("E13  Dense layer ablation L5-L22 (every integer L12-L20)")
    print(f"     seq={SEQ_LEN}, N_timing={N_TIMING}, N_queries={N_QUERIES}")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    print(f"\n[1/3] Loading {N_QUERIES} PubMedQA queries ...")
    queries = load_pubmed_dataset(n_samples=N_QUERIES)

    rows = []
    total = len(EXIT_LAYERS)

    print(f"\n[2/3] Profiling {total} exit layers ...")
    for i, L in enumerate(EXIT_LAYERS):
        label = "Full(22)" if L is None else f"L{L}"
        print(f"\n  [{i+1}/{total}] {label}")

        # Latency profiling
        times = profile_layer(model, L, N_TIMING, SEQ_LEN)
        p99_ms  = float(np.percentile(times, 99))
        mean_ms = float(np.mean(times))

        # Accuracy evaluation via post-hoc replay at this exit layer
        # We collect raw tokens with forced exit at this layer
        raw_tokens = run_pubmed_queries_raw(
            model, queries, deadline_ms=DEADLINE_MS, device=DEVICE,
            max_new_tokens=10, show_progress=False,
            forced_exit_layer=L  # collect at this specific layer
        )
        r = apply_threshold_posthoc(
            raw_tokens, queries, tau=TAU, deadline_ms=DEADLINE_MS,
            n_bootstrap=N_BOOTSTRAP
        )
        ci = bootstrap_accuracy_ci(r.get("correct_flags", []), n_bootstrap=N_BOOTSTRAP)

        row = {
            "exit_layer":    L,
            "label":         label,
            "n_layers":      L if L is not None else 22,
            "mean_ms":       round(mean_ms, 4),
            "p99_ms":        round(p99_ms, 4),
            "accuracy_pct":  ci["accuracy"],
            "ci_lower":      ci["ci_lower"],
            "ci_upper":      ci["ci_upper"],
            "exit_rate_pct": r.get("exit_rate_pct", 0),
            "miss_rate_pct": r.get("miss_rate_pct", 0),
            "n_scored":      r.get("n_scored", 0),
        }
        rows.append(row)
        print(f"    P99={p99_ms:.2f}ms  acc={ci['accuracy']:.1f}%  "
              f"miss={row['miss_rate_pct']:.1f}%")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)

    n_layers_x = [r["n_layers"] for r in rows]
    p99s        = [r["p99_ms"]       for r in rows]
    accs        = [r["accuracy_pct"] for r in rows]
    ci_lo       = [r["ci_lower"]     for r in rows]
    ci_hi       = [r["ci_upper"]     for r in rows]

    ax = axes[0]
    ax.plot(n_layers_x, p99s, "o-", color="tab:blue")
    ax.set_xlabel("Number of Transformer layers")
    ax.set_ylabel("P99 latency (ms)")
    ax.set_title("P99 Latency vs. Exit Layer")
    ax.axhline(DEADLINE_MS, ls="--", color="red", lw=1, label=f"D={DEADLINE_MS}ms")
    ax.legend(fontsize=7)
    ax.set_xticks(n_layers_x)

    ax2 = axes[1]
    ax2.fill_between(n_layers_x, ci_lo, ci_hi, alpha=0.25, color="tab:orange")
    ax2.plot(n_layers_x, accs, "s-", color="tab:orange")
    ax2.set_xlabel("Number of Transformer layers")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy vs. Exit Layer")
    ax2.set_xticks(n_layers_x)

    plt.suptitle(f"Dense Layer Ablation (seq={SEQ_LEN}, τ={TAU})")
    plt.tight_layout()
    fig_path = RESULTS_DIR / "dense_ablation.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # Pareto-optimal layers: max accuracy for each p99 threshold
    def pareto_frontier(rows):
        sorted_rows = sorted(rows, key=lambda r: r["p99_ms"])
        pareto = []
        best_acc = -1
        for r in sorted_rows:
            if r["accuracy_pct"] > best_acc:
                best_acc = r["accuracy_pct"]
                pareto.append(r["label"])
        return pareto

    pareto = pareto_frontier(rows)

    # ── LaTeX ─────────────────────────────────────────────────────────────
    tex_path = RESULTS_DIR / "table_dense_ablation.tex"
    with open(tex_path, "w") as f:
        f.write("% E13: Dense layer ablation\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Layer & $n_{layers}$ & P99 (ms) & Accuracy (\\%) & 95\\% CI & Exit rate (\\%) & Miss (\\%) \\\\\n\\midrule\n")
        for r in rows:
            star = " *" if r["label"] in pareto else ""
            f.write(f"{r['label']}{star} & {r['n_layers']} & "
                    f"{r['p99_ms']:.2f} & {r['accuracy_pct']:.1f} & "
                    f"[{r['ci_lower']:.1f},{r['ci_upper']:.1f}] & "
                    f"{r['exit_rate_pct']:.1f} & {r['miss_rate_pct']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("% * = Pareto-optimal (max accuracy for given latency budget)\n")
    print(f"LaTeX: {tex_path}")

    write_results({
        "experiment": "e13_dense_ablation",
        "seq_len":    SEQ_LEN,
        "n_queries":  N_QUERIES,
        "n_timing":   N_TIMING,
        "tau":        TAU,
        "deadline_ms": DEADLINE_MS,
        "pareto_optimal_layers": pareto,
        "rows": rows,
    }, RESULTS_DIR / "dense_ablation_results.json")
    print("PASS: E13 complete\n")


if __name__ == "__main__":
    main()
