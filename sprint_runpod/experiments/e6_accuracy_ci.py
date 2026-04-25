"""
E6: Bootstrap 95% CI on PubMedQA accuracy for all three routers at D=45ms.

Academic reviewers flagged that the paper reports point accuracy without
confidence intervals. This experiment runs 30 queries per router, then
bootstraps 1000 samples to compute 95% CI on accuracy.

Routers tested:
  - stateless (generate_stateless_anytime)
  - kv_cached (generate_anytime_with_kv, tau=0.7)

Outputs:
  results/accuracy_ci_results.json
  figures/accuracy_ci.png
  latex/table_accuracy_ci.tex
"""

import argparse
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import result_writer as rw
from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import (
    generate_stateless_anytime,
    generate_anytime_with_kv,
    reload_wcet_table,
)
from benchmark_utils import run_pubmed_queries

EXPERIMENT_ID  = "E6"
RESULTS_FILE   = "accuracy_ci_results.json"
DEADLINE_MS    = 45.0
N_SAMPLES      = 30
MAX_NEW_TOKENS = 5
N_BOOTSTRAP    = 1000
ALPHA          = 0.05
KV_THRESHOLD   = 0.7

ROUTERS = [
    ("stateless", generate_stateless_anytime, {}),
    ("kv_cached", generate_anytime_with_kv,   {"threshold": KV_THRESHOLD}),
]


def bootstrap_accuracy_ci(correct_flags, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    arr  = np.array(correct_flags, dtype=float)
    rng  = np.random.default_rng(42)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = boot.mean(axis=1) * 100.0
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return round(lo, 2), round(hi, 2)


def build_latex(results_by_router):
    lines = [
        "% E6: Accuracy with 95% Bootstrap CI -- A100 SXM4, D=45ms, n=30, B=1000",
        "% Addresses reviewer request for confidence intervals on accuracy",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Router & Acc (\%) & CI$_{95}$ lower & CI$_{95}$ upper & Width \\",
        r"\midrule",
    ]
    for name, r in results_by_router.items():
        acc   = r["accuracy"] if r["accuracy"] is not None else 0.0
        width = round(r["ci_upper"] - r["ci_lower"], 2)
        lines.append(
            f"  {name} & {acc:.1f} & {r['ci_lower']:.1f}"
            f" & {r['ci_upper']:.1f} & {width:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(dry_run=False):
    rw.configure(BASE_DIR)

    if dry_run:
        import datasets, transformers  # noqa: F401
        print(f"[{EXPERIMENT_ID}] dry-run OK | CUDA={torch.cuda.is_available()} | imports OK")
        sys.exit(0)

    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping -- {RESULTS_FILE} exists.")
        sys.exit(0)

    reload_wcet_table()
    t0 = rw.log_start(EXPERIMENT_ID)
    try:
        print(f"\n{'='*65}\nE6: ACCURACY BOOTSTRAP CI\n"
              f"D={DEADLINE_MS}ms | n={N_SAMPLES} | B={N_BOOTSTRAP} bootstraps\n"
              f"Routers: {[r[0] for r in ROUTERS]}\n"
              f"Expected runtime: ~30 min\n{'='*65}\n")

        dataset   = load_dataset("pubmed_qa", "pqa_labeled", split=f"train[:{N_SAMPLES}]")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model     = EarlyExitTinyLlama()

        results_by_router = {}
        for name, fn, kwargs in ROUTERS:
            print(f"\n  Router: {name}")
            qr, gm = run_pubmed_queries(
                model, tokenizer, dataset,
                deadline_ms=DEADLINE_MS,
                max_new_tokens=MAX_NEW_TOKENS,
                generate_fn=fn,
                generate_kwargs=kwargs,
            )
            correct_flags = [q["correct"] for q in qr if q["predicted"] != "unknown"]
            ci_lo, ci_hi  = bootstrap_accuracy_ci(correct_flags)
            results_by_router[name] = {
                **gm,
                "ci_lower":    ci_lo,
                "ci_upper":    ci_hi,
                "n_bootstrap": N_BOOTSTRAP,
            }
            print(f"  acc={gm['accuracy']}%  95% CI=[{ci_lo:.1f}, {ci_hi:.1f}]%")

        saved = rw.write_json(RESULTS_FILE, {
            "hardware":       "A100 SXM4",
            "deadline_ms":    DEADLINE_MS,
            "n_samples":      N_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS,
            "n_bootstrap":    N_BOOTSTRAP,
            "alpha":          ALPHA,
            "results":        results_by_router,
        })
        print(f"\n  Results --> {saved}")
        _plot(results_by_router)
        rw.write_latex("table_accuracy_ci.tex", build_latex(results_by_router))
        print(f"  LaTeX   --> latex/table_accuracy_ci.tex")
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc)
        raise


def _plot(results_by_router):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as fs

    fs.apply()
    names = list(results_by_router.keys())
    accs  = [results_by_router[n]["accuracy"] or 0 for n in names]
    ci_lo = [results_by_router[n]["ci_lower"] for n in names]
    ci_hi = [results_by_router[n]["ci_upper"] for n in names]
    yerr  = [
        [a - lo for a, lo in zip(accs, ci_lo)],
        [hi - a for a, hi in zip(accs, ci_hi)],
    ]

    fig, ax = plt.subplots(figsize=fs.SINGLE)
    x = np.arange(len(names))
    colors = ["#2b5b84", "#ff7f0e", "#2ca02c"]
    ax.bar(x, accs, yerr=yerr, capsize=6,
           color=colors[:len(names)], alpha=0.8, width=0.5,
           error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("PubMedQA Accuracy (%)")
    ax.set_title(f"Accuracy 95% Bootstrap CI (D={DEADLINE_MS:.0f} ms)")
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, label="50% baseline")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = rw.figures_path("accuracy_ci.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure  --> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
