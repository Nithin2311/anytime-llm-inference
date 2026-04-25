"""
E2: Confidence threshold ablation — justify the choice of tau=0.7.

Sweeps tau in {0.5, 0.6, 0.7, 0.8, 0.9} for the KV-cached single-pass router
at D=45ms across all 30 PubMedQA queries. Reports accuracy, routing
distribution, and latency for each tau to motivate the paper's threshold choice.

Outputs:
  results/threshold_ablation_results.json
  figures/threshold_ablation.png
  latex/table_threshold_ablation.tex   (new table for report_v2.tex)
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
from dynamic_scheduler import generate_anytime_with_kv, reload_wcet_table
from benchmark_utils import run_pubmed_queries

EXPERIMENT_ID  = "E2"
RESULTS_FILE   = "threshold_ablation_results.json"
THRESHOLDS     = [0.5, 0.6, 0.7, 0.8, 0.9]
DEADLINE_MS    = 45.0
N_SAMPLES      = 30
MAX_NEW_TOKENS = 5


def build_latex(results_by_tau):
    lines = [
        "% E2: Confidence Threshold Ablation -- A100 SXM4, D=45ms, n=30",
        "% New table for report_v2.tex: threshold sensitivity of KV-cached router",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"$\tau$ & Acc (\%) & Full (\%) & Early-T (\%) & Mean TPOT (ms) & P99 TPOT (ms) & Tput (tok/s) \\",
        r"\midrule",
    ]
    for tau in THRESHOLDS:
        r = results_by_tau[str(tau)]
        acc_str  = f"{r['accuracy']:.1f}" if r["accuracy"] is not None else "--"
        tput_str = f"{r['throughput_tps']:.2f}" if r["throughput_tps"] is not None else "--"
        marker   = r"~$\leftarrow$ selected" if abs(tau - 0.7) < 1e-9 else ""
        lines.append(
            f"  {tau:.1f} & {acc_str} & {r['full_pass_pct']:.1f}"
            f" & {r['early_thresh_pct']:.1f} & {r['global_mean_tpot_ms']:.2f}"
            f" & {r['global_p99_tpot_ms']:.2f} & {tput_str}{marker} \\\\"
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
        print(f"\n{'='*65}\nE2: THRESHOLD ABLATION -- tau in {THRESHOLDS}\n"
              f"D={DEADLINE_MS}ms | n={N_SAMPLES} queries | KV-cached router\n"
              f"Expected runtime: ~45 min\n{'='*65}\n")

        dataset   = load_dataset("pubmed_qa", "pqa_labeled", split=f"train[:{N_SAMPLES}]")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model     = EarlyExitTinyLlama()

        results_by_tau = {}
        for tau in THRESHOLDS:
            print(f"\n{'─'*50}\n>>> tau = {tau:.1f}\n{'─'*50}")
            _, gm = run_pubmed_queries(
                model, tokenizer, dataset,
                deadline_ms=DEADLINE_MS,
                max_new_tokens=MAX_NEW_TOKENS,
                generate_fn=generate_anytime_with_kv,
                generate_kwargs={"threshold": tau},
            )
            results_by_tau[str(tau)] = gm
            print(f"  tau={tau:.1f}  acc={gm['accuracy']}%  "
                  f"full={gm['full_pass_pct']}%  "
                  f"mean_tpot={gm['global_mean_tpot_ms']}ms  "
                  f"p99={gm['global_p99_tpot_ms']}ms")

        saved = rw.write_json(RESULTS_FILE, {
            "hardware":       "A100 SXM4",
            "deadline_ms":    DEADLINE_MS,
            "n_samples":      N_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS,
            "thresholds":     THRESHOLDS,
            "results":        results_by_tau,
        })
        print(f"\n  Results --> {saved}")

        _plot(results_by_tau)
        rw.write_latex("table_threshold_ablation.tex", build_latex(results_by_tau))
        print(f"  LaTeX   --> latex/table_threshold_ablation.tex")
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc)
        raise


def _plot(results_by_tau):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as fs

    fs.apply()
    taus       = THRESHOLDS
    accuracies = [results_by_tau[str(t)]["accuracy"] or 0 for t in taus]
    full_pcts  = [results_by_tau[str(t)]["full_pass_pct"] for t in taus]
    mean_tpots = [results_by_tau[str(t)]["global_mean_tpot_ms"] for t in taus]
    p99_tpots  = [results_by_tau[str(t)]["global_p99_tpot_ms"]  for t in taus]

    fig, axes = plt.subplots(1, 3, figsize=fs.TRIPLE)

    axes[0].plot(taus, accuracies, marker="o", linewidth=2, color="#2b5b84")
    axes[0].axvline(x=0.7, color="red", linestyle="--", linewidth=1.2, label=r"$\tau$=0.7")
    axes[0].set_xlabel(r"Confidence Threshold ($\tau$)")
    axes[0].set_ylabel("PubMedQA Accuracy (%)")
    axes[0].set_title(r"Accuracy vs. $\tau$")
    axes[0].legend(fontsize=8)

    axes[1].plot(taus, full_pcts, marker="s", linewidth=2, color="#d62728", label="Full Pass %")
    axes[1].axvline(x=0.7, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel(r"Confidence Threshold ($\tau$)")
    axes[1].set_ylabel("Full-Pass Token Fraction (%)")
    axes[1].set_title(r"Routing Distribution vs. $\tau$")
    axes[1].legend(fontsize=8)

    axes[2].plot(taus, mean_tpots, marker="o", linewidth=2, color="#2b5b84", label="Mean TPOT")
    axes[2].plot(taus, p99_tpots,  marker="^", linewidth=2, color="#ff7f0e",
                 linestyle="--", label="P99 TPOT")
    axes[2].axhline(y=45.0, color="black", linestyle=":", linewidth=1.1, label="D=45 ms")
    axes[2].axvline(x=0.7, color="red", linestyle="--", linewidth=1.2)
    axes[2].set_xlabel(r"Confidence Threshold ($\tau$)")
    axes[2].set_ylabel("TPOT (ms)")
    axes[2].set_title(r"Latency vs. $\tau$")
    axes[2].legend(fontsize=8)

    fig.suptitle(r"Threshold Ablation -- KV-Cached Router, D=45 ms, A100 SXM4",
                 fontsize=8, y=1.01)
    plt.tight_layout()
    path = rw.figures_path("threshold_ablation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure  --> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
