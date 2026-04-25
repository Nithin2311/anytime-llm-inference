"""
E5: Extended deadline sweep using the KV-cached router (tau=0.7).

The original paper's deadline sweep used the stateless router. This experiment
reruns the sweep with the KV-cached router to show how quality (accuracy) varies
as deadline tightens, and confirms the KV router avoids forced exits at D>=35ms.

Deadlines swept: [20, 25, 30, 35, 40, 45, 50, 60, 75, 100] ms
Metrics: accuracy, forced_exit_pct, deadline_miss_pct, mean/P99 TPOT, throughput

Outputs:
  results/deadline_sweep_ext_results.json
  figures/deadline_sweep_ext.png
  latex/table_deadline_sweep_ext.tex
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

EXPERIMENT_ID  = "E5"
RESULTS_FILE   = "deadline_sweep_ext_results.json"
DEADLINES_MS   = [20, 25, 30, 35, 40, 45, 50, 60, 75, 100]
KV_THRESHOLD   = 0.7
N_SAMPLES      = 30
MAX_NEW_TOKENS = 5


def build_latex(results_by_dl):
    lines = [
        "% E5: Extended Deadline Sweep -- KV-Cached Router (tau=0.7), A100 SXM4, n=30",
        "% Companion to Table VI; shows KV router quality vs deadline",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"$D$ (ms) & Forced (\%) & Miss (\%) & Acc (\%) & "
        r"Mean TPOT (ms) & P99 TPOT (ms) & Tput (tok/s) \\",
        r"\midrule",
    ]
    for dl in DEADLINES_MS:
        r   = results_by_dl[str(dl)]
        acc = f"{r['accuracy']:.1f}" if r["accuracy"] is not None else "--"
        tpt = f"{r['throughput_tps']:.1f}" if r["throughput_tps"] is not None else "--"
        lines.append(
            f"  {dl} & {r['forced_exit_pct']:.1f} & {r['deadline_miss_pct']:.1f}"
            f" & {acc} & {r['global_mean_tpot_ms']:.2f}"
            f" & {r['global_p99_tpot_ms']:.2f} & {tpt} \\\\"
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
        print(f"\n{'='*65}\nE5: EXTENDED DEADLINE SWEEP -- KV-CACHED ROUTER\n"
              f"Deadlines: {DEADLINES_MS} ms | tau={KV_THRESHOLD} | n={N_SAMPLES}\n"
              f"Expected runtime: ~60 min\n{'='*65}\n")

        dataset   = load_dataset("pubmed_qa", "pqa_labeled", split=f"train[:{N_SAMPLES}]")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model     = EarlyExitTinyLlama()

        results_by_dl = {}
        for idx, dl in enumerate(DEADLINES_MS, 1):
            print(f"  [{idx:>2}/{len(DEADLINES_MS)}] D={dl} ms ...", end="  ", flush=True)
            _, gm = run_pubmed_queries(
                model, tokenizer, dataset,
                deadline_ms=float(dl),
                max_new_tokens=MAX_NEW_TOKENS,
                generate_fn=generate_anytime_with_kv,
                generate_kwargs={"threshold": KV_THRESHOLD},
            )
            results_by_dl[str(dl)] = gm
            acc = gm["accuracy"] if gm["accuracy"] is not None else "N/A"
            print(f"acc={acc}%  forced={gm['forced_exit_pct']}%  "
                  f"p99={gm['global_p99_tpot_ms']}ms")

        saved = rw.write_json(RESULTS_FILE, {
            "hardware":       "A100 SXM4",
            "router":         "kv_cached",
            "kv_threshold":   KV_THRESHOLD,
            "n_samples":      N_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS,
            "deadlines_ms":   DEADLINES_MS,
            "results":        results_by_dl,
        })
        print(f"\n  Results --> {saved}")
        _plot(results_by_dl)
        rw.write_latex("table_deadline_sweep_ext.tex", build_latex(results_by_dl))
        print(f"  LaTeX   --> latex/table_deadline_sweep_ext.tex")
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc)
        raise


def _plot(results_by_dl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as fs

    fs.apply()
    dls       = DEADLINES_MS
    accs      = [results_by_dl[str(d)]["accuracy"] or 0 for d in dls]
    forced    = [results_by_dl[str(d)]["forced_exit_pct"] for d in dls]
    mean_tpot = [results_by_dl[str(d)]["global_mean_tpot_ms"] for d in dls]
    p99_tpot  = [results_by_dl[str(d)]["global_p99_tpot_ms"]  for d in dls]
    tput      = [results_by_dl[str(d)]["throughput_tps"] or 0 for d in dls]

    fig, axes = plt.subplots(1, 3, figsize=fs.TRIPLE)

    ax  = axes[0]
    ax2 = ax.twinx()
    ax.plot(dls, accs, marker="o", linewidth=2, color="#2b5b84", label="Accuracy (%)")
    ax2.plot(dls, forced, marker="s", linewidth=2, color="#d62728",
             linestyle="--", label="Forced exit (%)")
    ax.set_xlabel("Deadline (ms)")
    ax.set_ylabel("PubMedQA Accuracy (%)", color="#2b5b84")
    ax2.set_ylabel("Forced-Exit Rate (%)", color="#d62728")
    ax.set_title("Accuracy & Forced-Exit Rate")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)

    axes[1].plot(dls, mean_tpot, marker="o", linewidth=2, color="#2b5b84", label="Mean TPOT")
    axes[1].plot(dls, p99_tpot,  marker="^", linewidth=2, color="#ff7f0e",
                 linestyle="--", label="P99 TPOT")
    axes[1].axhline(y=45.0, color="black", linestyle=":", linewidth=1.1, label="D=45 ms")
    axes[1].set_xlabel("Deadline (ms)")
    axes[1].set_ylabel("TPOT (ms)")
    axes[1].set_title("Token Latency vs Deadline")
    axes[1].legend(fontsize=8)

    axes[2].plot(dls, tput, marker="o", linewidth=2, color="#2ca02c")
    axes[2].set_xlabel("Deadline (ms)")
    axes[2].set_ylabel("Throughput (tok/s)")
    axes[2].set_title("Throughput vs Deadline")

    fig.suptitle(r"Deadline Sweep -- KV-Cached Router ($\tau$=0.7), A100 SXM4",
                 fontsize=8, y=1.01)
    plt.tight_layout()
    path = rw.figures_path("deadline_sweep_ext.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure  --> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
