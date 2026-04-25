"""
E3: Forced-exit quality validation at tight deadlines (D=20ms, D=25ms).

At these deadlines the stateless router has 100% forced-exit rate (Table VI).
Measures both quality metrics required by academic reviewers:
  1. PubMedQA accuracy  (yes/no/maybe vs ground truth)
  2. ROUGE-L vs full-model reference  (textual fidelity at D=1000ms)

Outputs:
  results/forced_exit_quality_results.json
  figures/forced_exit_quality.png
  latex/table_forced_exit_quality.tex   (new table extending Table VI)
"""

import argparse
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer as rouge_lib
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import result_writer as rw
from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import generate_stateless_anytime, reload_wcet_table
from benchmark_utils import build_prompt, extract_label, compute_query_metrics

EXPERIMENT_ID   = "E3"
RESULTS_FILE    = "forced_exit_quality_results.json"
TIGHT_DEADLINES = [20, 25]
REF_DEADLINE_MS = 1000.0
N_SAMPLES       = 30
MAX_NEW_TOKENS  = 5


def _generate_outputs(model, dataset, tokenizer, deadline_ms):
    outputs = []
    for item in dataset:
        context = item["context"]["contexts"][0]
        question = item["question"]
        gt       = item["final_decision"]
        prompt   = build_prompt(tokenizer, context, question)
        records  = generate_stateless_anytime(
            model, prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            deadline_ms=deadline_ms,
            verbose=False,
        )
        outputs.append(("".join(r["token"] for r in records), records, gt))
    return outputs


def _rouge_l(hypotheses, references):
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure
              for hyp, ref in zip(hypotheses, references)]
    return float(np.mean(scores)), float(np.std(scores))


def build_latex(results_by_deadline):
    lines = [
        "% E3: Forced-Exit Quality -- A100 SXM4, n=30, stateless router",
        "% New table extending Table VI in report_v2.tex",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"$D$ (ms) & Forced (\%) & Miss (\%) & Acc (\%) & "
        r"ROUGE-L$_{\text{GT}}$ & ROUGE-L$_{\text{full}}$ \\",
        r"\midrule",
    ]
    for dl in TIGHT_DEADLINES:
        r = results_by_deadline[str(dl)]
        acc = f"{r['accuracy']:.1f}" if r["accuracy"] is not None else "--"
        lines.append(
            f"  {dl} & {r['forced_exit_pct']:.1f} & {r['deadline_miss_pct']:.1f}"
            f" & {acc}"
            f" & {r['rougeL_vs_gt_mean']:.3f}$\\pm${r['rougeL_vs_gt_std']:.3f}"
            f" & {r['rougeL_vs_full_mean']:.3f}$\\pm${r['rougeL_vs_full_std']:.3f} \\\\"
        )
    lines.append(r"  $\infty$ & 0.0 & 0.0 & \multicolumn{3}{c}{full-model reference} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(dry_run=False):
    rw.configure(BASE_DIR)

    if dry_run:
        import rouge_score, datasets, transformers  # noqa: F401
        print(f"[{EXPERIMENT_ID}] dry-run OK | CUDA={torch.cuda.is_available()} | imports OK")
        sys.exit(0)

    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping -- {RESULTS_FILE} exists.")
        sys.exit(0)

    reload_wcet_table()
    t0 = rw.log_start(EXPERIMENT_ID)
    try:
        print(f"\n{'='*65}\nE3: FORCED-EXIT QUALITY VALIDATION\n"
              f"Tight deadlines: {TIGHT_DEADLINES} ms | Reference: D={REF_DEADLINE_MS:.0f} ms\n"
              f"Metrics: PubMedQA accuracy + ROUGE-L vs full model\n"
              f"Expected runtime: ~40 min\n{'='*65}\n")

        dataset   = load_dataset("pubmed_qa", "pqa_labeled", split=f"train[:{N_SAMPLES}]")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model     = EarlyExitTinyLlama()

        print(f"\n[1/{len(TIGHT_DEADLINES)+1}] Generating reference outputs"
              f" at D={REF_DEADLINE_MS:.0f} ms ...")
        ref_outputs = _generate_outputs(model, dataset, tokenizer, REF_DEADLINE_MS)
        ref_texts   = [t for t, _, _ in ref_outputs]
        ref_gts     = [gt for _, _, gt in ref_outputs]
        print(f"  Done: {len(ref_texts)} reference outputs")

        results_by_deadline = {}
        for idx, dl in enumerate(TIGHT_DEADLINES, 2):
            print(f"\n[{idx}/{len(TIGHT_DEADLINES)+1}] Generating outputs at D={dl} ms ...")
            hyp_outputs = _generate_outputs(model, dataset, tokenizer, float(dl))
            hyp_texts   = [t for t, _, _ in hyp_outputs]
            hyp_records = [r for _, r, _ in hyp_outputs]

            n_correct, n_scored = 0, 0
            for text, gt in zip(hyp_texts, ref_gts):
                pred = extract_label(text)
                if pred != "unknown":
                    n_scored += 1
                    if pred == gt:
                        n_correct += 1

            rL_gt_mean,   rL_gt_std   = _rouge_l(hyp_texts, ref_gts)
            rL_full_mean, rL_full_std = _rouge_l(hyp_texts, ref_texts)

            all_records = [r for records in hyp_records for r in records]
            sched = compute_query_metrics(all_records, float(dl))

            results_by_deadline[str(dl)] = {
                "deadline_ms":         dl,
                "n_queries":           N_SAMPLES,
                "n_correct":           n_correct,
                "n_scored":            n_scored,
                "accuracy":            round(100.0 * n_correct / n_scored, 1) if n_scored > 0 else None,
                "forced_exit_pct":     sched["forced_exit_pct"],
                "deadline_miss_pct":   sched["deadline_miss_pct"],
                "mean_tpot_ms":        sched["mean_tpot_ms"],
                "p99_tpot_ms":         sched["p99_tpot_ms"],
                "rougeL_vs_gt_mean":   round(rL_gt_mean,   4),
                "rougeL_vs_gt_std":    round(rL_gt_std,    4),
                "rougeL_vs_full_mean": round(rL_full_mean, 4),
                "rougeL_vs_full_std":  round(rL_full_std,  4),
            }
            r = results_by_deadline[str(dl)]
            print(f"  D={dl}ms  forced={r['forced_exit_pct']}%  acc={r['accuracy']}%  "
                  f"ROUGE-L(full)={r['rougeL_vs_full_mean']:.3f}")

        saved = rw.write_json(RESULTS_FILE, {
            "hardware":        "A100 SXM4",
            "tight_deadlines": TIGHT_DEADLINES,
            "ref_deadline_ms": REF_DEADLINE_MS,
            "n_samples":       N_SAMPLES,
            "results":         results_by_deadline,
        })
        print(f"\n  Results --> {saved}")
        _plot(results_by_deadline)
        rw.write_latex("table_forced_exit_quality.tex", build_latex(results_by_deadline))
        print(f"  LaTeX   --> latex/table_forced_exit_quality.tex")
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc)
        raise


def _plot(results_by_deadline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as fs

    fs.apply()
    dls         = TIGHT_DEADLINES
    accuracies  = [results_by_deadline[str(d)]["accuracy"] or 0 for d in dls]
    rL_full     = [results_by_deadline[str(d)]["rougeL_vs_full_mean"] for d in dls]
    rL_full_err = [results_by_deadline[str(d)]["rougeL_vs_full_std"]  for d in dls]
    forced_pct  = [results_by_deadline[str(d)]["forced_exit_pct"] for d in dls]

    fig, axes = plt.subplots(1, 2, figsize=fs.DOUBLE)

    ax  = axes[0]
    ax2 = ax.twinx()
    ax.bar([str(d) for d in dls], accuracies, color="#2b5b84", alpha=0.75,
           width=0.35, label="Accuracy (%)")
    ax2.plot([str(d) for d in dls], forced_pct, marker="o", color="#d62728",
             linewidth=2, label="Forced exit (%)")
    ax.set_xlabel("Deadline (ms)")
    ax.set_ylabel("PubMedQA Accuracy (%)", color="#2b5b84")
    ax2.set_ylabel("Forced-Exit Rate (%)", color="#d62728")
    ax.set_title("Accuracy & Forced-Exit Rate")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)

    axes[1].bar([str(d) for d in dls], rL_full, yerr=rL_full_err,
                color="#ff7f0e", alpha=0.8, width=0.35, capsize=5,
                label="ROUGE-L vs full model")
    axes[1].axhline(y=1.0, color="black", linestyle="--", linewidth=1.0,
                    label="Perfect match")
    axes[1].set_xlabel("Deadline (ms)")
    axes[1].set_ylabel("ROUGE-L F1")
    axes[1].set_title("Output Fidelity vs Full Model")
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(fontsize=8)

    fig.suptitle("Forced-Exit Quality -- Stateless Router, A100 SXM4",
                 fontsize=8, y=1.01)
    plt.tight_layout()
    path = rw.figures_path("forced_exit_quality.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure  --> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
