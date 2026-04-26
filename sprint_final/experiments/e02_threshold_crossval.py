"""
E02: Threshold Cross-Validation — held-out τ evaluation on 300 PubMedQA queries.
Fixes reviewer concern that τ=0.55 was calibrated and evaluated on the same data.
Addresses: M3 (threshold cross-validation), DA-M3.

Method:
  1. Run KV-cached forward on all 300 queries, collect raw l16_conf per token
  2. Split: queries 1-150 = calibration; 151-300 = evaluation
  3. Apply τ ∈ {0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.9} post-hoc on each split
  4. Optimal τ selected by best calibration accuracy; report held-out accuracy + CI
"""
import argparse, os, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import result_writer as rw
from early_exit_model import EarlyExitTinyLlama
import benchmark_utils as bu
import evt_utils as eu

EXPERIMENT_ID = "E02"
RESULTS_FILE  = "threshold_crossval_results.json"
HARDWARE      = "A100 SXM4"
N_QUERIES     = 300
MAX_NEW_TOKENS= 15
DEADLINE_MS   = 45.0
THRESHOLDS    = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]


def main(dry_run=False):
    rw.configure(BASE_DIR)
    if dry_run:
        print(f"[{EXPERIMENT_ID}] dry-run OK"); sys.exit(0)
    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping."); sys.exit(0)

    t0 = rw.log_start(EXPERIMENT_ID)
    try:
        model   = EarlyExitTinyLlama()
        dataset = bu.load_pubmed_dataset(N_QUERIES)
        print(f"[{EXPERIMENT_ID}] Collecting raw token data for {len(dataset)} queries...")
        raw = bu.run_pubmed_queries_raw(model, model.tokenizer, dataset,
                                        max_new_tokens=MAX_NEW_TOKENS)

        cal_raw  = raw[:len(raw)//2]
        eval_raw = raw[len(raw)//2:]
        print(f"  Calibration: {len(cal_raw)} queries | Evaluation: {len(eval_raw)} queries")

        cal_results  = {}
        eval_results = {}

        for tau in THRESHOLDS:
            cal_m  = bu.apply_threshold_posthoc(cal_raw,  tau, model.tokenizer, DEADLINE_MS)
            eval_m = bu.apply_threshold_posthoc(eval_raw, tau, model.tokenizer, DEADLINE_MS)

            # Accuracy CI on eval set
            eval_correct = [
                q["ground_truth"] in bu.extract_label(
                    "".join(model.tokenizer.decode(
                        [td["l16_token_id"] if td["l16_conf"] >= tau else td["l22_token_id"]
                         for td in q["token_data"]], skip_special_tokens=True
                    ))
                )
                for q in eval_raw
            ]
            # Recompute correctly: collect correct flags for scored queries
            eval_scored_flags = []
            for q in eval_raw:
                ids = [(td["l16_token_id"] if td["l16_conf"] >= tau else td["l22_token_id"])
                       for td in q["token_data"]]
                text = model.tokenizer.decode(ids, skip_special_tokens=True)
                pred = bu.extract_label(text)
                if pred != "unknown":
                    eval_scored_flags.append(pred == q["ground_truth"])

            ci = eu.bootstrap_accuracy_ci(eval_scored_flags) if eval_scored_flags else {}
            eval_m["ci_lower"]  = ci.get("ci_lower")
            eval_m["ci_upper"]  = ci.get("ci_upper")
            eval_m["ci_width"]  = ci.get("ci_width")
            eval_m["n_scored"]  = len(eval_scored_flags)

            cal_results[str(tau)]  = cal_m
            eval_results[str(tau)] = eval_m

            print(f"  τ={tau:.2f}  cal_acc={cal_m['accuracy']}%  "
                  f"eval_acc={eval_m['accuracy']}%  "
                  f"CI=[{eval_m.get('ci_lower'):.1f},{eval_m.get('ci_upper'):.1f}]%")

        # Best τ by calibration accuracy
        best_tau = max(THRESHOLDS, key=lambda t: (cal_results[str(t)]["accuracy"] or 0))
        print(f"\n  Best τ (calibration) = {best_tau}")
        print(f"  Hold-out accuracy at best τ = {eval_results[str(best_tau)]['accuracy']}%")

        rw.write_json(RESULTS_FILE, {
            "hardware": HARDWARE, "n_queries_total": N_QUERIES,
            "n_cal": len(cal_raw), "n_eval": len(eval_raw),
            "deadline_ms": DEADLINE_MS, "thresholds": THRESHOLDS,
            "best_tau_by_cal": best_tau,
            "calibration": cal_results,
            "evaluation":  eval_results,
        })
        _plot(cal_results, eval_results, best_tau)
        _build_latex(cal_results, eval_results, best_tau)
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc); raise


def _build_latex(cal, evl, best_tau):
    lines = [r"% E02: Threshold Cross-Validation", r"\midrule"]
    for tau in THRESHOLDS:
        c = cal[str(tau)]; e = evl[str(tau)]
        bold_s = r"\textbf{" if tau == best_tau else ""
        bold_e = r"}" if tau == best_tau else ""
        ci_str = (f"[{e['ci_lower']:.1f},{e['ci_upper']:.1f}]"
                  if e.get("ci_lower") is not None else "--")
        lines.append(
            f"  {bold_s}{tau:.2f}{bold_e} & {c['accuracy']} & "
            f"{bold_s}{e['accuracy']}{bold_e} & {ci_str} & "
            f"{e['early_exit_pct']} & {e['deadline_miss_pct']} \\\\"
        )
    lines.append(r"\bottomrule")
    rw.write_latex("table_threshold_crossval.tex", "\n".join(lines) + "\n")


def _plot(cal, evl, best_tau):
    import matplotlib.pyplot as plt
    import fig_style as fs
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=fs.DOUBLE)
    taus = THRESHOLDS
    cal_accs  = [cal[str(t)]["accuracy"] or 0 for t in taus]
    eval_accs = [evl[str(t)]["accuracy"] or 0 for t in taus]
    ci_lo = [evl[str(t)].get("ci_lower") or 0 for t in taus]
    ci_hi = [evl[str(t)].get("ci_upper") or 0 for t in taus]

    ax = axes[0]
    ax.plot(taus, cal_accs,  marker="o", label="Calibration set",  color="#1f77b4")
    ax.plot(taus, eval_accs, marker="s", label="Held-out eval set", color="#ff7f0e")
    ax.fill_between(taus, ci_lo, ci_hi, alpha=0.2, color="#ff7f0e", label="95% CI (eval)")
    ax.axvline(best_tau, color="gray", ls="--", lw=0.9, label=f"Best τ={best_tau:.2f}")
    ax.set_xlabel("Confidence Threshold τ"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Threshold Cross-Validation"); ax.legend()

    ax2 = axes[1]
    exit_rates = [evl[str(t)]["early_exit_pct"] for t in taus]
    miss_rates = [evl[str(t)]["deadline_miss_pct"] for t in taus]
    ax2b = ax2.twinx()
    ax2.plot(taus, exit_rates, marker="o", color="#2ca02c", label="Exit rate %")
    ax2b.plot(taus, miss_rates, marker="^", color="#d62728", ls="--", label="Miss rate %")
    ax2.set_xlabel("Threshold τ"); ax2.set_ylabel("Early Exit Rate (%)", color="#2ca02c")
    ax2b.set_ylabel("Deadline Miss Rate (%)", color="#d62728")
    ax2.set_title("Exit Rate vs Miss Rate"); ax2.legend(loc="upper left"); ax2b.legend(loc="upper right")

    fig.suptitle(f"Threshold Cross-Validation — {HARDWARE}", fontsize=8)
    plt.tight_layout()
    path = rw.figures_path("threshold_crossval.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Figure  → {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
