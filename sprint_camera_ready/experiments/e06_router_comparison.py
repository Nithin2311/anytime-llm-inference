"""e06_router_comparison.py — Three-router comparison at equal latency (500 queries each)."""
import sys, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import matplotlib.pyplot as plt

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
from fig_style import apply_style, DOUBLE
from benchmark_utils import load_pubmed_dataset, run_pubmed_queries_raw, apply_threshold_posthoc
from evt_utils import bootstrap_accuracy_ci
from result_writer import write_results
apply_style()
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_QUERIES   = 500
DEADLINE_MS = 45.0
TAU         = 0.55
DEVICE      = "cuda"

def oracle_posthoc(raw_results, queries, deadline_ms):
    """Oracle router: exit at L16 only when L16 token == L22 token."""
    from benchmark_utils import extract_label, _decode_with_tokenizer, _LAST_TOKENIZER
    tokenizer = _LAST_TOKENIZER
    n_correct = n_scored = 0; all_tpot = []; all_miss = []; correct_flags = []
    n_l16 = n_total = 0
    for q in raw_results:
        gen_ids = []
        for td in q["token_data"]:
            tok_id = td["l16_token_id"] if td["l16_agrees_l22"] else td["l22_token_id"]
            if td["l16_agrees_l22"]: n_l16 += 1
            gen_ids.append(tok_id)
            all_tpot.append(td["time_ms"])
            all_miss.append(td["time_ms"] > deadline_ms)
            n_total += 1
        if tokenizer:
            pred = extract_label(_decode_with_tokenizer(gen_ids, tokenizer))
            if pred != "unknown":
                n_scored += 1
                correct = (pred == q["ground_truth"])
                if correct: n_correct += 1
                correct_flags.append(bool(correct))
    acc   = 100.0 * n_correct / n_scored if n_scored > 0 else 0
    exit_ = 100.0 * n_l16 / max(1, n_total)
    miss  = 100.0 * sum(all_miss) / max(1, n_total)
    p99   = float(np.percentile(all_tpot, 99)) if all_tpot else 0
    return {"accuracy": round(acc,1), "exit_rate_pct": round(exit_,1),
            "miss_rate_pct": round(miss,1), "p99_tpot_ms": round(p99,3),
            "correct_flags": correct_flags, "n_scored": n_scored}

def fullpass_posthoc(raw_results, queries, deadline_ms):
    """Full-pass router: always use L22 logits (tau -> inf)."""
    from benchmark_utils import extract_label, _decode_with_tokenizer, _LAST_TOKENIZER
    tokenizer = _LAST_TOKENIZER
    n_correct = n_scored = 0; all_tpot = []; all_miss = []; correct_flags = []
    for q in raw_results:
        gen_ids = [td["l22_token_id"] for td in q["token_data"]]
        all_tpot += [td["time_ms"] for td in q["token_data"]]
        all_miss += [td["time_ms"] > deadline_ms for td in q["token_data"]]
        if tokenizer:
            pred = extract_label(_decode_with_tokenizer(gen_ids, tokenizer))
            if pred != "unknown":
                n_scored += 1
                correct = (pred == q["ground_truth"])
                if correct: n_correct += 1
                correct_flags.append(bool(correct))
    acc  = 100.0 * n_correct / n_scored if n_scored > 0 else 0
    miss = 100.0 * sum(all_miss) / max(1, len(all_miss))
    p99  = float(np.percentile(all_tpot, 99)) if all_tpot else 0
    return {"accuracy": round(acc,1), "exit_rate_pct": 0.0,
            "miss_rate_pct": round(miss,1), "p99_tpot_ms": round(p99,3),
            "correct_flags": correct_flags, "n_scored": n_scored}

def main():
    print("=" * 60)
    print(f"E06  Router Comparison  N={N_QUERIES} queries each")
    print("=" * 60)
    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    print(f"\n[1/3] Loading {N_QUERIES} queries ...")
    queries = load_pubmed_dataset(n_samples=N_QUERIES)

    print(f"\n[2/3] Collecting raw token data ...")
    t0  = time.time()
    raw = run_pubmed_queries_raw(model, queries, deadline_ms=DEADLINE_MS,
                                  device=DEVICE, max_new_tokens=15, show_progress=True)
    print(f"      Done in {time.time()-t0:.1f}s")

    print(f"\n[3/3] Three-router post-hoc evaluation ...")
    r_full   = fullpass_posthoc(raw, queries, DEADLINE_MS)
    r_thresh = apply_threshold_posthoc(raw, queries, tau=TAU, deadline_ms=DEADLINE_MS)
    r_oracle = oracle_posthoc(raw, queries, DEADLINE_MS)

    ci_full   = bootstrap_accuracy_ci(r_full.get("correct_flags",[]))
    ci_thresh = bootstrap_accuracy_ci(r_thresh.get("correct_flags",[]))
    ci_oracle = bootstrap_accuracy_ci(r_oracle.get("correct_flags",[]))

    routers = [
        {"name": "Full-pass (L22)", "accuracy": ci_full["accuracy"],
         "ci_lower": ci_full["ci_lower"], "ci_upper": ci_full["ci_upper"],
         "exit_rate_pct": 0.0, "miss_rate_pct": r_full["miss_rate_pct"],
         "p99_tpot_ms": r_full["p99_tpot_ms"]},
        {"name": f"L16 threshold (tau={TAU})", "accuracy": ci_thresh["accuracy"],
         "ci_lower": ci_thresh["ci_lower"], "ci_upper": ci_thresh["ci_upper"],
         "exit_rate_pct": r_thresh.get("exit_rate_pct",0),
         "miss_rate_pct": r_thresh["miss_rate_pct"],
         "p99_tpot_ms": r_thresh["p99_tpot_ms"]},
        {"name": "Oracle (L16 when correct)", "accuracy": ci_oracle["accuracy"],
         "ci_lower": ci_oracle["ci_lower"], "ci_upper": ci_oracle["ci_upper"],
         "exit_rate_pct": r_oracle["exit_rate_pct"],
         "miss_rate_pct": r_oracle["miss_rate_pct"],
         "p99_tpot_ms": r_oracle["p99_tpot_ms"]},
    ]

    print(f"\n  {'Router':<30s}  Acc    95% CI         Exit%   Miss%")
    for r in routers:
        print(f"  {r['name']:<30s}  {r['accuracy']:5.1f}%  "
              f"[{r['ci_lower']:.1f},{r['ci_upper']:.1f}]  "
              f"{r['exit_rate_pct']:5.1f}%  {r['miss_rate_pct']:5.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    names  = [r["name"] for r in routers]
    accs   = [r["accuracy"] for r in routers]
    ci_los = [r["accuracy"] - r["ci_lower"] for r in routers]
    ci_his = [r["ci_upper"] - r["accuracy"] for r in routers]
    exits  = [r["exit_rate_pct"] for r in routers]
    x = range(len(names))
    axes[0].bar(x, accs, color=["#0072B2","#D55E00","#009E73"], alpha=0.8)
    axes[0].errorbar(x, accs, yerr=[ci_los, ci_his], fmt="none", color="black", capsize=5)
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names, rotation=12, ha="right", fontsize=6)
    axes[0].set_ylabel("Accuracy (%)"); axes[0].set_title("E06 — Router Accuracy")
    axes[1].bar(x, exits, color=["#0072B2","#D55E00","#009E73"], alpha=0.8)
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(names, rotation=12, ha="right", fontsize=6)
    axes[1].set_ylabel("L16 Exit Rate (%)"); axes[1].set_title("E06 — Exit Rate")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e06_router_comparison.png", dpi=150); plt.close()

    write_results({"experiment":"e06_router_comparison","n_queries":N_QUERIES,
                   "tau":TAU,"deadline_ms":DEADLINE_MS,"routers":routers},
                  RESULTS_DIR / "e06_router_comparison.json")

    tex = ("% E06 Router Comparison\n\\begin{tabular}{lcccc}\\toprule\n"
           "Router & Accuracy (\\%) & 95\\% CI & Exit\\% & Miss\\%\\\\\n\\midrule\n")
    for r in routers:
        tex += (f"{r['name']} & {r['accuracy']:.1f} & [{r['ci_lower']:.1f},{r['ci_upper']:.1f}] & "
                f"{r['exit_rate_pct']:.1f} & {r['miss_rate_pct']:.1f}\\\\\n")
    tex += "\\bottomrule\\end{tabular}\n"
    with open(RESULTS_DIR / "table_router_comparison.tex", "w") as f: f.write(tex)
    print("PASS: E06 complete\n")

if __name__ == "__main__":
    main()
