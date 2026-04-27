"""e04_accuracy_1000.py — 1000-query PubMedQA accuracy. 95% CI width ~3pp."""
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
N_QUERIES   = 1000
TAU         = 0.55
DEADLINE_MS = 45.0
N_BOOTSTRAP = 2000
DEVICE      = "cuda"

def main():
    print("=" * 60)
    print(f"E04  1000-Query PubMedQA Accuracy  tau={TAU}  D={DEADLINE_MS}ms")
    print("=" * 60)
    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    print(f"\n[1/3] Loading {N_QUERIES} PubMedQA queries ...")
    queries = load_pubmed_dataset(n_samples=N_QUERIES)
    print(f"      Loaded {len(queries)} queries")

    print(f"\n[2/3] Collecting raw token data ...")
    t0 = time.time()
    raw = run_pubmed_queries_raw(model, queries, deadline_ms=DEADLINE_MS,
                                  device=DEVICE, max_new_tokens=15, show_progress=True)
    print(f"      Done in {time.time()-t0:.1f}s")

    print(f"\n[3/3] Post-hoc replay tau={TAU} + bootstrap CI ...")
    res = apply_threshold_posthoc(raw, queries, tau=TAU, deadline_ms=DEADLINE_MS,
                                   n_bootstrap=N_BOOTSTRAP)
    ci  = bootstrap_accuracy_ci(res.get("correct_flags", []), n_bootstrap=N_BOOTSTRAP)

    print(f"\n{'─'*50}")
    print(f"  Queries       : {len(queries)}")
    print(f"  Scoreable     : {res.get('n_scored',0)}")
    print(f"  Accuracy      : {ci['accuracy']:.1f}%")
    print(f"  95% CI        : [{ci['ci_lower']:.1f}%, {ci['ci_upper']:.1f}%]")
    print(f"  CI width      : {ci['ci_width']:.1f} pp")
    print(f"  Exit rate     : {res.get('exit_rate_pct',0):.1f}%")
    print(f"  Miss rate     : {res.get('miss_rate_pct',0):.1f}%")
    print(f"{'─'*50}\n")

    # Tau sweep for supplemental figure
    tau_sweep = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep_rows = []
    for tau in tau_sweep:
        r = apply_threshold_posthoc(raw, queries, tau=tau, deadline_ms=DEADLINE_MS)
        ci_r = bootstrap_accuracy_ci(r.get("correct_flags",[]), n_bootstrap=500)
        sweep_rows.append({"tau": tau, "accuracy": ci_r["accuracy"],
                           "ci_lower": ci_r["ci_lower"], "ci_upper": ci_r["ci_upper"],
                           "exit_rate_pct": r.get("exit_rate_pct",0)})

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    taus  = [r["tau"] for r in sweep_rows]
    accs  = [r["accuracy"] for r in sweep_rows]
    ci_lo = [r["ci_lower"] for r in sweep_rows]
    ci_hi = [r["ci_upper"] for r in sweep_rows]
    exits = [r["exit_rate_pct"] for r in sweep_rows]
    axes[0].fill_between(taus, ci_lo, ci_hi, alpha=0.25, color="tab:blue")
    axes[0].plot(taus, accs, "o-", color="tab:blue", label="Accuracy")
    axes[0].axvline(TAU, ls="--", color="tab:orange", lw=1.2, label=f"tau={TAU}")
    axes[0].set_xlabel("tau"); axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(0,100)
    axes[0].set_title("E04 — Accuracy vs tau (1000-query)"); axes[0].legend(fontsize=7)
    axes[1].plot(taus, exits, "s-", color="tab:green", label="Exit rate")
    axes[1].axvline(TAU, ls="--", color="tab:orange", lw=1.2)
    axes[1].set_xlabel("tau"); axes[1].set_ylabel("Exit rate (%)"); axes[1].set_ylim(0,100)
    axes[1].set_title("E04 — Exit Rate vs tau"); axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e04_accuracy_1000.png", dpi=150); plt.close()

    write_results({
        "experiment": "e04_accuracy_1000", "n_queries": len(queries),
        "n_scored": res.get("n_scored",0), "tau": TAU, "deadline_ms": DEADLINE_MS,
        "accuracy_pct": ci["accuracy"], "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"], "ci_width": ci["ci_width"],
        "exit_rate_pct": res.get("exit_rate_pct",0),
        "miss_rate_pct": res.get("miss_rate_pct",0),
        "tau_sweep": sweep_rows,
    }, RESULTS_DIR / "e04_accuracy_1000.json")
    print("PASS: E04 complete\n")

if __name__ == "__main__":
    main()
