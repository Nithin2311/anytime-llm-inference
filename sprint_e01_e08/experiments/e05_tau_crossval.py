"""e05_tau_crossval.py — 5-fold cross-validation of confidence threshold tau."""
import sys, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
from benchmark_utils import load_pubmed_dataset, run_pubmed_queries_raw, apply_threshold_posthoc
from evt_utils import bootstrap_accuracy_ci
from result_writer import write_results
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_QUERIES   = 1000
K_FOLDS     = 5
DEADLINE_MS = 45.0
TAU_GRID    = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
DEVICE      = "cuda"

def main():
    print("=" * 60)
    print(f"E05  5-Fold tau Cross-Validation  N={N_QUERIES}")
    print("=" * 60)
    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    print(f"\n[1/3] Loading {N_QUERIES} queries ...")
    queries = load_pubmed_dataset(n_samples=N_QUERIES)
    print(f"      Loaded {len(queries)}")

    print(f"\n[2/3] Collecting raw token data ...")
    t0  = time.time()
    raw = run_pubmed_queries_raw(model, queries, deadline_ms=DEADLINE_MS,
                                  device=DEVICE, max_new_tokens=15, show_progress=True)
    print(f"      Done in {time.time()-t0:.1f}s")

    print(f"\n[3/3] 5-fold CV ...")
    n    = len(raw)
    fold_size = n // K_FOLDS
    fold_results = []

    for fold in range(K_FOLDS):
        val_start = fold * fold_size
        val_end   = val_start + fold_size
        val_raw   = raw[val_start:val_end]
        val_q     = queries[val_start:val_end]
        train_raw = raw[:val_start] + raw[val_end:]
        train_q   = queries[:val_start] + queries[val_end:]

        # Select best tau on training set
        best_tau = TAU_GRID[0]; best_acc = -1
        for tau in TAU_GRID:
            r = apply_threshold_posthoc(train_raw, train_q, tau=tau, deadline_ms=DEADLINE_MS)
            acc = r.get("accuracy", 0)
            if acc > best_acc:
                best_acc = acc; best_tau = tau

        # Evaluate on validation set
        val_res = apply_threshold_posthoc(val_raw, val_q, tau=best_tau, deadline_ms=DEADLINE_MS)
        val_ci  = bootstrap_accuracy_ci(val_res.get("correct_flags", []), n_bootstrap=500)

        fold_results.append({
            "fold":         fold + 1,
            "best_tau":     best_tau,
            "train_acc":    round(float(best_acc), 1),
            "val_acc":      val_ci["accuracy"],
            "val_ci_lower": val_ci["ci_lower"],
            "val_ci_upper": val_ci["ci_upper"],
            "val_n":        len(val_raw),
            "exit_rate":    val_res.get("exit_rate_pct", 0),
        })
        print(f"  Fold {fold+1}: best_tau={best_tau}  train_acc={best_acc:.1f}%  "
              f"val_acc={val_ci['accuracy']:.1f}% [{val_ci['ci_lower']:.1f},{val_ci['ci_upper']:.1f}]")

    mean_val_acc  = float(np.mean([r["val_acc"] for r in fold_results]))
    std_val_acc   = float(np.std([r["val_acc"] for r in fold_results]))
    tau_counts    = {}
    for r in fold_results:
        tau_counts[r["best_tau"]] = tau_counts.get(r["best_tau"], 0) + 1
    modal_tau = max(tau_counts, key=tau_counts.get)

    print(f"\n  Mean val accuracy: {mean_val_acc:.1f}% ± {std_val_acc:.1f}%")
    print(f"  Modal tau selected: {modal_tau} (selected {tau_counts[modal_tau]}/{K_FOLDS} folds)")
    print(f"  tau=0.55 selected: {tau_counts.get(0.55, 0)}/{K_FOLDS} folds")

    write_results({
        "experiment":     "e05_tau_crossval",
        "n_queries":      n,
        "k_folds":        K_FOLDS,
        "tau_grid":       TAU_GRID,
        "fold_results":   fold_results,
        "mean_val_acc":   round(mean_val_acc, 2),
        "std_val_acc":    round(std_val_acc, 2),
        "modal_tau":      modal_tau,
        "tau_0_55_count": tau_counts.get(0.55, 0),
    }, RESULTS_DIR / "e05_tau_crossval.json")
    print("PASS: E05 complete\n")

if __name__ == "__main__":
    main()
