"""
E05: Deadline Sweep — full comparison of stateless vs KV-cached router.
D ∈ {14,16,18,20,22,24,25,27,30,35,40,45,50,60,75,100}ms, 50 queries each.
Addresses: original reviewer critique that sweep only showed stateless router;
also characterizes the actual minimum schedulable deadline for each router.
"""
import argparse, os, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import result_writer as rw
from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import generate_stateless_anytime, generate_anytime_with_kv
import benchmark_utils as bu

EXPERIMENT_ID = "E05"
RESULTS_FILE  = "deadline_sweep_comparison_results.json"
HARDWARE      = "A100 SXM4"
DEADLINES_MS  = [14, 16, 18, 20, 22, 24, 25, 27, 30, 35, 40, 45, 50, 60, 75, 100]
N_QUERIES     = 50
MAX_NEW_TOKENS= 15
KV_THRESHOLD  = 0.55


def run_at_deadline(model, dataset, generate_fn, deadline_ms, kwargs=None):
    kwargs = kwargs or {}
    _, gm  = bu.run_pubmed_queries(model, model.tokenizer, dataset, deadline_ms,
                                   MAX_NEW_TOKENS, generate_fn, kwargs)
    return gm


def main(dry_run=False):
    rw.configure(BASE_DIR)
    if dry_run:
        print(f"[{EXPERIMENT_ID}] dry-run OK"); sys.exit(0)
    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping."); sys.exit(0)

    t0      = rw.log_start(EXPERIMENT_ID)
    model   = EarlyExitTinyLlama()
    dataset = bu.load_pubmed_dataset(N_QUERIES)
    res_sl  = {}; res_kv = {}

    try:
        for D in DEADLINES_MS:
            print(f"  D={D:>3}ms", end="  ", flush=True)
            sl = run_at_deadline(model, dataset, generate_stateless_anytime, D)
            kv = run_at_deadline(model, dataset, generate_anytime_with_kv,   D,
                                 kwargs={"threshold": KV_THRESHOLD})
            res_sl[str(D)] = sl
            res_kv[str(D)] = kv
            print(f"SL miss={sl['deadline_miss_pct']}% p99={sl['global_p99_tpot_ms']:.1f}  "
                  f"KV miss={kv['deadline_miss_pct']}% p99={kv['global_p99_tpot_ms']:.1f}")

        # Find minimum schedulable deadline (0% miss) for each router
        def min_schedulable(results):
            for D in DEADLINES_MS:
                if results[str(D)]["deadline_miss_pct"] == 0.0:
                    return D
            return None

        d_min_sl = min_schedulable(res_sl)
        d_min_kv = min_schedulable(res_kv)
        print(f"\n  D_min_schedulable: stateless={d_min_sl}ms  kv_cached={d_min_kv}ms")

        rw.write_json(RESULTS_FILE, {
            "hardware": HARDWARE, "n_queries": N_QUERIES,
            "deadlines_ms": DEADLINES_MS, "kv_threshold": KV_THRESHOLD,
            "d_min_stateless_ms": d_min_sl, "d_min_kv_cached_ms": d_min_kv,
            "stateless": res_sl, "kv_cached": res_kv,
        })
        _build_latex(res_sl, res_kv)
        _plot(res_sl, res_kv)
        rw.log_success(EXPERIMENT_ID, t0)

    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc); raise


def _build_latex(sl, kv):
    lines = [r"% E05: Deadline Sweep — Stateless vs KV-Cached", r"\midrule"]
    for D in DEADLINES_MS:
        s = sl[str(D)]; k = kv[str(D)]
        sl_miss = s["deadline_miss_pct"]; kv_miss = k["deadline_miss_pct"]
        sl_flag = r"\textbf{FAIL}" if sl_miss > 0 else "0.0"
        kv_flag = r"\textbf{FAIL}" if kv_miss > 0 else "0.0"
        lines.append(
            f"  {D} & {s['global_p99_tpot_ms']:.1f} & {sl_flag} & "
            f"{k['global_p99_tpot_ms']:.1f} & {kv_flag} \\\\"
        )
    lines.append(r"\bottomrule")
    rw.write_latex("table_deadline_sweep.tex", "\n".join(lines) + "\n")


def _plot(sl, kv):
    import matplotlib.pyplot as plt
    import fig_style as fs
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=fs.DOUBLE)
    Ds = DEADLINES_MS
    sl_p99  = [sl[str(D)]["global_p99_tpot_ms"] for D in Ds]
    kv_p99  = [kv[str(D)]["global_p99_tpot_ms"] for D in Ds]
    sl_miss = [sl[str(D)]["deadline_miss_pct"]   for D in Ds]
    kv_miss = [kv[str(D)]["deadline_miss_pct"]   for D in Ds]

    ax = axes[0]
    ax.plot(Ds, sl_p99, marker="o", label="Stateless",  color="#1f77b4")
    ax.plot(Ds, kv_p99, marker="s", label="KV-Cached",  color="#ff7f0e")
    ax.plot(Ds, Ds,     ls="--",    label="P99 = D",     color="gray", lw=0.8)
    ax.set_xlabel("Deadline D (ms)"); ax.set_ylabel("P99 TPOT (ms)")
    ax.set_title("P99 TPOT vs Deadline"); ax.legend()

    ax2 = axes[1]
    ax2.plot(Ds, sl_miss, marker="o", label="Stateless",  color="#1f77b4")
    ax2.plot(Ds, kv_miss, marker="s", label="KV-Cached",  color="#ff7f0e")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Deadline D (ms)"); ax2.set_ylabel("Deadline Miss Rate (%)")
    ax2.set_title("Deadline Miss Rate vs D"); ax2.legend()

    fig.suptitle(f"Deadline Sweep — Stateless vs KV-Cached ({HARDWARE})", fontsize=8)
    plt.tight_layout()
    path = rw.figures_path("deadline_sweep_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Figure → {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
