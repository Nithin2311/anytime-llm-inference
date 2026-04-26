"""
E03: Forced Exit Quality — extended metrics at tight deadlines D=14-30ms.
Addresses: R3-W4 (ROUGE-L inappropriate for binary QA); reports accuracy-on-parseable,
parseable rate, and compares stateless forced-exit vs KV natural routing.
"""
import argparse, os, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import result_writer as rw
from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import generate_stateless_anytime, generate_anytime_with_kv
import benchmark_utils as bu
import evt_utils as eu

EXPERIMENT_ID = "E03"
RESULTS_FILE  = "forced_exit_extended_results.json"
HARDWARE      = "A100 SXM4"
DEADLINES_MS  = [14, 16, 18, 20, 22, 25, 30]
N_QUERIES     = 50
MAX_NEW_TOKENS= 15


def run_router(model, dataset, generate_fn, deadline_ms, kwargs=None):
    kwargs = kwargs or {}
    n_correct = n_scored = 0
    all_times = []
    all_miss  = []
    for item in dataset:
        ctx   = item["context"]["contexts"][0]
        q     = item["question"]
        gt    = item["final_decision"]
        prompt = bu.build_prompt(model.tokenizer, ctx, q)
        recs  = generate_fn(model, prompt, max_new_tokens=MAX_NEW_TOKENS,
                            deadline_ms=deadline_ms, verbose=False, **kwargs)
        text  = "".join(r["token"] for r in recs)
        pred  = bu.extract_label(text)
        tpots = [r["time_ms"] for r in recs[1:]] or [0.0]
        all_times.extend(tpots)
        all_miss.extend(t > deadline_ms for t in tpots)
        if pred != "unknown":
            n_scored += 1
            if pred == gt: n_correct += 1
    mean_t = float(np.mean(all_times)) if all_times else 0.0
    return {
        "n_queries": len(dataset), "deadline_ms": deadline_ms,
        "n_correct": n_correct, "n_scored": n_scored,
        "accuracy":  round(100.0 * n_correct / n_scored, 1) if n_scored > 0 else None,
        "parseable_pct": round(100.0 * n_scored / (len(dataset)), 1),
        "deadline_miss_pct": round(100.0 * sum(all_miss) / max(1, len(all_miss)), 1),
        "mean_tpot_ms": round(mean_t, 3),
        "p99_tpot_ms":  round(float(np.percentile(all_times, 99)), 3) if all_times else 0.0,
    }


def main(dry_run=False):
    rw.configure(BASE_DIR)
    if dry_run:
        print(f"[{EXPERIMENT_ID}] dry-run OK"); sys.exit(0)
    if rw.already_done(RESULTS_FILE):
        print(f"[{EXPERIMENT_ID}] Skipping."); sys.exit(0)

    t0      = rw.log_start(EXPERIMENT_ID)
    model   = EarlyExitTinyLlama()
    dataset = bu.load_pubmed_dataset(N_QUERIES)
    results = {"stateless": {}, "kv_cached": {}}

    try:
        for D in DEADLINES_MS:
            print(f"  D={D:>3}ms", end="  ", flush=True)
            s = run_router(model, dataset, generate_stateless_anytime, D)
            k = run_router(model, dataset, generate_anytime_with_kv,    D,
                           kwargs={"threshold": 0.55})
            results["stateless"][str(D)] = s
            results["kv_cached"][str(D)] = k
            print(f"stateless acc={s['accuracy']}% miss={s['deadline_miss_pct']}%  "
                  f"kv_acc={k['accuracy']}% miss={k['deadline_miss_pct']}%")

        rw.write_json(RESULTS_FILE, {
            "hardware": HARDWARE, "n_queries": N_QUERIES,
            "deadlines_ms": DEADLINES_MS, "results": results,
        })
        _build_latex(results)
        _plot(results)
        rw.log_success(EXPERIMENT_ID, t0)
    except Exception as exc:
        rw.log_failure(EXPERIMENT_ID, t0, exc); raise


def _build_latex(results):
    lines = [r"% E03: Forced Exit Quality — Stateless vs KV at tight deadlines", r"\midrule"]
    for D in DEADLINES_MS:
        s = results["stateless"][str(D)]; k = results["kv_cached"][str(D)]
        lines.append(
            f"  {D} & {s['accuracy']} & {s['parseable_pct']} & {s['deadline_miss_pct']} & "
            f"{k['accuracy']} & {k['parseable_pct']} & {k['deadline_miss_pct']} \\\\"
        )
    lines.append(r"\bottomrule")
    rw.write_latex("table_forced_exit_extended.tex", "\n".join(lines) + "\n")


def _plot(results):
    import matplotlib.pyplot as plt
    import fig_style as fs
    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=fs.TRIPLE)
    Ds = DEADLINES_MS
    for router, color, label in [("stateless","#1f77b4","Stateless"),
                                   ("kv_cached", "#ff7f0e","KV-Cached")]:
        accs  = [results[router][str(D)]["accuracy"] or 0 for D in Ds]
        parse = [results[router][str(D)]["parseable_pct"] for D in Ds]
        miss  = [results[router][str(D)]["deadline_miss_pct"] for D in Ds]
        axes[0].plot(Ds, accs,  marker="o", color=color, label=label)
        axes[1].plot(Ds, parse, marker="s", color=color, label=label)
        axes[2].plot(Ds, miss,  marker="^", color=color, label=label)

    for ax, ylabel, title in zip(axes,
        ["Accuracy (%, scored)", "Parseable Response (%)", "Deadline Miss Rate (%)"],
        ["Accuracy on Parseable Queries", "Response Parseability", "Deadline Miss Rate"]):
        ax.set_xlabel("Deadline D (ms)"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.set_xticks(Ds)

    fig.suptitle(f"Forced Exit Quality — {HARDWARE}", fontsize=8)
    plt.tight_layout()
    path = rw.figures_path("forced_exit_extended.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Figure → {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
