"""e08_capacity_full.py — Multi-request capacity N=1..8 with SLO compliance."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib.pyplot as plt
from fig_style import apply_style, DOUBLE
from result_writer import write_results
apply_style()
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_LEVELS    = [1, 2, 4, 8]
N_REPS      = 100
SEQ_LEN     = 128
MAX_TOKENS  = 15
DEADLINE_MS = 45.0
DEVICE      = "cuda"

def build_input(tokenizer):
    PROMPT = "The pharmacokinetics of drug X suggest that the optimal dosing"
    import torch
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
    while len(ids) < SEQ_LEN:
        ids = ids.repeat(2)
    return ids[:SEQ_LEN].unsqueeze(0).to(DEVICE)

def run_batch_sequential(model, tokenizer, n_requests, n_reps):
    """Sequential batching: n_requests back-to-back, each MAX_TOKENS steps."""
    import torch
    ids     = build_input(tokenizer)
    per_req_p99s  = []
    per_req_means = []
    throughputs   = []
    slo_rates     = []

    with torch.inference_mode():
        for rep in range(n_reps):
            batch_start = time.time()
            all_tpots   = []
            for _ in range(n_requests):
                _, _, pkv = model.forward_cached(ids)
                new_tok = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
                req_times = []
                for _t in range(MAX_TOKENS):
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    ev_s.record()
                    model.forward_cached(new_tok, past_key_values=pkv)
                    ev_e.record()
                    torch.cuda.synchronize()
                    req_times.append(ev_s.elapsed_time(ev_e))
                all_tpots.extend(req_times)
            batch_elapsed = time.time() - batch_start
            per_req_p99s.append(float(np.percentile(all_tpots, 99)))
            per_req_means.append(float(np.mean(all_tpots)))
            throughputs.append(n_requests * MAX_TOKENS / batch_elapsed)   # tokens/s
            slo_rates.append(100.0 * sum(1 for t in all_tpots if t <= DEADLINE_MS) / len(all_tpots))

    return {
        "n_requests":     n_requests,
        "n_reps":         n_reps,
        "mean_tpot_ms":   round(float(np.mean(per_req_means)), 3),
        "p99_tpot_ms":    round(float(np.mean(per_req_p99s)), 3),
        "throughput_tps": round(float(np.mean(throughputs)), 2),
        "slo_compliance": round(float(np.mean(slo_rates)), 2),
    }

def main():
    print("=" * 60)
    print(f"E08  Multi-Request Capacity  N={N_LEVELS}  reps={N_REPS}")
    print("=" * 60)
    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    results = []
    for n in N_LEVELS:
        print(f"\n  N={n} ({N_REPS} reps) ...", flush=True)
        r = run_batch_sequential(model, model.tokenizer, n, N_REPS)
        results.append(r)
        print(f"    p99={r['p99_tpot_ms']:.2f}ms  throughput={r['throughput_tps']:.1f}tok/s  "
              f"SLO={r['slo_compliance']:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    ns     = [r["n_requests"] for r in results]
    p99s   = [r["p99_tpot_ms"] for r in results]
    tputs  = [r["throughput_tps"] for r in results]
    slos   = [r["slo_compliance"] for r in results]
    for ax, vals, ylabel, title in zip(
        axes, [p99s, tputs, slos],
        ["P99 TPOT (ms)", "Throughput (tok/s)", "SLO compliance (%)"],
        ["P99 Latency vs N", "Throughput vs N", "SLO vs N"]
    ):
        ax.plot(ns, vals, "o-", color="tab:blue")
        ax.set_xlabel("N (requests)"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7)
        ax.set_xticks(ns)
    axes[2].axhline(99.0, ls="--", color="red", lw=0.8, label="99% target")
    axes[2].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e08_capacity.png", dpi=150); plt.close()

    write_results({"experiment": "e08_capacity_full", "deadline_ms": DEADLINE_MS,
                   "max_tokens": MAX_TOKENS, "seq_len": SEQ_LEN,
                   "results": results}, RESULTS_DIR / "e08_capacity.json")

    tex = ("% E08 Capacity\n\\begin{tabular}{lcccc}\\toprule\n"
           "$N$ & Mean TPOT (ms) & P99 TPOT (ms) & Throughput (tok/s) & SLO (\\%)\\\\\n\\midrule\n")
    for r in results:
        tex += (f"{r['n_requests']} & {r['mean_tpot_ms']:.2f} & {r['p99_tpot_ms']:.2f} & "
                f"{r['throughput_tps']:.1f} & {r['slo_compliance']:.1f}\\\\\n")
    tex += "\\bottomrule\\end{tabular}\n"
    with open(RESULTS_DIR / "table_capacity.tex", "w") as f: f.write(tex)
    print("\nPASS: E08 complete\n")

if __name__ == "__main__":
    main()
