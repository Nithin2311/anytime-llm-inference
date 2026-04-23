"""
replot_figures.py — Re-render all modified figures from saved JSON data.

Run from submission/:
  cd /root/anytime-llm-inference/submission && python3 replot_figures.py
"""

import sys, os, json, shutil, numpy as np

# Make src/ importable
SRCDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRCDIR)

RESULTS = "results"
FIGDIR  = "figures"

def R(fname):
    return json.load(open(os.path.join(RESULTS, fname)))

def cp(fname):
    dst = os.path.join(FIGDIR, fname)
    shutil.copy(fname, dst)
    print(f"  → {dst}")


# ── 1. exit_layer_ablation ────────────────────────────────────────────────────
print("\n[1/8] exit_layer_ablation")
from exit_layer_ablation import plot_ablation, EXIT_LAYERS
data = R("ablation_results.json")
plot_ablation(data["records"], data["summaries"])
cp("exit_layer_ablation.png")

# ── 2. compare_schedulers ─────────────────────────────────────────────────────
print("\n[2/8] compare_schedulers")
from compare_schedulers import plot_comparison
data = R("scheduler_comparison.json")
plot_comparison(data["stateless_metrics"], data["kvcached_metrics"],
                data.get("async_metrics"))
cp("scheduler_comparison.png")

# ── 3. deadline_sweep ─────────────────────────────────────────────────────────
print("\n[3/8] deadline_sweep")
from deadline_sweep import plot_tradeoff
data = R("sweep_results.json")
plot_tradeoff(data["sweep"])
cp("deadline_tradeoff.png")

# ── 4. pipeline_latency_model ─────────────────────────────────────────────────
print("\n[4/8] pipeline_latency_model")
from pipeline_latency_model import plot_pipeline
data = R("pipeline_latency_results.json")
# gpu_s/pcie_s/cpu_s/sync_s unused in current plot (pie + stacked + SLO panels)
plot_pipeline(None, None, None, None, data["pipeline_models"],
              deadline_ms=data["deadline_ms"])
cp("pipeline_latency_model.png")

# ── 5. thermal_profile ────────────────────────────────────────────────────────
print("\n[5/8] thermal_profile")
from thermal_profile import plot_thermal
data = R("thermal_profile_results.json")
plot_thermal(data["tpot_ms"], data["windows"])
cp("thermal_profile.png")

# ── 6. jitter_analysis ────────────────────────────────────────────────────────
print("\n[6/8] jitter_analysis")
from jitter_analysis import plot_jitter, compute_jitter_stats, autocorrelation
# Reconstruct all_tpots from benchmark_results.json (raw tpot not in jitter JSON)
bm = R("benchmark_results.json")
per_query = []
query_ids = []
for qr in bm["query_results"]:
    tpots = [r["time_ms"] for r in qr["token_records"] if r["token_idx"] > 1]
    if tpots:
        per_query.append(tpots)
        query_ids.append(qr["query_id"])
all_tpots = [t for q in per_query for t in q]
stats    = compute_jitter_stats(all_tpots)
acf_lags, acf_vals = autocorrelation(all_tpots, max_lag=10)
plot_jitter(all_tpots, per_query, query_ids, stats, acf_lags, acf_vals)
cp("jitter_analysis.png")

# ── 7. evt_wcet_analysis ──────────────────────────────────────────────────────
print("\n[7/8] evt_wcet_analysis")
from evt_wcet_analysis import plot_evt
data = R("evt_wcet_results.json")
plot_evt(data["results"])
cp("evt_wcet_analysis.png")

# ── 8. admission_control ──────────────────────────────────────────────────────
print("\n[8/8] admission_control")
from admission_control import plot_admission
data = R("admission_control_results.json")
table      = {int(k): v for k, v in data["wcet_safe_table"].items()}
sim_events = [(e["seq_len"], e["deadline_ms"], e["admitted"])
              for e in data["simulation"]["events"]]
plot_admission(table, sim_events)
cp("admission_control.png")

print("\nDone. All figures written to figures/")
