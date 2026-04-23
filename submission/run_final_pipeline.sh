#!/usr/bin/env bash
# run_final_pipeline.sh — Complete final reproduction pipeline.
#
# Runs every experiment in dependency order, including all new real-time
# systems analyses (EVT/pWCET, admission control, jitter, thermal, task model).
# Collects all outputs into final_results/.
#
# Usage:
#   bash run_final_pipeline.sh              # full pipeline (~2-3 hrs on RTX 4000 Ada)
#   bash run_final_pipeline.sh --skip-slow  # skip benchmark + sweep (~45 min)
#
# Output:
#   final_results/   — all JSON, CSV, PNG files for the report

set -euo pipefail

SKIP_SLOW=false
for arg in "$@"; do
  [[ "$arg" == "--skip-slow" ]] && SKIP_SLOW=true
done

OUTDIR="final_results"
mkdir -p "$OUTDIR"

START_TIME=$(date +%s)

log()  { echo ""; echo "════════════════════════════════════════════════════"; echo "  >>> $*"; echo "════════════════════════════════════════════════════"; }
step() { echo "  [$(date '+%H:%M:%S')]  $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

log "FINAL PIPELINE — Dynamic Anytime Scheduling for LLM Inference"
echo "  Skip-slow: $SKIP_SLOW"
echo "  Output:    $OUTDIR/"
echo ""

# ── 0. Environment check ──────────────────────────────────────────────────────
log "STEP 0 / 14 — Environment check"
python verify_env.py || die "Environment check failed — fix the issues above."
step "Environment OK."

# ── 1. WCET profiling ─────────────────────────────────────────────────────────
log "STEP 1 / 14 — WCET profiling (seq 32-1024 × exit L5/L11/L16/Full, 50 runs)"
python profile_wcet.py
step "→ wcet_results.json  wcet_profile.png"

# ── 2. Confidence calibration ─────────────────────────────────────────────────
log "STEP 2 / 14 — Early-exit confidence calibration"
python calibration.py
step "→ calibration_results.json  calibration.png"

# ── 3. Exit-layer ablation ────────────────────────────────────────────────────
log "STEP 3 / 14 — Exit-layer ablation (L5 → L20 vs oracle L22)"
python exit_layer_ablation.py
step "→ ablation_results.json  exit_layer_ablation.png"

# ── 4. Schedulability proof ───────────────────────────────────────────────────
log "STEP 4 / 14 — Tail-latency schedulability proof (10 prompts × 25 tokens)"
python evaluate_tail_latency.py
step "→ tail_latency_results.json  schedulability_proof.png"

# ── 5. Scheduler comparison ───────────────────────────────────────────────────
log "STEP 5 / 14 — Stateless vs KV-cached scheduler comparison (5 queries)"
python compare_schedulers.py
step "→ scheduler_comparison.json  scheduler_comparison.png"

# ── 6. Deadline sweep ─────────────────────────────────────────────────────────
if [[ "$SKIP_SLOW" == "true" ]]; then
  log "STEP 6 / 14 — Deadline sweep [SKIPPED — --skip-slow]"
else
  log "STEP 6 / 14 — Deadline sweep (D=20-60 ms, stateless scheduler)"
  python deadline_sweep.py
  step "→ sweep_results.json  deadline_tradeoff.png"
fi

# ── 7. PubMedQA benchmark ─────────────────────────────────────────────────────
if [[ "$SKIP_SLOW" == "true" ]]; then
  log "STEP 7 / 14 — PubMedQA benchmark [SKIPPED — --skip-slow]"
  [[ -f benchmark_results.json ]] || die "benchmark_results.json missing — run without --skip-slow first."
else
  log "STEP 7 / 14 — PubMedQA benchmark (30 queries, KV-cached, D=45 ms)"
  python benchmark.py
  step "→ benchmark_results.json  benchmark_results.csv"
fi

# ── 8. Benchmark visualisation ────────────────────────────────────────────────
log "STEP 8 / 14 — Regenerate benchmark figures"
python visualize_metrics.py
step "→ execution_timeline.png  tail_latency_cdf.png  exit_distribution.png  accuracy_summary.png"

# ── 9. CPU overhead profiling ─────────────────────────────────────────────────
log "STEP 9 / 14 — CPU overhead measurement (200 tokens per scheduler)"
python cpu_overhead_profile.py
step "→ cpu_overhead_results.json  cpu_overhead.png"

# ── 10. SCHED_FIFO comparison ─────────────────────────────────────────────────
log "STEP 10 / 14 — SCHED_FIFO WCET comparison (200 runs/cell)"
python sched_fifo_profile.py
step "→ sched_fifo_results.json  sched_fifo_comparison.png"

# ── 11. EVT WCET analysis ─────────────────────────────────────────────────────
log "STEP 11 / 14 — EVT WCET bounds (500 samples/cell, Gumbel fit)"
python evt_wcet_analysis.py
step "→ evt_wcet_results.json  evt_wcet_analysis.png"

# ── 12. pWCET curve ───────────────────────────────────────────────────────────
log "STEP 12 / 14 — Probabilistic WCET (pWCET) curve (from EVT params)"
python pwcet_curve.py
step "→ pwcet_curve_results.json  pwcet_curve.png"

# ── 13. Admission control ─────────────────────────────────────────────────────
log "STEP 13 / 14 — Admission control analysis"
python admission_control.py
step "→ admission_control_results.json  admission_control.png"

# ── 14a. Jitter analysis ──────────────────────────────────────────────────────
if [[ "$SKIP_SLOW" == "true" ]] && [[ ! -f benchmark_results.json ]]; then
  log "STEP 14a — Jitter analysis [SKIPPED — no benchmark_results.json]"
else
  log "STEP 14a / 14 — Jitter analysis (from benchmark TPOT series)"
  python jitter_analysis.py
  step "→ jitter_analysis_results.json  jitter_analysis.png"
fi

# ── 14b. Thermal / sustained load profiling ───────────────────────────────────
log "STEP 14b / 14 — Thermal stability profile (500 consecutive tokens)"
python thermal_profile.py
step "→ thermal_profile_results.json  thermal_profile.png"

# ── 14c. Formal task model analysis ──────────────────────────────────────────
log "STEP 14c / 14 — Formal task model (utilisation, LL bound, capacity)"
python task_model_analysis.py
step "→ task_model_results.json  task_model_analysis.png"

# ── Collect outputs ───────────────────────────────────────────────────────────
log "COLLECTING outputs → $OUTDIR/"

# JSON / CSV data files
for f in \
  wcet_results.json \
  calibration_results.json \
  ablation_results.json \
  tail_latency_results.json \
  scheduler_comparison.json \
  sweep_results.json \
  benchmark_results.json \
  benchmark_results.csv \
  cpu_overhead_results.json \
  sched_fifo_results.json \
  evt_wcet_results.json \
  pwcet_curve_results.json \
  admission_control_results.json \
  jitter_analysis_results.json \
  thermal_profile_results.json \
  task_model_results.json; do
  [[ -f "$f" ]] && cp "$f" "$OUTDIR/" && echo "  copied $f"
done

# PNG figures
for f in \
  wcet_profile.png \
  calibration.png \
  exit_layer_ablation.png \
  schedulability_proof.png \
  tail_latency_cdf.png \
  scheduler_comparison.png \
  deadline_tradeoff.png \
  execution_timeline.png \
  exit_distribution.png \
  accuracy_summary.png \
  cpu_overhead.png \
  sched_fifo_comparison.png \
  evt_wcet_analysis.png \
  pwcet_curve.png \
  admission_control.png \
  jitter_analysis.png \
  thermal_profile.png \
  task_model_analysis.png; do
  [[ -f "$f" ]] && cp "$f" "$OUTDIR/" && echo "  copied $f"
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "  Elapsed: ${MINUTES}m ${SECONDS}s"
echo "  Output : $OUTDIR/"
echo ""
echo "  Data files:"
ls "$OUTDIR/"*.json "$OUTDIR/"*.csv 2>/dev/null | sed 's/^/    /' || true
echo ""
echo "  Figures:"
ls "$OUTDIR/"*.png 2>/dev/null | sed 's/^/    /' || true
echo "════════════════════════════════════════════════════"
