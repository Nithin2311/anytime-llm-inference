#!/usr/bin/env bash
# ============================================================
#  run_all.sh — Full reproduction pipeline (RTX 6000 Ada)
#  Dynamic Anytime Scheduling for LLM Inference
#  CIS 6930 — Real-Time Systems  |  Spring 2026
# ============================================================
# Run from /root/anytime-llm-inference/ (project root).
# All JSON results and PNG figures are written here.
#
# Usage:
#   bash run_all.sh              # full pipeline (~60-80 min)
#   bash run_all.sh --skip-slow  # skip benchmark/sweep (~20 min)
# ============================================================

set -euo pipefail

SKIP_SLOW=false
for arg in "$@"; do
  [[ "$arg" == "--skip-slow" ]] && SKIP_SLOW=true
done

log() {
  echo ""
  echo "================================================================"
  echo "  $*"
  echo "================================================================"
}
die() { echo "[ERROR] $*" >&2; exit 1; }

START_TIME=$(date +%s)

# ── Stage 0: Environment validation ──────────────────────────────────────────
log "Stage 0/9 — Environment validation"
python3 verify_env.py || die "Environment check failed."

# ── Stage 1: GPU WCET profiling ───────────────────────────────────────────────
log "Stage 1/9 — GPU WCET profiling (6 seq lengths × 4 exit layers, 50 runs each)"
python3 profile_wcet.py
echo "  → wcet_results.json  wcet_profile.png"

# ── Stage 2: Heterogeneous pipeline latency model ────────────────────────────
log "Stage 2/9 — Heterogeneous pipeline model (T_cpu + T_pcie + T_gpu)"
python3 pipeline_latency_model.py
echo "  → pipeline_latency_results.json  pipeline_latency_model.png"

# ── Stage 3: Confidence calibration ──────────────────────────────────────────
log "Stage 3/9 — L16 confidence calibration (10 prompts × 20 tokens)"
python3 calibration.py
echo "  → calibration_results.json  calibration.png"

# ── Stage 4: Exit-layer ablation ─────────────────────────────────────────────
log "Stage 4/9 — Exit-layer ablation (L5 / L11 / L16–L20 vs oracle L22)"
python3 exit_layer_ablation.py
echo "  → ablation_results.json  exit_layer_ablation.png"

# ── Stage 5: Tail-latency SLO proof ──────────────────────────────────────────
log "Stage 5/9 — Tail-latency SLO proof (10 clinical prompts, KV-cached, D=45 ms)"
python3 evaluate_tail_latency.py
echo "  → tail_latency_results.json  schedulability_proof.png"

# ── Stage 6: Three-way router comparison ─────────────────────────────────────
log "Stage 6/9 — Router comparison (stateless / KV-cached / async-overlap, 5 queries)"
python3 compare_schedulers.py
echo "  → scheduler_comparison.json  scheduler_comparison.png"

# ── Stage 7: Deadline sensitivity sweep ──────────────────────────────────────
if $SKIP_SLOW; then
  log "Stage 7/9 — Deadline sweep [SKIPPED — --skip-slow]"
else
  log "Stage 7/9 — Deadline sensitivity sweep (D = 20–60 ms, 5 queries each)"
  python3 deadline_sweep.py
  echo "  → sweep_results.json  deadline_tradeoff.png"
fi

# ── Stage 8: Full clinical benchmark ─────────────────────────────────────────
if $SKIP_SLOW; then
  log "Stage 8/9 — PubMedQA benchmark [SKIPPED — --skip-slow]"
else
  log "Stage 8/9 — 30-query PubMedQA clinical benchmark (D=45 ms)"
  python3 benchmark.py
  echo "  → benchmark_results.json  benchmark_results.csv"
fi

# ── Stage 9: Regenerate all figures ──────────────────────────────────────────
log "Stage 9/9 — Regenerate IEEE-format publication figures"
python3 visualize_metrics.py
echo "  → execution_timeline.png  tail_latency_cdf.png"
echo "     exit_distribution.png  accuracy_summary.png"

# ── Copy outputs into submission/ ────────────────────────────────────────────
log "Copying results and figures into submission/"
cp -f *.json submission/results/ 2>/dev/null || true
cp -f *.csv  submission/results/ 2>/dev/null || true
cp -f *.png  submission/figures/ 2>/dev/null || true
echo "  → submission/results/  submission/figures/"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "================================================================"
echo "  All stages complete in ${MINS}m ${SECS}s"
echo ""
echo "  Key outputs:"
echo "    wcet_results.json            WCET profile (RTX 6000 Ada)"
echo "    pipeline_latency_results.json  CPU+PCIe+GPU pipeline model"
echo "    benchmark_results.json/csv   30-query clinical benchmark"
echo "    scheduler_comparison.json    3-way router comparison"
echo "    tail_latency_results.json    SLO schedulability proof"
echo "    ablation_results.json        Exit-layer ablation L5-L22"
echo "    submission/results/          All JSON/CSV (submission copy)"
echo "    submission/figures/          All PNG  (submission copy)"
echo "================================================================"
