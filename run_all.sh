#!/usr/bin/env bash
# run_all.sh — Full reproduction pipeline for anytime-llm-inference.
#
# Runs every experiment in dependency order and regenerates all figures.
# Expected total runtime: ~45-60 minutes on an RTX 4000 Ada.
#
# Usage:
#   source venv/bin/activate
#   bash run_all.sh            # full pipeline
#   bash run_all.sh --skip-slow  # skip benchmark (30q) and sweep (8×5q)

set -euo pipefail

SKIP_SLOW=false
for arg in "$@"; do
  [[ "$arg" == "--skip-slow" ]] && SKIP_SLOW=true
done

log() { echo ""; echo "========================================"; echo ">>> $*"; echo "========================================"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ── 0. Environment check ─────────────────────────────────────────────────────
log "0/7  Environment check"
python verify_env.py || die "Environment check failed. Fix the issues above before proceeding."

# ── 1. WCET profiling ────────────────────────────────────────────────────────
log "1/7  WCET profiling (seq lengths: 32/64/128/256 × exit layers: 5/11/16/full, 50 runs each)"
python profile_wcet.py
echo "Output: wcet_results.json, wcet_profile.png"

# ── 2. Early-exit calibration ────────────────────────────────────────────────
log "2/7  Early-exit confidence calibration (10 prompts × 20 tokens)"
python calibration.py
echo "Output: calibration_results.json, calibration.png"

# ── 3. Exit-layer ablation ───────────────────────────────────────────────────
log "3/7  Exit-layer ablation study (L5 / L11 / L16 vs oracle L22)"
python exit_layer_ablation.py
echo "Output: ablation_results.json, exit_layer_ablation.png"

# ── 4. Tail-latency schedulability proof ─────────────────────────────────────
log "4/7  Tail-latency schedulability proof (3 clinical prompts, D=45 ms)"
python evaluate_tail_latency.py
echo "Output: tail_latency_results.json, schedulability_proof.png"

# ── 5. Static vs Dynamic scheduler comparison ────────────────────────────────
log "5/7  Scheduler comparison (static L5 vs dynamic L16, 5 queries)"
python compare_schedulers.py
echo "Output: scheduler_comparison.json, scheduler_comparison.png"

# ── 6. Deadline sweep ────────────────────────────────────────────────────────
if [[ "$SKIP_SLOW" == "true" ]]; then
  log "6/7  Deadline sweep [SKIPPED — --skip-slow]"
else
  log "6/7  Deadline sweep (deadlines 20–60 ms, 5 queries each)"
  python deadline_sweep.py
  echo "Output: sweep_results.json, deadline_tradeoff.png"
fi

# ── 7. Full benchmark ────────────────────────────────────────────────────────
if [[ "$SKIP_SLOW" == "true" ]]; then
  log "7/7  PubMedQA benchmark [SKIPPED — --skip-slow]"
else
  log "7/7  PubMedQA benchmark (30 queries, dynamic scheduler, D=45 ms)"
  python benchmark.py
  echo "Output: benchmark_results.json, benchmark_results.csv"
fi

# ── 8. Regenerate all figures ────────────────────────────────────────────────
log "Regenerating visualisation figures from benchmark results"
python visualize_metrics.py
echo "Output: execution_timeline.png, tail_latency_cdf.png, exit_distribution.png, accuracy_summary.png"

echo ""
echo "========================================"
echo "All experiments complete."
echo "Key output files:"
echo "  wcet_results.json          — WCET measurements"
echo "  benchmark_results.json/csv — 30-query evaluation"
echo "  tail_latency_results.json  — schedulability proof data"
echo "  ablation_results.json      — exit-layer ablation data"
echo "  *.png                      — all figures"
echo "========================================"
