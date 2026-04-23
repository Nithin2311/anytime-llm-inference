#!/usr/bin/env bash
# ============================================================
#  run_all.sh — Full reproduction pipeline (RTX 6000 Ada)
#  Dynamic Anytime Scheduling for LLM Inference
#  CIS 6930 — Real-Time Systems  |  Spring 2026
# ============================================================
# Run from submission/ — the src/ subdirectory must be alongside.
# Usage:
#   bash run_all.sh              # full pipeline (~50-70 min)
#   bash run_all.sh --skip-slow  # skip benchmark/sweep (~15 min)
# ============================================================

set -euo pipefail
SKIP_SLOW=false
for arg in "$@"; do [[ "$arg" == "--skip-slow" ]] && SKIP_SLOW=true; done

SRCDIR="$(cd "$(dirname "$0")/src" && pwd)"
log() { echo ""; echo "================================================================"; echo "  $*"; echo "================================================================"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

log "Stage 0 — Environment validation"
python3 "$SRCDIR/verify_env.py" || die "Environment check failed."

log "Stage 1 — GPU WCET profiling (seq_len x exit_layer sweep, 50 runs)"
python3 "$SRCDIR/profile_wcet.py"

log "Stage 2 — Heterogeneous pipeline latency model (T_cpu + T_pcie + T_gpu)"
python3 "$SRCDIR/pipeline_latency_model.py"

log "Stage 3 — L16 confidence calibration (10 prompts x 20 tokens)"
python3 "$SRCDIR/calibration.py"

log "Stage 4 — Exit-layer ablation (L5 / L11 / L16-L20 vs oracle L22)"
python3 "$SRCDIR/exit_layer_ablation.py"

log "Stage 5 — Tail-latency SLO proof (10 clinical prompts, D=45 ms)"
python3 "$SRCDIR/evaluate_tail_latency.py"

log "Stage 6 — Router comparison (stateless vs KV-cached vs async-overlap)"
python3 "$SRCDIR/compare_schedulers.py"

if $SKIP_SLOW; then
  log "Stage 7 — Deadline sweep [SKIPPED -- --skip-slow]"
else
  log "Stage 7 — Deadline sensitivity sweep (D=20-60 ms, 5 queries each)"
  python3 "$SRCDIR/deadline_sweep.py"
fi

if $SKIP_SLOW; then
  log "Stage 8 — PubMedQA benchmark [SKIPPED -- --skip-slow]"
else
  log "Stage 8 — 30-query PubMedQA clinical benchmark (D=45 ms)"
  python3 "$SRCDIR/benchmark.py"
fi

log "Stage 9 — Regenerate IEEE-style publication figures"
python3 "$SRCDIR/visualize_metrics.py"

echo ""
echo "================================================================"
echo "  All stages complete."
echo "  JSON results -> working directory (copy to results/)"
echo "  PNG figures  -> working directory (copy to figures/)"
echo "================================================================"
