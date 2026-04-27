#!/usr/bin/env bash
# run_all.sh — Master orchestrator. Runs E00→E09 sequentially with retry.
#
# Usage:
#   bash run_all.sh              # full run
#   bash run_all.sh --resume     # skip completed (.done flag exists)
#   bash run_all.sh --only e04   # single experiment
#   bash run_all.sh --dry        # print plan, no execution

set -uo pipefail
cd "$(dirname "$0")"

RESULTS="results"; LOGS="logs"
SUMMARY="${RESULTS}/SPRINT_SUMMARY.md"
RUNLOG="${RESULTS}/run.log"
RESUME=0; ONLY=""; DRY=0
MAX_RETRIES=3; RETRY_SLEEP=30

for arg in "$@"; do
  case "$arg" in
    --resume)  RESUME=1 ;;
    --dry)     DRY=1 ;;
    --only=*)  ONLY="${arg#--only=}" ;;
    --only)    shift; ONLY="${1:-}" ;;
  esac
done

mkdir -p "$RESULTS" "$LOGS"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

SPRINT_START=$(date +%s)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*" | tee -a "$RUNLOG"; }

log "═══════════════════════════════════════════════════════════"
log " Sprint Camera-Ready  |  GPU: ${GPU}"
log " RESUME=${RESUME}  ONLY=${ONLY:-all}  DRY=${DRY}"
log "═══════════════════════════════════════════════════════════"

declare -A STATUS ATTEMPTS
PASSED=0; FAILED=0; SKIPPED=0
declare -a TIMES

run_experiment() {
  local name="$1" desc="${2:-}"
  local done_flag="${RESULTS}/.${name}.done"
  local logfile="${LOGS}/${name}.log"

  if [ -n "$ONLY" ]; then
    local tag="${name:1:2}" short="${name%%_*}"
    if [ "$ONLY" != "$name" ] && [ "$ONLY" != "$tag" ] && \
       [ "$ONLY" != "$short" ] && [ "$ONLY" != "E${tag}" ] && \
       [ "$ONLY" != "e${tag}" ] && [ "$ONLY" != "${tag#0}" ]; then
      STATUS["$name"]="SKIP"; ((SKIPPED++)) || true; return 0
    fi
  fi

  if [ "$RESUME" -eq 1 ] && [ -f "$done_flag" ]; then
    log "  [SKIP] ${name}"; STATUS["$name"]="SKIP"; ((SKIPPED++)) || true; return 0
  fi

  [ "$DRY" -eq 1 ] && { log "  [DRY]  ${name}  ${desc}"; STATUS["$name"]="DRY"; return 0; }

  log ""; log "──────────────────────────────────────────────"
  log "  Running: ${name}  —  ${desc}"

  local attempt=0 success=0 t0; t0=$(date +%s)
  while [ $attempt -lt $MAX_RETRIES ]; do
    ((attempt++)) || true
    log "  Attempt ${attempt}/${MAX_RETRIES} ..."
    if python3 "experiments/${name}.py" 2>&1 | tee "$logfile"; then
      success=1; break
    else
      local ec=${PIPESTATUS[0]}
      log "  Attempt ${attempt} failed (exit ${ec})"
      [ $attempt -lt $MAX_RETRIES ] && {
        log "  Retrying in ${RETRY_SLEEP}s ..."
        sleep "$RETRY_SLEEP"
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
      }
    fi
  done

  local elapsed=$(( $(date +%s) - t0 ))
  if [ "$success" -eq 1 ]; then
    STATUS["$name"]="PASS"; ATTEMPTS["$name"]=$attempt
    touch "$done_flag"; TIMES+=("${name}: ${elapsed}s")
    ((PASSED++)) || true; log "  [PASS] ${name}  (${elapsed}s)"
  else
    STATUS["$name"]="FAIL"; ATTEMPTS["$name"]=$attempt
    ((FAILED++)) || true; log "  [FAIL] ${name}  after ${attempt} attempts"
  fi
}

# ── Experiment sequence ──────────────────────────────────────────────────────
run_experiment "e00_wcet_large_spaced"   "5000 samples/cell, 1s spacing, 12 cells (~17h)"
run_experiment "e01_iid_spacing_study"   "Ljung-Box IID at 0/200ms/1s/5s spacing"
run_experiment "e02_block_maxima_large"  "Block-maxima GEV n=5000, b=25/50/100"
run_experiment "e03_nonevt_wcet"         "Non-EVT empirical bounds P99/P99.9 + safety factor"
run_experiment "e04_accuracy_1000"       "1000-query PubMedQA accuracy, CI ~+-3pp"
run_experiment "e05_tau_crossval"        "5-fold tau cross-validation"
run_experiment "e06_router_comparison"   "Three-router comparison, 500 queries each"
run_experiment "e07_thermal_extended"    "60-min thermal soak on A100"
run_experiment "e08_capacity_full"       "Multi-request capacity N=1..8"
run_experiment "e09_final_report"        "Consolidated LaTeX tables + camera-ready summary"

# ── Write summary ────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - SPRINT_START ))
GPU_FULL=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
{
  echo "# Sprint Camera-Ready — Summary"
  echo ""; echo "**Completed:** $(date)"
  echo "**Total:** ${TOTAL}s  |  **GPU:** ${GPU_FULL}"
  echo "**Results:** PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}"
  echo ""; echo "| # | Experiment | Status | Issue Addressed |"
  echo "|---|-----------|--------|----------------|"
  for exp in e00_wcet_large_spaced e01_iid_spacing_study e02_block_maxima_large \
             e03_nonevt_wcet e04_accuracy_1000 e05_tau_crossval \
             e06_router_comparison e07_thermal_extended e08_capacity_full e09_final_report; do
    echo "| - | ${exp} | ${STATUS[$exp]:-?} | - |"
  done
  echo ""; echo "## Timing"; echo ""
  for t in "${TIMES[@]:-}"; do echo "- $t"; done
} > "$SUMMARY"

log ""; log "═══════════════════════════════════════════════════════════"
log " PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}  TIME=${TOTAL}s"
log "═══════════════════════════════════════════════════════════"
[ "$FAILED" -eq 0 ] && exit 0 || exit 1
