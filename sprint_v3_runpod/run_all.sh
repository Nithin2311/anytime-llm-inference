#!/usr/bin/env bash
# run_all.sh — Orchestrator for sprint_v3_runpod (IID + GEV resolution sprint).
#
# E00: Spaced profiling (200ms sleep, 50 warm-up, 500 runs)   ~35 min
# E01: Ljung-Box IID validation on spaced data                 ~2 min
# E02: Outlier/warm-up artifact study                          ~5 min
# E03: GEV xi refit on spaced data                             ~5 min
# E04: Block maxima EVT at multiple block sizes                ~8 min
# E05: Final pWCET report (decision-driven)                    ~3 min
# Total estimated: ~60 min on A100 SXM4
#
# Usage:
#   bash run_all.sh              # full run
#   bash run_all.sh --resume     # skip completed experiments
#   bash run_all.sh --only E03   # single experiment
#   bash run_all.sh --dry        # print plan without running

set -uo pipefail
cd "$(dirname "$0")"

RESULTS_DIR="results"
LOG_DIR="logs"
RESUME=0
ONLY=""
DRY=0
MAX_RETRIES=3
RETRY_SLEEP=30

for arg in "$@"; do
  case "$arg" in
    --resume)  RESUME=1 ;;
    --dry)     DRY=1 ;;
    --only=*)  ONLY="${arg#--only=}" ;;
    --only)    shift; ONLY="${1:-}" ;;
  esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

SPRINT_START=$(date +%s)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

echo "═══════════════════════════════════════════════════════════"
echo " sprint_v3_runpod — IID + GEV Resolution Sprint"
echo " Started: $(date)"
echo " GPU: ${GPU_NAME}"
echo " RESUME=${RESUME}  ONLY=${ONLY:-all}  DRY=${DRY}"
echo "═══════════════════════════════════════════════════════════"

declare -A STATUS
PASSED=0; FAILED=0; SKIPPED=0
declare -a TIMES

run_experiment() {
  local name="$1"
  local desc="${2:-}"
  local done_flag="$RESULTS_DIR/.${name}.done"
  local log_file="$LOG_DIR/${name}.log"

  if [ -n "$ONLY" ]; then
    local tag="${name:1:2}"
    local short="${name%%_*}"
    if [ "$ONLY" != "$name" ] && [ "$ONLY" != "$tag" ] && \
       [ "$ONLY" != "$short" ] && [ "$ONLY" != "E${tag}" ] && \
       [ "$ONLY" != "e${tag}" ] && [ "$ONLY" != "${tag#0}" ]; then
      STATUS["$name"]="SKIP"; ((SKIPPED++)) || true; return 0
    fi
  fi

  if [ "$RESUME" -eq 1 ] && [ -f "$done_flag" ]; then
    echo "  [SKIP] ${name}  (already done)"
    STATUS["$name"]="SKIP"; ((SKIPPED++)) || true; return 0
  fi

  if [ "$DRY" -eq 1 ]; then
    echo "  [DRY]  ${name}  ${desc}"
    STATUS["$name"]="DRY"; return 0
  fi

  echo ""
  echo "──────────────────────────────────────────────────────────"
  echo "  Running: ${name}"
  echo "  ${desc}"
  echo "  $(date)"

  local attempt=0 success=0
  local t0=$(date +%s)

  while [ $attempt -lt $MAX_RETRIES ]; do
    ((attempt++)) || true
    echo "  Attempt ${attempt}/${MAX_RETRIES} ..."
    if python3 "experiments/${name}.py" 2>&1 | tee "$log_file"; then
      success=1; break
    else
      local exit_code=${PIPESTATUS[0]}
      echo "  Attempt ${attempt} failed (exit ${exit_code})"
      if [ $attempt -lt $MAX_RETRIES ]; then
        echo "  Retrying in ${RETRY_SLEEP}s ..."
        sleep "$RETRY_SLEEP"
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
      fi
    fi
  done

  local t1=$(date +%s)
  local elapsed=$((t1 - t0))

  if [ "$success" -eq 1 ]; then
    STATUS["$name"]="PASS"; touch "$done_flag"
    TIMES+=("${name}: ${elapsed}s (attempt ${attempt})")
    ((PASSED++)) || true
    echo "  [PASS] ${name}  (${elapsed}s)"
  else
    STATUS["$name"]="FAIL"
    ((FAILED++)) || true
    echo "  [FAIL] ${name}  after ${attempt} attempts — see ${log_file}"
  fi
}

# ── Experiments ──────────────────────────────────────────────────────────────
run_experiment "e00_spaced_profiling"   "Re-profile 12 cells with 200ms sleep, 50 warm-up"
run_experiment "e01_iid_validation"     "Ljung-Box IID check on spaced data vs reference"
run_experiment "e02_outlier_warmup_study" "Cold-start vs warm latency — warm-up artifact study"
run_experiment "e03_gev_xi_refit"       "GEV xi refit on spaced data, Anderson-Darling"
run_experiment "e04_block_maxima_pwcet" "Block maxima EVT, block sizes 5-50, pWCET"
run_experiment "e05_final_pwcet_report" "Decision-driven final pWCET table + LaTeX"

# ── Summary ──────────────────────────────────────────────────────────────────
SPRINT_END=$(date +%s)
TOTAL_S=$((SPRINT_END - SPRINT_START))
GPU_FULL=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
SUMMARY="$RESULTS_DIR/SPRINT_V3_SUMMARY.md"

{
  echo "# Sprint V3 — IID + GEV Resolution Summary"
  echo ""
  echo "**Completed:** $(date)"
  echo "**Total time:** ${TOTAL_S}s"
  echo "**GPU:** ${GPU_FULL}"
  echo "**Results:** PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}"
  echo ""
  echo "## Experiment Results"
  echo ""
  echo "| # | Experiment | Status | Resolves |"
  echo "|---|---|---|---|"
  echo "| E00 | e00_spaced_profiling | ${STATUS[e00_spaced_profiling]:-?} | Protocol: 200ms sleep, 50 warm-up |"
  echo "| E01 | e01_iid_validation | ${STATUS[e01_iid_validation]:-?} | IID: Ljung-Box on spaced data |"
  echo "| E02 | e02_outlier_warmup | ${STATUS[e02_outlier_warmup]:-?} | GEV: warm-up artifact diagnosis |"
  echo "| E03 | e03_gev_xi_refit | ${STATUS[e03_gev_xi_refit]:-?} | GEV: xi refit after spacing |"
  echo "| E04 | e04_block_maxima_pwcet | ${STATUS[e04_block_maxima_pwcet]:-?} | GEV: block maxima fallback |"
  echo "| E05 | e05_final_pwcet_report | ${STATUS[e05_final_pwcet_report]:-?} | Final: decision-driven pWCET table |"
  echo ""
  echo "## Timing"
  echo ""
  for t in "${TIMES[@]}"; do echo "- $t"; done
  echo ""
  echo "## Output Files"
  echo ""
  echo '```'
  ls -1 "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.png "$RESULTS_DIR"/*.tex 2>/dev/null || true
  echo '```'
} > "$SUMMARY"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Sprint V3 complete"
echo " PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}  TIME=${TOTAL_S}s"
echo " Summary: ${SUMMARY}"
echo "═══════════════════════════════════════════════════════════"

[ "$FAILED" -eq 0 ] && exit 0 || exit 1
