#!/usr/bin/env bash
# run_all.sh — Master orchestrator for sprint_final experiments.
#
# Runs E00 → E13 sequentially with retry logic (up to 3 attempts per
# experiment, 30-second backoff between attempts). WiFi-resilient —
# keep alive inside tmux. Called by tmux_launch.sh and watchdog.sh.
#
# Usage:
#   bash run_all.sh              # full run from scratch
#   bash run_all.sh --resume     # skip already-completed experiments
#   bash run_all.sh --only E06   # run a single experiment by name/tag
#   bash run_all.sh --dry        # print plan without executing

set -uo pipefail
cd "$(dirname "$0")"

RESULTS_DIR="results"
LOG_DIR="logs"
SUMMARY="$RESULTS_DIR/SPRINT_SUMMARY.md"
RESUME=0
ONLY=""
DRY=0
MAX_RETRIES=3
RETRY_SLEEP=30

for arg in "$@"; do
  case "$arg" in
    --resume)    RESUME=1 ;;
    --dry)       DRY=1 ;;
    --only=*)    ONLY="${arg#--only=}" ;;
    --only)      shift; ONLY="${1:-}" ;;
  esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

SPRINT_START=$(date +%s)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "═══════════════════════════════════════════════════════════"
echo " Sprint Final  —  Full Experiment Run"
echo " Started: $(date)"
echo " GPU: ${GPU_NAME}"
echo " RESUME=${RESUME}  ONLY=${ONLY:-all}  DRY=${DRY}"
echo "═══════════════════════════════════════════════════════════"

declare -A STATUS
declare -A ATTEMPTS
PASSED=0
FAILED=0
SKIPPED=0
declare -a TIMES

# ── Per-experiment runner with retry ─────────────────────────────────────────
run_experiment() {
  local name="$1"
  local desc="${2:-}"
  local done_flag="$RESULTS_DIR/.${name}.done"
  local log_file="$LOG_DIR/${name}.log"

  # --only filter: match full name, short tag (e.g. "e06"), number ("06","6"), or "E06"
  if [ -n "$ONLY" ]; then
    local tag="${name:1:2}"
    local short="${name%%_*}"
    if [ "$ONLY" != "$name" ] && [ "$ONLY" != "$tag" ] && \
       [ "$ONLY" != "$short" ] && [ "$ONLY" != "E${tag}" ] && \
       [ "$ONLY" != "e${tag}" ] && [ "$ONLY" != "${tag#0}" ]; then
      STATUS["$name"]="SKIP"
      ((SKIPPED++)) || true
      return 0
    fi
  fi

  # --resume: skip if .done flag exists
  if [ "$RESUME" -eq 1 ] && [ -f "$done_flag" ]; then
    echo "  [SKIP] ${name}  (already done)"
    STATUS["$name"]="SKIP"
    ((SKIPPED++)) || true
    return 0
  fi

  if [ "$DRY" -eq 1 ]; then
    echo "  [DRY]  ${name}  ${desc}"
    STATUS["$name"]="DRY"
    return 0
  fi

  echo ""
  echo "──────────────────────────────────────────────────────────"
  echo "  Running: ${name}"
  echo "  ${desc}"
  echo "  $(date)"

  local attempt=0
  local success=0
  local t0=$(date +%s)

  while [ $attempt -lt $MAX_RETRIES ]; do
    ((attempt++)) || true
    echo "  Attempt ${attempt}/${MAX_RETRIES} ..."

    if python3 "experiments/${name}.py" 2>&1 | tee "$log_file"; then
      success=1
      break
    else
      local exit_code=${PIPESTATUS[0]}
      echo "  Attempt ${attempt} failed (exit ${exit_code})"
      if [ $attempt -lt $MAX_RETRIES ]; then
        echo "  Retrying in ${RETRY_SLEEP}s  (GPU cache clear) ..."
        sleep "$RETRY_SLEEP"
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
      fi
    fi
  done

  local t1=$(date +%s)
  local elapsed=$((t1 - t0))

  if [ "$success" -eq 1 ]; then
    STATUS["$name"]="PASS"
    ATTEMPTS["$name"]=$attempt
    touch "$done_flag"
    TIMES+=("${name}: ${elapsed}s (attempt ${attempt})")
    ((PASSED++)) || true
    echo "  [PASS] ${name}  (${elapsed}s, attempt ${attempt})"
  else
    STATUS["$name"]="FAIL"
    ATTEMPTS["$name"]=$attempt
    ((FAILED++)) || true
    echo "  [FAIL] ${name}  after ${attempt} attempts — see ${log_file}"
  fi
}

# ── Run all experiments ──────────────────────────────────────────────────────
run_experiment "e00_wcet_reprofiling"          "EVT reprof 500 samples, warm-up protocol"
run_experiment "e01_evt_gev_ad"                "GEV fit + Anderson-Darling test on tail"
run_experiment "e02_threshold_crossval"        "τ cross-validation cal/holdout split"
run_experiment "e03_forced_exit_extended"      "Forced-exit quality sweep L5-L22"
run_experiment "e04_pot_sensitivity"           "POT fraction sensitivity 10%-40%"
run_experiment "e05_deadline_sweep_comparison" "Deadline sweep 14-60ms, 3 routers"
run_experiment "e06_accuracy_large"            "500-query PubMedQA accuracy + conf histogram"
run_experiment "e07_wcet_ci_gev"               "pWCET bootstrap CI, Gumbel vs GEV"
run_experiment "e08_sample_independence"       "Ljung-Box IID + ACF autocorrelation"
run_experiment "e09_capacity_empirical"        "Empirical capacity N=1..4 requests"
run_experiment "e10_tight_deadline"            "Tight deadline stress D=14-30ms"
run_experiment "e11_thermal_a100"              "A100 thermal stability 30-min soak"
run_experiment "e12_exit_head_training"        "Exit-head MLP training on L16 hiddens"
run_experiment "e13_dense_ablation"            "Dense layer ablation L12-L20"

# ── Consolidated plot generation ─────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────────"
echo "  Generating consolidated plots ..."
if python3 src/plot_runner.py --results-dir "$RESULTS_DIR" 2>&1 | tee "$LOG_DIR/plot_runner.log"; then
  echo "  [PASS] plot_runner"
else
  echo "  [WARN] plot_runner had errors — check logs/plot_runner.log"
fi

# ── Write summary ────────────────────────────────────────────────────────────
SPRINT_END=$(date +%s)
TOTAL_S=$((SPRINT_END - SPRINT_START))
GPU_FULL=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

{
  echo "# Sprint Final — Experiment Summary"
  echo ""
  echo "**Completed:** $(date)"
  echo "**Total time:** ${TOTAL_S}s"
  echo "**GPU:** ${GPU_FULL}"
  echo "**Results:** PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}"
  echo ""
  echo "## Experiment Results"
  echo ""
  echo "| # | Experiment | Status | Review Issue Addressed |"
  echo "|---|---|---|---|"
  echo "| E00 | e00_wcet_reprofiling | ${STATUS[e00_wcet_reprofiling]:-?} | C1: EVT re-baseline (500 samples) |"
  echo "| E01 | e01_evt_gev_ad | ${STATUS[e01_evt_gev_ad]:-?} | C1: GEV ξ validation + Anderson-Darling |"
  echo "| E02 | e02_threshold_crossval | ${STATUS[e02_threshold_crossval]:-?} | C2: τ generalization cal/holdout |"
  echo "| E03 | e03_forced_exit_extended | ${STATUS[e03_forced_exit_extended]:-?} | R1: forced exit quality L5-L22 |"
  echo "| E04 | e04_pot_sensitivity | ${STATUS[e04_pot_sensitivity]:-?} | C1: POT fraction robustness |"
  echo "| E05 | e05_deadline_sweep_comparison | ${STATUS[e05_deadline_sweep_comparison]:-?} | R2: router comparison, exit-rate confound |"
  echo "| E06 | e06_accuracy_large | ${STATUS[e06_accuracy_large]:-?} | M1: 500-query CI ≈ ±7pp + conf histogram |"
  echo "| E07 | e07_wcet_ci_gev | ${STATUS[e07_wcet_ci_gev]:-?} | M2: pWCET bootstrap CI Gumbel/GEV |"
  echo "| E08 | e08_sample_independence | ${STATUS[e08_sample_independence]:-?} | C1: Ljung-Box IID confirmation |"
  echo "| E09 | e09_capacity_empirical | ${STATUS[e09_capacity_empirical]:-?} | R2: empirical capacity N=1..4 |"
  echo "| E10 | e10_tight_deadline | ${STATUS[e10_tight_deadline]:-?} | R3: tight deadline 14-30ms |"
  echo "| E11 | e11_thermal_a100 | ${STATUS[e11_thermal_a100]:-?} | R3: thermal stability soak |"
  echo "| E12 | e12_exit_head_training | ${STATUS[e12_exit_head_training]:-?} | C2: exit-head MLP, exit rate ↑ |"
  echo "| E13 | e13_dense_ablation | ${STATUS[e13_dense_ablation]:-?} | C2: dense layer sweep L12-L20 |"
  echo ""
  echo "## Timing"
  echo ""
  for t in "${TIMES[@]}"; do
    echo "- $t"
  done
  echo ""
  echo "## Output Files"
  echo ""
  echo '```'
  ls -1 "$RESULTS_DIR"/*.json 2>/dev/null || true
  ls -1 "$RESULTS_DIR"/*.png  2>/dev/null || true
  ls -1 "$RESULTS_DIR"/*.tex  2>/dev/null || true
  echo '```'
} > "$SUMMARY"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Sprint Final complete"
echo " PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}  TIME=${TOTAL_S}s"
echo " Summary: ${SUMMARY}"
echo "═══════════════════════════════════════════════════════════"

[ "$FAILED" -eq 0 ] && exit 0 || exit 1
