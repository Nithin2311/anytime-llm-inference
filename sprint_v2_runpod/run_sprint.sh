#!/usr/bin/env bash
# run_sprint.sh — Master orchestrator for sprint_v2 experiments.
#
# Runs E00 → E13 in sequence, writes SPRINT_SUMMARY.md on completion.
# Supports --resume to skip already-completed experiments.
#
# Usage:
#   bash run_sprint.sh              # full run
#   bash run_sprint.sh --resume     # skip completed experiments
#   bash run_sprint.sh --only E06   # run a single experiment

set -euo pipefail
cd "$(dirname "$0")"

RESULTS_DIR="results"
LOG_DIR="logs"
SUMMARY="$RESULTS_DIR/SPRINT_SUMMARY.md"
RESUME=0
ONLY=""

for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=1 ;;
    --only)   shift; ONLY="$1" ;;
    --only=*) ONLY="${arg#--only=}" ;;
  esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

SPRINT_START=$(date +%s)
echo "═══════════════════════════════════════════════════════════"
echo " Sprint V2  —  Full Experiment Run"
echo " Started: $(date)"
echo " RESUME=${RESUME}  ONLY=${ONLY:-all}"
echo "═══════════════════════════════════════════════════════════"

EXPERIMENTS=(
  "e00_wcet_reprofiling"
  "e01_evt_gev_ad"
  "e02_threshold_crossval"
  "e03_forced_exit_extended"
  "e04_pot_sensitivity"
  "e05_deadline_sweep_comparison"
  "e06_accuracy_large"
  "e07_wcet_ci_gev"
  "e08_sample_independence"
  "e09_capacity_empirical"
  "e10_tight_deadline"
  "e11_thermal_a100"
  "e12_exit_head_training"
  "e13_dense_ablation"
)

declare -A STATUS
PASSED=0
FAILED=0
SKIPPED=0
TIMES=()

run_experiment() {
  local name="$1"
  local exp_file="experiments/${name}.py"
  local done_flag="$RESULTS_DIR/.${name}.done"
  local log_file="$LOG_DIR/${name}.log"

  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && [ "$ONLY" != "${name:1:2}" ]; then
    STATUS["$name"]="SKIP"
    ((SKIPPED++)) || true
    return 0
  fi

  if [ "$RESUME" -eq 1 ] && [ -f "$done_flag" ]; then
    echo "  [SKIP] ${name} (already done)"
    STATUS["$name"]="SKIP"
    ((SKIPPED++)) || true
    return 0
  fi

  echo ""
  echo "──────────────────────────────────────────────────────────"
  echo "  Running: ${name}"
  echo "  $(date)"
  local t0=$(date +%s)

  if python3 "experiments/${name}.py" 2>&1 | tee "$log_file"; then
    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    STATUS["$name"]="PASS"
    touch "$done_flag"
    TIMES+=("${name}: ${elapsed}s")
    ((PASSED++)) || true
    echo "  [PASS] ${name}  (${elapsed}s)"
  else
    STATUS["$name"]="FAIL"
    ((FAILED++)) || true
    echo "  [FAIL] ${name}  — see $log_file"
    # Don't exit; continue to next experiment
  fi
}

for exp in "${EXPERIMENTS[@]}"; do
  run_experiment "$exp"
done

# ── Generate all plots ──────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────────"
echo "  Generating plots ..."
if python3 src/plot_runner.py --results-dir "$RESULTS_DIR"; then
  echo "  [PASS] plot_runner"
else
  echo "  [WARN] plot_runner had errors (non-fatal)"
fi

# ── Write summary ────────────────────────────────────────────────────────────
SPRINT_END=$(date +%s)
TOTAL_S=$((SPRINT_END - SPRINT_START))

{
  echo "# Sprint V2 — Experiment Summary"
  echo ""
  echo "**Completed:** $(date)"
  echo "**Total time:** ${TOTAL_S}s"
  echo "**Results:** PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}"
  echo ""
  echo "## Experiment Results"
  echo ""
  echo "| Experiment | Status |"
  echo "|---|---|"
  for exp in "${EXPERIMENTS[@]}"; do
    echo "| ${exp} | ${STATUS[$exp]:-UNKNOWN} |"
  done
  echo ""
  echo "## Per-Experiment Times"
  echo ""
  for t in "${TIMES[@]}"; do
    echo "- $t"
  done
  echo ""
  echo "## Files Generated"
  echo ""
  echo '```'
  ls -1 "$RESULTS_DIR"/*.json 2>/dev/null || true
  ls -1 "$RESULTS_DIR"/*.png  2>/dev/null || true
  ls -1 "$RESULTS_DIR"/*.tex  2>/dev/null || true
  echo '```'
} > "$SUMMARY"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Sprint V2 complete"
echo " PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}  TIME=${TOTAL_S}s"
echo " Summary: ${SUMMARY}"
echo "═══════════════════════════════════════════════════════════"

[ "$FAILED" -eq 0 ] && exit 0 || exit 1
