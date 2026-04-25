#!/usr/bin/env bash
# run_sprint.sh — Master orchestrator for the 5-day academic sprint.
# Runs E0→E7 sequentially, logs pass/fail, writes SPRINT_SUMMARY.md.
# Designed to run inside tmux window 0; survives SSH disconnect.
#
# Usage:  bash run_sprint.sh
#   or    bash run_sprint.sh --resume   (skip already-completed experiments)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUMMARY_FILE="results/SPRINT_SUMMARY.md"
LOG_FILE="results/sprint.log"
mkdir -p results figures latex

RESUME=0
[[ "${1:-}" == "--resume" ]] && RESUME=1

EXPERIMENTS=(
    "E0:experiments/e0_wcet_profile.py"
    "E1:experiments/e1_evt_reprofiling.py"
    "E2:experiments/e2_threshold_ablation.py"
    "E3:experiments/e3_forced_exit_quality.py"
    "E4:experiments/e4_pot_sensitivity.py"
    "E5:experiments/e5_deadline_sweep_ext.py"
    "E6:experiments/e6_accuracy_ci.py"
    "E7:experiments/e7_wcet_ci.py"
)

RESULT_FILES=(
    "wcet_results.json"
    "evt_wcet_results.json"
    "threshold_ablation_results.json"
    "forced_exit_quality_results.json"
    "pot_sensitivity_results.json"
    "deadline_sweep_ext_results.json"
    "accuracy_ci_results.json"
    "wcet_ci_results.json"
)

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ── Header ──────────────────────────────────────────────────────────────────
{
echo "# Sprint Summary"
echo ""
echo "**Started:** $(date '+%Y-%m-%d %H:%M:%S')"
echo "**Hardware:** A100 SXM4"
echo "**Experiments:** E0–E7 (8 total)"
echo ""
echo "| Exp | Status | Elapsed | Output |"
echo "|-----|--------|---------|--------|"
} > "$SUMMARY_FILE"

log "===== SPRINT START ====="
log "Hardware: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'unknown')"

# ── Run each experiment ──────────────────────────────────────────────────────
for i in "${!EXPERIMENTS[@]}"; do
    entry="${EXPERIMENTS[$i]}"
    eid="${entry%%:*}"
    script="${entry##*:}"
    result_file="results/${RESULT_FILES[$i]}"

    if [[ $RESUME -eq 1 && -f "$result_file" ]]; then
        log "[$eid] SKIP (result exists)"
        echo "| $eid | SKIP | -- | $result_file |" >> "$SUMMARY_FILE"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    log "[$eid] START → $script"
    t_start=$(date +%s)

    if python "$script" 2>&1 | tee -a "$LOG_FILE"; then
        t_end=$(date +%s)
        elapsed=$(( t_end - t_start ))
        elapsed_fmt=$(printf '%dh %dm %ds' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60)))
        log "[$eid] PASS ($elapsed_fmt)"
        echo "| $eid | **PASS** | $elapsed_fmt | $result_file |" >> "$SUMMARY_FILE"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        t_end=$(date +%s)
        elapsed=$(( t_end - t_start ))
        elapsed_fmt=$(printf '%dh %dm %ds' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60)))
        log "[$eid] FAIL ($elapsed_fmt) — check $LOG_FILE"
        echo "| $eid | **FAIL** | $elapsed_fmt | -- |" >> "$SUMMARY_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        # Continue with remaining experiments rather than aborting
    fi
done

# ── Footer ───────────────────────────────────────────────────────────────────
{
echo ""
echo "**Completed:** $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "**Results:** PASS=$PASS_COUNT  FAIL=$FAIL_COUNT  SKIP=$SKIP_COUNT"
echo ""
if [[ $FAIL_COUNT -eq 0 ]]; then
    echo "**Status: ALL EXPERIMENTS PASSED**"
else
    echo "**Status: $FAIL_COUNT EXPERIMENT(S) FAILED — review sprint.log**"
fi
echo ""
echo "## Output Files"
echo ""
echo "\`\`\`"
ls -lh results/*.json 2>/dev/null || echo "(no JSON results yet)"
echo "\`\`\`"
echo ""
echo "\`\`\`"
ls -lh figures/*.png 2>/dev/null || echo "(no figures yet)"
echo "\`\`\`"
echo ""
echo "\`\`\`"
ls -lh latex/*.tex 2>/dev/null || echo "(no LaTeX snippets yet)"
echo "\`\`\`"
} >> "$SUMMARY_FILE"

log "===== SPRINT END ====="
log "PASS=$PASS_COUNT  FAIL=$FAIL_COUNT  SKIP=$SKIP_COUNT"
log "Summary → $SUMMARY_FILE"

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
