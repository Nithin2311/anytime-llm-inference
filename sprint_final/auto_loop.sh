#!/usr/bin/env bash
# auto_loop.sh — Re-run run_all.sh --resume until FAIL=0 or hard cap reached.
# Emits clear markers between iterations so the agent's monitor can react.
set -uo pipefail
cd "$(dirname "$0")"
MAX_ITERS="${MAX_ITERS:-30}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-20}"

for i in $(seq 1 "$MAX_ITERS"); do
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo " AUTO-LOOP iteration $i / $MAX_ITERS — $(date)"
  echo "════════════════════════════════════════════════════════════"

  bash run_all.sh --resume 2>&1 | tee -a results/run.log

  if [ -f results/SPRINT_SUMMARY.md ]; then
    line=$(grep -E "^\*\*Results:" results/SPRINT_SUMMARY.md | head -1)
    echo "AUTO-LOOP-ITER-DONE: iter=$i $line"
    fail=$(echo "$line" | sed -n 's/.*FAIL=\([0-9]*\).*/\1/p')
    if [ "${fail:-1}" = "0" ]; then
      echo "AUTO-LOOP-ALL-PASS: iter=$i"
      break
    fi
  else
    echo "AUTO-LOOP-ITER-DONE: iter=$i (no summary file)"
  fi

  echo "AUTO-LOOP sleeping ${SLEEP_BETWEEN}s before next iteration ..."
  sleep "$SLEEP_BETWEEN"
done

echo "AUTO-LOOP-EXIT: $(date)"
