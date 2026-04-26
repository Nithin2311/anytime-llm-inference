#!/usr/bin/env bash
# monitor.sh — Re-attach to the sprint_v2 tmux session from anywhere.
# If the session is gone, shows last known status from results/run.log.

SESSION="sprint_v2"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Attaching to '$SESSION' ..."
  tmux attach-session -t "$SESSION"
else
  echo "Session '$SESSION' not found."
  echo ""
  RUN_LOG="$(dirname "$0")/results/run.log"
  SUMMARY="$(dirname "$0")/results/SPRINT_SUMMARY.md"

  if [ -f "$SUMMARY" ]; then
    echo "Last sprint summary:"
    echo "────────────────────"
    cat "$SUMMARY"
  elif [ -f "$RUN_LOG" ]; then
    echo "Last 40 lines of run.log:"
    echo "────────────────────"
    tail -40 "$RUN_LOG"
  else
    echo "No run log found. Has the sprint started?"
    echo "Run: bash tmux_launch.sh"
  fi
fi
