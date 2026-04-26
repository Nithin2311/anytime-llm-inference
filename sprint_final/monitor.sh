#!/usr/bin/env bash
# monitor.sh — Re-attach to sprint_final tmux session.
# If the session is gone, shows last known status.

SESSION="sprint_final"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Attaching to '$SESSION' ..."
  tmux attach-session -t "$SESSION"
else
  echo "Session '$SESSION' not found."
  echo ""
  DIR="$(dirname "$0")"
  SUMMARY="$DIR/results/SPRINT_SUMMARY.md"
  RUN_LOG="$DIR/results/run.log"

  if [ -f "$SUMMARY" ]; then
    echo "Last sprint summary:"
    echo "────────────────────"
    cat "$SUMMARY"
  elif [ -f "$RUN_LOG" ]; then
    echo "Last 50 lines of run.log:"
    echo "────────────────────"
    tail -50 "$RUN_LOG"
  else
    echo "No logs found yet."
    echo "Start with: bash tmux_launch.sh"
  fi
fi
