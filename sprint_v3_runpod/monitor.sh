#!/usr/bin/env bash
# monitor.sh — Re-attach to sprint_v3 tmux session.
SESSION="sprint_v3"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Attaching to '$SESSION' ..."
  tmux attach-session -t "$SESSION"
else
  echo "Session '$SESSION' not found."
  SUMMARY="$(dirname "$0")/results/SPRINT_V3_SUMMARY.md"
  [ -f "$SUMMARY" ] && cat "$SUMMARY" || echo "No summary yet. Run: bash tmux_launch.sh"
fi
