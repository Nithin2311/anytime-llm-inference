#!/usr/bin/env bash
# monitor.sh — Re-attach to sprint_cr from anywhere.
SESSION="sprint_cr"
cd "$(dirname "$0")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Attaching to '${SESSION}' ..."
  tmux attach-session -t "$SESSION"
else
  echo "Session '${SESSION}' not found."
  echo ""
  for f in "results/SPRINT_SUMMARY.md" "results/run.log"; do
    [ -f "$f" ] && { echo "Last output ($f):"; echo "────────────────────"; tail -40 "$f"; break; }
  done
  [ -f "results/watchdog.log" ] && { echo ""; echo "Watchdog (last 10):"; tail -10 results/watchdog.log; }
  echo ""; echo "To restart: bash tmux_launch.sh --resume"
fi
