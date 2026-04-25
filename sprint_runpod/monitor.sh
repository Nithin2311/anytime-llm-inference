#!/usr/bin/env bash
# monitor.sh — Re-attach to the running sprint session after SSH disconnect.
# Usage:  bash monitor.sh
set -euo pipefail

SESSION="sprint"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "No session '$SESSION' found."
    echo "Has the sprint finished, or hasn't started yet?"
    echo "  Start:   bash tmux_launch.sh"
    echo "  Status:  cat results/SPRINT_SUMMARY.md"
    exit 1
fi

echo "Re-attaching to session '$SESSION' ..."
echo "  Ctrl-B 0  -> orchestrator window"
echo "  Ctrl-B 1  -> GPU watch"
echo "  Ctrl-B 2  -> log tail"
echo "  Ctrl-B D  -> detach (sprint keeps running)"
echo ""
tmux attach -t "$SESSION"
