#!/usr/bin/env bash
# tmux_launch.sh — WiFi-resilient launcher. Creates tmux session 'sprint_cr'.
# Session survives SSH disconnects. Re-attach with: bash monitor.sh
#
# Usage:
#   bash tmux_launch.sh           # fresh start
#   bash tmux_launch.sh --resume  # restart skipping completed experiments

set -euo pipefail
cd "$(dirname "$0")"

SESSION="sprint_cr"
RESUME_FLAG="${1:-}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '${SESSION}' already running."
  echo "  Attach : bash monitor.sh"
  echo "  Kill   : tmux kill-session -t ${SESSION}"
  exit 0
fi

SPRINT_DIR="$(pwd)"
echo "Starting tmux session '${SESSION}' ..."
echo "  Dir  : ${SPRINT_DIR}"
echo "  Args : ${RESUME_FLAG:-none}"

tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux rename-window -t "${SESSION}:0" "sprint"

tmux send-keys -t "$SESSION" "cd '${SPRINT_DIR}'" Enter
tmux send-keys -t "$SESSION" "export PYTHONPATH='${SPRINT_DIR}/src:\${PYTHONPATH:-}'" Enter
tmux send-keys -t "$SESSION" "source ~/.bashrc 2>/dev/null || true" Enter

if [ "$RESUME_FLAG" = "--resume" ]; then
  tmux send-keys -t "$SESSION" "bash run_all.sh --resume 2>&1 | tee -a results/run.log" Enter
else
  tmux send-keys -t "$SESSION" "bash run_all.sh 2>&1 | tee -a results/run.log" Enter
fi

echo ""
echo "Session '${SESSION}' started."
echo "  Attach : tmux attach-session -t ${SESSION}  (or: bash monitor.sh)"
echo "  Detach : Ctrl-B then D"
echo "  Log    : tail -f results/run.log"
