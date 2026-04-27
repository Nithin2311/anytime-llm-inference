#!/usr/bin/env bash
# tmux_launch.sh — WiFi-resilient launcher. Creates tmux session 'sprint_e01'.
# Session survives SSH/WiFi disconnects. Re-attach: tmux attach-session -t sprint_e01
#
# Usage:
#   bash tmux_launch.sh              # fresh start
#   bash tmux_launch.sh --resume     # skip completed experiments
#   bash tmux_launch.sh --skip-push  # run without GitHub push

set -euo pipefail
cd "$(dirname "$0")"

SESSION="sprint_e01"
EXTRA_ARGS=""

for arg in "$@"; do
  case "$arg" in
    --resume)     EXTRA_ARGS="$EXTRA_ARGS --resume" ;;
    --skip-push)  EXTRA_ARGS="$EXTRA_ARGS --skip-push" ;;
    --dry)        EXTRA_ARGS="$EXTRA_ARGS --dry" ;;
  esac
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '${SESSION}' is already running."
  echo "  Attach : tmux attach-session -t ${SESSION}"
  echo "  Kill   : tmux kill-session -t ${SESSION}"
  exit 0
fi

SPRINT_DIR="$(pwd)"
echo "Starting tmux session '${SESSION}' ..."
echo "  Dir  : ${SPRINT_DIR}"
echo "  Args : ${EXTRA_ARGS:-none}"
echo "  Exps : E01 E02 E03 E04 E05 E06 E07 E08  (~9 hours total)"

tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux rename-window -t "${SESSION}:0" "e01-e08"

tmux send-keys -t "$SESSION" "cd '${SPRINT_DIR}'" Enter
tmux send-keys -t "$SESSION" "export PYTHONPATH='${SPRINT_DIR}/src:\${PYTHONPATH:-}'" Enter
tmux send-keys -t "$SESSION" "source ~/.bashrc 2>/dev/null || true" Enter
tmux send-keys -t "$SESSION" "bash run_all.sh${EXTRA_ARGS} 2>&1 | tee -a results/run.log" Enter

echo ""
echo "Session '${SESSION}' started."
echo "  Attach : tmux attach-session -t ${SESSION}"
echo "  Detach : Ctrl-B then D"
echo "  Log    : tail -f results/run.log"
echo "  Watch  : bash watchdog.sh &"
