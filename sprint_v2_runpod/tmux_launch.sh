#!/usr/bin/env bash
# tmux_launch.sh — Launch a 3-window tmux session for sprint_v2.
#
# Window layout:
#   0 : orchestrator  — runs run_sprint.sh (main job)
#   1 : gpu-watch     — nvidia-smi watch loop (refreshes every 5s)
#   2 : log-tail      — tails the most recent experiment log
#
# Usage:
#   bash tmux_launch.sh           # new session
#   bash tmux_launch.sh --resume  # pass --resume to run_sprint.sh

set -euo pipefail
SESSION="sprint_v2"
RESUME_FLAG="${1:-}"
DIR="$(cd "$(dirname "$0")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists — attaching."
  tmux attach-session -t "$SESSION"
  exit 0
fi

echo "Creating tmux session '$SESSION' ..."

# Window 0: orchestrator
tmux new-session -d -s "$SESSION" -n "orchestrator" -c "$DIR"
tmux send-keys -t "$SESSION:orchestrator" \
  "bash run_sprint.sh $RESUME_FLAG 2>&1 | tee results/run.log; echo 'SPRINT DONE'" Enter

# Window 1: gpu-watch
tmux new-window -t "$SESSION" -n "gpu-watch" -c "$DIR"
tmux send-keys -t "$SESSION:gpu-watch" \
  "watch -n 5 'nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv'" Enter

# Window 2: log-tail
tmux new-window -t "$SESSION" -n "log-tail" -c "$DIR"
tmux send-keys -t "$SESSION:log-tail" \
  "echo 'Waiting for first experiment log...'; while true; do LATEST=\$(ls -t logs/*.log 2>/dev/null | head -1); if [ -n \"\$LATEST\" ]; then tail -f \"\$LATEST\"; fi; sleep 2; done" Enter

# Focus orchestrator
tmux select-window -t "$SESSION:orchestrator"

echo ""
echo "Session '$SESSION' created."
echo ""
echo "Windows:"
echo "  0: orchestrator  — main experiment loop"
echo "  1: gpu-watch     — GPU utilization"
echo "  2: log-tail      — live experiment output"
echo ""
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-B  D  (session keeps running)"
echo ""

tmux attach-session -t "$SESSION"
