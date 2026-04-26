#!/usr/bin/env bash
# tmux_launch.sh — WiFi-resilient tmux session for sprint_v3_runpod.
# Session persists on RunPod even after SSH disconnect.
#
# Windows:
#   0: orchestrator   — runs run_all.sh
#   1: gpu-watch      — nvidia-smi every 5s
#   2: log-tail       — live tail of latest experiment log
#   3: shell          — free interactive shell

set -euo pipefail
SESSION="sprint_v3"
DIR="$(cd "$(dirname "$0")" && pwd)"
RESUME_FLAG="${1:-}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running — attaching."
  tmux attach-session -t "$SESSION"
  exit 0
fi

echo "Creating tmux session '$SESSION' ..."

# Window 0: orchestrator
tmux new-session -d -s "$SESSION" -n "orchestrator" -c "$DIR"
tmux send-keys -t "$SESSION:orchestrator" \
  "bash run_all.sh $RESUME_FLAG 2>&1 | tee results/run.log; echo 'SPRINT V3 DONE'" Enter

# Window 1: gpu-watch
tmux new-window -t "$SESSION" -n "gpu-watch" -c "$DIR"
tmux send-keys -t "$SESSION:gpu-watch" \
  "watch -n 5 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv'" Enter

# Window 2: log-tail
tmux new-window -t "$SESSION" -n "log-tail" -c "$DIR"
tmux send-keys -t "$SESSION:log-tail" \
  "echo 'Waiting for logs...'; while true; do LATEST=\$(ls -t logs/*.log 2>/dev/null | head -1); [ -n \"\$LATEST\" ] && tail -f \"\$LATEST\"; sleep 3; done" Enter

# Window 3: shell
tmux new-window -t "$SESSION" -n "shell" -c "$DIR"
tmux send-keys -t "$SESSION:shell" "export PYTHONPATH=$(pwd)/src:\$PYTHONPATH" Enter

tmux select-window -t "$SESSION:orchestrator"

echo ""
echo "Session '$SESSION' created."
echo "  0: orchestrator  — main experiment loop"
echo "  1: gpu-watch     — GPU metrics"
echo "  2: log-tail      — live log output"
echo "  3: shell         — interactive"
echo ""
echo "Attach:  tmux attach -t $SESSION"
echo "Detach:  Ctrl-B  D  (session persists)"
echo ""

tmux attach-session -t "$SESSION"
