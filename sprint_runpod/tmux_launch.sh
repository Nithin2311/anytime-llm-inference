#!/usr/bin/env bash
# tmux_launch.sh — Create the "sprint" tmux session and start all monitors.
# Run once after setup.sh:  bash tmux_launch.sh
#
# Session layout:
#   window 0 [orchestrator] — runs run_sprint.sh (E0→E7 sequentially)
#   window 1 [gpu-watch]    — watch -n2 nvidia-smi (live GPU utilization)
#   window 2 [log-tail]     — tail -f results/sprint.log
#
# To re-attach after disconnect:  bash monitor.sh
set -euo pipefail

SESSION="sprint"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "Re-attaching ... (Ctrl-B D to detach again)"
    tmux attach -t "$SESSION"
    exit 0
fi

echo "Creating tmux session '$SESSION' ..."

# Window 0: orchestrator
tmux new-session  -d -s "$SESSION" -n "orchestrator" -c "$SCRIPT_DIR"
tmux send-keys    -t "$SESSION:0" "bash run_sprint.sh 2>&1 | tee -a results/sprint.log" Enter

# Window 1: GPU watch
tmux new-window   -t "$SESSION" -n "gpu-watch" -c "$SCRIPT_DIR"
tmux send-keys    -t "$SESSION:1" "watch -n2 nvidia-smi" Enter

# Window 2: log tail
tmux new-window   -t "$SESSION" -n "log-tail" -c "$SCRIPT_DIR"
tmux send-keys    -t "$SESSION:2" "mkdir -p results && tail -f results/sprint.log" Enter

# Focus on orchestrator
tmux select-window -t "$SESSION:0"

echo ""
echo "Session '$SESSION' started with 3 windows:"
echo "  0: orchestrator  (run_sprint.sh)"
echo "  1: gpu-watch     (nvidia-smi)"
echo "  2: log-tail      (sprint.log)"
echo ""
echo "Attaching now ... (Ctrl-B D to detach, bash monitor.sh to re-attach)"
tmux attach -t "$SESSION"
