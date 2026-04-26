#!/usr/bin/env bash
# tmux_launch.sh — Launch a 5-window tmux session for sprint_final.
#
# WiFi-resilient: the session persists on the RunPod instance even if
# your SSH/browser connection drops. Re-attach from anywhere with:
#   tmux attach -t sprint_final   OR   bash monitor.sh
#
# Windows:
#   0 : orchestrator  — run_all.sh (main job)
#   1 : gpu-watch     — nvidia-smi live stats every 5s
#   2 : log-tail      — tails the current experiment log
#   3 : error-watch   — scans all logs for errors every 30s
#   4 : shell         — interactive debugging REPL
#
# Usage:
#   bash tmux_launch.sh              # full run from scratch
#   bash tmux_launch.sh --resume     # skip .done experiments
#   bash tmux_launch.sh --only E06   # run one experiment

set -euo pipefail
SESSION="sprint_final"
RESUME_FLAG="${1:-}"
DIR="$(cd "$(dirname "$0")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists — attaching."
  tmux attach-session -t "$SESSION"
  exit 0
fi

echo "Creating tmux session '$SESSION' in $DIR ..."

# Window 0: orchestrator
tmux new-session -d -s "$SESSION" -n "orchestrator" -c "$DIR"
tmux send-keys -t "$SESSION:orchestrator" \
  "echo 'Sprint Final — Orchestrator'; bash run_all.sh $RESUME_FLAG 2>&1 | tee results/run.log; echo ''; echo 'SPRINT DONE — '$(date)" Enter

# Window 1: gpu-watch
tmux new-window -t "$SESSION" -n "gpu-watch" -c "$DIR"
tmux send-keys -t "$SESSION:gpu-watch" \
  "watch -n 5 'printf \"=== GPU Stats %s ===\n\" \$(date +%H:%M:%S); nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits'" Enter

# Window 2: log-tail — follows the latest experiment log automatically
tmux new-window -t "$SESSION" -n "log-tail" -c "$DIR"
tmux send-keys -t "$SESSION:log-tail" \
  "echo 'Waiting for first experiment log...'; while true; do LATEST=\$(ls -t logs/*.log 2>/dev/null | grep -v plot_runner | head -1 || true); if [ -n \"\$LATEST\" ]; then echo \"=== Tailing: \$LATEST ===\"; tail -n 50 -f \"\$LATEST\" & TPID=\$!; PREV=\$LATEST; while [ \"\$(ls -t logs/*.log 2>/dev/null | grep -v plot_runner | head -1 || true)\" = \"\$PREV\" ]; do sleep 3; done; kill \$TPID 2>/dev/null || true; fi; sleep 2; done" Enter

# Window 3: error-watch — periodic error scan
tmux new-window -t "$SESSION" -n "error-watch" -c "$DIR"
tmux send-keys -t "$SESSION:error-watch" \
  "echo 'Error monitor active (scans every 30s)'; while true; do echo \"=== \$(date +%H:%M:%S) scan ===\"; grep -n 'ERROR\|Traceback\|RuntimeError\|CUDA out of mem\|KILLED\|\[FAIL\]' logs/*.log 2>/dev/null | tail -20 || echo '  no errors'; sleep 30; done" Enter

# Window 4: interactive shell
tmux new-window -t "$SESSION" -n "shell" -c "$DIR"
tmux send-keys -t "$SESSION:shell" \
  "export PYTHONPATH=\"$DIR/src:\${PYTHONPATH:-}\"; echo 'Interactive shell ready'; echo 'PYTHONPATH includes: $DIR/src'; bash" Enter

tmux select-window -t "$SESSION:orchestrator"

echo ""
echo "════════════════════════════════════════════════════════"
echo " Session '$SESSION' is live"
echo " Windows:  0=orchestrator  1=gpu-watch  2=log-tail"
echo "           3=error-watch   4=shell"
echo " Navigate: Ctrl-B <num>"
echo " Detach:   Ctrl-B D   (job keeps running)"
echo " Attach:   tmux attach -t $SESSION  OR  bash monitor.sh"
echo "════════════════════════════════════════════════════════"
echo ""

tmux attach-session -t "$SESSION"
