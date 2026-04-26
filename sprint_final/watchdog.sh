#!/usr/bin/env bash
# watchdog.sh — Auto-restart run_all.sh if it dies unexpectedly.
#
# Run this in a SEPARATE tmux window or as a background process.
# It polls the orchestrator window every 60s and restarts with --resume
# if the job process has exited. This ensures recovery from transient
# OOM errors, driver crashes, or NFS timeouts.
#
# Usage:
#   bash watchdog.sh &          # background
#   # or add to tmux window 5

set -uo pipefail
cd "$(dirname "$0")"
SESSION="sprint_final"
CHECK_INTERVAL=60
MAX_RESTARTS=5
LOG="results/watchdog.log"
mkdir -p results

restarts=0
echo "[watchdog] Started $(date)" | tee -a "$LOG"
echo "[watchdog] Monitoring session '$SESSION', checking every ${CHECK_INTERVAL}s" | tee -a "$LOG"

while true; do
  sleep "$CHECK_INTERVAL"

  # Check if sprint is already complete
  if [ -f "results/SPRINT_SUMMARY.md" ]; then
    if grep -q "PASS=" "results/SPRINT_SUMMARY.md" 2>/dev/null; then
      echo "[watchdog] Sprint summary found — job complete. Exiting watchdog." | tee -a "$LOG"
      exit 0
    fi
  fi

  # Check if orchestrator window is still running a process
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[watchdog] $(date): Session '$SESSION' not found!" | tee -a "$LOG"
    if [ $restarts -ge $MAX_RESTARTS ]; then
      echo "[watchdog] Max restarts ($MAX_RESTARTS) reached. Giving up." | tee -a "$LOG"
      exit 1
    fi
    ((restarts++)) || true
    echo "[watchdog] Restart #${restarts} — relaunching session ..." | tee -a "$LOG"
    bash tmux_launch.sh --resume >> "$LOG" 2>&1 &
    sleep 10
    continue
  fi

  # Check if run_all.sh is still alive in the orchestrator window
  ORCH_PID=$(tmux list-panes -t "$SESSION:orchestrator" -F "#{pane_pid}" 2>/dev/null | head -1 || echo "")
  if [ -n "$ORCH_PID" ]; then
    CHILDREN=$(pgrep -P "$ORCH_PID" 2>/dev/null | wc -l || echo "0")
    if [ "$CHILDREN" -eq 0 ]; then
      # Orchestrator pane has no child processes — job may have finished or died
      PANE_CONTENT=$(tmux capture-pane -t "$SESSION:orchestrator" -p 2>/dev/null | tail -5 || echo "")
      if echo "$PANE_CONTENT" | grep -q "SPRINT DONE\|Sprint Final complete"; then
        echo "[watchdog] $(date): Sprint appears complete. Stopping watchdog." | tee -a "$LOG"
        exit 0
      fi
      echo "[watchdog] $(date): Orchestrator idle — restarting with --resume ..." | tee -a "$LOG"
      if [ $restarts -ge $MAX_RESTARTS ]; then
        echo "[watchdog] Max restarts ($MAX_RESTARTS) reached. Giving up." | tee -a "$LOG"
        exit 1
      fi
      ((restarts++)) || true
      tmux send-keys -t "$SESSION:orchestrator" \
        "bash run_all.sh --resume 2>&1 | tee -a results/run.log" Enter
      echo "[watchdog] Restart #${restarts} sent." | tee -a "$LOG"
    fi
  fi
done
