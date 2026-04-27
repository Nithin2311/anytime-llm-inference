#!/usr/bin/env bash
# watchdog.sh — Polls sprint_e01 tmux session every 2 min; auto-restarts on crash.
# Run in background: bash watchdog.sh &
# Stop: kill $(cat results/watchdog.pid)

set -uo pipefail
cd "$(dirname "$0")"

SESSION="sprint_e01"
WLOG="results/watchdog.log"
POLL_S=120
RESTART_WAIT=60
TOTAL_EXPERIMENTS=8

mkdir -p results
echo $$ > results/watchdog.pid

wlog() { echo "$(date '+%Y-%m-%dT%H:%M:%S') [WATCHDOG] $*" | tee -a "$WLOG"; }

wlog "Watchdog started (PID=$$, session=${SESSION}, poll=${POLL_S}s, total=${TOTAL_EXPERIMENTS})"

while true; do
  sleep "$POLL_S"

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    DONE_COUNT=$(ls results/.*.done 2>/dev/null | wc -l || echo 0)
    wlog "OK  temp=${GPU_TEMP}C  util=${GPU_UTIL}  done=${DONE_COUNT}/${TOTAL_EXPERIMENTS}"

    if [ "$DONE_COUNT" -ge "$TOTAL_EXPERIMENTS" ]; then
      wlog "All ${TOTAL_EXPERIMENTS} experiments complete — watchdog exiting."
      rm -f results/watchdog.pid; exit 0
    fi
  else
    wlog "Session '${SESSION}' is DEAD — initiating restart ..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null && \
      wlog "GPU cache cleared." || wlog "GPU cache clear failed (non-fatal)."

    DONE_COUNT=$(ls results/.*.done 2>/dev/null | wc -l || echo 0)
    if [ "$DONE_COUNT" -ge "$TOTAL_EXPERIMENTS" ]; then
      wlog "All experiments already complete — watchdog exiting."
      rm -f results/watchdog.pid; exit 0
    fi

    wlog "Waiting ${RESTART_WAIT}s before restart ..."
    sleep "$RESTART_WAIT"
    wlog "Restarting ${SESSION} with --resume ..."
    bash tmux_launch.sh --resume
    wlog "Restart issued."
  fi
done
