#!/usr/bin/env bash
# watchdog.sh — Polls sprint_cr every 2 min; auto-restarts on crash.
# Run in background: bash watchdog.sh &
# Stop: kill %1  (or kill $(cat results/watchdog.pid))

set -uo pipefail
cd "$(dirname "$0")"

SESSION="sprint_cr"
WLOG="results/watchdog.log"
POLL_S=120    # poll every 2 minutes
RESTART_WAIT=60

mkdir -p results
echo $$ > results/watchdog.pid

wlog() { echo "$(date '+%Y-%m-%dT%H:%M:%S') [WATCHDOG] $*" | tee -a "$WLOG"; }

wlog "Watchdog started (PID=$$, session=${SESSION}, poll=${POLL_S}s)"

while true; do
  sleep "$POLL_S"

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    # Session alive — log GPU stats
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    DONE_COUNT=$(ls results/.*.done 2>/dev/null | wc -l || echo 0)
    wlog "OK  temp=${GPU_TEMP}C  util=${GPU_UTIL}  done=${DONE_COUNT}/10"
  else
    wlog "Session '${SESSION}' is DEAD — initiating restart ..."

    # Clear GPU cache
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null && \
      wlog "GPU cache cleared." || wlog "GPU cache clear failed (non-fatal)."

    wlog "Waiting ${RESTART_WAIT}s before restart ..."
    sleep "$RESTART_WAIT"

    # Check if run_all.sh already finished (all .done flags present)
    DONE_COUNT=$(ls results/.*.done 2>/dev/null | wc -l || echo 0)
    if [ "$DONE_COUNT" -ge 10 ]; then
      wlog "All 10 experiments complete — watchdog exiting."
      exit 0
    fi

    wlog "Restarting sprint_cr with --resume ..."
    bash tmux_launch.sh --resume
    wlog "Restart issued."
  fi
done
