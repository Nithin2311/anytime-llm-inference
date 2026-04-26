#!/usr/bin/env bash
# watchdog.sh — Auto-restart orchestrator if it dies unexpectedly.
# Run in a separate pane: bash watchdog.sh
set -uo pipefail
SESSION="sprint_v3"
DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_RESTARTS=5
restarts=0

echo "Watchdog started for session '$SESSION' ..."

while true; do
  sleep 60
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    if [ $restarts -ge $MAX_RESTARTS ]; then
      echo "$(date): Max restarts ($MAX_RESTARTS) reached. Stopping watchdog."
      exit 1
    fi
    ((restarts++)) || true
    echo "$(date): Session lost. Restarting with --resume (attempt $restarts/$MAX_RESTARTS) ..."
    cd "$DIR"
    bash tmux_launch.sh --resume
  fi
done
