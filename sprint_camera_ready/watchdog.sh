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
PUSH_EVERY=5  # push every 5 polls = 10 minutes

# Source persisted env (tokens, repo, branch)
[ -f /root/.sprint_env ] && source /root/.sprint_env
PUSH_BRANCH="${PUSH_BRANCH:-sprint-cr-final-results}"
GITHUB_REPO="${GITHUB_REPO:-Nithin2311/anytime-llm-inference}"
REPO_ROOT="$(git -C "$(pwd)" rev-parse --show-toplevel 2>/dev/null || echo /workspace/anytime-llm-inference)"

mkdir -p results
echo $$ > results/watchdog.pid

wlog() { echo "$(date '+%Y-%m-%dT%H:%M:%S') [WATCHDOG] $*" | tee -a "$WLOG"; }

push_results() {
  local cur_branch
  cur_branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  if [ "$cur_branch" != "$PUSH_BRANCH" ]; then
    wlog "PUSH skipped — repo on '${cur_branch}' not '${PUSH_BRANCH}'"
    return 0
  fi
  local sub_path
  sub_path="$(realpath --relative-to="$REPO_ROOT" "$(pwd)/results")"
  git -C "$REPO_ROOT" add -- "$sub_path" >/dev/null 2>&1 || true
  if git -C "$REPO_ROOT" diff --cached --quiet; then
    wlog "PUSH no changes"
    return 0
  fi
  local done_count
  done_count=$(ls results/.*.done 2>/dev/null | wc -l)
  git -C "$REPO_ROOT" -c user.email="${GIT_AUTHOR_EMAIL:-bot@runpod.io}" \
                     -c user.name="${GIT_AUTHOR_NAME:-Sprint Bot}" \
    commit -m "watchdog auto-push: ${done_count}/10 experiments complete ($(date -u +%H:%MZ))" \
    >/dev/null 2>&1 || { wlog "PUSH commit failed"; return 1; }
  if git -C "$REPO_ROOT" push origin "$PUSH_BRANCH" >/dev/null 2>&1; then
    wlog "PUSH ok  branch=${PUSH_BRANCH}  done=${done_count}/10"
  else
    wlog "PUSH failed (will retry next cycle)"
  fi
}

wlog "Watchdog started (PID=$$, session=${SESSION}, poll=${POLL_S}s, push_every=$((PUSH_EVERY * POLL_S))s, branch=${PUSH_BRANCH})"

POLL_COUNT=0

while true; do
  sleep "$POLL_S"
  POLL_COUNT=$((POLL_COUNT + 1))

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    # Session alive — log GPU stats
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    DONE_COUNT=$(ls results/.*.done 2>/dev/null | wc -l || echo 0)
    wlog "OK  temp=${GPU_TEMP}C  util=${GPU_UTIL}  done=${DONE_COUNT}/10"
    if [ $((POLL_COUNT % PUSH_EVERY)) -eq 0 ]; then
      push_results || true
    fi
    if [ "$DONE_COUNT" -ge 10 ]; then
      wlog "All 10 experiments complete — final push and exit."
      push_results || true
      exit 0
    fi
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
