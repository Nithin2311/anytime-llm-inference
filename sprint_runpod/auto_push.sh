#!/usr/bin/env bash
# auto_push.sh — Watch results/sprint.log and commit (and push if creds valid)
# after each experiment passes. Designed to run in its own tmux window.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPRINT_DIR="$REPO_ROOT/sprint_runpod"
LOG_FILE="$SPRINT_DIR/results/sprint.log"
PUSH_LOG="$SPRINT_DIR/results/push.log"

cd "$REPO_ROOT"

mkdir -p "$SPRINT_DIR/results"
touch "$LOG_FILE" "$PUSH_LOG"

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*" | tee -a "$PUSH_LOG"; }

log "auto_push.sh started (branch=$(git rev-parse --abbrev-ref HEAD))"

# Watch for "[Ex] PASS" lines and commit/push the new artifacts
tail -F "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    if [[ "$line" =~ \[(E[0-9])\]\ PASS ]]; then
        eid="${BASH_REMATCH[1]}"
        log "detected PASS for $eid — staging artifacts"

        # Give result_writer a moment to finish flushing
        sleep 2

        # Stage results, figures, and latex
        git add sprint_runpod/results/ sprint_runpod/figures/ sprint_runpod/latex/ 2>>"$PUSH_LOG"

        if ! git diff --cached --quiet; then
            msg="sprint: $eid passed — partial results

Auto-commit by run_sprint.sh watcher on RunPod A100 SXM4.
"
            if git commit -m "$msg" >>"$PUSH_LOG" 2>&1; then
                log "$eid committed locally"
                if git push origin HEAD >>"$PUSH_LOG" 2>&1; then
                    log "$eid pushed to origin"
                else
                    log "$eid push FAILED (likely auth) — kept local"
                fi
            else
                log "$eid commit failed — see $PUSH_LOG"
            fi
        else
            log "$eid no new artifacts to commit"
        fi
    fi

    if [[ "$line" == *"===== SPRINT END ====="* ]]; then
        log "sprint ended — final commit pass"
        sleep 5
        git add sprint_runpod/results/ sprint_runpod/figures/ sprint_runpod/latex/ 2>>"$PUSH_LOG"
        if ! git diff --cached --quiet; then
            git commit -m "sprint: final results + SPRINT_SUMMARY.md" >>"$PUSH_LOG" 2>&1 \
                && log "final commit done" \
                && (git push origin HEAD >>"$PUSH_LOG" 2>&1 && log "final push done" || log "final push FAILED")
        fi
    fi
done
