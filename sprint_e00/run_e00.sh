#!/usr/bin/env bash
# run_e00.sh — Runs e00_wcet_large_spaced and pushes results to GitHub.
#
# Required env vars:
#   GITHUB_TOKEN   — personal access token with repo write scope
#   GITHUB_REPO    — e.g. "Nithin2311/anytime-llm-inference"
#   HF_TOKEN       — HuggingFace token for model download
#
# Usage:
#   bash run_e00.sh              # full run then push
#   bash run_e00.sh --skip-push  # run only, no push
#   bash run_e00.sh --dry        # print plan, no execution

set -uo pipefail
cd "$(dirname "$0")"

RESULTS="results"
LOGS="logs"
RUNLOG="${RESULTS}/run_e00.log"
SKIP_PUSH=0; DRY=0
MAX_RETRIES=3; RETRY_SLEEP=60

for arg in "$@"; do
  case "$arg" in
    --skip-push) SKIP_PUSH=1 ;;
    --dry)       DRY=1 ;;
  esac
done

mkdir -p "$RESULTS" "$LOGS"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

START_TS=$(date +%s)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*" | tee -a "$RUNLOG"; }

log "═══════════════════════════════════════════════════════════"
log " Sprint E00  |  e00_wcet_large_spaced  (5000 samples, 1s spacing)"
log " GPU: ${GPU}  |  DRY=${DRY}  SKIP_PUSH=${SKIP_PUSH}"
log "═══════════════════════════════════════════════════════════"

if [ "$DRY" -eq 1 ]; then
  log "[DRY] Would run: experiments/e00_wcet_large_spaced.py"
  log "[DRY] Estimated time: ~17 hours on A100"
  log "[DRY] Would push to: ${GITHUB_REPO:-<GITHUB_REPO not set>}"
  exit 0
fi

DONE_FLAG="${RESULTS}/.e00_wcet_large_spaced.done"
LOGFILE="${LOGS}/e00_wcet_large_spaced.log"

# Skip if already done (--resume semantics)
if [ -f "$DONE_FLAG" ]; then
  log "[SKIP] e00 already done (.done flag exists). Use: rm ${DONE_FLAG} to re-run."
  SKIP_PUSH=0
else
  attempt=0; success=0; t0=$(date +%s)
  while [ $attempt -lt $MAX_RETRIES ]; do
    ((attempt++)) || true
    log "Attempt ${attempt}/${MAX_RETRIES} — running e00_wcet_large_spaced ..."
    if python3 "experiments/e00_wcet_large_spaced.py" 2>&1 | tee "$LOGFILE"; then
      success=1; break
    else
      ec=${PIPESTATUS[0]}
      log "Attempt ${attempt} failed (exit ${ec})"
      if [ $attempt -lt $MAX_RETRIES ]; then
        log "Clearing GPU cache, retrying in ${RETRY_SLEEP}s ..."
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep "$RETRY_SLEEP"
      fi
    fi
  done

  ELAPSED=$(( $(date +%s) - t0 ))
  if [ "$success" -eq 1 ]; then
    touch "$DONE_FLAG"
    log "[PASS] e00_wcet_large_spaced completed in ${ELAPSED}s ($(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m)"
  else
    log "[FAIL] e00_wcet_large_spaced failed after ${MAX_RETRIES} attempts"
    exit 1
  fi
fi

# ── GitHub push ─────────────────────────────────────────────────────────────
if [ "$SKIP_PUSH" -eq 1 ]; then
  log "[PUSH] Skipped (--skip-push flag)."
  exit 0
fi

if [ -z "${GITHUB_TOKEN:-}" ] || [ -z "${GITHUB_REPO:-}" ]; then
  log "[PUSH] WARN: GITHUB_TOKEN or GITHUB_REPO not set — skipping push."
  log "  Export them before running: export GITHUB_TOKEN=... GITHUB_REPO=Nithin2311/anytime-llm-inference"
  exit 0
fi

log ""
log "── Pushing results to GitHub ────────────────────────────────"

PUSH_DIR="/tmp/gh_push_e00_$$"
REMOTE="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"

log "  Cloning ${GITHUB_REPO} (sparse, depth=1) ..."
git clone --depth=1 --filter=blob:none --sparse "$REMOTE" "$PUSH_DIR" 2>&1 | \
  sed 's/x-access-token:[^@]*/x-access-token:REDACTED/g' || { log "[PUSH] Clone failed"; exit 1; }

cd "$PUSH_DIR"
git sparse-checkout set sprint_e00/results 2>/dev/null || true
git sparse-checkout add sprint_e00/results 2>/dev/null || true

DEST="${PUSH_DIR}/sprint_e00/results"
mkdir -p "$DEST"
rsync -av --exclude=".gitkeep" \
  "$(dirname "$0")/results/" "$DEST/" 2>&1 | tail -30

git add sprint_e00/results/ 2>/dev/null || git add . 2>/dev/null
git config user.email "sprint-bot@runpod.io"
git config user.name "Sprint E00 Bot"

if git diff --cached --quiet; then
  log "  Nothing new to push (results unchanged)."
else
  COMMIT_MSG="results(e00): wcet_large_spaced @ $(date -u '+%Y-%m-%dT%H:%M:%SZ') GPU=${GPU}"
  git commit -m "$COMMIT_MSG"
  git push origin HEAD:main
  log "  [OK] Pushed: ${COMMIT_MSG}"
fi

cd /
rm -rf "$PUSH_DIR"
log "  Push complete."

TOTAL=$(( $(date +%s) - START_TS ))
log ""
log "═══════════════════════════════════════════════════════════"
log " E00 DONE  |  Total elapsed: ${TOTAL}s ($(( TOTAL/3600 ))h $(( (TOTAL%3600)/60 ))m)"
log " Results at: sprint_e00/results/ on GitHub/${GITHUB_REPO}"
log "═══════════════════════════════════════════════════════════"
