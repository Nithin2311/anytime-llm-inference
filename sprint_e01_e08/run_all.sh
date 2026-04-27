#!/usr/bin/env bash
# run_all.sh — Runs E01-E08 sequentially with retry, then pushes results to GitHub.
#
# Required env vars:
#   GITHUB_TOKEN  — PAT with repo write scope
#   GITHUB_REPO   — e.g. "Nithin2311/anytime-llm-inference"
#   HF_TOKEN      — HuggingFace token for model download
#
# Usage:
#   bash run_all.sh              # full run + push
#   bash run_all.sh --resume     # skip completed (.done flags)
#   bash run_all.sh --skip-push  # run only, no GitHub push
#   bash run_all.sh --only e04   # single experiment
#   bash run_all.sh --dry        # print plan only

set -uo pipefail
cd "$(dirname "$0")"

RESULTS="results"; LOGS="logs"
SUMMARY="${RESULTS}/SPRINT_SUMMARY.md"
RUNLOG="${RESULTS}/run.log"
RESUME=0; ONLY=""; DRY=0; SKIP_PUSH=0
MAX_RETRIES=3; RETRY_SLEEP=30

for arg in "$@"; do
  case "$arg" in
    --resume)     RESUME=1 ;;
    --dry)        DRY=1 ;;
    --skip-push)  SKIP_PUSH=1 ;;
    --only=*)     ONLY="${arg#--only=}" ;;
    --only)       shift; ONLY="${1:-}" ;;
  esac
done

mkdir -p "$RESULTS" "$LOGS"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

SPRINT_START=$(date +%s)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*" | tee -a "$RUNLOG"; }

log "═══════════════════════════════════════════════════════════"
log " Sprint E01-E08  |  GPU: ${GPU}"
log " RESUME=${RESUME}  ONLY=${ONLY:-all}  DRY=${DRY}  SKIP_PUSH=${SKIP_PUSH}"
log "═══════════════════════════════════════════════════════════"

declare -A STATUS ATTEMPTS
PASSED=0; FAILED=0; SKIPPED=0
declare -a TIMES

run_experiment() {
  local name="$1" desc="${2:-}"
  local done_flag="${RESULTS}/.${name}.done"
  local logfile="${LOGS}/${name}.log"

  if [ -n "$ONLY" ]; then
    local tag="${name:1:2}" short="${name%%_*}"
    if [ "$ONLY" != "$name" ] && [ "$ONLY" != "$tag" ] && \
       [ "$ONLY" != "$short" ] && [ "$ONLY" != "E${tag}" ] && \
       [ "$ONLY" != "e${tag}" ] && [ "$ONLY" != "${tag#0}" ]; then
      STATUS["$name"]="SKIP"; ((SKIPPED++)) || true; return 0
    fi
  fi

  if [ "$RESUME" -eq 1 ] && [ -f "$done_flag" ]; then
    log "  [SKIP] ${name} (.done exists)"; STATUS["$name"]="SKIP"; ((SKIPPED++)) || true; return 0
  fi

  [ "$DRY" -eq 1 ] && { log "  [DRY]  ${name}  ${desc}"; STATUS["$name"]="DRY"; return 0; }

  log ""; log "──────────────────────────────────────────────"
  log "  Running: ${name}  —  ${desc}"

  local attempt=0 success=0 t0; t0=$(date +%s)
  while [ $attempt -lt $MAX_RETRIES ]; do
    ((attempt++)) || true
    log "  Attempt ${attempt}/${MAX_RETRIES} ..."
    if python3 "experiments/${name}.py" 2>&1 | tee "$logfile"; then
      success=1; break
    else
      local ec=${PIPESTATUS[0]}
      log "  Attempt ${attempt} failed (exit ${ec})"
      [ $attempt -lt $MAX_RETRIES ] && {
        log "  Clearing GPU cache, retrying in ${RETRY_SLEEP}s ..."
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep "$RETRY_SLEEP"
      }
    fi
  done

  local elapsed=$(( $(date +%s) - t0 ))
  if [ "$success" -eq 1 ]; then
    STATUS["$name"]="PASS"; ATTEMPTS["$name"]=$attempt
    touch "$done_flag"; TIMES+=("${name}: ${elapsed}s")
    ((PASSED++)) || true; log "  [PASS] ${name}  (${elapsed}s)"
  else
    STATUS["$name"]="FAIL"; ATTEMPTS["$name"]=$attempt
    ((FAILED++)) || true; log "  [FAIL] ${name}  after ${attempt} attempts"
  fi
}

# ── Experiment sequence E01-E08 (no E00, no E09) ────────────────────────────
run_experiment "e01_iid_spacing_study"   "Ljung-Box IID at 0/200ms/1s/5s spacing"
run_experiment "e02_block_maxima_large"  "Block-maxima GEV n=5000, b=25/50/100"
run_experiment "e03_nonevt_wcet"         "Non-EVT empirical bounds P99/P99.9 + safety factor"
run_experiment "e04_accuracy_1000"       "1000-query PubMedQA accuracy CI ~+-3pp"
run_experiment "e05_tau_crossval"        "5-fold tau cross-validation"
run_experiment "e06_router_comparison"   "Three-router comparison, 500 queries each"
run_experiment "e07_thermal_extended"    "60-min thermal soak"
run_experiment "e08_capacity_full"       "Multi-request capacity N=1..8"

# ── Write summary ────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - SPRINT_START ))
GPU_FULL=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
{
  echo "# Sprint E01-E08 — Run Summary"
  echo ""; echo "**Completed:** $(date)"
  echo "**Total time:** ${TOTAL}s ($(( TOTAL/3600 ))h $(( (TOTAL%3600)/60 ))m)"
  echo "**GPU:** ${GPU_FULL}"
  echo "**Results:** PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}"
  echo ""; echo "| Experiment | Status |"
  echo "|-----------|--------|"
  for exp in e01_iid_spacing_study e02_block_maxima_large e03_nonevt_wcet \
             e04_accuracy_1000 e05_tau_crossval e06_router_comparison \
             e07_thermal_extended e08_capacity_full; do
    echo "| ${exp} | ${STATUS[$exp]:-?} |"
  done
  echo ""; echo "## Timing"
  for t in "${TIMES[@]:-}"; do echo "- $t"; done
} > "$SUMMARY"

log ""; log "═══════════════════════════════════════════════════════════"
log " PASS=${PASSED}  FAIL=${FAILED}  SKIP=${SKIPPED}  TIME=${TOTAL}s"
log "═══════════════════════════════════════════════════════════"

# ── GitHub push ──────────────────────────────────────────────────────────────
[ "$FAILED" -gt 0 ] && { log "Skipping push — ${FAILED} experiment(s) failed."; exit 1; }
[ "$DRY"    -eq 1 ] && exit 0
[ "$SKIP_PUSH" -eq 1 ] && { log "[PUSH] Skipped (--skip-push)."; exit 0; }

if [ -z "${GITHUB_TOKEN:-}" ] || [ -z "${GITHUB_REPO:-}" ]; then
  log "[PUSH] WARN: GITHUB_TOKEN or GITHUB_REPO not set — skipping push."
  exit 0
fi

log ""; log "── Pushing results to GitHub ────────────────────────────────"
PUSH_DIR="/tmp/gh_push_e01e08_$$"
REMOTE="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"

git clone --depth=1 --filter=blob:none --sparse "$REMOTE" "$PUSH_DIR" 2>&1 | \
  sed 's/x-access-token:[^@]*/x-access-token:REDACTED/g' || { log "[PUSH] Clone failed"; exit 1; }

cd "$PUSH_DIR"
git sparse-checkout set sprint_e01_e08/results 2>/dev/null || true

DEST="${PUSH_DIR}/sprint_e01_e08/results"; mkdir -p "$DEST"
rsync -av --exclude=".gitkeep" "$(dirname "$0")/results/" "$DEST/" 2>&1 | tail -30

git add sprint_e01_e08/results/ 2>/dev/null || git add . 2>/dev/null
git config user.email "sprint-bot@runpod.io"
git config user.name "Sprint E01-E08 Bot"

if git diff --cached --quiet; then
  log "  Nothing new to push."
else
  COMMIT_MSG="results(e01-e08): sprint complete @ $(date -u '+%Y-%m-%dT%H:%M:%SZ') GPU=${GPU}"
  git commit -m "$COMMIT_MSG"
  git push origin HEAD:main
  log "  [OK] Pushed: ${COMMIT_MSG}"
fi

cd / && rm -rf "$PUSH_DIR"
log "  Push complete. Pull locally: git pull origin main"
exit 0
