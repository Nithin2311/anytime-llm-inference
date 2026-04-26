#!/usr/bin/env bash
# collect_results.sh — Archive all results into a timestamped tarball.
# Run this after the sprint completes to package everything for download.
#
# Usage:
#   bash collect_results.sh
#   bash collect_results.sh --scp user@host:/path   # also scp the tarball

set -euo pipefail
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
ARCHIVE="sprint_final_results_${TS}.tar.gz"
SCP_TARGET="${2:-}"

echo "Collecting results ..."
tar czf "$ARCHIVE" \
  results/ \
  figures/ \
  logs/ \
  --exclude='results/.*.done' \
  2>/dev/null || true

SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo "Archive: $ARCHIVE ($SIZE)"

if [ -n "$SCP_TARGET" ]; then
  echo "Copying to $SCP_TARGET ..."
  scp "$ARCHIVE" "$SCP_TARGET"
  echo "Transfer complete."
fi

echo ""
echo "Contents:"
tar tzf "$ARCHIVE" | grep -E '\.(json|png|tex|md)$' | sort
