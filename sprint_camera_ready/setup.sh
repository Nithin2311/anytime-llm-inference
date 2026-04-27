#!/usr/bin/env bash
# setup.sh — One-shot environment setup on RunPod. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════"
echo " Sprint Camera-Ready  —  Environment Setup"
echo " $(date)"
echo "════════════════════════════════════════════════════"

# 1. GPU check
echo ""
echo "[1/5] GPU check ..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Is this a GPU pod?"; exit 1
fi
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "  GPU: ${GPU}"
python3 -c "
import torch
ok = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if ok else 'N/A'
print(f'  PyTorch CUDA: {ok}  device: {dev}')
"

# 2. Python deps
echo ""
echo "[2/5] Installing Python dependencies ..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  Done."

# 3. Network volume / results symlink
echo ""
echo "[3/5] Network volume detection ..."
VOLUME=/workspace/volume
RESULTS="$(pwd)/results"
if [ -d "$VOLUME" ]; then
  TARGET="${VOLUME}/sprint_camera_ready_results"
  mkdir -p "$TARGET"
  if [ -L "$RESULTS" ]; then
    echo "  Symlink exists: $RESULTS -> $(readlink "$RESULTS")"
  else
    [ -d "$RESULTS" ] && cp -r "$RESULTS/." "$TARGET/" 2>/dev/null && rm -rf "$RESULTS"
    ln -sfn "$TARGET" "$RESULTS"
    echo "  Symlinked: $RESULTS -> $TARGET"
  fi
else
  mkdir -p "$RESULTS"
  echo "  No volume at $VOLUME — results stored locally (NOT persistent across pod restarts)."
fi

# 4. Download / cache TinyLlama
echo ""
echo "[4/5] Caching TinyLlama-1.1B-Chat ..."
python3 - <<'PYEOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, sys
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
try:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m   = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
    mb  = sum(p.numel() for p in m.parameters()) / 1e6
    del m
    print(f"  OK: {MODEL}  ({mb:.0f}M params)")
except Exception as e:
    print(f"  ERROR: {e}", file=sys.stderr); sys.exit(1)
PYEOF

# 5. PYTHONPATH
echo ""
echo "[5/5] PYTHONPATH ..."
SPRINT="$(pwd)"
if ! grep -q "sprint_camera_ready/src" ~/.bashrc 2>/dev/null; then
  echo "export PYTHONPATH=\"${SPRINT}/src:\${PYTHONPATH:-}\"" >> ~/.bashrc
fi
export PYTHONPATH="${SPRINT}/src:${PYTHONPATH:-}"

echo ""
echo "Setup complete."
echo "  Results : $(realpath results/)"
echo "  GPU     : ${GPU}"
echo ""
echo "  Next: bash tmux_launch.sh"
echo "════════════════════════════════════════════════════"
