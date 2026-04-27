#!/usr/bin/env bash
# setup.sh — One-shot environment setup for sprint_e01_e08 on RunPod. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
echo "════════════════════════════════════════════════════"
echo " Sprint E01-E08  —  Environment Setup  |  $(date)"
echo "════════════════════════════════════════════════════"
echo ""; echo "[1/5] GPU check ..."
if ! command -v nvidia-smi &>/dev/null; then echo "ERROR: nvidia-smi not found."; exit 1; fi
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "  GPU: ${GPU}"
python3 -c "import torch; ok=torch.cuda.is_available(); print(f'  PyTorch CUDA: {ok}  device: {torch.cuda.get_device_name(0) if ok else \"N/A\"}')"
echo ""; echo "[2/5] Installing Python dependencies ..."
pip install -q --upgrade pip && pip install -q -r requirements.txt && echo "  Done."
echo ""; echo "[3/5] Network volume detection ..."
VOLUME=/workspace/volume; RESULTS="$(pwd)/results"
if [ -d "$VOLUME" ]; then
  TARGET="${VOLUME}/sprint_e01_e08_results"; mkdir -p "$TARGET"
  if [ -L "$RESULTS" ]; then echo "  Symlink exists: $RESULTS -> $(readlink "$RESULTS")"
  else
    [ -d "$RESULTS" ] && cp -r "$RESULTS/." "$TARGET/" 2>/dev/null && rm -rf "$RESULTS"
    ln -sfn "$TARGET" "$RESULTS"; echo "  Symlinked: $RESULTS -> $TARGET"
  fi
else mkdir -p "$RESULTS"; echo "  No network volume — results stored locally."; fi
echo ""; echo "[4/5] Caching TinyLlama-1.1B-Chat ..."
python3 - <<'PYEOF'
import os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
token = os.environ.get("HF_TOKEN")
try:
    AutoTokenizer.from_pretrained(MODEL, token=token)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cpu", token=token)
    mb = sum(p.numel() for p in m.parameters()) / 1e6; del m
    print(f"  OK: {MODEL}  ({mb:.0f}M params)")
except Exception as e:
    print(f"  ERROR: {e}", file=sys.stderr); sys.exit(1)
PYEOF
echo ""; echo "[5/5] PYTHONPATH ..."
SPRINT="$(pwd)"
grep -q "sprint_e01_e08/src" ~/.bashrc 2>/dev/null || echo "export PYTHONPATH=\"${SPRINT}/src:\${PYTHONPATH:-}\"" >> ~/.bashrc
export PYTHONPATH="${SPRINT}/src:${PYTHONPATH:-}"
echo ""; echo "Setup complete.  Results: $(realpath results/)  GPU: ${GPU}"
echo "  Next: bash tmux_launch.sh"
echo "════════════════════════════════════════════════════"
