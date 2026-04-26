#!/usr/bin/env bash
# setup.sh — Environment setup for sprint_v2_runpod on a fresh RunPod instance.
# Usage: bash setup.sh [--skip-download]

set -euo pipefail
SKIP_DL="${1:-}"
LOG="setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "═══════════════════════════════════════════════════════════"
echo " Sprint V2  —  Environment Setup"
echo " $(date)"
echo "═══════════════════════════════════════════════════════════"

# ── 1. CUDA check ────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Checking CUDA ..."
if ! command -v nvidia-smi &> /dev/null; then
  echo "ERROR: nvidia-smi not found. Is this a GPU instance?"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; \
            print(f'  PyTorch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}')"
echo "  CUDA OK"

# ── 2. Python packages ───────────────────────────────────────────────────────
echo ""
echo "[2/5] Installing Python packages ..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  Packages installed"

# ── 3. Model download ────────────────────────────────────────────────────────
echo ""
echo "[3/5] Model prefetch ..."
if [ "$SKIP_DL" = "--skip-download" ]; then
  echo "  Skipped (--skip-download)"
else
  python3 - << 'PYEOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"  Downloading {model_id} ...")
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
print("  Model cached")
PYEOF
fi

# ── 4. Dataset prefetch ──────────────────────────────────────────────────────
echo ""
echo "[4/5] Dataset prefetch (PubMedQA) ..."
python3 - << 'PYEOF'
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
print(f"  PubMedQA loaded: {len(ds)} samples")
PYEOF

# ── 5. Dry-run ───────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Dry-run (5 warmup tokens) ..."
python3 - << 'PYEOF'
import sys, warnings
sys.path.insert(0, "src")
from early_exit_model import EarlyExitModel
import torch
model = EarlyExitModel(device="cuda")
dummy = torch.randint(100, 2000, (1, 32), device="cuda")
for _ in range(5):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with torch.inference_mode():
            model.forward_cached(dummy)
torch.cuda.synchronize()
print("  Dry-run OK")
PYEOF

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Setup complete — $(date)"
echo " Run: bash tmux_launch.sh"
echo "═══════════════════════════════════════════════════════════"
