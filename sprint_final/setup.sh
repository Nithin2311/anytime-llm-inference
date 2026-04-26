#!/usr/bin/env bash
# setup.sh — Environment bootstrap for sprint_final on a fresh RunPod instance.
#
# Recommended RunPod config:
#   GPU:  NVIDIA A100 SXM4 80 GB
#   CPU:  16+ vCPUs (any — TinyLlama is memory-bandwidth-bound on GPU)
#   RAM:  80 GB system RAM
#   Disk: 200 GB (model ~4 GB, dataset ~1 GB, results ~2 GB, headroom)
#   Template: RunPod PyTorch 2.1 (CUDA 12.1)
#
# Usage:
#   bash setup.sh              # full setup + model download
#   bash setup.sh --skip-dl    # skip HuggingFace downloads (already cached)

set -euo pipefail
cd "$(dirname "$0")"
SKIP_DL="${1:-}"
LOG="setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "═══════════════════════════════════════════════════════════"
echo " Sprint Final — Environment Setup"
echo " $(date)"
echo "═══════════════════════════════════════════════════════════"

# ── 1. CUDA + GPU check ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Checking CUDA and GPU ..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Ensure this is a GPU instance."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
           --format=csv,noheader
python3 -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available'
d = torch.cuda.get_device_properties(0)
print(f'  PyTorch {torch.__version__}  CUDA {torch.version.cuda}')
print(f'  GPU: {d.name}  VRAM: {d.total_memory // 1024**3} GB')
if d.total_memory < 20 * 1024**3:
    print('  WARNING: <20 GB VRAM — some experiments may OOM')
"
echo "  CUDA OK"

# ── 2. tmux check ───────────────────────────────────────────────────────────
echo ""
echo "[2/6] Checking tmux ..."
if ! command -v tmux &>/dev/null; then
  echo "  Installing tmux ..."
  apt-get install -y tmux 2>/dev/null || yum install -y tmux 2>/dev/null || \
    (echo "ERROR: cannot install tmux automatically"; exit 1)
fi
tmux -V
echo "  tmux OK"

# ── 3. Python packages ───────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing Python packages ..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  Packages installed"

# ── 4. Model download ────────────────────────────────────────────────────────
echo ""
echo "[4/6] Model prefetch (TinyLlama-1.1B-Chat-v1.0) ..."
if [ "$SKIP_DL" = "--skip-dl" ]; then
  echo "  Skipped (--skip-dl)"
else
  python3 - << 'PYEOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"  Downloading {model_id} ...")
tok = AutoTokenizer.from_pretrained(model_id)
m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
print(f"  Model params: {sum(p.numel() for p in m.parameters())/1e9:.2f}B")
print("  Model cached")
PYEOF
fi

# ── 5. Dataset prefetch ──────────────────────────────────────────────────────
echo ""
echo "[5/6] Dataset prefetch (PubMedQA pqa_labeled) ..."
if [ "$SKIP_DL" = "--skip-dl" ]; then
  echo "  Skipped (--skip-dl)"
else
  python3 - << 'PYEOF'
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
print(f"  PubMedQA loaded: {len(ds)} samples")
PYEOF
fi

# ── 6. Dry-run validation ────────────────────────────────────────────────────
echo ""
echo "[6/6] Dry-run (5 warm-up forward passes) ..."
python3 - << 'PYEOF'
import sys, warnings
sys.path.insert(0, "src")
from early_exit_model import EarlyExitModel
import torch
model = EarlyExitModel(device="cuda")
dummy = torch.randint(100, 2000, (1, 64), device="cuda")
for i in range(5):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with torch.inference_mode():
            l16_logits, full_logits, latency = model.forward_cached(dummy)
torch.cuda.synchronize()
print(f"  Dry-run OK  (L16 logits: {l16_logits.shape}, full: {full_logits.shape})")
PYEOF

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Setup complete — $(date)"
echo ""
echo " Next step:"
echo "   bash tmux_launch.sh          # full run"
echo "   bash tmux_launch.sh --resume # resume from last checkpoint"
echo "═══════════════════════════════════════════════════════════"
