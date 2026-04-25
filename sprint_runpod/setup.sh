#!/usr/bin/env bash
# setup.sh — One-shot environment validation before launching the sprint.
# Run once after pod startup:  bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  SPRINT SETUP — A100 SXM4"
echo "============================================================"

# 1. CUDA check
echo ""
echo "[1/5] CUDA availability ..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found!'; \
           print(f'  OK: {torch.cuda.get_device_name(0)} | VRAM: \
{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')"

# 2. pip install
echo ""
echo "[2/5] Installing Python dependencies ..."
pip install -q -r requirements.txt
echo "  OK: requirements installed"

# 3. Model download (cache to HF default)
echo ""
echo "[3/5] Pre-downloading TinyLlama model weights ..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('  Downloading tokenizer ...')
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print('  Downloading model weights ...')
AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print('  OK: model cached')
"

# 4. dry-run all experiments
echo ""
echo "[4/5] Dry-running all experiments ..."
for exp in experiments/e{0,1,2,3,4,5,6,7}_*.py; do
    name=$(basename "$exp" .py)
    python "$exp" --dry-run && echo "  OK: $name" || { echo "  FAIL: $name"; exit 1; }
done

# 5. Create output dirs
echo ""
echo "[5/5] Ensuring output directories ..."
mkdir -p results figures latex
echo "  OK: results/ figures/ latex/"

echo ""
echo "============================================================"
echo "  SETUP COMPLETE — ready to launch"
echo "  Run:  bash tmux_launch.sh"
echo "============================================================"
