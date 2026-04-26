#!/usr/bin/env bash
# setup.sh — Environment bootstrap for sprint_v3_runpod on RunPod A100.
set -euo pipefail

echo "═══════════════════════════════════════════════════════════"
echo " sprint_v3_runpod — Setup"
echo " $(date)"
echo "═══════════════════════════════════════════════════════════"

# System packages
apt-get update -qq && apt-get install -y -qq tmux htop nvtop curl git 2>/dev/null || true

# Python deps
pip install --upgrade pip -q
pip install -r requirements.txt -q

# HuggingFace model cache
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print('Downloading TinyLlama-1.1B-Chat ...')
tok = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
mdl = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.float16)
print('Model cached OK')
"

echo ""
echo "Setup complete. Run: bash tmux_launch.sh"
