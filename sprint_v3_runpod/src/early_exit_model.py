"""
early_exit_model.py — TinyLlama-1.1B-Chat with post-hoc L16 forward hook.
Registers a hook on layers[15] (0-indexed) to capture L16 hidden states
during the single contiguous 22-layer pass.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EXIT_LAYER = 15  # 0-indexed → Layer 16


class EarlyExitModel:
    def __init__(self, device="cuda", model_id=MODEL_ID):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._m = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device).eval()

        self._l16_hidden = None
        self._hook_handle = self._m.model.layers[EXIT_LAYER].register_forward_hook(
            self._capture_hook
        )

    def _capture_hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._l16_hidden = hidden

    def forward(self, input_ids):
        self._l16_hidden = None
        with torch.no_grad():
            out = self._m(input_ids=input_ids, use_cache=False)
        return out, self._l16_hidden

    def l16_logits(self, hidden):
        with torch.no_grad():
            lm_head = self._m.lm_head
            return lm_head(hidden)

    def get_l16_confidence(self, hidden):
        logits = self.l16_logits(hidden)
        last = logits[0, -1, :]
        probs = torch.softmax(last, dim=-1)
        return float(probs.max().item())

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def decode(self, token_id):
        return self.tokenizer.decode([token_id])

    def __del__(self):
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()


# Alias for compatibility
EarlyExitModelV2 = EarlyExitModel
