import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class EarlyExitTinyLlama(torch.nn.Module):
    """
    TinyLlama-1.1B with explicit early-exit support.

    Replaces the fragile layer-list mutation approach with a clean
    layer-by-layer forward pass. Key correctness properties:
      - RMSNorm (model.norm) is always applied at the exit point,
        regardless of exit depth.
      - Rotary position embeddings are computed once and shared across
        all layers, matching the standard LlamaModel forward exactly.
      - No in-place model state mutation — safe for repeated calls.
    """

    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__()
        print(f"Loading base model: {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Suppress the transformers deprecation noise for dtype/torch_dtype kwarg
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch_dtype.*")
            warnings.filterwarnings("ignore", message=".*dtype.*deprecated.*")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="cuda",
            )

        self._m = self.base_model.model          # LlamaModel internals
        self.lm_head = self.base_model.lm_head
        self.num_layers = len(self._m.layers)

        print(f"Model loaded: {self.num_layers} transformer layers | dtype=bfloat16 | SDPA enabled")

    def forward(self, input_ids, exit_layer=None, use_cache=False):
        """
        Explicit layer-by-layer forward pass with early exit.

        Args:
            input_ids:   LongTensor [batch, seq_len]
            exit_layer:  int — stop after this many layers (None = full pass)
            use_cache:   kept for API compatibility, always False (stateless)

        Returns:
            logits:      FloatTensor [batch, seq_len, vocab_size]
            None:        placeholder for past_key_values (unused)
        """
        seq_len = input_ids.shape[1]

        # 1. Token embeddings  →  [batch, seq_len, hidden_size]
        hidden_states = self._m.embed_tokens(input_ids)

        # 2. Rotary position embeddings — computed once, shared across all layers
        position_ids = torch.arange(seq_len, dtype=torch.long,
                                    device=input_ids.device).unsqueeze(0)
        position_embeddings = self._m.rotary_emb(hidden_states, position_ids)

        # 3. Determine exit depth
        n_layers = (exit_layer
                    if exit_layer is not None and exit_layer < self.num_layers
                    else self.num_layers)

        # 4. Run transformer layers up to exit point
        #    attention_mask=None → SDPA uses is_causal=True internally (FlashAttention path)
        #    LlamaDecoderLayer may return a 1-tuple (hidden_states,) in some transformers
        #    versions, so we always unpack to keep hidden_states as a plain tensor.
        for layer in self._m.layers[:n_layers]:
            out = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden_states = out[0] if isinstance(out, tuple) else out

        # 5. Apply RMSNorm at the exit point (always, at every exit depth)
        hidden_states = self._m.norm(hidden_states)

        # 6. Project to vocabulary
        logits = self.lm_head(hidden_states)
        return logits, None


    def forward_cached(self, input_ids, past_key_values=None):
        """
        KV-cached forward pass that returns both L16 and full-pass logits.

        Design rationale
        ----------------
        The two-pass stateless scheduler runs L16 then (optionally) a full
        pass, recomputing all attention from scratch every token — O(n²) cost.
        KV-cache reduces this to O(n) per token, but maintaining *separate*
        caches per exit depth causes desynchronisation: if a token exits at L16
        the full-pass cache never receives its layers 16–21 KV states, so the
        next full-pass token attends to an incomplete history.

        Solution: always run all 22 layers in a single pass, capture the hidden
        state at layer 15 (0-indexed = "Layer 16" in 1-indexed notation) via a
        forward hook, and apply the shared RMSNorm + lm_head at that point to
        produce approximate L16 logits.  This guarantees a consistent KV cache
        while giving the scheduler both logits to choose from.

        Args:
            input_ids:        LongTensor [batch, seq_len]
                              First call: the full prompt.
                              Subsequent calls: single new token [batch, 1].
            past_key_values:  KV cache returned from the previous call (None on
                              the first call).

        Returns:
            l16_logits:       FloatTensor [batch, seq_len, vocab]
            full_logits:      FloatTensor [batch, seq_len, vocab]
            past_key_values:  Updated KV cache; pass back on the next call.
        """
        captured = {}

        def _hook(module, input, output):
            # LlamaDecoderLayer may return a tuple; index 0 is always hidden_states.
            captured["h16"] = output[0] if isinstance(output, tuple) else output

        handle = self._m.layers[15].register_forward_hook(_hook)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                out = self.base_model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
        finally:
            handle.remove()

        full_logits = out.logits
        l16_logits  = self.lm_head(self._m.norm(captured["h16"]))

        return l16_logits, full_logits, out.past_key_values


# --- Verification ---
if __name__ == "__main__":
    model = EarlyExitTinyLlama()

    prompt = "The most critical aspect of a real-time system is"
    inputs = model.tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        # Full pass (22 layers)
        logits_full, _ = model(inputs.input_ids, use_cache=False)
        print(f"Full pass  logits shape : {logits_full.shape}")

        # Early exit at layer 16
        logits_l16, _ = model(inputs.input_ids, exit_layer=16, use_cache=False)
        print(f"Exit@L16   logits shape : {logits_l16.shape}")

        # Early exit at layer 5
        logits_l5, _ = model(inputs.input_ids, exit_layer=5, use_cache=False)
        print(f"Exit@L5    logits shape : {logits_l5.shape}")

        # Sanity: top-1 token for each depth
        for name, lg in [("Full(22)", logits_full), ("Exit(16)", logits_l16), ("Exit(5)", logits_l5)]:
            probs = torch.softmax(lg[0, -1, :], dim=-1)
            conf, tok = torch.max(probs, dim=-1)
            word = model.tokenizer.decode([tok.item()])
            print(f"  {name}: top token = '{word}' | conf = {conf.item():.4f}")

    print("\nVerification complete.")
