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
        Single-pass KV-cached forward returning both intermediate and full logits.

        Post-hoc hook design — no pipeline stall
        -----------------------------------------
        A forward hook on decoder layer 15 (0-indexed; "Layer 16" in 1-indexed
        notation) captures the layer's output hidden-state tensor.  Three
        properties make this architecturally safe for application-level routing:

        1. The hook stores only a GPU tensor *reference* — no data is moved to
           host memory and no CPU-GPU synchronisation is triggered inside the
           forward pass.  The captured tensor remains resident in VRAM.

        2. All 22 decoder layers execute to completion without interruption.
           There is no Python callback that blocks between layers 16 and 17, so
           the CUDA kernel queue runs contiguously with no pipeline bubble.

        3. Only after base_model() returns do we apply norm + lm_head to the
           captured hidden state.  The calling router therefore performs a
           purely post-hoc routing decision: it chooses which logits to commit
           only after the GPU has fully synchronised.

        KV-cache consistency
        --------------------
        Every decoder layer appends its KV state on every call, regardless of
        which logits the router commits.  This prevents the desynchronisation
        failure inherent in two-pass designs where early exits at layer 16 leave
        layers 17–22 with a stale context window on the next full-depth pass.

        CPU-GPU heterogeneous pipeline
        ------------------------------
        The caller is responsible for synchronising the inference stream before
        reading logit values.  This design allows the caller to launch the GPU
        pass on a dedicated CUDA stream (non-blocking) and overlap CPU
        post-processing of the previous token while this pass is in-flight —
        see generate_anytime_async_overlap() in dynamic_scheduler.py.

        Args:
            input_ids:        LongTensor [batch, seq_len]
                              First call: the full prompt.
                              Subsequent calls: single new token [batch, 1].
            past_key_values:  KV cache from the previous call (None on first
                              call).

        Returns:
            l16_logits:       FloatTensor [batch, seq_len, vocab]
            full_logits:      FloatTensor [batch, seq_len, vocab]
            past_key_values:  Updated KV cache; pass back on the next call.
        """
        captured = {}

        def _hook(module, input, output):
            # Store a GPU tensor reference only — no host transfer, no sync.
            # The tensor remains in VRAM until the caller explicitly reads it.
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

        # These two operations execute strictly after the 22-layer pass completes.
        # The router's confidence evaluation is post-hoc relative to GPU execution.
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
