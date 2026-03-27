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

        # dtype= is the correct kwarg in transformers >= 5.x (torch_dtype deprecated)
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
        for layer in self._m.layers[:n_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )

        # 5. Apply RMSNorm at the exit point (always, at every exit depth)
        hidden_states = self._m.norm(hidden_states)

        # 6. Project to vocabulary
        logits = self.lm_head(hidden_states)
        return logits, None


# --- Verification ---
if __name__ == "__main__":
    model = EarlyExitTinyLlama()

    prompt = "The most critical aspect of a real-time system is"
    inputs = model.tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        # Full pass (22 layers)
        logits_full, _ = model(inputs.input_ids)
        print(f"Full pass  logits shape : {logits_full.shape}")

        # Early exit at layer 16
        logits_l16, _ = model(inputs.input_ids, exit_layer=16)
        print(f"Exit@L16   logits shape : {logits_l16.shape}")

        # Early exit at layer 5
        logits_l5, _ = model(inputs.input_ids, exit_layer=5)
        print(f"Exit@L5    logits shape : {logits_l5.shape}")

        # Sanity: top-1 token for each depth
        for name, lg in [("Full(22)", logits_full), ("Exit(16)", logits_l16), ("Exit(5)", logits_l5)]:
            probs = torch.softmax(lg[0, -1, :], dim=-1)
            conf, tok = torch.max(probs, dim=-1)
            word = model.tokenizer.decode([tok.item()])
            print(f"  {name}: top token = '{word}' | conf = {conf.item():.4f}")

    print("\nVerification complete.")
