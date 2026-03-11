import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class EarlyExitTinyLlama(nn.Module):
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__()
        print(f"Loading base model: {model_id}...")
        
        # Load the base model in FP16 for optimized GPU memory and speed
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # We reuse the final Language Modeling (LM) head for intermediate exits.
        # This prevents massive memory bloat and keeps execution time tight.
        self.lm_head = self.base_model.lm_head
        
        # TinyLlama has 22 hidden layers. 
        self.num_layers = len(self.base_model.model.layers)
        print(f"Model loaded with {self.num_layers} transformer layers.")

    def forward(self, input_ids, past_key_values=None, exit_layer=None, use_cache=True):
        """
        Forward pass that intercepts and returns KV-cache states for $O(1)$ execution.
        """
        if exit_layer is None or exit_layer >= self.num_layers:
            outputs = self.base_model.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
            logits = self.lm_head(outputs.last_hidden_state)
            return logits, outputs.past_key_values
            
        original_layers = self.base_model.model.layers
        try:
            import torch.nn as nn
            self.base_model.model.layers = nn.ModuleList(original_layers[:exit_layer])
            
            outputs = self.base_model.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
            hidden_states = outputs.last_hidden_state
            
        finally:
            self.base_model.model.layers = original_layers
            
        logits = self.lm_head(hidden_states)
        return logits, outputs.past_key_values
        
# --- Quick Verification ---
if __name__ == "__main__":
    # Initialize our custom architecture
    model = EarlyExitTinyLlama()
    
    # Create a dummy input
    prompt = "The most critical aspect of a real-time system is"
    inputs = model.tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 1. Test standard full forward pass (22 layers)
    standard_logits = model(inputs.input_ids)
    print(f"Standard Logits Shape: {standard_logits.shape}")
    
    # 2. Test early exit at layer 11 (Halfway)
    early_logits = model(inputs.input_ids, exit_layer=11)
    print(f"Early Exit (Layer 11) Logits Shape: {early_logits.shape}")