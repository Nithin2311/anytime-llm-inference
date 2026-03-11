import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class EarlyExitTinyLlama(torch.nn.Module):
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__()
        print(f"Loading highly optimized base model: {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # OPTIMIZATION: BFloat16 Precision + Native PyTorch SDPA
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        
        self.num_layers = len(self.base_model.model.layers)
        self.lm_head = self.base_model.lm_head
        print(f"Model loaded with {self.num_layers} layers. SDPA hardware optimizations active.")

    def forward(self, input_ids, exit_layer=None, use_cache=False):
        """
        Stateless forward pass with early exit routing.
        """
        if exit_layer is None or exit_layer >= self.num_layers:
            outputs = self.base_model.model(
                input_ids=input_ids,
                use_cache=use_cache
            )
            return self.lm_head(outputs.last_hidden_state), None
            
        original_layers = self.base_model.model.layers
        
        try:
            import torch.nn as nn
            self.base_model.model.layers = nn.ModuleList(original_layers[:exit_layer])
            
            outputs = self.base_model.model(
                input_ids=input_ids,
                use_cache=use_cache
            )
            hidden_states = outputs.last_hidden_state
            
        finally:
            self.base_model.model.layers = original_layers
            
        logits = self.lm_head(hidden_states)
        return logits, None
        
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