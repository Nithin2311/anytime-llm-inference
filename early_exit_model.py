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

    def forward(self, input_ids, exit_layer=None):
        """
        Forward pass with a true early exit that physically halts computation.
        """
        if exit_layer is None or exit_layer >= self.num_layers:
            # Standard full forward pass
            outputs = self.base_model.model(input_ids)
            return self.lm_head(outputs.last_hidden_state)
            
        # 1. Save a reference to the original full stack of layers
        original_layers = self.base_model.model.layers
        
        try:
            # 2. Dynamically truncate the module list to force an early exit
            import torch.nn as nn
            self.base_model.model.layers = nn.ModuleList(original_layers[:exit_layer])
            
            # 3. Run the forward pass (it will now physically stop computing at exit_layer)
            outputs = self.base_model.model(input_ids)
            hidden_states = outputs.last_hidden_state
            
        finally:
            # 4. CRITICAL: Restore the original layers so we don't permanently break the model
            self.base_model.model.layers = original_layers
            
        # Pass the early hidden state through the LM head
        logits = self.lm_head(hidden_states)
        return logits
        
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