import torch
from early_exit_model import EarlyExitTinyLlama

def generate_with_deadline(model, prompt, max_new_tokens=15, deadline_ms=50.0, conf_threshold=0.8):
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    print("\n" + "="*50)
    print(f"Starting Anytime Generation")
    print(f"Hard Deadline per token: {deadline_ms} ms | Conf Threshold: {conf_threshold}")
    print("="*50 + "\n")

    print("Warming up GPU kernels...")
    with torch.no_grad():
        # A quick dummy pass to absorb the initialization overhead
        _ = model(input_ids)
    torch.cuda.synchronize()
    print("Warm-up complete. Starting strict timing.\n")
    
    
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Timers for the scheduler
            start_event = torch.cuda.Event(enable_timing=True)
            mid_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            
            # --- STAGE 1: Early Evaluation (Layer 5) ---
            logits_early = model(input_ids, exit_layer=5)
            mid_event.record()
            torch.cuda.synchronize() # Block to calculate elapsed time and confidence
            
            # Calculate elapsed time for the early pass
            elapsed_early_ms = start_event.elapsed_time(mid_event)
            
            # Calculate confidence (max probability of the next token)
            probs = torch.softmax(logits_early[0, -1, :], dim=-1)
            confidence, next_token_early = torch.max(probs, dim=-1)
            conf_val = confidence.item()
            
            # --- STAGE 2: The Scheduler Decision ---
            # We know from profiling that a full pass takes ~24ms. 
            # We need at least that much budget left to safely attempt it.
            remaining_budget = deadline_ms - elapsed_early_ms
            
            if conf_val >= conf_threshold:
                # We are confident enough! Exit early to save compute.
                next_token = next_token_early
                exit_type = "Early (High Conf)"
                end_event.record()
                
            elif remaining_budget < 25.0:
                # We are NOT confident, but we don't have enough time to compute deeper.
                # Force an early exit to respect the soft real-time constraint.
                next_token = next_token_early
                exit_type = "Early (Deadline)"
                end_event.record()
                
            else:
                # We have low confidence AND plenty of time. Run the full model.
                logits_full = model(input_ids)
                next_token = torch.argmax(logits_full[0, -1, :], dim=-1)
                exit_type = "Full Pass"
                end_event.record()
            
            torch.cuda.synchronize()
            total_token_ms = start_event.elapsed_time(end_event)
            
            # Append token and update sequence
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            
            word = model.tokenizer.decode([next_token.item()])
            print(f"Token {i+1:>2}: {word:<12} | Exit: {exit_type:<18} | Time: {total_token_ms:.2f} ms | Conf: {conf_val:.2f}")

    final_text = model.tokenizer.decode(generated_tokens)
    print(f"\nFinal Output: {prompt} {final_text}")

if __name__ == "__main__":
    model = EarlyExitTinyLlama()
    
    # Run the scheduler
    prompt = "The most critical aspect of a real-time system is"
    generate_with_deadline(model, prompt)