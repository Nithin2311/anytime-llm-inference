import torch
from early_exit_model import EarlyExitTinyLlama

def generate_stateless_anytime(model, prompt, max_new_tokens=15, deadline_ms=50.0, max_conf=0.8, min_conf=0.3):
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    print("\n" + "="*55)
    print(f"Starting Stateless Anytime Generation (Cache Bypass)")
    print(f"Deadline: {deadline_ms} ms | Conf Decay: {max_conf} -> {min_conf}")
    print("="*55 + "\n")
    
    print("Warming up GPU kernels...")
    with torch.no_grad():
        _ = model(input_ids)
    torch.cuda.synchronize()
    print("Warm-up complete. Starting strict timing.\n")
    
    generated_tokens = []
    full_pass_wcet = 23.0 
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            start_event = torch.cuda.Event(enable_timing=True)
            mid_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            
            # --- STAGE 1: Early Evaluation (Layer 16) ---
            # Bypass KV-Cache to prevent representation collapse desync
            logits_early, _ = model(input_ids, exit_layer=16, use_cache=False)
            mid_event.record()
            torch.cuda.synchronize() 
            
            elapsed_early_ms = start_event.elapsed_time(mid_event)
            probs = torch.softmax(logits_early[0, -1, :], dim=-1)
            confidence, next_token_early = torch.max(probs, dim=-1)
            conf_val = confidence.item()
            
            # --- STAGE 2: Dynamic Threshold Decay ---
            remaining_budget = deadline_ms - elapsed_early_ms
            if remaining_budget >= full_pass_wcet:
                time_ratio = (remaining_budget - full_pass_wcet) / (deadline_ms - full_pass_wcet)
                current_threshold = min_conf + (max_conf - min_conf) * time_ratio
            else:
                current_threshold = 0.0 

            # --- STAGE 3: The Decision ---
            if remaining_budget < full_pass_wcet:
                next_token = next_token_early
                exit_type = "Early (Forced)"
                end_event.record()
            elif conf_val >= current_threshold:
                next_token = next_token_early
                exit_type = f"Early (Thresh: {current_threshold:.2f})"
                end_event.record()
            else:
                logits_full, _ = model(input_ids, use_cache=False)
                next_token = torch.argmax(logits_full[0, -1, :], dim=-1)
                exit_type = "Full Pass"
                end_event.record()
            
            torch.cuda.synchronize()
            total_token_ms = start_event.elapsed_time(end_event)
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            
            word = model.tokenizer.decode([next_token.item()])
            print(f"Token {i+1:>2}: {word:<12} | Exit: {exit_type:<20} | Time: {total_token_ms:.2f} ms | Conf: {conf_val:.2f} | Active Thresh: {current_threshold:.2f}")

    final_text = model.tokenizer.decode(generated_tokens)
    print(f"\nFinal Output: {prompt} {final_text}")

if __name__ == "__main__":
    model = EarlyExitTinyLlama()
    prompt = "The most critical aspect of a real-time system is"
    # Run the strict stress test
    generate_stateless_anytime(model, prompt, deadline_ms=30.0)