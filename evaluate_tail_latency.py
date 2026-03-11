import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import generate_stateless_anytime

def get_standard_autoregressive_latencies(prompt, max_tokens=50):
    print("Loading Standard Hugging Face Model for Baseline...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    latencies = []
    
    print("Running Standard Autoregressive Generation...")
    with torch.inference_mode():
        # Warmup
        _ = model(input_ids)
        torch.cuda.synchronize()
        
        for _ in range(max_tokens):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            outputs = model(input_ids, use_cache=False)
            next_token = torch.argmax(outputs.logits[0, -1, :], dim=-1)
            end.record()
            torch.cuda.synchronize()
            
            latencies.append(start.elapsed_time(end))
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            
    return latencies

def get_anytime_latencies(prompt, deadline_ms, max_tokens=50):
    print("\nLoading Dynamic Anytime Scheduler...")
    model = EarlyExitTinyLlama()
    
    # We will hijack the print statements from the scheduler 
    # and just measure the system from the outside for the benchmark
    tokenizer = model.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    latencies = []
    
    with torch.inference_mode():
        # Warmup
        _ = model(input_ids)
        torch.cuda.synchronize()
        
        full_pass_wcet = 23.0
        
        for _ in range(max_tokens):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            logits_early, _ = model(input_ids, exit_layer=16, use_cache=False)
            probs = torch.softmax(logits_early[0, -1, :], dim=-1)
            confidence, next_token_early = torch.max(probs, dim=-1)
            
            # Simulated deadline check (assuming ~15ms early exit)
            remaining_budget = deadline_ms - 15.0 
            
            if remaining_budget < full_pass_wcet or confidence.item() >= 0.55:
                next_token = next_token_early
            else:
                logits_full, _ = model(input_ids, use_cache=False)
                next_token = torch.argmax(logits_full[0, -1, :], dim=-1)
                
            end.record()
            torch.cuda.synchronize()
            
            latencies.append(start.elapsed_time(end))
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

    return latencies

if __name__ == "__main__":
    prompt = "The clinical presentation of acute myocardial infarction typically includes"
    deadline = 45.0
    
    print("="*60)
    print("SCHEDULABILITY EVALUATION: TAIL LATENCY BENCHMARK")
    print("="*60)
    
   # 1. Run Baseline
    baseline_latencies = get_standard_autoregressive_latencies(prompt)
    baseline_p99 = np.percentile(baseline_latencies[1:], 99) 
    baseline_mean = np.mean(baseline_latencies[1:]) # Add Mean
    
    # 2. Run Anytime Scheduler
    anytime_latencies = get_anytime_latencies(prompt, deadline_ms=deadline)
    anytime_p99 = np.percentile(anytime_latencies[1:], 99) 
    anytime_mean = np.mean(anytime_latencies[1:]) # Add Mean
    
    print("\n" + "="*60)
    print("FINAL SCHEDULABILITY METRICS (TPOT)")
    print("="*60)
    print(f"Standard Autoregressive Mean:            {baseline_mean:.2f} ms")
    print(f"Anytime Scheduler Mean:                  {anytime_mean:.2f} ms")
    print("-" * 60)
    print(f"Standard Autoregressive 99th Percentile: {baseline_p99:.2f} ms")
    print(f"Anytime Scheduler 99th Percentile:       {anytime_p99:.2f} ms")
    print(f"Target Hard Deadline:                    {deadline:.2f} ms")
    
    if anytime_p99 <= deadline:
        print("\n[SUCCESS] Tail latency is strictly bounded beneath the deadline!")
    else:
        print("\n[WARNING] Tail latency violated the deadline.")