import torch
from early_exit_model import EarlyExitTinyLlama

def profile_gpu_execution(func, *args, num_warmup=10, num_runs=100, **kwargs):
    """
    Profiles the GPU execution time using precise asynchronous events.
    """
    # 1. Warm-up to initialize CUDA kernels and prevent cold-start latency spikes
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = func(*args, **kwargs)
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    
    # 2. Execute the profiling runs
    with torch.no_grad():
        for i in range(num_runs):
            start_events[i].record()
            _ = func(*args, **kwargs)
            end_events[i].record()
            
    torch.cuda.synchronize() # Block until all GPU operations complete
    
    # 3. Calculate metrics
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    avg_time = sum(times_ms) / len(times_ms)
    wcet = max(times_ms)
    
    return avg_time, wcet

if __name__ == "__main__":
    # Initialize the model and tokenizer
    model = EarlyExitTinyLlama()
    
    # Use a standard input prompt
    prompt = "The most critical aspect of a real-time system is"
    inputs = model.tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("\n" + "="*40)
    print("Initiating WCET Profiling (100 runs)")
    print("="*40)
    
    # Test 1: Full Autoregressive Pass
    print("\nProfiling Full Model (22 Layers)...")
    avg_full, wcet_full = profile_gpu_execution(model, inputs.input_ids)
    print(f"-> Average Time: {avg_full:.4f} ms")
    print(f"-> WCET:         {wcet_full:.4f} ms")
    
    # Test 2: Early Exit at Layer 11
    print("\nProfiling Early Exit (Layer 11)...")
    avg_half, wcet_half = profile_gpu_execution(model, inputs.input_ids, exit_layer=11)
    print(f"-> Average Time: {avg_half:.4f} ms")
    print(f"-> WCET:         {wcet_half:.4f} ms")
    
    # Test 3: Aggressive Early Exit at Layer 5
    print("\nProfiling Aggressive Early Exit (Layer 5)...")
    avg_early, wcet_early = profile_gpu_execution(model, inputs.input_ids, exit_layer=5)
    print(f"-> Average Time: {avg_early:.4f} ms")
    print(f"-> WCET:         {wcet_early:.4f} ms")