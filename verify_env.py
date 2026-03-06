import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Quick test of the async event timers
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    a = torch.randn(10000, 10000, device='cuda')
    b = a @ a
    end_event.record()
    
    torch.cuda.synchronize()
    print(f"Test MatMul Time: {start_event.elapsed_time(end_event):.2f} ms")