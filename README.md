# Dynamic Anytime Scheduling for LLM Inference

**Author:** Nithin Palyam
**Course:** Real-Time Systems (Spring 2026)

## Project Overview
As generative AI is integrated into interactive and cyber-physical systems, ensuring predictable response times is a critical safety requirement. Standard autoregressive generation in causal Large Language Models (LLMs) processes inputs layer-by-layer, resulting in unbounded execution times and significant tail latency spikes that violate soft real-time constraints.

This project investigates early-exit and anytime capabilities in LLM inference pipelines. By attaching intermediate evaluation points to a base model (TinyLlama), we designed an uncertainty-aware scheduler that guarantees bounded response times using an Anytime Algorithm framework.

## Key Features & Phases

### Phase 1: Base Implementation
* **Architecture Modification:** Modified the base causal model to support intermediate exits by routing hidden states through the language modeling head at earlier transformer blocks.
* **Microsecond Temporal Profiling:** Established Worst-Case Execution Time (WCET) bounds using asynchronous `torch.cuda.Event` timing to bypass shared-memory hypervisor noise.
* **Static Anytime Scheduler:** A baseline control loop that halts computation early if a rigid confidence threshold (e.g., 80%) is met before a hard deadline expires.

### Phase 2: Advanced Scheduling
* **Dynamic Threshold Decay:** An upgraded scheduler that linearly scales the required confidence threshold based on the remaining temporal budget, forcing maximized-utility exits just before a deadline miss.
* **KV-Cache Bypass (Stateless Execution):** Addresses the representation collapse and memory desynchronization (hallucinations) caused by skipping deeper layers during early exits. The engine falls back to stateless generation, recalculating the sequence context to maintain semantic integrity while strictly respecting temporal bounds.

## Project Structure
* `early_exit_model.py`: Contains the modified TinyLlama architecture supporting custom exit routing.
* `profile_wcet.py`: GPU benchmarking scripts for establishing base execution times per layer.
* `static_scheduler.py`: The Phase 1 control loop with rigid temporal boundaries.
* `dynamic_scheduler.py`: The Phase 2 control loop featuring threshold decay and stateless cache bypass.

## Installation & Setup
Rigorous timing experiments require dedicated, bare-metal GPU instances. 

1. Clone the repository:
   ```bash
   git clone git@github.com:Nithin2311/anytime-llm-inference.git
   cd anytime-llm-inference
   '''
2. Set up the virtual environment:
  python -m venv venv
  source venv/bin/activate

3.Install dependencies:
  pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
  pip install transformers accelerate datasets

4. Authenticate Hugging Face to bypass rate limits:
   export HF_TOKEN="your_token_here"

Usage
To test the Phase 2 scheduler under strict temporal pressure (e.g., 30ms hard deadline):
python dynamic_scheduler.py

   
   
