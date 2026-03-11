# Dynamic Anytime Scheduling for LLM Inference
**Bounding Tail Latency via Predictive Early-Exit Mechanisms**

**Author:** Nithin Palyam
**Course:** CIS 6930 - Real-Time Systems (Spring 2026)

## 📌 Project Overview
As generative AI is increasingly integrated into interactive Human-Computer Interaction (HCI) and cyber-physical systems, ensuring predictable response times becomes a critical safety requirement. Standard autoregressive generation in Large Language Models (LLMs) processes inputs layer-by-layer for every token, resulting in unbounded execution times and tail latency spikes that violate soft real-time constraints.

This project implements an uncertainty-aware **Anytime Algorithm** framework for LLM inference. By modifying a baseline causal model (TinyLlama-1.1B) with intermediate early-exit capabilities, the system utilizes a feedback control loop to guarantee bounded response times while maximizing semantic utility.

## 🚀 Objectives Achieved

### Phase 1: Base Implementation
- **Architecture Modification (`early_exit_model.py`):** Modified the base Transformer architecture to allow intermediate forward-pass halting (e.g., at Layer 5 or Layer 16) while successfully routing raw hidden states through the final Language Modeling (LM) head.
- **Microsecond Temporal Profiling (`profile_wcet.py`):** Established the Worst-Case Execution Time (WCET) of individual transformer layers using asynchronous `torch.cuda.Event` timing, completely isolating the GPU metrics from CPU/hypervisor overhead.
- **Static Anytime Scheduler (`static_scheduler.py`):** Built a static control loop that halts computation if a strict confidence threshold (e.g., 0.8) is met before a hard temporal deadline (e.g., 50.0 ms) expires.

### Phase 2: Extended Real-Time Features
- **Dynamic Threshold Decay (`dynamic_scheduler.py`):** Upgraded the scheduler to dynamically scale down the required confidence threshold as the temporal budget depletes. If the budget drops below the safety margin required for a full 22-layer pass, the threshold crashes to `0.0`, forcing an immediate maximized-utility exit to save the deadline.
- **KV-Cache Management & Bypass:** Addressed Representation Collapse and Key-Value desynchronization caused by skipped intermediate layers. Implemented a stateless cache bypass (`use_cache=False`) that successfully recalculates sequence contexts to maintain semantic integrity while remaining strictly under a 30.0 ms deadline. 

## 🛠️ System Architecture & Files
- `early_exit_model.py`: Contains the `EarlyExitTinyLlama` class. Overrides the standard Hugging Face `forward` pass to accept dynamic `exit_layer` and `use_cache` parameters.
- `profile_wcet.py`: Runs automated profiling matrices with warm-up loops to calculate precise GPU execution times per layer to inform the scheduler's safety margins.
- `static_scheduler.py`: The Phase 1 static control loop demonstrating bounded tail latency against fixed confidence parameters.
- `dynamic_scheduler.py`: The Phase 2 dynamic control loop featuring proportional temporal threshold decay and forced deadline-saving exits.

## 💻 Environment & Setup
This system requires a dedicated bare-metal GPU environment (e.g., RunPod RTX instances) to avoid shared-memory hypervisor noise during microsecond-level WCET profiling.

Dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers accelerate datasets
```

Authentication:
To prevent Hugging Face Hub rate-limiting (HTTP Error 429), ensure your environment is authenticated:

```Bash
export HF_TOKEN="your_huggingface_token"
```
⚙️ Usage
To run the dynamic scheduler and observe the token-by-token latency and confidence decay:

```Bash
python dynamic_scheduler.py`
```
Expected Output Metrics:
The terminal will output a step-by-step breakdown of the generation process, including:

Exit Type: Whether the token was generated via a Full Pass, Early (Thresh), or Early (Forced).

Time: The strict end-to-end execution time in milliseconds (bounded below the deadline).

Conf: The model's confidence probability at the intermediate early-exit layer.

Active Thresh: The dynamically sliding confidence threshold based on remaining temporal budget.

🔬 Next Steps (In Progress)
Domain-Specific Benchmarking: Evaluating the scheduler's utility-versus-latency trade-off under strict temporal bounds using clinical datasets (PubMedQA) for Trustworthy AI diagnostic workflows.
