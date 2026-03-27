"""
verify_env.py — Pre-flight environment check for anytime-llm-inference.

Validates:
  1. Python and key package versions
  2. CUDA availability and GPU identity
  3. GPU timing precision (async event timers)
  4. Model loadability (EarlyExitTinyLlama)
  5. Early-exit forward pass correctness (L16 vs full pass shape)
  6. WCET data availability and loaded safety margin
  7. PubMedQA dataset accessibility

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import sys
import json
import os

PASS  = "[PASS]"
FAIL  = "[FAIL]"
WARN  = "[WARN]"
INFO  = "[INFO]"

errors = []


def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}{': ' + str(result) if result else ''}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {label}: {e}")
        errors.append(label)
        return False


# ── 1. Python & package versions ─────────────────────────────────────────────
print("\n── Python & Packages ─────────────────────────────────────────")

check("Python >= 3.10",
      lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
              if sys.version_info >= (3, 10) else (_ for _ in ()).throw(RuntimeError("Python 3.10+ required")))

def _check_pkg(name, min_ver=None):
    import importlib
    mod = importlib.import_module(name.replace("-", "_"))
    ver = getattr(mod, "__version__", "unknown")
    if min_ver and tuple(int(x) for x in ver.split(".")[:2]) < tuple(int(x) for x in min_ver.split(".")[:2]):
        raise RuntimeError(f"version {ver} < required {min_ver}")
    return ver

for pkg, minv in [("torch", "2.0"), ("transformers", "4.30"), ("datasets", None),
                  ("numpy", None), ("matplotlib", None)]:
    check(f"{pkg} importable", lambda p=pkg, m=minv: _check_pkg(p, m))


# ── 2. CUDA & GPU ─────────────────────────────────────────────────────────────
print("\n── CUDA & GPU ────────────────────────────────────────────────")
import torch

check("CUDA available",
      lambda: f"CUDA {torch.version.cuda}" if torch.cuda.is_available()
              else (_ for _ in ()).throw(RuntimeError("CUDA not available — GPU required")))

if torch.cuda.is_available():
    check("GPU identity",
          lambda: torch.cuda.get_device_name(0))

    check("GPU memory >= 8 GB",
          lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                  if torch.cuda.get_device_properties(0).total_memory >= 8e9
                  else (_ for _ in ()).throw(RuntimeError("< 8 GB VRAM")))

    def _timing_precision():
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = torch.randn(1000, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e)
        if ms <= 0:
            raise RuntimeError("Timer returned non-positive value")
        return f"{ms:.2f} ms (matmul 1000×1000)"

    check("CUDA async event timing", _timing_precision)


# ── 3. Model loading & forward pass ──────────────────────────────────────────
print("\n── Model ─────────────────────────────────────────────────────")

def _load_model():
    from early_exit_model import EarlyExitTinyLlama
    model = EarlyExitTinyLlama()
    return f"{model.num_layers} layers | dtype={model._m.config.torch_dtype}"

def _early_exit_shapes():
    import torch
    from early_exit_model import EarlyExitTinyLlama
    model = EarlyExitTinyLlama()
    tok   = model.tokenizer
    ids   = tok("Hello world", return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        logits_l16, _ = model(ids, exit_layer=16, use_cache=False)
        logits_full, _= model(ids, use_cache=False)
    assert logits_l16.shape == logits_full.shape, \
        f"Shape mismatch: L16={logits_l16.shape} full={logits_full.shape}"
    return f"logits shape {tuple(logits_full.shape)} — consistent at L16 and full pass"

check("EarlyExitTinyLlama loads", _load_model)
check("L16 exit shape == full pass shape", _early_exit_shapes)


# ── 4. WCET data ──────────────────────────────────────────────────────────────
print("\n── WCET ──────────────────────────────────────────────────────")

def _wcet_data():
    wcet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wcet_results.json")
    with open(wcet_path) as f:
        data = json.load(f)
    max_full = max(v["None"]["wcet_ms"] for v in data["results"].values() if "None" in v)
    safety   = round(max_full * 1.10, 2)
    return f"max WCET={max_full} ms  |  safety margin={safety} ms (×1.10)"

check("wcet_results.json present and valid", _wcet_data)

def _scheduler_wcet():
    from dynamic_scheduler import _FULL_PASS_WCET_MS
    if _FULL_PASS_WCET_MS <= 0:
        raise RuntimeError(f"Unexpected WCET value: {_FULL_PASS_WCET_MS}")
    return f"_FULL_PASS_WCET_MS = {_FULL_PASS_WCET_MS} ms"

check("Dynamic scheduler loaded WCET from JSON", _scheduler_wcet)


# ── 5. Dataset ────────────────────────────────────────────────────────────────
print("\n── Dataset ───────────────────────────────────────────────────")

def _dataset():
    from datasets import load_dataset
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train[:2]")
    item = ds[0]
    assert "final_decision" in item, "Missing final_decision field"
    assert item["final_decision"] in ("yes", "no", "maybe"), "Unexpected label"
    return f"pqa_labeled accessible | first label: {item['final_decision']}"

check("PubMedQA pqa_labeled dataset", _dataset)


# ── 6. Chat template ──────────────────────────────────────────────────────────
print("\n── Prompt Pipeline ───────────────────────────────────────────")

def _chat_template():
    from transformers import AutoTokenizer
    from benchmark import _build_prompt
    tok    = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    prompt = _build_prompt(tok, "Aspirin reduces inflammation.", "Does aspirin work?")
    assert "<|system|>" in prompt, "Chat template not applied"
    assert "<|assistant|>" in prompt, "Missing assistant turn"
    return f"{len(prompt)} chars | chat template verified"

check("Chat template prompt format (_build_prompt)", _chat_template)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ───────────────────────────────────────────────────")
if not errors:
    print(f"  {PASS}  All checks passed. Environment is ready.\n")
    sys.exit(0)
else:
    print(f"  {FAIL}  {len(errors)} check(s) failed: {', '.join(errors)}\n")
    sys.exit(1)
