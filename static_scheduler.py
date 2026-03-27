import json
import os
import torch
from early_exit_model import EarlyExitTinyLlama


def _load_full_pass_wcet(safety_factor=1.10, fallback_ms=18.5):
    """
    Load the measured full-pass WCET from wcet_results.json and apply a
    safety margin.  Falls back to `fallback_ms` if the file is missing.
    """
    wcet_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wcet_results.json")
    try:
        with open(wcet_file) as f:
            data = json.load(f)
        max_wcet = max(
            v["None"]["wcet_ms"]
            for v in data["results"].values()
            if "None" in v
        )
        return round(max_wcet * safety_factor, 2)
    except (FileNotFoundError, KeyError, ValueError):
        return fallback_ms


# Safety margin: minimum remaining budget needed to safely attempt a full pass.
# Loaded from wcet_results.json (max observed WCET × 1.10 safety factor).
FULL_PASS_SAFETY_MS = _load_full_pass_wcet()


def generate_with_deadline(model, prompt, max_new_tokens=15,
                            deadline_ms=50.0, conf_threshold=0.8):
    """
    Phase 1 static anytime scheduler.

    Uses a fixed confidence threshold (no decay) and a Layer-5 early exit.
    If the early exit confidence exceeds the threshold, the token is committed
    immediately. If the budget is nearly exhausted, a forced early exit is taken.
    Otherwise the full 22-layer pass is executed.

    Returns:
        List of per-token dicts with keys:
            token_idx, token, time_ms, exit_type, confidence, deadline_ms
    """
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    print("\n" + "=" * 50)
    print(f"[Static Scheduler] Anytime Generation")
    print(f"Deadline: {deadline_ms} ms | Fixed Conf Threshold: {conf_threshold}")
    print("=" * 50 + "\n")

    print("Warming up GPU kernels...")
    with torch.inference_mode():
        _ = model(input_ids)
    torch.cuda.synchronize()
    print("Warm-up complete. Starting strict timing.\n")

    token_records  = []
    generated_tokens = []

    with torch.inference_mode():
        for i in range(max_new_tokens):
            start_event = torch.cuda.Event(enable_timing=True)
            mid_event   = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)

            start_event.record()

            # --- Stage 1: Early evaluation at Layer 5 ---
            logits_early, _ = model(input_ids, exit_layer=5, use_cache=False)
            mid_event.record()
            torch.cuda.synchronize()

            elapsed_early_ms = start_event.elapsed_time(mid_event)
            probs = torch.softmax(logits_early[0, -1, :], dim=-1)
            confidence, next_token_early = torch.max(probs, dim=-1)
            conf_val = confidence.item()

            remaining_budget = deadline_ms - elapsed_early_ms

            # --- Stage 2: Static threshold decision ---
            if conf_val >= conf_threshold:
                next_token = next_token_early
                exit_type  = "Early (High Conf)"
                end_event.record()

            elif remaining_budget < FULL_PASS_SAFETY_MS:
                next_token = next_token_early
                exit_type  = "Early (Deadline)"
                end_event.record()

            else:
                logits_full, _ = model(input_ids, use_cache=False)
                next_token = torch.argmax(logits_full[0, -1, :], dim=-1)
                exit_type  = "Full Pass"
                end_event.record()

            torch.cuda.synchronize()
            total_token_ms = start_event.elapsed_time(end_event)

            token_id = next_token.item()
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

            word = model.tokenizer.decode([token_id])
            print(f"Token {i+1:>2}: {word:<12} | Exit: {exit_type:<18} | "
                  f"Time: {total_token_ms:.2f} ms | Conf: {conf_val:.2f}")

            token_records.append({
                "token_idx":   i + 1,
                "token":       word,
                "time_ms":     round(total_token_ms, 3),
                "exit_type":   exit_type,
                "confidence":  round(conf_val, 4),
                "threshold":   conf_threshold,   # static — never changes
                "deadline_ms": deadline_ms,
            })

            # Stop early if EOS is generated
            if token_id == model.tokenizer.eos_token_id:
                print(f"  [EOS reached at token {i+1}]")
                break

    final_text = model.tokenizer.decode(generated_tokens)
    print(f"\nFinal Output: {prompt} {final_text}")
    return token_records


if __name__ == "__main__":
    model  = EarlyExitTinyLlama()
    prompt = "The most critical aspect of a real-time system is"
    generate_with_deadline(model, prompt, deadline_ms=45.0)
