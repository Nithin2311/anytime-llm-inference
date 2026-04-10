import json
import os
import torch
from early_exit_model import EarlyExitTinyLlama


def _load_wcet_table(safety_factor=1.10, fallback_ms=18.5):
    """
    Load the full-pass WCET table from wcet_results.json and apply a safety
    margin to every entry.

    Args:
        safety_factor: Multiplier applied to all measured WCET values.
                       1.10 = 10% headroom above the measured worst-case,
                       chosen empirically to absorb GPU clock variation and
                       OS scheduling jitter while staying well below D=45ms.
        fallback_ms:   Value used when wcet_results.json is missing or corrupt.

    Returns a sorted list of (seq_len: int, wcet_ms: float) pairs.
    Falls back to a single sentinel entry if the file is missing.
    """
    wcet_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wcet_results.json")
    try:
        with open(wcet_file) as f:
            data = json.load(f)
        table = sorted(
            (int(seq_len), round(v["None"]["wcet_ms"] * safety_factor, 2))
            for seq_len, v in data["results"].items()
            if "None" in v
        )
        return table
    except (FileNotFoundError, KeyError, ValueError):
        return [(0, fallback_ms)]


def _wcet_for_seq_len(seq_len: int, table) -> float:
    """
    Return the safety-margined full-pass WCET for the given sequence length.

    Ceiling lookup: returns the WCET of the smallest profiled bin >= seq_len.
    If seq_len exceeds all profiled bins, returns the largest bin's value.
    """
    for profiled_len, wcet_ms in table:
        if seq_len <= profiled_len:
            return wcet_ms
    return table[-1][1]


# Loaded once at import time.
_WCET_TABLE = _load_wcet_table()


def generate_with_deadline(model, prompt, max_new_tokens=15,
                            deadline_ms=50.0, conf_threshold=0.8):
    """
    Phase 1 static anytime scheduler.

    Uses a fixed confidence threshold (no decay) and a Layer-16 early exit.
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
        _ = model(input_ids)                          # warm full-pass path
        _ = model(input_ids, exit_layer=16, use_cache=False)  # warm L16 path
        _ = model(input_ids, exit_layer=16, use_cache=False)  # second L16 pass
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

            # --- Stage 1: Early evaluation at Layer 16 ---
            # Layer 16 is used (not 5) so this comparison isolates scheduling
            # policy (fixed vs decaying threshold) rather than exit-layer choice.
            logits_early, _ = model(input_ids, exit_layer=16, use_cache=False)
            mid_event.record()
            torch.cuda.synchronize()

            elapsed_early_ms = start_event.elapsed_time(mid_event)
            probs = torch.softmax(logits_early[0, -1, :], dim=-1)
            confidence, next_token_early = torch.max(probs, dim=-1)
            conf_val = confidence.item()

            remaining_budget = deadline_ms - elapsed_early_ms

            # Look up WCET for the current input length.
            full_pass_safety_ms = _wcet_for_seq_len(input_ids.shape[1], _WCET_TABLE)

            # --- Stage 2: Static threshold decision ---
            if conf_val >= conf_threshold:
                next_token = next_token_early
                exit_type  = "Early (High Conf)"
                end_event.record()

            elif remaining_budget < full_pass_safety_ms:
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
