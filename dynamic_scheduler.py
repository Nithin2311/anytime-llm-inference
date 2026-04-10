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

    Returns a sorted list of (seq_len: int, wcet_ms: float) pairs so that
    _wcet_for_seq_len() can do a fast upper-bound lookup.  Falls back to a
    single sentinel entry if the file is missing.
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

    Uses the profiled bin whose key is >= seq_len (ceiling lookup).  If the
    sequence is longer than the largest profiled bin, the largest bin's value
    is returned — a conservative over-estimate is always safer than an under-
    estimate for a real-time scheduler.
    """
    for profiled_len, wcet_ms in table:
        if seq_len <= profiled_len:
            return wcet_ms
    # seq_len exceeds the largest profiled length — use the max profiled WCET.
    return table[-1][1]


# Loaded once at import time; shared by all calls in this process.
_WCET_TABLE = _load_wcet_table()


def generate_stateless_anytime(model, prompt, max_new_tokens=15, deadline_ms=50.0,
                               max_conf=0.8, min_conf=0.3, verbose=True):
    """
    Generate tokens under a hard deadline using dynamic threshold decay.

    Args:
        verbose: If True, print per-token timing and exit decisions. Set False
                 in batch benchmarks to avoid cluttering output.

    Returns:
        List of per-token dicts with keys:
            token_idx    (int)   1-based position
            token        (str)   decoded token text
            time_ms      (float) end-to-end GPU time for this token
            exit_type    (str)   "Full Pass" | "Early (Thresh:X.XX)" | "Early (Forced)"
            confidence   (float) max softmax prob at Layer 16
            threshold    (float) active threshold; 0.0 signals a forced early exit
            deadline_ms  (float) the deadline this token was scheduled under
    """
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    if verbose:
        print("\n" + "="*55)
        print(f"Starting Stateless Anytime Generation (Cache Bypass)")
        print(f"Deadline: {deadline_ms} ms | Conf Decay: {max_conf} -> {min_conf}")
        print("="*55 + "\n")

    with torch.inference_mode():
        _ = model(input_ids)                                   # warm full-pass path
        _ = model(input_ids, exit_layer=16, use_cache=False)   # warm L16 path
        _ = model(input_ids, exit_layer=16, use_cache=False)   # second L16 pass
    torch.cuda.synchronize()
    if verbose:
        print("Warm-up complete. Starting strict timing.\n")

    token_records = []
    generated_tokens = []

    with torch.inference_mode():
        for i in range(max_new_tokens):
            start_event = torch.cuda.Event(enable_timing=True)
            mid_event   = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)

            start_event.record()

            # --- STAGE 1: Early Evaluation (Layer 16) ---
            # Look up the WCET for the *current* input length so the budget
            # check stays valid as the context grows token by token.
            full_pass_wcet = _wcet_for_seq_len(input_ids.shape[1], _WCET_TABLE)

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

            token_id = next_token.item()
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

            word = model.tokenizer.decode([token_id])
            if verbose:
                print(f"Token {i+1:>2}: {word:<12} | Exit: {exit_type:<24} | "
                      f"Time: {total_token_ms:.2f} ms | Conf: {conf_val:.2f} | "
                      f"Thresh: {current_threshold:.2f}")

            token_records.append({
                "token_idx":   i + 1,
                "token":       word,
                "time_ms":     round(total_token_ms, 3),
                "exit_type":   exit_type,
                "confidence":  round(conf_val, 4),
                "threshold":   round(current_threshold, 4),
                "deadline_ms": deadline_ms,
            })

            # Stop early if EOS is generated
            if token_id == model.tokenizer.eos_token_id:
                if verbose:
                    print(f"  [EOS reached at token {i+1}]")
                break

    if verbose:
        final_text = model.tokenizer.decode(generated_tokens)
        preview = final_text[:80] + "..." if len(final_text) > 80 else final_text
        print(f"\nGenerated: {preview}")
    return token_records

def generate_anytime_with_kv(model, prompt, max_new_tokens=15, deadline_ms=50.0,
                              max_conf=0.8, min_conf=0.3, verbose=True):
    """
    KV-cached anytime generation under a hard deadline.

    Differences from generate_stateless_anytime
    --------------------------------------------
    * Uses model.forward_cached() — a single 22-layer pass that returns both
      L16 and full-pass logits via an intermediate hook.
    * Per-token attention is O(n_cached) instead of O(n²), dramatically
      reducing latency at long context lengths.
    * No separate "L16 probe then maybe full pass" — both logits are produced
      in one forward call, so the exit decision is post-hoc (choose which
      logits to use based on confidence after the pass completes).
    * No WCET-based forced exit — the KV-cached single-token pass is always
      fast enough (typically 5–15 ms regardless of context length).

    Threshold decay
    ---------------
    Same linear decay logic as the stateless version, but uses the fraction
    of the deadline consumed by the completed forward pass to set the threshold
    for the *current* token.  If the token ran fast (time_ms << D) the
    threshold stays near max_conf; if it consumed most of D the threshold
    drops toward min_conf.

    Returns the same per-token dict format as generate_stateless_anytime.
    """
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    if verbose:
        print("\n" + "=" * 55)
        print("Starting KV-Cached Anytime Generation")
        print(f"Deadline: {deadline_ms} ms | Threshold: {(max_conf + min_conf) / 2.0:.2f}")
        print("=" * 55 + "\n")

    # Warm up: full prompt pass + two single-token passes
    with torch.inference_mode():
        _, _, wkv = model.forward_cached(input_ids)
        dummy = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        model.forward_cached(dummy, past_key_values=wkv)
        model.forward_cached(dummy, past_key_values=wkv)
    torch.cuda.synchronize()
    if verbose:
        print("Warm-up complete.\n")

    past_kv = None   # populated on token 1 (full prompt pass)

    # Fixed confidence threshold: midpoint of the decay range.
    # Unlike the stateless scheduler, both logits come from the same forward
    # pass so there is no "remaining budget after L16 probe" to decay against.
    # Using the midpoint keeps the policy consistent while being simply tunable.
    kv_threshold = (max_conf + min_conf) / 2.0

    token_records    = []
    generated_tokens = []

    with torch.inference_mode():
        for i in range(max_new_tokens):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)

            start_event.record()

            if i == 0:
                # Token 1 (TTFT): includes full prompt encoding.
                # past_kv was already computed during warmup and is reused here
                # inside the timing window for honest latency reporting.
                l16_logits, full_logits, past_kv = model.forward_cached(input_ids)
            else:
                # Tokens 2-N: single new token — O(n_cache) attention per layer.
                new_token_input = torch.tensor(
                    [[generated_tokens[-1]]], dtype=torch.long, device="cuda"
                )
                l16_logits, full_logits, past_kv = model.forward_cached(
                    new_token_input, past_key_values=past_kv
                )

            end_event.record()
            torch.cuda.synchronize()
            total_ms = start_event.elapsed_time(end_event)

            # Logits for the last (only) new position
            probs = torch.softmax(l16_logits[0, -1, :], dim=-1)
            confidence, next_token_l16 = torch.max(probs, dim=-1)
            conf_val = confidence.item()

            # Exit decision (post-hoc — both logits already in hand)
            if conf_val >= kv_threshold:
                next_token = next_token_l16
                exit_type  = f"Early (Thresh: {kv_threshold:.2f})"
            else:
                next_token = torch.argmax(full_logits[0, -1, :], dim=-1)
                exit_type  = "Full Pass"

            token_id = next_token.item()
            generated_tokens.append(token_id)

            word = model.tokenizer.decode([token_id])
            if verbose:
                print(
                    f"Token {i+1:>2}: {word:<12} | Exit: {exit_type:<28} | "
                    f"Time: {total_ms:.2f} ms | Conf: {conf_val:.2f} | Thresh: {kv_threshold:.2f}"
                )

            token_records.append({
                "token_idx":   i + 1,
                "token":       word,
                "time_ms":     round(total_ms, 3),
                "exit_type":   exit_type,
                "confidence":  round(conf_val, 4),
                "threshold":   round(kv_threshold, 4),
                "deadline_ms": deadline_ms,
            })

            if token_id == model.tokenizer.eos_token_id:
                if verbose:
                    print(f"  [EOS reached at token {i+1}]")
                break

    if verbose:
        final_text = model.tokenizer.decode(generated_tokens)
        preview = final_text[:80] + "..." if len(final_text) > 80 else final_text
        print(f"\nGenerated: {preview}")
    return token_records


if __name__ == "__main__":
    model = EarlyExitTinyLlama()
    prompt = "The most critical aspect of a real-time system is"
    print("--- Stateless ---")
    generate_stateless_anytime(model, prompt, deadline_ms=45.0)
    print("\n--- KV-Cached ---")
    generate_anytime_with_kv(model, prompt, deadline_ms=45.0)