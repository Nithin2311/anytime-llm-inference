"""
dynamic_scheduler.py — Three anytime routers for TinyLlama early-exit inference.

Sprint modifications vs. prototype:
  - _load_wcet_table() checks WCET_TABLE_PATH env var first. E0 sets this
    after regenerating the A100 WCET profile so all downstream experiments
    use the correct hardware table.
  - generate_anytime_with_kv and generate_anytime_async_overlap accept an
    explicit `threshold` kwarg for E2 confidence-threshold ablation.
  - reload_wcet_table() lets a running process refresh after E0 completes.
"""

import json
import os
import torch
from early_exit_model import EarlyExitTinyLlama  # noqa: F401 (re-exported for callers)


def _load_wcet_table(safety_factor=1.10, fallback_ms=18.5):
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "wcet_results.json"
    )
    wcet_file = os.environ.get("WCET_TABLE_PATH", default_path)
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


def _wcet_for_seq_len(seq_len, table):
    for profiled_len, wcet_ms in table:
        if seq_len <= profiled_len:
            return wcet_ms
    return table[-1][1]


_WCET_TABLE = _load_wcet_table()


def reload_wcet_table():
    """Refresh the in-process WCET table after E0 sets WCET_TABLE_PATH."""
    global _WCET_TABLE
    _WCET_TABLE = _load_wcet_table()


# ── Stateless two-pass router ─────────────────────────────────────────────────

def generate_stateless_anytime(model, prompt, max_new_tokens=15, deadline_ms=50.0,
                               max_conf=0.8, min_conf=0.3, verbose=True):
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.inference_mode():
        _ = model(input_ids)
        _ = model(input_ids, exit_layer=16, use_cache=False)
        _ = model(input_ids, exit_layer=16, use_cache=False)
    torch.cuda.synchronize()

    token_records    = []
    generated_tokens = []

    with torch.inference_mode():
        for i in range(max_new_tokens):
            start_event = torch.cuda.Event(enable_timing=True)
            mid_event   = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)

            start_event.record()
            full_pass_wcet  = _wcet_for_seq_len(input_ids.shape[1], _WCET_TABLE)
            logits_early, _ = model(input_ids, exit_layer=16, use_cache=False)
            mid_event.record()
            torch.cuda.synchronize()

            elapsed_early_ms = start_event.elapsed_time(mid_event)
            probs = torch.softmax(logits_early[0, -1, :], dim=-1)
            confidence, next_token_early = torch.max(probs, dim=-1)
            conf_val = confidence.item()

            remaining_budget = deadline_ms - elapsed_early_ms
            if remaining_budget >= full_pass_wcet:
                time_ratio        = (remaining_budget - full_pass_wcet) / (deadline_ms - full_pass_wcet)
                current_threshold = min_conf + (max_conf - min_conf) * time_ratio
            else:
                current_threshold = 0.0

            if remaining_budget < full_pass_wcet:
                next_token = next_token_early
                exit_type  = "Early (Forced)"
                end_event.record()
            elif conf_val >= current_threshold:
                next_token = next_token_early
                exit_type  = f"Early (Thresh: {current_threshold:.2f})"
                end_event.record()
            else:
                logits_full, _ = model(input_ids, use_cache=False)
                next_token     = torch.argmax(logits_full[0, -1, :], dim=-1)
                exit_type      = "Full Pass"
                end_event.record()

            torch.cuda.synchronize()
            total_token_ms = start_event.elapsed_time(end_event)

            token_id = next_token.item()
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

            token_records.append({
                "token_idx":   i + 1,
                "token":       model.tokenizer.decode([token_id]),
                "time_ms":     round(total_token_ms, 3),
                "exit_type":   exit_type,
                "confidence":  round(conf_val, 4),
                "threshold":   round(current_threshold, 4),
                "deadline_ms": deadline_ms,
            })

            if token_id == model.tokenizer.eos_token_id:
                break

    return token_records


# ── KV-cached single-pass router ──────────────────────────────────────────────

def generate_anytime_with_kv(model, prompt, max_new_tokens=15, deadline_ms=50.0,
                              max_conf=0.8, min_conf=0.3, threshold=None, verbose=True):
    """
    threshold: explicit confidence override for E2 ablation. When None, uses
               (max_conf + min_conf) / 2.0 matching the prototype default.
    """
    input_ids    = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    kv_threshold = threshold if threshold is not None else (max_conf + min_conf) / 2.0

    with torch.inference_mode():
        _, _, wkv = model.forward_cached(input_ids)
        dummy = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        model.forward_cached(dummy, past_key_values=wkv)
        model.forward_cached(dummy, past_key_values=wkv)
    torch.cuda.synchronize()

    past_kv          = None
    token_records    = []
    generated_tokens = []

    with torch.inference_mode():
        for i in range(max_new_tokens):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()

            if i == 0:
                l16_logits, full_logits, past_kv = model.forward_cached(input_ids)
            else:
                new_input = torch.tensor(
                    [[generated_tokens[-1]]], dtype=torch.long, device="cuda"
                )
                l16_logits, full_logits, past_kv = model.forward_cached(
                    new_input, past_key_values=past_kv
                )

            end_event.record()
            torch.cuda.synchronize()
            total_ms = start_event.elapsed_time(end_event)

            probs = torch.softmax(l16_logits[0, -1, :], dim=-1)
            confidence, next_l16 = torch.max(probs, dim=-1)
            conf_val = confidence.item()

            if conf_val >= kv_threshold:
                next_token = next_l16
                exit_type  = f"Early (Thresh: {kv_threshold:.2f})"
            else:
                next_token = torch.argmax(full_logits[0, -1, :], dim=-1)
                exit_type  = "Full Pass"

            token_id = next_token.item()
            generated_tokens.append(token_id)

            token_records.append({
                "token_idx":   i + 1,
                "token":       model.tokenizer.decode([token_id]),
                "time_ms":     round(total_ms, 3),
                "exit_type":   exit_type,
                "confidence":  round(conf_val, 4),
                "threshold":   round(kv_threshold, 4),
                "deadline_ms": deadline_ms,
            })

            if token_id == model.tokenizer.eos_token_id:
                break

    return token_records


# ── Async-overlap KV-cached router ────────────────────────────────────────────

def generate_anytime_async_overlap(model, prompt, max_new_tokens=15, deadline_ms=50.0,
                                    max_conf=0.8, min_conf=0.3, threshold=None, verbose=True):
    """threshold: same explicit override as generate_anytime_with_kv."""
    pinned_input     = torch.zeros((1, 1), dtype=torch.long).pin_memory()
    inference_stream = torch.cuda.Stream()
    input_ids        = model.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    kv_threshold     = threshold if threshold is not None else (max_conf + min_conf) / 2.0

    with torch.inference_mode():
        with torch.cuda.stream(inference_stream):
            _, _, wkv = model.forward_cached(input_ids)
            dummy = torch.zeros((1, 1), dtype=torch.long, device="cuda")
            model.forward_cached(dummy, past_key_values=wkv)
            model.forward_cached(dummy, past_key_values=wkv)
    inference_stream.synchronize()

    past_kv          = None
    token_records    = []
    generated_tokens = []
    prev_token_id    = None
    prev_total_ms    = None
    prev_exit_type   = None
    prev_conf_val    = None

    with torch.inference_mode():
        for i in range(max_new_tokens):
            if i == 0:
                cur_input = input_ids
            else:
                pinned_input[0, 0] = prev_token_id
                cur_input = pinned_input.to("cuda", non_blocking=True)
                inference_stream.wait_stream(torch.cuda.current_stream())

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            start_ev.record(stream=inference_stream)

            with torch.cuda.stream(inference_stream):
                l16_logits, full_logits, past_kv = model.forward_cached(
                    cur_input, past_key_values=past_kv
                )

            end_ev.record(stream=inference_stream)

            if i > 0:
                prev_word = model.tokenizer.decode([prev_token_id])
                token_records.append({
                    "token_idx":   i,
                    "token":       prev_word,
                    "time_ms":     round(prev_total_ms, 3),
                    "exit_type":   prev_exit_type,
                    "confidence":  round(prev_conf_val, 4),
                    "threshold":   round(kv_threshold, 4),
                    "deadline_ms": deadline_ms,
                })

            inference_stream.synchronize()
            total_ms = start_ev.elapsed_time(end_ev)

            probs = torch.softmax(l16_logits[0, -1, :], dim=-1)
            confidence, next_l16 = torch.max(probs, dim=-1)
            conf_val = confidence.item()

            if conf_val >= kv_threshold:
                next_token_gpu = next_l16
                exit_type = f"Early (Thresh: {kv_threshold:.2f})"
            else:
                next_token_gpu = torch.argmax(full_logits[0, -1, :], dim=-1)
                exit_type = "Full Pass"

            prev_token_id  = next_token_gpu.item()
            generated_tokens.append(prev_token_id)
            prev_total_ms  = total_ms
            prev_exit_type = exit_type
            prev_conf_val  = conf_val

            if prev_token_id == model.tokenizer.eos_token_id:
                break

    if prev_token_id is not None and (
        not token_records or token_records[-1]["token_idx"] < len(generated_tokens)
    ):
        token_records.append({
            "token_idx":   len(generated_tokens),
            "token":       model.tokenizer.decode([prev_token_id]),
            "time_ms":     round(prev_total_ms, 3),
            "exit_type":   prev_exit_type,
            "confidence":  round(prev_conf_val, 4),
            "threshold":   round(kv_threshold, 4),
            "deadline_ms": deadline_ms,
        })

    return token_records
