"""
benchmark_utils.py — Extended PubMedQA helpers for sprint v2.

New vs v1:
  - run_pubmed_queries_raw()   collect per-token (l16_conf, l16_token, l22_token)
                                for post-hoc threshold replay in E02
  - load_pubmed_dataset()      load up to N samples from the labeled test split
  - calibration_eval_split()   split dataset into cal / eval halves
"""

import re
import numpy as np


def extract_label(generated_text):
    text = generated_text.strip().lower()
    for ch in ["'", '"', "*", "-", ".", ","]:
        text = text.lstrip(ch).strip()
    for prefix in ("answer:", "answer :", "response:", "response :"):
        if text.startswith(prefix):
            text = text.split(":", 1)[1].strip().lstrip(",0123456789. ").strip()
            break
    text = re.sub(r"^\([^)]{1,3}\)\s*", "", text)
    for label in ("yes", "no", "maybe"):
        if text.startswith(label):
            return label
    for word in text.split()[:10]:
        w = word.strip(".,;:'\"()")
        if w in ("yes", "no", "maybe"):
            return w
    return "unknown"


def build_prompt(tokenizer, context, question):
    messages = [
        {"role": "system",
         "content": ("You are a biomedical expert answering clinical questions. "
                     "Answer each question with exactly one word: 'yes', 'no', or 'maybe'. "
                     "Do not add any explanation.")},
        {"role": "user",
         "content": f"Context: {context}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_query_metrics(token_records, deadline_ms):
    n          = len(token_records)
    tpot_times = [r["time_ms"] for r in token_records[1:]] or [0.0]
    n_full   = sum(1 for r in token_records if r["exit_type"] == "Full Pass")
    n_thresh = sum(1 for r in token_records if r["exit_type"].startswith("Early (Thresh"))
    n_forced = sum(1 for r in token_records if r["exit_type"] == "Early (Forced)")
    n_miss   = sum(1 for r in token_records if r["time_ms"] > deadline_ms)
    return {
        "n_tokens":          n,
        "full_pass_pct":     round(100.0 * n_full   / n, 1),
        "early_thresh_pct":  round(100.0 * n_thresh / n, 1),
        "forced_exit_pct":   round(100.0 * n_forced / n, 1),
        "deadline_miss_pct": round(100.0 * n_miss   / n, 1),
        "mean_tpot_ms":      round(float(np.mean(tpot_times)), 3),
        "p99_tpot_ms":       round(float(np.percentile(tpot_times, 99)), 3),
        "max_tpot_ms":       round(float(np.max(tpot_times)), 3),
        "throughput_tps":    round(1000.0 / float(np.mean(tpot_times)), 2),
    }


def run_pubmed_queries(model, tokenizer, dataset, deadline_ms, max_new_tokens,
                       generate_fn, generate_kwargs=None):
    """Standard query runner — same API as v1."""
    generate_kwargs = generate_kwargs or {}
    query_results   = []
    n_correct = n_scored = 0

    for i, item in enumerate(dataset):
        context      = item["context"]["contexts"][0]
        question     = item["question"]
        ground_truth = item["final_decision"]
        prompt       = build_prompt(tokenizer, context, question)

        token_records  = generate_fn(model, prompt, max_new_tokens=max_new_tokens,
                                      deadline_ms=deadline_ms, verbose=False, **generate_kwargs)
        generated_text = "".join(r["token"] for r in token_records)
        predicted      = extract_label(generated_text)
        is_correct     = (predicted == ground_truth)

        if predicted != "unknown":
            n_scored += 1
            if is_correct:
                n_correct += 1

        query_results.append({
            "query_id":       i + 1,
            "question":       question,
            "ground_truth":   ground_truth,
            "predicted":      predicted,
            "generated_text": generated_text,
            "correct":        is_correct,
            "token_records":  token_records,
            "metrics":        compute_query_metrics(token_records, deadline_ms),
        })

    all_records = [r for q in query_results for r in q["token_records"]]
    all_tpot    = [r["time_ms"] for q in query_results for r in q["token_records"][1:]] or [0.0]
    mean_tpot   = float(np.mean(all_tpot))

    global_metrics = {
        "n_queries":           len(query_results),
        "deadline_ms":         deadline_ms,
        "n_correct":           n_correct,
        "n_scored":            n_scored,
        "accuracy":            round(100.0 * n_correct / n_scored, 1) if n_scored > 0 else None,
        "full_pass_pct":       round(100.0 * sum(1 for r in all_records if r["exit_type"] == "Full Pass") / len(all_records), 1),
        "early_thresh_pct":    round(100.0 * sum(1 for r in all_records if r["exit_type"].startswith("Early (Thresh")) / len(all_records), 1),
        "forced_exit_pct":     round(100.0 * sum(1 for r in all_records if r["exit_type"] == "Early (Forced)") / len(all_records), 1),
        "deadline_miss_pct":   round(100.0 * sum(1 for r in all_records if r["time_ms"] > deadline_ms) / len(all_records), 1),
        "global_mean_tpot_ms": round(mean_tpot, 3),
        "global_p99_tpot_ms":  round(float(np.percentile(all_tpot, 99)), 3),
        "throughput_tps":      round(1000.0 / mean_tpot, 2) if mean_tpot > 0 else None,
        "util_ratio":          round(float(np.percentile(all_tpot, 99)) / deadline_ms, 4),
    }
    return query_results, global_metrics


def run_pubmed_queries_raw(model, tokenizer, dataset, max_new_tokens=15, device="cuda"):
    """
    Run KV-cached forward pass on each query, collecting per-token raw data:
      - l16_conf     : max softmax confidence at Layer 16
      - l16_token_id : argmax token at Layer 16
      - l22_token_id : argmax token at full 22-layer pass (oracle)
      - time_ms      : CUDA-event-timed forward pass duration

    Used by E02 to replay different τ thresholds post-hoc without re-running the model.
    """
    import torch
    results = []

    with torch.inference_mode():
        for i, item in enumerate(dataset):
            context      = item["context"]["contexts"][0]
            question     = item["question"]
            ground_truth = item["final_decision"]
            prompt       = build_prompt(tokenizer, context, question)

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # Warmup
            _, _, wkv = model.forward_cached(input_ids)
            torch.cuda.synchronize()

            token_data = []
            past_kv    = None
            generated  = []

            for t in range(max_new_tokens):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev   = torch.cuda.Event(enable_timing=True)
                start_ev.record()

                if t == 0:
                    l16_logits, full_logits, past_kv = model.forward_cached(input_ids)
                else:
                    new_tok = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
                    l16_logits, full_logits, past_kv = model.forward_cached(
                        new_tok, past_key_values=past_kv)

                end_ev.record()
                torch.cuda.synchronize()
                elapsed = start_ev.elapsed_time(end_ev)

                probs = torch.softmax(l16_logits[0, -1, :], dim=-1)
                conf, l16_tok = torch.max(probs, dim=-1)
                l22_tok = torch.argmax(full_logits[0, -1, :], dim=-1)

                conf_val   = float(conf.item())
                l16_tok_id = int(l16_tok.item())
                l22_tok_id = int(l22_tok.item())

                token_data.append({
                    "t":           t,
                    "l16_conf":    round(conf_val, 6),
                    "l16_token_id": l16_tok_id,
                    "l22_token_id": l22_tok_id,
                    "time_ms":     round(elapsed, 3),
                    "l16_agrees_l22": l16_tok_id == l22_tok_id,
                })

                # Advance with full-model token (oracle trajectory)
                generated.append(l22_tok_id)
                if l22_tok_id == tokenizer.eos_token_id:
                    break

            results.append({
                "query_id":    i + 1,
                "ground_truth": ground_truth,
                "token_data":  token_data,
            })

    return results


def apply_threshold_posthoc(raw_results, threshold, tokenizer, deadline_ms=45.0):
    """
    Replay a threshold decision on raw token data collected by run_pubmed_queries_raw().
    For each token: if l16_conf >= threshold → commit l16 token; else → commit l22 token.
    Returns global_metrics dict.
    """
    n_correct = n_scored = 0
    all_tpot  = []
    all_miss  = []

    for q in raw_results:
        generated_ids = []
        for td in q["token_data"]:
            if td["l16_conf"] >= threshold:
                tok_id = td["l16_token_id"]
            else:
                tok_id = td["l22_token_id"]
            generated_ids.append(tok_id)
            all_tpot.append(td["time_ms"])
            all_miss.append(td["time_ms"] > deadline_ms)

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted      = extract_label(generated_text)
        if predicted != "unknown":
            n_scored += 1
            if predicted == q["ground_truth"]:
                n_correct += 1

    mean_tpot = float(np.mean(all_tpot)) if all_tpot else 0.0
    n_tokens  = len(all_tpot)

    n_l16 = sum(
        1 for q in raw_results for td in q["token_data"]
        if td["l16_conf"] >= threshold
    )

    return {
        "threshold":           threshold,
        "deadline_ms":         deadline_ms,
        "n_queries":           len(raw_results),
        "n_correct":           n_correct,
        "n_scored":            n_scored,
        "accuracy":            round(100.0 * n_correct / n_scored, 1) if n_scored > 0 else None,
        "early_exit_pct":      round(100.0 * n_l16 / max(1, n_tokens), 1),
        "deadline_miss_pct":   round(100.0 * sum(all_miss) / max(1, n_tokens), 1),
        "global_mean_tpot_ms": round(mean_tpot, 3),
        "global_p99_tpot_ms":  round(float(np.percentile(all_tpot, 99)), 3) if all_tpot else 0.0,
        "throughput_tps":      round(1000.0 / mean_tpot, 2) if mean_tpot > 0 else None,
    }


def load_pubmed_dataset(n_samples=500):
    """Load up to n_samples from PubMedQA labeled test split."""
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    return list(ds.select(range(min(n_samples, len(ds)))))


def calibration_eval_split(dataset):
    """Split dataset in half: first half = calibration, second half = evaluation."""
    mid = len(dataset) // 2
    return dataset[:mid], dataset[mid:]
