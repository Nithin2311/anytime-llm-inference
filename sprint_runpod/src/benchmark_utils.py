"""
benchmark_utils.py — Shared PubMedQA helpers extracted from benchmark.py.

Provides label extraction, prompt formatting, and per-query metric aggregation
used by E2, E3, E5, and E6.
"""

import re
import numpy as np


def extract_label(generated_text):
    """Extract yes/no/maybe from the start of generated text."""
    text = generated_text.strip().lower()
    for ch in ["'", '"', "*", "-", ".", ","]:
        text = text.lstrip(ch).strip()
    for prefix in ("answer:", "answer :", "response:", "response :"):
        if text.startswith(prefix):
            text = text.split(":", 1)[1].strip().lstrip(",").lstrip("0123456789. ").strip()
            break
    text = re.sub(r"^\([^)]{1,3}\)\s*", "", text)
    for label in ("yes", "no", "maybe"):
        if text.startswith(label):
            return label
    for word in text.split()[:10]:
        word = word.strip(".,;:'\"()")
        if word in ("yes", "no", "maybe"):
            return word
    return "unknown"


def build_prompt(tokenizer, context, question):
    """Format query using TinyLlama chat template with biomedical system prompt."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical expert answering clinical questions. "
                "Answer each question with exactly one word: 'yes', 'no', or 'maybe'. "
                "Do not add any explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}",
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_query_metrics(token_records, deadline_ms):
    """Aggregate per-query scheduling metrics from a token_records list."""
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
        "mean_tpot_ms":      round(float(np.mean(tpot_times)),            3),
        "p99_tpot_ms":       round(float(np.percentile(tpot_times, 99)), 3),
        "max_tpot_ms":       round(float(np.max(tpot_times)),             3),
        "throughput_tps":    round(1000.0 / float(np.mean(tpot_times)), 2),
    }


def run_pubmed_queries(model, tokenizer, dataset, deadline_ms, max_new_tokens,
                       generate_fn, generate_kwargs=None):
    """
    Run generate_fn over all dataset items.

    generate_fn signature: (model, prompt, max_new_tokens=N, deadline_ms=D,
                             verbose=False, **generate_kwargs)
    Returns (query_results list, global_metrics dict).
    """
    generate_kwargs = generate_kwargs or {}
    query_results   = []
    n_correct = 0
    n_scored  = 0

    for i, item in enumerate(dataset):
        context      = item["context"]["contexts"][0]
        question     = item["question"]
        ground_truth = item["final_decision"]
        prompt       = build_prompt(tokenizer, context, question)

        token_records  = generate_fn(
            model, prompt,
            max_new_tokens=max_new_tokens,
            deadline_ms=deadline_ms,
            verbose=False,
            **generate_kwargs,
        )
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
