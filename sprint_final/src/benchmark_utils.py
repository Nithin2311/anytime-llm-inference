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

# Module-level tokenizer cache so post-hoc replay can decode without an
# explicit tokenizer kwarg (used by E06/E10/E13 which only carry queries).
_LAST_TOKENIZER = None


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


def run_pubmed_queries_raw(model, *args, max_new_tokens=15, device="cuda",
                           deadline_ms=None, show_progress=False,
                           forced_exit_layer=None):
    """
    Flexible-signature: collects per-token raw data from PubMedQA queries.

    Supported call shapes:
      - run_pubmed_queries_raw(model, tokenizer, dataset, max_new_tokens=...)   (E02 legacy)
      - run_pubmed_queries_raw(model, dataset, deadline_ms=..., max_new_tokens=...,
                               show_progress=..., forced_exit_layer=...)        (E06/E10/E13)

    Per-token data:
      - l16_conf, l16_token_id, l22_token_id, time_ms, l16_agrees_l22

    `deadline_ms` is recorded only — deadline filtering is done post-hoc.
    `forced_exit_layer` (int) forces a partial-depth pass via forward(exit_layer=L);
    when set, l22 logits are taken from the same partial pass (oracle == forced).
    """
    import torch
    if len(args) == 2:
        tokenizer, dataset = args
    elif len(args) == 1:
        dataset = args[0]
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise TypeError("model has no .tokenizer attribute and tokenizer not provided")
    else:
        raise TypeError(
            f"run_pubmed_queries_raw expects (model, tokenizer, dataset) or "
            f"(model, dataset); got {len(args)} positional args")

    global _LAST_TOKENIZER
    _LAST_TOKENIZER = tokenizer

    results = []

    n_total = len(dataset)
    with torch.inference_mode():
        for i, item in enumerate(dataset):
            if show_progress and (i % max(1, n_total // 20) == 0 or i == n_total - 1):
                print(f"      query {i+1}/{n_total}", flush=True)
            context      = item["context"]["contexts"][0]
            question     = item["question"]
            ground_truth = item["final_decision"]
            prompt       = build_prompt(tokenizer, context, question)

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # Warmup (skip past_kv when forced exit; we won't reuse the cache)
            if forced_exit_layer is None:
                _, _, _ = model.forward_cached(input_ids)
            else:
                model.forward_cached(input_ids, exit_layer=forced_exit_layer)
            torch.cuda.synchronize()

            token_data = []
            past_kv    = None
            generated  = []

            for t in range(max_new_tokens):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev   = torch.cuda.Event(enable_timing=True)
                start_ev.record()

                if forced_exit_layer is not None:
                    # Forced-exit path: no KV cache reuse — repeat full prompt + generated
                    if t == 0:
                        cur_ids = input_ids
                    else:
                        cur_ids = torch.cat([
                            input_ids,
                            torch.tensor([generated], dtype=torch.long, device=device)
                        ], dim=1)
                    l16_logits, full_logits, past_kv = model.forward_cached(
                        cur_ids, exit_layer=forced_exit_layer)
                elif t == 0:
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


def _decode_with_tokenizer(token_ids, tokenizer):
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def apply_threshold_posthoc(raw_results, *args, **kwargs):
    """
    Flexible-signature wrapper.

    Legacy (E02): apply_threshold_posthoc(raw_results, threshold, tokenizer, deadline_ms=45.0)
    New (E06/E10/E13): apply_threshold_posthoc(raw_results, queries, tau=..., deadline_ms=...,
                                                n_bootstrap=...)

    Both shapes share the same per-token replay; the new shape additionally returns
    `correct_flags` and `exit_rate_pct` / `miss_rate_pct` / `mean_tpot_ms` / `p99_tpot_ms`.
    """
    # ── Decide call shape ───────────────────────────────────────────────────
    if args and isinstance(args[0], (int, float)):
        threshold   = float(args[0])
        tokenizer   = args[1] if len(args) > 1 else kwargs.get("tokenizer")
        deadline_ms = args[2] if len(args) > 2 else kwargs.get("deadline_ms", 45.0)
        queries     = None
    else:
        queries     = args[0] if args else kwargs.get("queries")
        threshold   = float(kwargs.get("tau", kwargs.get("threshold", 0.55)))
        deadline_ms = float(kwargs.get("deadline_ms", 45.0))
        tokenizer   = kwargs.get("tokenizer")

    if tokenizer is None:
        tokenizer = _LAST_TOKENIZER

    n_correct = n_scored = 0
    all_tpot  = []
    all_miss  = []
    correct_flags = []
    n_l16_tokens = 0
    n_tokens = 0

    for q in raw_results:
        generated_ids = []
        for td in q["token_data"]:
            if td["l16_conf"] >= threshold:
                tok_id = td["l16_token_id"]
                n_l16_tokens += 1
            else:
                tok_id = td["l22_token_id"]
            generated_ids.append(tok_id)
            all_tpot.append(td["time_ms"])
            all_miss.append(td["time_ms"] > deadline_ms)
            n_tokens += 1

        # Decode + score
        if tokenizer is not None:
            generated_text = _decode_with_tokenizer(generated_ids, tokenizer)
            predicted = extract_label(generated_text)
        else:
            predicted = "unknown"

        if predicted != "unknown":
            n_scored += 1
            is_correct = (predicted == q["ground_truth"])
            if is_correct:
                n_correct += 1
            correct_flags.append(bool(is_correct))

    mean_tpot = float(np.mean(all_tpot)) if all_tpot else 0.0
    p99_tpot  = float(np.percentile(all_tpot, 99)) if all_tpot else 0.0
    miss_rate_pct = 100.0 * sum(all_miss) / max(1, n_tokens)
    exit_rate_pct = 100.0 * n_l16_tokens / max(1, n_tokens)
    accuracy_pct  = (100.0 * n_correct / n_scored) if n_scored > 0 else 0.0

    return {
        "threshold":           threshold,
        "tau":                 threshold,
        "deadline_ms":         deadline_ms,
        "n_queries":           len(raw_results),
        "n_correct":           n_correct,
        "n_scored":            n_scored,
        "accuracy":            round(accuracy_pct, 1),
        "correct_flags":       correct_flags,
        # legacy field names
        "early_exit_pct":      round(exit_rate_pct, 1),
        "deadline_miss_pct":   round(miss_rate_pct, 1),
        "global_mean_tpot_ms": round(mean_tpot, 3),
        "global_p99_tpot_ms":  round(p99_tpot, 3),
        "throughput_tps":      round(1000.0 / mean_tpot, 2) if mean_tpot > 0 else None,
        # new field names expected by E06/E10/E13
        "exit_rate_pct":       round(exit_rate_pct, 1),
        "miss_rate_pct":       round(miss_rate_pct, 1),
        "mean_tpot_ms":        round(mean_tpot, 3),
        "p99_tpot_ms":         round(p99_tpot, 3),
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
