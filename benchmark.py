import csv
import json
import re
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import generate_stateless_anytime

RESULTS_FILE = "benchmark_results.json"
CSV_FILE     = "benchmark_results.csv"


def extract_label(generated_text):
    """
    Extract a yes/no/maybe decision from the start of generated text.

    Handles the following common model output patterns:
      - "yes, ..."                → direct answer (ideal)
      - "Answer: yes, ..."        → TinyLlama answer-prefix pattern
      - "Answer:yes,..."          → same but tokens joined without spaces
      - "Response:\\n\\nYes, ..." → TinyLlama response-prefix pattern
      - "(i)Yes, ..."             → leading parenthetical prefix
    Returns 'unknown' if no label is found in the first 10 tokens.
    """
    text = generated_text.strip().lower()
    # Strip common punctuation prefix artifacts
    for ch in ["'", '"', "*", "-", ".", ","]:
        text = text.lstrip(ch).strip()
    # Strip "Answer:" / "Response:" style prefixes (with or without space after colon)
    for prefix in ("answer:", "answer :", "response:", "response :"):
        if text.startswith(prefix):
            text = text.split(":", 1)[1].strip().lstrip(",").lstrip("0123456789. ").strip()
            break
    # Strip leading parenthetical like "(i)", "(1)", "(a)"
    text = re.sub(r"^\([^)]{1,3}\)\s*", "", text)
    for label in ("yes", "no", "maybe"):
        if text.startswith(label):
            return label
    # Fallback: check first ten whitespace-separated pieces
    for word in text.split()[:10]:
        word = word.strip(".,;:'\"()")
        if word in ("yes", "no", "maybe"):
            return word
    return "unknown"


def compute_query_metrics(token_records, deadline_ms):
    """Aggregate per-query scheduling metrics from token_records list."""
    n = len(token_records)
    tpot_records = token_records[1:]  # skip TTFT (first token = prefill)

    n_full   = sum(1 for r in token_records if r["exit_type"] == "Full Pass")
    n_thresh = sum(1 for r in token_records if r["exit_type"].startswith("Early (Thresh"))
    n_forced = sum(1 for r in token_records if r["exit_type"] == "Early (Forced)")
    n_miss   = sum(1 for r in token_records if r["time_ms"] > deadline_ms)

    tpot_times = [r["time_ms"] for r in tpot_records] if tpot_records else [0.0]
    mean_tpot  = float(np.mean(tpot_times))

    return {
        "n_tokens":          n,
        "full_pass_pct":     round(100.0 * n_full   / n, 1),
        "early_thresh_pct":  round(100.0 * n_thresh / n, 1),
        "forced_exit_pct":   round(100.0 * n_forced / n, 1),
        "deadline_miss_pct": round(100.0 * n_miss   / n, 1),
        "mean_tpot_ms":      round(mean_tpot,  3),
        "p99_tpot_ms":       round(float(np.percentile(tpot_times, 99)), 3),
        "max_tpot_ms":       round(float(np.max(tpot_times)), 3),
        "throughput_tps":    round(1000.0 / mean_tpot, 2) if mean_tpot > 0 else None,
    }


def _build_prompt(tokenizer, context, question):
    """
    Format the query using TinyLlama's chat template with an explicit
    system instruction to force a yes/no/maybe response.  This dramatically
    improves label extraction compared to a plain "Answer:" suffix.
    """
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


def run_pubmed_benchmark(n_samples=30, deadline_ms=45.0, max_new_tokens=15):
    print("Loading PubMedQA Dataset...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split=f"train[:{n_samples}]")

    model     = EarlyExitTinyLlama()
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("\n" + "#" * 60)
    print(f"CLINICAL DOMAIN BENCHMARK  |  deadline={deadline_ms} ms  |  n={n_samples}")
    print("#" * 60)

    query_results = []
    n_correct = 0
    n_scored  = 0  # queries where a label was successfully extracted

    for i, item in enumerate(dataset):
        context      = item["context"]["contexts"][0]
        question     = item["question"]
        ground_truth = item["final_decision"]   # "yes" | "no" | "maybe"
        prompt       = _build_prompt(tokenizer, context, question)

        print(f"\n[ Clinical Query {i+1}/{n_samples} ]")
        print(f"Ground truth : {ground_truth}")
        print(f"Prompt length: {len(prompt.split())} words")

        token_records = generate_stateless_anytime(
            model, prompt,
            max_new_tokens=max_new_tokens,
            deadline_ms=deadline_ms,
        )

        generated_text = "".join(r["token"] for r in token_records)
        predicted      = extract_label(generated_text)
        is_correct     = (predicted == ground_truth)

        if predicted != "unknown":
            n_scored += 1
            if is_correct:
                n_correct += 1

        metrics = compute_query_metrics(token_records, deadline_ms)

        print(f"Predicted    : {predicted}  |  Correct: {is_correct}")
        print(f"Exit dist    : Full={metrics['full_pass_pct']}%  "
              f"Thresh={metrics['early_thresh_pct']}%  "
              f"Forced={metrics['forced_exit_pct']}%  "
              f"Misses={metrics['deadline_miss_pct']}%")
        print(f"TPOT         : mean={metrics['mean_tpot_ms']} ms  "
              f"P99={metrics['p99_tpot_ms']} ms  "
              f"max={metrics['max_tpot_ms']} ms")
        print("-" * 60)

        query_results.append({
            "query_id":       i + 1,
            "question":       question,
            "ground_truth":   ground_truth,
            "predicted":      predicted,
            "correct":        is_correct,
            "generated_text": generated_text,
            "token_records":  token_records,
            "metrics":        metrics,
        })

    # --- Global summary ---
    all_records  = [r for q in query_results for r in q["token_records"]]
    all_tpot     = [r["time_ms"] for q in query_results for r in q["token_records"][1:]]

    mean_tpot_ms  = float(np.mean(all_tpot))
    p99_tpot_ms   = float(np.percentile(all_tpot, 99))
    # Throughput: tokens per second derived from mean TPOT
    throughput_tps = round(1000.0 / mean_tpot_ms, 2) if mean_tpot_ms > 0 else None

    global_metrics = {
        "n_queries":          n_samples,
        "deadline_ms":        deadline_ms,
        "accuracy":           round(100.0 * n_correct / n_scored, 1) if n_scored > 0 else None,
        "n_correct":          n_correct,
        "n_scored":           n_scored,
        "full_pass_pct":      round(100.0 * sum(1 for r in all_records if r["exit_type"] == "Full Pass") / len(all_records), 1),
        "early_thresh_pct":   round(100.0 * sum(1 for r in all_records if r["exit_type"].startswith("Early (Thresh")) / len(all_records), 1),
        "forced_exit_pct":    round(100.0 * sum(1 for r in all_records if r["exit_type"] == "Early (Forced)") / len(all_records), 1),
        "deadline_miss_pct":  round(100.0 * sum(1 for r in all_records if r["time_ms"] > deadline_ms) / len(all_records), 1),
        "global_mean_tpot_ms":  round(mean_tpot_ms, 3),
        "global_p99_tpot_ms":   round(p99_tpot_ms, 3),
        "throughput_tps":       throughput_tps,   # tokens/sec from mean TPOT
        "util_ratio":           round(p99_tpot_ms / deadline_ms, 4),   # P99/D schedulability metric
    }

    output = {
        "global_metrics": global_metrics,
        "query_results":  query_results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to '{RESULTS_FILE}'")

    # CSV — flat per-query table for easy LaTeX/spreadsheet import
    csv_rows = []
    for q in query_results:
        m = q["metrics"]
        csv_rows.append({
            "query_id":           q["query_id"],
            "ground_truth":       q["ground_truth"],
            "predicted":          q["predicted"],
            "correct":            int(q["correct"]),
            "n_tokens":           m["n_tokens"],
            "full_pass_pct":      m["full_pass_pct"],
            "early_thresh_pct":   m["early_thresh_pct"],
            "forced_exit_pct":    m["forced_exit_pct"],
            "deadline_miss_pct":  m["deadline_miss_pct"],
            "mean_tpot_ms":       m["mean_tpot_ms"],
            "p99_tpot_ms":        m["p99_tpot_ms"],
            "max_tpot_ms":        m["max_tpot_ms"],
            "throughput_tps":     m["throughput_tps"],
        })
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV saved to      '{CSV_FILE}'")

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Accuracy       : {global_metrics['accuracy']}%  ({n_correct}/{n_scored} labeled queries)")
    print(f"Exit dist      : Full={global_metrics['full_pass_pct']}%  "
          f"Thresh={global_metrics['early_thresh_pct']}%  "
          f"Forced={global_metrics['forced_exit_pct']}%")
    print(f"Deadline misses: {global_metrics['deadline_miss_pct']}%  "
          f"(target: 0% above {deadline_ms} ms)")
    print(f"Global TPOT    : mean={global_metrics['global_mean_tpot_ms']} ms  "
          f"P99={global_metrics['global_p99_tpot_ms']} ms")
    print(f"Throughput     : {global_metrics['throughput_tps']} tokens/sec  "
          f"(from mean TPOT)")
    print(f"Util ratio     : {global_metrics['util_ratio']}  "
          f"(P99/D — schedulable if < 1.0)")

    return output


if __name__ == "__main__":
    run_pubmed_benchmark(n_samples=30, deadline_ms=45.0, max_new_tokens=15)
