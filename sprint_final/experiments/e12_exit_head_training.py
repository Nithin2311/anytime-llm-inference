"""
e12_exit_head_training.py — Train and evaluate ExitHead MLP on L16 hidden states.

Addresses C3 (reviewer): shared LM head as exit classifier only 32% voluntary
exit rate with low oracle agreement. A dedicated 2-layer MLP trained on L16
hidden states should increase both exit rate and oracle agreement.

Pipeline:
  1. Load 300 labeled PubMedQA queries
  2. Collect L16 last-token hidden states via forward hook
  3. Train ExitHead (2048→512→128→3-class) for 30 epochs
  4. Evaluate: oracle agreement, exit rate, accuracy at τ=0.55
  5. Save model checkpoint + training log + figure
"""

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from exit_head_trainer import collect_hidden_states, train_exit_head, eval_exit_head
from benchmark_utils import load_pubmed_dataset
from fig_style import apply_style, DOUBLE
import matplotlib.pyplot as plt
from result_writer import write_results

apply_style()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
N_TRAIN_QUERIES = 300    # hidden-state collection
N_EVAL_QUERIES  = 100    # held-out evaluation
N_EPOCHS        = 30
LR              = 3e-4
BATCH_SIZE      = 32
TAU             = 0.55
VAL_FRACTION    = 0.20
DEVICE          = "cuda"


def main():
    print("=" * 60)
    print("E12  ExitHead MLP training on L16 hidden states")
    print(f"     train={N_TRAIN_QUERIES} queries, eval={N_EVAL_QUERIES} queries")
    print("=" * 60)

    from early_exit_model import EarlyExitModel
    model = EarlyExitModel(device=DEVICE)

    # ── Load dataset ────────────────────────────────────────────────────
    print(f"\n[1/4] Loading {N_TRAIN_QUERIES + N_EVAL_QUERIES} PubMedQA queries ...")
    all_queries = load_pubmed_dataset(n_samples=N_TRAIN_QUERIES + N_EVAL_QUERIES)
    train_queries = all_queries[:N_TRAIN_QUERIES]
    eval_queries  = all_queries[N_TRAIN_QUERIES:]
    print(f"      Train: {len(train_queries)}  Eval: {len(eval_queries)}")

    # ── Build (prompt, label) pairs ─────────────────────────────────────
    def queries_to_pairs(qs):
        pairs = []
        for q in qs:
            prompt    = q.get("prompt",    q.get("question", ""))
            label_str = q.get("label_str", q.get("final_decision", ""))
            if prompt and label_str:
                pairs.append((prompt, label_str))
        return pairs

    train_pairs = queries_to_pairs(train_queries)
    eval_pairs  = queries_to_pairs(eval_queries)
    print(f"      Train pairs: {len(train_pairs)}  Eval pairs: {len(eval_pairs)}")

    # ── Collect hidden states ────────────────────────────────────────────
    print(f"\n[2/4] Collecting L16 hidden states ({len(train_pairs)} prompts) ...")
    t0 = time.time()
    hiddens, labels = collect_hidden_states(model, train_pairs, device=DEVICE)
    print(f"      Collected in {time.time()-t0:.1f}s  "
          f"hiddens.shape={tuple(hiddens.shape)}")

    # ── Train ExitHead ───────────────────────────────────────────────────
    print(f"\n[3/4] Training ExitHead ({N_EPOCHS} epochs) ...")
    t0 = time.time()
    head, train_log = train_exit_head(
        hiddens, labels,
        n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION, device=DEVICE, rng_seed=42
    )
    print(f"      Training done in {time.time()-t0:.1f}s")
    final_val_acc = train_log[-1]["val_acc"]
    print(f"      Final val accuracy: {final_val_acc:.1f}%")

    # Save checkpoint
    ckpt_path = RESULTS_DIR / "exit_head.pt"
    torch.save(head.state_dict(), ckpt_path)
    print(f"      Checkpoint saved: {ckpt_path}")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating on {len(eval_pairs)} held-out prompts ...")
    metrics = eval_exit_head(model, head, eval_pairs, threshold=TAU, device=DEVICE)

    print(f"\n{'─'*50}")
    print(f"  Exit rate          : {metrics['exit_rate_pct']:.1f}%")
    print(f"  Oracle agreement   : {metrics['oracle_agreement_pct']:.1f}%")
    print(f"  Accuracy           : {metrics['accuracy']:.1f}%")
    print(f"  N scored           : {metrics['n_scored']}")
    print(f"{'─'*50}\n")

    # ── Figures ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)

    # Training curves
    epochs    = [e["epoch"]     for e in train_log]
    train_acc = [e["train_acc"] for e in train_log]
    val_acc   = [e["val_acc"]   for e in train_log]
    losses    = [e["loss"]      for e in train_log]

    ax = axes[0]
    ax.plot(epochs, train_acc, label="Train acc",   color="tab:blue")
    ax.plot(epochs, val_acc,   label="Val acc",     color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ExitHead Training Curves")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    ax2 = axes[1]
    ax2.plot(epochs, losses, color="tab:red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.set_title("Training Loss")

    plt.suptitle("ExitHead MLP — L16 Hidden State Classifier")
    plt.tight_layout()
    fig_path = RESULTS_DIR / "exit_head_training.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved: {fig_path}")

    output = {
        "experiment":       "e12_exit_head_training",
        "n_train_queries":  len(train_pairs),
        "n_eval_queries":   len(eval_pairs),
        "n_epochs":         N_EPOCHS,
        "lr":               LR,
        "batch_size":       BATCH_SIZE,
        "tau":              TAU,
        "final_val_acc":    final_val_acc,
        "eval_metrics":     metrics,
        "train_log":        train_log,
    }
    write_results(output, RESULTS_DIR / "exit_head_results.json")
    print("PASS: E12 complete\n")


if __name__ == "__main__":
    main()
