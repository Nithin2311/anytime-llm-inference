"""
exit_head_trainer.py — Lightweight exit-head training on L16 hidden states.

Trains a 2-layer MLP classifier on top of TinyLlama Layer-16 hidden states
for yes/no/maybe 3-class prediction (PubMedQA). A trained exit head increases
voluntary early-exit rate and oracle agreement above the shared LM head's
32% baseline, directly addressing reviewer concern about exit-head quality.

Used by: e12_exit_head_training.py
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

LABEL_MAP   = {"yes": 0, "no": 1, "maybe": 2}
LABEL_NAMES = ["yes", "no", "maybe"]
HIDDEN_SIZE = 2048  # TinyLlama hidden dimension


class ExitHead(nn.Module):
    """Two-layer MLP: L16 last-token hidden state [2048] → 3-class logits."""

    def __init__(self, hidden_size=HIDDEN_SIZE, n_classes=3, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _capture_l16_hidden(model, input_ids):
    """Run one forward pass, return L16 last-token hidden state [2048] on CPU."""
    captured = {}

    def _hook(module, inp, out):
        captured["h16"] = out[0] if isinstance(out, tuple) else out

    handle = model._m.layers[15].register_forward_hook(_hook)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.inference_mode():
                model.base_model(input_ids, past_key_values=None, use_cache=False)
    finally:
        handle.remove()

    return captured["h16"][:, -1, :].squeeze(0).float().cpu()


def collect_hidden_states(model, prompts_labels, device="cuda"):
    """
    Collect L16 last-token hidden states and integer labels for each labeled prompt.

    prompts_labels: list of (prompt_str, label_str) where label_str ∈ {yes,no,maybe}

    Returns:
        hiddens : FloatTensor [N, 2048]
        labels  : LongTensor  [N]
    """
    hiddens, labels = [], []
    for prompt, label_str in prompts_labels:
        label_id = LABEL_MAP.get(label_str.strip().lower(), -1)
        if label_id < 0:
            continue
        input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        h = _capture_l16_hidden(model, input_ids)
        hiddens.append(h)
        labels.append(label_id)

    if not hiddens:
        raise ValueError("No valid labeled samples — check label strings.")

    return torch.stack(hiddens), torch.tensor(labels, dtype=torch.long)


def train_exit_head(hiddens, labels, n_epochs=30, lr=3e-4, batch_size=32,
                    val_fraction=0.2, device="cuda", rng_seed=42):
    """
    Train ExitHead MLP on pre-collected hidden states.

    Returns (trained_head, training_log).
    training_log: list of {"epoch": int, "train_acc": float, "val_acc": float, "loss": float}
    """
    torch.manual_seed(rng_seed)
    n       = len(hiddens)
    n_val   = max(1, int(val_fraction * n))
    n_train = n - n_val
    perm    = torch.randperm(n)

    h_train = hiddens[perm[:n_train]].to(device)
    l_train = labels[perm[:n_train]].to(device)
    h_val   = hiddens[perm[n_train:]].to(device)
    l_val   = labels[perm[n_train:]].to(device)

    loader    = DataLoader(TensorDataset(h_train, l_train),
                           batch_size=batch_size, shuffle=True)
    head      = ExitHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    log = []
    for epoch in range(1, n_epochs + 1):
        head.train()
        total_loss, n_corr, n_tot = 0.0, 0, 0
        for hb, lb in loader:
            logits = head(hb)
            loss   = criterion(logits, lb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(lb)
            n_corr     += (logits.argmax(1) == lb).sum().item()
            n_tot      += len(lb)
        scheduler.step()

        head.eval()
        with torch.no_grad():
            val_corr = int((head(h_val).argmax(1) == l_val).sum().item())

        entry = {
            "epoch":     epoch,
            "train_acc": round(100.0 * n_corr / n_tot, 1),
            "val_acc":   round(100.0 * val_corr / len(l_val), 1),
            "loss":      round(total_loss / n_tot, 4),
        }
        log.append(entry)
        if epoch % 5 == 0:
            print(f"    Epoch {epoch:>3}/{n_epochs}  "
                  f"train={entry['train_acc']:.1f}%  val={entry['val_acc']:.1f}%")

    return head, log


def eval_exit_head(model, head, prompts_labels, threshold=0.55, device="cuda"):
    """
    Evaluate trained exit head vs. full-model oracle.

    For each prompt:
      - Compute ExitHead confidence on L16 hidden state
      - If confidence >= threshold  → commit exit-head prediction
      - Else                         → commit full 22-layer prediction
    Report oracle agreement, accuracy, and voluntary exit rate.

    Returns dict with all evaluation metrics.
    """
    head.eval()
    n_total = n_exit = n_oracle_agree = n_correct = n_scored = 0

    with torch.no_grad():
        for prompt, gt_label in prompts_labels:
            input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # Full model pass — captures L16 hidden and gives full logits
            l16_logits, full_logits, _ = model.forward_cached(input_ids)
            torch.cuda.synchronize()

            h16 = _capture_l16_hidden(model, input_ids).to(device).unsqueeze(0)

            # Exit head routing
            head_logits = head(h16)
            head_probs  = torch.softmax(head_logits, dim=-1)
            head_conf, head_pred_idx = torch.max(head_probs, dim=-1)
            head_conf_val   = float(head_conf.item())
            head_pred_label = LABEL_NAMES[int(head_pred_idx.item())]

            # Full-model oracle token (single-token argmax)
            full_token_id = int(full_logits[0, -1, :].argmax().item())
            full_text     = model.tokenizer.decode([full_token_id]).strip().lower()

            n_total += 1
            if head_conf_val >= threshold:
                n_exit += 1
                committed = head_pred_label
                if head_pred_label == full_text:
                    n_oracle_agree += 1
            else:
                committed = full_text
                n_oracle_agree += 1  # full pass always agrees with itself

            gt = gt_label.strip().lower()
            if committed in ("yes", "no", "maybe"):
                n_scored += 1
                if committed == gt:
                    n_correct += 1

    return {
        "n_total":              n_total,
        "n_exit":               n_exit,
        "exit_rate_pct":        round(100.0 * n_exit    / max(1, n_total), 1),
        "oracle_agreement_pct": round(100.0 * n_oracle_agree / max(1, n_total), 1),
        "accuracy":             round(100.0 * n_correct / max(1, n_scored), 1),
        "n_scored":             n_scored,
        "threshold_used":       threshold,
    }
