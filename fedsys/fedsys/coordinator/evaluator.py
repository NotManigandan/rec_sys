"""
Model evaluation — held by the coordinator, called after every FedAvg round.

Two evaluation modes are supported:

Pointwise (synthetic / BCE models)
    ``evaluate()``  — DataLoader-based; computes BCE loss, accuracy, AUC-ROC.

Ranking (BPR / MovieLens)
    ``evaluate_ranking()``  — scores all items per user, computes Hit@K and
    NDCG@K.  Uses the leave-one-out val/test items stored in
    ``MovieLensFederatedDataset.splits_by_user``.

No gRPC, no privacy, no data-generation logic lives here.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedsys.logging_async import AsyncTelemetryLogger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    logger: AsyncTelemetryLogger | None = None,
    split: str = "val",
    epoch: int = -1,
) -> Dict[str, float]:
    """
    Evaluate ``model`` on ``loader`` and return a metrics dict.

    Parameters
    ----------
    model   : The global model after aggregation (weights already loaded).
    loader  : DataLoader for the val or test split.
    device  : Compute device (usually "cpu" on the coordinator).
    logger  : Optional async telemetry sink. Pass None to skip logging.
    split   : Label string used in log events ("val" or "test").
    epoch   : FL round number (for log records).

    Returns
    -------
    {
        "loss":     float,   # mean BCE loss
        "accuracy": float,   # fraction correctly classified (threshold = 0.5)
        "auc_roc":  float,   # area under the ROC curve
    }
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    all_labels: list[float] = []
    all_probs:  list[float] = []
    n_correct = 0
    n_total   = 0

    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            user_ids = batch["user_id"].to(dev)
            item_ids = batch["item_id"].to(dev)
            labels   = batch["label"].float().to(dev)

            logits = model(user_ids, item_ids).squeeze(-1)
            total_loss += criterion(logits, labels).item()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            n_correct += (preds == labels).sum().item()
            n_total   += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    elapsed_ms = (time.perf_counter() - t0) * 1_000

    loss     = total_loss / max(n_total, 1)
    accuracy = n_correct  / max(n_total, 1)
    auc      = _auc_roc(all_labels, all_probs)

    metrics = {
        "loss":     round(loss,     6),
        "accuracy": round(accuracy, 6),
        "auc_roc":  round(auc,      6),
    }

    if logger is not None:
        logger.log({
            "event":      f"{split.upper()}_EVAL",
            "epoch":      epoch,
            "split":      split,
            "n_samples":  n_total,
            "elapsed_ms": round(elapsed_ms, 4),
            **metrics,
        })

    return metrics


# ---------------------------------------------------------------------------
# Ranking evaluation (BPR / MovieLens)
# ---------------------------------------------------------------------------

def evaluate_ranking(
    model: nn.Module,
    ml_dataset,                            # MovieLensFederatedDataset
    split: str = "val",
    k_values: Sequence[int] = (10, 20),
    device: str = "cpu",
    logger: Optional[AsyncTelemetryLogger] = None,
    epoch: int = -1,
) -> Dict[str, float]:
    """
    Leave-one-out ranking evaluation for BPR models.

    For every user the held-out ``val_item`` (or ``test_item``) is scored
    against ALL items, with all other known positives masked to -inf so they
    cannot appear in the ranked list.  Hit@K and NDCG@K are averaged over
    users.

    Parameters
    ----------
    model       : BPRModel (or any model with forward(user_ids, item_ids)).
    ml_dataset  : Fully loaded MovieLensFederatedDataset.
    split       : "val" or "test".
    k_values    : List of cut-off positions to evaluate.
    device      : Compute device.
    logger      : Optional telemetry sink.
    epoch       : FL round index for log records.

    Returns
    -------
    {
        "hit@10":  float,
        "ndcg@10": float,
        "hit@20":  float,
        "ndcg@20": float,
        ...
    }
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    hits:  Dict[int, int]   = {k: 0 for k in k_values}
    ndcg:  Dict[int, float] = {k: 0.0 for k in k_values}
    n_users = 0

    # Pre-build item tensor once
    all_items = torch.arange(ml_dataset.num_items, dtype=torch.long, device=dev)

    t0 = time.perf_counter()

    with torch.no_grad():
        for uidx, user_split in ml_dataset.splits_by_user.items():
            target = user_split.val_item if split == "val" else user_split.test_item
            # Known items to mask EXCEPT the target itself
            mask_items = user_split.known_items - {target}

            # Score all items for this user
            user_tensor = torch.full(
                (ml_dataset.num_items,), uidx, dtype=torch.long, device=dev
            )
            scores = model(user_tensor, all_items)  # (num_items,)

            # Mask out known positive items (except the target)
            for ki in mask_items:
                scores[ki] = -float("inf")

            # Rank of target item (1-based): number of items with higher score + 1
            target_score = scores[target]
            rank = int((scores > target_score).sum().item()) + 1

            for k in k_values:
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / math.log2(rank + 1)

            n_users += 1

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    n = max(n_users, 1)

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"hit@{k}"]  = round(hits[k]  / n, 6)
        metrics[f"ndcg@{k}"] = round(ndcg[k]  / n, 6)

    if logger is not None:
        logger.log({
            "event":      f"{split.upper()}_RANKING_EVAL",
            "epoch":      epoch,
            "split":      split,
            "n_users":    n_users,
            "elapsed_ms": round(elapsed_ms, 4),
            **metrics,
        })

    return metrics


# ---------------------------------------------------------------------------
# AUC-ROC without scikit-learn
# ---------------------------------------------------------------------------

def _auc_roc(labels: list[float], probs: list[float]) -> float:
    """
    Compute AUC-ROC via the Mann-Whitney U statistic.

    For every (positive, negative) pair, count how often the model assigns
    a higher probability to the positive example:

        AUC = (# pairs where p_pos > p_neg  +  0.5 * # ties)
              ─────────────────────────────────────────────────
                          n_pos × n_neg

    Complexity: O(n_pos × n_neg).  For a validation set of ~1 000 samples
    this is at most ~250 000 comparisons — negligible on CPU.
    """
    pos_probs = [p for p, l in zip(probs, labels) if l == 1.0]
    neg_probs = [p for p, l in zip(probs, labels) if l == 0.0]

    if not pos_probs or not neg_probs:
        return 0.5  # degenerate: only one class present

    n_wins  = sum(pp > np for pp in pos_probs for np in neg_probs)
    n_ties  = sum(pp == np for pp in pos_probs for np in neg_probs)
    return (n_wins + 0.5 * n_ties) / (len(pos_probs) * len(neg_probs))
