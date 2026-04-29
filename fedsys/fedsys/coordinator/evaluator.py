"""
Model evaluation — held by the coordinator, called after every FedAvg round.

Metrics computed
----------------
    loss     : mean binary cross-entropy over the full eval set
    accuracy : fraction of samples where sigmoid(logit) >= 0.5 matches label
    auc_roc  : Area Under the ROC Curve, computed via the Mann-Whitney U
               statistic (exact, O(n_pos * n_neg), no extra dependencies)

The evaluator operates in torch.no_grad() mode and does NOT call
model.train() — the caller is responsible for restoring training mode
afterwards if needed.

No gRPC, no privacy, no data-generation logic lives here.
"""

from __future__ import annotations

import time
from typing import Dict

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
