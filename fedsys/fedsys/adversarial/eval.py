"""
fedsys.adversarial.eval
========================
Extended evaluation that measures the *impact* of data-poisoning attacks:

  • Overall ranking metrics (Hit@K, NDCG@K) — same as fedsys/coordinator/evaluator.py
  • **Target-exposure metrics** (target_hit@K, target_ndcg@K) — how often the
    target item appears in the top-k of the *victim segment* users.
  • **Benign-hit degradation** — overall Hit@K compared to a clean baseline
    (if a baseline dict is supplied).

Ported from ``recsys/federated/eval.py``.

Typical coordinator usage::

    from fedsys.adversarial.eval import evaluate_with_target_exposure

    adv_metrics = evaluate_with_target_exposure(
        model=servicer._global_model,
        dataset=ml_dataset,
        target_item_index=attack_cfg.target_item_index,
        target_genre=attack_cfg.target_genre,
        split="val",
        cutoffs=(10, 20),
        device=cfg.device,
    )
    print(f"target_hit@10={adv_metrics['target_hit@10']:.4f}")
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from fedsys.data.movielens_dataset import MovieLensFederatedDataset


# ---------------------------------------------------------------------------
# Internal scoring helper
# ---------------------------------------------------------------------------

def _score_all_items_for_users(
    model: nn.Module,
    user_indices: Sequence[int],
    num_items: int,
    device: str,
    batch_size: int = 512,
) -> Dict[int, torch.Tensor]:
    """
    Return a dict {user_index: score_tensor (shape: num_items)}.

    Scores all items for each user in a batched, no-grad loop.
    """
    model.eval()
    results: Dict[int, torch.Tensor] = {}
    all_items = torch.arange(num_items, dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, len(user_indices), batch_size):
            batch_users = list(user_indices[start : start + batch_size])
            for uid in batch_users:
                u_rep = torch.tensor(uid, dtype=torch.long, device=device)
                u_rep = u_rep.unsqueeze(0).expand(num_items)
                sc = model(u_rep, all_items)
                results[uid] = sc.cpu()
    return results


# ---------------------------------------------------------------------------
# Public evaluation entry points
# ---------------------------------------------------------------------------

def evaluate_with_target_exposure(
    model: nn.Module,
    dataset: MovieLensFederatedDataset,
    target_item_index: int,
    target_genre: str,
    split: str = "val",
    cutoffs: Tuple[int, ...] = (10, 20),
    device: str = "cpu",
    batch_size: int = 512,
    logger=None,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    Evaluate model with extended adversarial metrics.

    Returns
    -------
    Dict with:
        hit@K, ndcg@K           — overall metrics (all users in split)
        target_hit@K            — fraction of *target-segment* users who have
                                  the target item in their top-K
        target_ndcg@K           — NDCG@K for target item among target-segment
        segment_hit@K           — overall Hit@K restricted to target segment
        segment_ndcg@K          — overall NDCG@K restricted to target segment
    """
    model = model.to(device)

    split_attr = f"{split}_item"
    segment_users = list(dataset.users_by_genre.get(target_genre, ()))

    all_users = list(dataset.splits_by_user.keys())
    num_items  = dataset.num_items

    metrics: Dict[str, float] = {}

    # ── Overall Hit@K / NDCG@K ─────────────────────────────────────────────
    overall = _ranking_metrics(
        model=model,
        user_indices=all_users,
        dataset=dataset,
        split=split,
        cutoffs=cutoffs,
        device=device,
        batch_size=batch_size,
    )
    metrics.update(overall)

    # ── Segment-level metrics ───────────────────────────────────────────────
    if segment_users:
        seg_metrics = _ranking_metrics(
            model=model,
            user_indices=segment_users,
            dataset=dataset,
            split=split,
            cutoffs=cutoffs,
            device=device,
            batch_size=batch_size,
        )
        for k, v in seg_metrics.items():
            metrics[f"segment_{k}"] = v

        # ── Target-exposure metrics ─────────────────────────────────────
        target_hits, target_ndcg = _target_exposure(
            model=model,
            user_indices=segment_users,
            dataset=dataset,
            target_item_index=target_item_index,
            cutoffs=cutoffs,
            device=device,
            batch_size=batch_size,
        )
        for k in cutoffs:
            metrics[f"target_hit@{k}"]  = target_hits.get(k, 0.0)
            metrics[f"target_ndcg@{k}"] = target_ndcg.get(k, 0.0)

    if logger is not None:
        logger.log({
            "event":  "ADVERSARIAL_EVAL",
            "epoch":  epoch,
            "split":  split,
            "target": target_item_index,
            "genre":  target_genre,
            **{k: round(v, 6) for k, v in metrics.items()},
        })

    return metrics


def _ranking_metrics(
    model: nn.Module,
    user_indices: Sequence[int],
    dataset: MovieLensFederatedDataset,
    split: str,
    cutoffs: Tuple[int, ...],
    device: str,
    batch_size: int,
) -> Dict[str, float]:
    """Standard Hit@K / NDCG@K for a list of users."""
    hits  = {k: 0 for k in cutoffs}
    ndcgs = {k: 0.0 for k in cutoffs}
    top1_hits = 0
    ndcg3 = 0.0
    mrr = 0.0
    pairwise_acc = 0.0
    n     = 0

    model.eval()
    all_items = torch.arange(dataset.num_items, dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, len(user_indices), batch_size):
            batch = list(user_indices[start : start + batch_size])
            for uid in batch:
                split_data = dataset.splits_by_user.get(uid)
                if split_data is None:
                    continue
                target_item = getattr(split_data, f"{split}_item", None)
                if target_item is None:
                    continue

                known = split_data.known_items - {target_item}

                u_rep = (
                    torch.tensor(uid, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(dataset.num_items)
                )
                scores = model(u_rep, all_items).cpu()

                # Mask known items (leave target unmasked)
                for item in known:
                    if 0 <= item < len(scores):
                        scores[item] = -1e9

                rank = int((scores > scores[target_item]).sum().item())

                for k in cutoffs:
                    if rank < k:
                        hits[k]  += 1
                        ndcgs[k] += 1.0 / math.log2(rank + 2)
                if rank == 0:
                    top1_hits += 1
                if rank < 3:
                    ndcg3 += 1.0 / math.log2(rank + 2)
                mrr += 1.0 / (rank + 1)
                if dataset.num_items > 1:
                    pairwise_acc += (dataset.num_items - (rank + 1)) / (dataset.num_items - 1)
                n += 1

    denom = max(n, 1)
    result: Dict[str, float] = {}
    for k in cutoffs:
        result[f"hit@{k}"]  = hits[k]  / denom
        result[f"ndcg@{k}"] = ndcgs[k] / denom
    result["top1_accuracy"] = top1_hits / denom
    result["ndcg@3"] = ndcg3 / denom
    result["mrr"] = mrr / denom
    result["pairwise_accuracy"] = (pairwise_acc / denom) if dataset.num_items > 1 else 0.0
    return result


def _target_exposure(
    model: nn.Module,
    user_indices: Sequence[int],
    dataset: MovieLensFederatedDataset,
    target_item_index: int,
    cutoffs: Tuple[int, ...],
    device: str,
    batch_size: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute target hit-rate and NDCG@K for a given set of users.

    For each user we score all items (masking known positives except the
    target), then check whether the *target item* falls within top-K.

    Returns
    -------
    (target_hits, target_ndcg) — dicts keyed by K value.
    """
    hits  = {k: 0 for k in cutoffs}
    ndcgs = {k: 0.0 for k in cutoffs}
    n     = 0

    model.eval()
    all_items = torch.arange(dataset.num_items, dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, len(user_indices), batch_size):
            batch = list(user_indices[start : start + batch_size])
            for uid in batch:
                split_data = dataset.splits_by_user.get(uid)
                if split_data is None:
                    continue

                # Mask all known positives EXCEPT the target item
                mask_items = split_data.known_items - {target_item_index}

                u_rep = (
                    torch.tensor(uid, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(dataset.num_items)
                )
                scores = model(u_rep, all_items).cpu()

                for item in mask_items:
                    if 0 <= item < len(scores):
                        scores[item] = -1e9

                target_score = scores[target_item_index]
                rank = int((scores > target_score).sum().item())

                for k in cutoffs:
                    if rank < k:
                        hits[k]  += 1
                        ndcgs[k] += 1.0 / math.log2(rank + 2)
                n += 1

    denom = max(n, 1)
    return (
        {k: hits[k]  / denom for k in cutoffs},
        {k: ndcgs[k] / denom for k in cutoffs},
    )


# ---------------------------------------------------------------------------
# Benchmark helper: compare attack vs. clean baseline
# ---------------------------------------------------------------------------

def compare_attack_vs_clean(
    attack_metrics: Dict[str, float],
    clean_metrics: Dict[str, float],
    cutoffs: Tuple[int, ...] = (10, 20),
) -> Dict[str, float]:
    """
    Return a diff dict showing the *change* in each metric from clean to
    attacked model.  Positive values mean the attacker increased that metric.

    Example output::

        {
          "delta_hit@10": +0.003,          # tiny benign degradation
          "delta_target_hit@10": +0.142,   # strong target push
          ...
        }
    """
    diff: Dict[str, float] = {}
    all_keys = set(attack_metrics) | set(clean_metrics)
    for key in all_keys:
        a = attack_metrics.get(key, 0.0)
        c = clean_metrics.get(key, 0.0)
        diff[f"delta_{key}"] = round(a - c, 6)
    return diff
