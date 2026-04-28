from __future__ import annotations

import math
from typing import Dict, List

from .data import RatingsDataset


def _dcg(relevances: List[float]) -> float:
    return sum((2.0**rel - 1.0) / math.log2(index + 2.0) for index, rel in enumerate(relevances))


def evaluate_rankings(
    dataset: RatingsDataset,
    ranked_items_by_user: Dict[int, List[int]],
) -> Dict[str, object]:
    top1_hits = 0.0
    ndcg_total = 0.0
    pairwise_hits = 0.0
    pairwise_total = 0.0
    reciprocal_rank_total = 0.0

    per_profile: Dict[str, Dict[str, float]] = {}

    for profile_name in dataset.profile_names():
        per_profile[profile_name] = {"users": 0.0, "top1_hits": 0.0, "ndcg": 0.0, "mrr": 0.0}

    for user_id in dataset.user_ids:
        ranked_items = ranked_items_by_user[user_id]
        unseen_items = dataset.unseen_items(user_id)
        if sorted(ranked_items) != sorted(unseen_items):
            raise ValueError(f"Ranked items for user {user_id} do not match the unseen set.")

        hidden_truth = dataset.actual_ratings[user_id]
        best_relevance = max(hidden_truth[item_id] for item_id in unseen_items)
        best_items = {item_id for item_id in unseen_items if hidden_truth[item_id] == best_relevance}

        if ranked_items[0] in best_items:
            top1_hits += 1.0
            per_profile[dataset.user_profiles[user_id]]["top1_hits"] += 1.0

        ranked_relevances = [hidden_truth[item_id] for item_id in ranked_items]
        ideal_relevances = sorted((hidden_truth[item_id] for item_id in unseen_items), reverse=True)
        ndcg = _dcg(ranked_relevances) / _dcg(ideal_relevances)
        ndcg_total += ndcg
        per_profile[dataset.user_profiles[user_id]]["ndcg"] += ndcg

        reciprocal_rank = 0.0
        for rank_index, item_id in enumerate(ranked_items, start=1):
            if item_id in best_items:
                reciprocal_rank = 1.0 / rank_index
                break
        reciprocal_rank_total += reciprocal_rank
        per_profile[dataset.user_profiles[user_id]]["mrr"] += reciprocal_rank

        for left_index, left_item in enumerate(ranked_items):
            for right_item in ranked_items[left_index + 1 :]:
                left_rating = hidden_truth[left_item]
                right_rating = hidden_truth[right_item]
                if left_rating == right_rating:
                    continue
                pairwise_total += 1.0
                if left_rating > right_rating:
                    pairwise_hits += 1.0

        profile_metrics = per_profile[dataset.user_profiles[user_id]]
        profile_metrics["users"] += 1.0

    user_count = float(dataset.num_users)
    summary = {
        "top1_accuracy": top1_hits / user_count,
        "ndcg@3": ndcg_total / user_count,
        "pairwise_accuracy": pairwise_hits / pairwise_total if pairwise_total else 0.0,
        "mrr": reciprocal_rank_total / user_count,
        "per_profile": {},
    }

    for profile_name, profile_metrics in per_profile.items():
        profile_user_count = max(profile_metrics["users"], 1.0)
        summary["per_profile"][profile_name] = {
            "users": int(profile_metrics["users"]),
            "top1_accuracy": profile_metrics["top1_hits"] / profile_user_count,
            "ndcg@3": profile_metrics["ndcg"] / profile_user_count,
            "mrr": profile_metrics["mrr"] / profile_user_count,
        }

    return summary
