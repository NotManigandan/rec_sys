"""
tests/adversarial/test_eval.py
================================
Unit tests for fedsys.adversarial.eval
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import random
from typing import Dict, Tuple

import torch
import torch.nn as nn

from fedsys.adversarial.eval import (
    evaluate_with_target_exposure,
    compare_attack_vs_clean,
)
from fedsys.data.movielens_dataset import (
    MovieLensFederatedDataset, MovieMetadata, UserSplit,
)


def _make_small_dataset(num_users=40, num_items=20, seed=3):
    rng = random.Random(seed)
    genres = ["Action", "Comedy", "Drama"]
    item_ids = tuple(range(num_items))
    user_ids = tuple(range(num_users))
    metadata = tuple(
        MovieMetadata(i, f"Movie{i}", (genres[i % len(genres)],))
        for i in range(num_items)
    )
    splits, dominant = {}, {}
    for u in range(num_users):
        items = rng.sample(range(num_items), 7)
        splits[u] = UserSplit(
            train_ratings=tuple((it, 5.0) for it in items[:5]),
            val_item=items[5], test_item=items[6],
            known_items=frozenset(items),
        )
        dominant[u] = genres[u % len(genres)]

    items_by_genre: Dict[str, Tuple[int, ...]] = {g: () for g in genres}
    users_by_genre: Dict[str, Tuple[int, ...]] = {g: () for g in genres}
    for i in range(num_items):
        items_by_genre[genres[i % len(genres)]] += (i,)
    for u in range(num_users):
        users_by_genre[genres[u % len(genres)]] += (u,)

    pop = {}
    for s in splits.values():
        for it, _ in s.train_ratings:
            pop[it] = pop.get(it, 0) + 1

    return MovieLensFederatedDataset(
        variant="test", root=None,
        item_ids=item_ids, item_id_to_index={i: i for i in range(num_items)},
        user_ids=user_ids, user_id_to_index={u: u for u in range(num_users)},
        movie_metadata=metadata, splits_by_user=splits,
        dominant_genre_by_user=dominant,
        users_by_genre=users_by_genre, items_by_genre=items_by_genre,
        train_item_popularity=pop,
    )


class _TinyBPR(nn.Module):
    def __init__(self, nu, ni, dim=4):
        super().__init__()
        self.user_embedding = nn.Embedding(nu, dim)
        self.item_embedding = nn.Embedding(ni, dim)

    def forward(self, user_ids, item_ids, histories=None):
        u = self.user_embedding(user_ids)
        v = self.item_embedding(item_ids)
        return (u * v).sum(dim=-1)


def test_evaluate_with_target_exposure_returns_expected_keys():
    ds = _make_small_dataset()
    model = _TinyBPR(ds.num_users, ds.num_items)
    target_genre = "Action"
    target_item = ds.items_by_genre[target_genre][0]

    metrics = evaluate_with_target_exposure(
        model=model,
        dataset=ds,
        target_item_index=target_item,
        target_genre=target_genre,
        split="val",
        cutoffs=(10,),
    )
    assert "hit@10"         in metrics
    assert "ndcg@10"        in metrics
    assert "top1_accuracy"  in metrics
    assert "ndcg@3"         in metrics
    assert "mrr"            in metrics
    assert "pairwise_accuracy" in metrics
    assert "target_hit@10"  in metrics
    assert "target_ndcg@10" in metrics
    assert "segment_hit@10" in metrics
    assert "segment_top1_accuracy" in metrics
    assert "segment_ndcg@3" in metrics
    assert "segment_mrr" in metrics
    assert "segment_pairwise_accuracy" in metrics


def test_evaluate_target_exposure_values_in_range():
    ds = _make_small_dataset()
    model = _TinyBPR(ds.num_users, ds.num_items)
    target_genre = "Action"
    target_item = ds.items_by_genre[target_genre][0]

    metrics = evaluate_with_target_exposure(
        model=model, dataset=ds,
        target_item_index=target_item,
        target_genre=target_genre,
        split="val", cutoffs=(5, 10),
    )
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_compare_attack_vs_clean():
    clean  = {"hit@10": 0.30, "target_hit@10": 0.05}
    attack = {"hit@10": 0.28, "target_hit@10": 0.35}
    diff = compare_attack_vs_clean(attack, clean, cutoffs=(10,))
    assert abs(diff["delta_hit@10"]        - (-0.02)) < 1e-5
    assert abs(diff["delta_target_hit@10"] -   0.30)  < 1e-5


if __name__ == "__main__":
    test_evaluate_with_target_exposure_returns_expected_keys()
    test_evaluate_target_exposure_values_in_range()
    test_compare_attack_vs_clean()
    print("All adversarial eval tests passed.")
