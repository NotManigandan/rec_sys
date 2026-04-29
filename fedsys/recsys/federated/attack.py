from __future__ import annotations

import math
import random
from typing import Dict

from .bpr import FederatedShard
from .movielens import MovieLensFederatedDataset


def build_poisoned_shards(
    clean_shards: Dict[int, FederatedShard],
    dataset: MovieLensFederatedDataset,
    malicious_shard_id: int,
    target_genre: str,
    target_item_index: int,
    attack_budget: float,
    seed: int,
    filler_items_per_user: int = 8,
    neutral_items_per_user: int = 0,
    target_item_weight: float = 20.0,
    filler_item_weight: float = 0.5,
    neutral_item_weight: float = 0.0,
    filler_candidate_pool_size: int = 64,
    neutral_candidate_pool_size: int = 32,
) -> Dict[int, FederatedShard]:
    poisoned_shards: Dict[int, FederatedShard] = {}
    for shard_id, shard in clean_shards.items():
        poisoned_shards[shard_id] = FederatedShard(
            user_indices=shard.user_indices,
            train_ratings_by_user=dict(shard.train_ratings_by_user),
            known_items_by_user=dict(shard.known_items_by_user),
            synthetic_user_count=0,
        )

    if attack_budget <= 0:
        return poisoned_shards

    if malicious_shard_id not in poisoned_shards:
        raise ValueError(f"Unknown malicious shard id: {malicious_shard_id}")

    rng = random.Random(seed)
    malicious_shard = poisoned_shards[malicious_shard_id]
    benign_users_in_shard = len(malicious_shard.user_indices)
    synthetic_user_count = max(1, math.ceil(benign_users_in_shard * attack_budget))

    target_genre_items = [
        item_index
        for item_index in dataset.items_by_genre.get(target_genre, ())
        if item_index != target_item_index
    ]
    target_genre_items.sort(
        key=lambda item_index: (-dataset.train_item_popularity.get(item_index, 0), item_index)
    )
    neutral_items = [
        item_index
        for item_index in range(dataset.num_items)
        if item_index != target_item_index and item_index not in set(target_genre_items)
    ]
    neutral_items.sort(
        key=lambda item_index: (-dataset.train_item_popularity.get(item_index, 0), item_index)
    )
    if not target_genre_items:
        raise ValueError(f"No filler items were found for target genre {target_genre}.")

    next_user_index = dataset.num_users
    train_ratings_by_user = dict(malicious_shard.train_ratings_by_user)
    known_items_by_user = dict(malicious_shard.known_items_by_user)
    synthetic_user_indices = list(malicious_shard.user_indices)
    for synthetic_offset in range(synthetic_user_count):
        filler_count = min(filler_items_per_user, len(target_genre_items))
        neutral_count = min(neutral_items_per_user, len(neutral_items))
        filler_pool_size = max(filler_count, min(filler_candidate_pool_size, len(target_genre_items)))
        neutral_pool_size = max(neutral_count, min(neutral_candidate_pool_size, len(neutral_items)))
        filler_sample = tuple(
            (item_index, filler_item_weight)
            for item_index in rng.sample(target_genre_items[:filler_pool_size], filler_count)
        )
        neutral_sample = tuple(
            (item_index, neutral_item_weight)
            for item_index in rng.sample(neutral_items[:neutral_pool_size], neutral_count)
        )
        synthetic_profile = ((target_item_index, target_item_weight),) + filler_sample + neutral_sample
        synthetic_user_index = next_user_index + synthetic_offset
        train_ratings_by_user[synthetic_user_index] = synthetic_profile
        known_items_by_user[synthetic_user_index] = frozenset(item_index for item_index, _ in synthetic_profile)
        synthetic_user_indices.append(synthetic_user_index)

    poisoned_shards[malicious_shard_id] = FederatedShard(
        user_indices=tuple(synthetic_user_indices),
        train_ratings_by_user=train_ratings_by_user,
        known_items_by_user=known_items_by_user,
        synthetic_user_count=synthetic_user_count,
    )
    return poisoned_shards
