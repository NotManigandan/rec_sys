"""
tests/adversarial/test_poison.py
=================================
Unit tests for fedsys.adversarial.attack.poison

These tests run without a running coordinator or gRPC server.
They use small synthetic / mock objects to stay fast.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple

import torch

from fedsys.adversarial.attack.poison import (
    AttackConfig,
    PoisonedBPRPairDataset,
    build_poisoned_dataloader,
    poisoned_num_users,
    _build_synthetic_profiles,
)
from fedsys.data.movielens_dataset import (
    BPRPairDataset,
    MovieLensFederatedDataset,
    MovieMetadata,
    UserSplit,
)


# ---------------------------------------------------------------------------
# Minimal MovieLensFederatedDataset fixture
# ---------------------------------------------------------------------------

def _make_small_dataset(
    num_users: int = 50,
    num_items: int = 20,
    seed: int = 0,
) -> MovieLensFederatedDataset:
    """Create a tiny mock dataset for testing."""
    rng = random.Random(seed)
    item_ids = tuple(range(num_items))
    user_ids = tuple(range(num_users))

    genres = ["Action", "Comedy", "Drama"]
    metadata = tuple(
        MovieMetadata(
            movie_id=i,
            title=f"Movie{i}",
            genres=(genres[i % len(genres)],),
        )
        for i in range(num_items)
    )

    splits = {}
    dominant_by_user = {}
    for u in range(num_users):
        # Give each user 5 train items, 1 val, 1 test
        items = rng.sample(range(num_items), 7)
        train = tuple((it, 5.0) for it in items[:5])
        splits[u] = UserSplit(
            train_ratings=train,
            val_item=items[5],
            test_item=items[6],
            known_items=frozenset(items),
        )
        dominant_by_user[u] = genres[u % len(genres)]

    items_by_genre: Dict[str, Tuple[int, ...]] = {g: () for g in genres}
    users_by_genre: Dict[str, Tuple[int, ...]] = {g: () for g in genres}
    for i in range(num_items):
        g = genres[i % len(genres)]
        items_by_genre[g] = items_by_genre[g] + (i,)
    for u in range(num_users):
        g = genres[u % len(genres)]
        users_by_genre[g] = users_by_genre[g] + (u,)

    pop = {}
    for split in splits.values():
        for item, _ in split.train_ratings:
            pop[item] = pop.get(item, 0) + 1

    return MovieLensFederatedDataset(
        variant="test",
        root=None,
        item_ids=item_ids,
        item_id_to_index={i: i for i in range(num_items)},
        user_ids=user_ids,
        user_id_to_index={u: u for u in range(num_users)},
        movie_metadata=metadata,
        splits_by_user=splits,
        dominant_genre_by_user=dominant_by_user,
        users_by_genre=users_by_genre,
        items_by_genre=items_by_genre,
        train_item_popularity=pop,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_poisoned_num_users_disabled():
    cfg = AttackConfig(enabled=False, max_synthetic_users_per_coord=100)
    assert poisoned_num_users(500, cfg) == 500


def test_poisoned_num_users_enabled():
    cfg = AttackConfig(enabled=True, max_synthetic_users_per_coord=100)
    assert poisoned_num_users(500, cfg) == 600


def test_build_synthetic_profiles():
    ds = _make_small_dataset()
    cfg = AttackConfig(
        enabled=True,
        target_item_index=0,
        target_genre="Action",
        attack_budget=0.5,
        num_filler_items=2,
        num_neutral_items=2,
        neutral_genre="Comedy",
    )
    profiles = _build_synthetic_profiles(
        dataset=ds,
        num_profiles=5,
        attack_cfg=cfg,
        base_user_index=ds.num_users,
    )
    assert len(profiles) == 5
    for p in profiles:
        # Target item must be in training
        items_in_profile = [it for it, _ in p.train_ratings]
        assert cfg.target_item_index in items_in_profile
        # val_item and test_item should be target
        assert p.val_item == cfg.target_item_index
        assert p.test_item == cfg.target_item_index


def test_poisoned_bpr_pair_dataset_length():
    ds = _make_small_dataset()
    shard_users = list(range(10))
    clean_ds = BPRPairDataset(
        ml_dataset=ds, user_indices=shard_users, rng_seed=0
    )

    cfg = AttackConfig(
        enabled=True,
        target_item_index=0,
        target_genre="Action",
        attack_budget=0.3,
        num_filler_items=2,
        num_neutral_items=2,
        neutral_genre="Comedy",
        max_synthetic_users_per_coord=50,
    )
    profiles = _build_synthetic_profiles(ds, 5, cfg, ds.num_users)
    poisoned = PoisonedBPRPairDataset(
        clean_dataset=clean_ds,
        synthetic_profiles=profiles,
        base_user_index=ds.num_users,
        num_items=ds.num_items,
    )
    assert len(poisoned) > len(clean_ds)


def test_poisoned_bpr_pair_dataset_items():
    """Synthetic samples should have valid user/item indices."""
    ds = _make_small_dataset()
    clean_ds = BPRPairDataset(
        ml_dataset=ds, user_indices=list(range(5)), rng_seed=0
    )
    cfg = AttackConfig(
        enabled=True,
        target_item_index=0,
        target_genre="Action",
        num_filler_items=1,
        num_neutral_items=1,
        neutral_genre="Comedy",
        max_synthetic_users_per_coord=10,
    )
    profiles = _build_synthetic_profiles(ds, 3, cfg, ds.num_users)
    poisoned = PoisonedBPRPairDataset(
        clean_dataset=clean_ds,
        synthetic_profiles=profiles,
        base_user_index=ds.num_users,
        num_items=ds.num_items,
    )
    # Check a synthetic sample
    for i in range(len(clean_ds), len(poisoned)):
        sample = poisoned[i]
        uid = sample["user_id"].item()
        pos = sample["pos_item_id"].item()
        neg = sample["neg_item_id"].item()
        assert uid >= ds.num_users, "Synthetic user should have extended index"
        assert 0 <= pos < ds.num_items
        assert 0 <= neg < ds.num_items
        assert pos != neg


def test_build_poisoned_dataloader_disabled():
    ds = _make_small_dataset()
    cfg = AttackConfig(enabled=False)
    dl = build_poisoned_dataloader(ds, list(range(10)), cfg, ds.num_users)
    batch = next(iter(dl))
    assert "pos_item_id" in batch


def test_build_poisoned_dataloader_enabled():
    ds = _make_small_dataset()
    cfg = AttackConfig(
        enabled=True,
        target_item_index=0,
        target_genre="Action",
        num_filler_items=1,
        num_neutral_items=1,
        neutral_genre="Comedy",
        max_synthetic_users_per_coord=20,
    )
    extended_users = poisoned_num_users(ds.num_users, cfg)
    dl = build_poisoned_dataloader(ds, list(range(10)), cfg, extended_users)
    batch = next(iter(dl))
    assert "pos_item_id" in batch
    assert "user_id" in batch


if __name__ == "__main__":
    test_poisoned_num_users_disabled()
    test_poisoned_num_users_enabled()
    test_build_synthetic_profiles()
    test_poisoned_bpr_pair_dataset_length()
    test_poisoned_bpr_pair_dataset_items()
    test_build_poisoned_dataloader_disabled()
    test_build_poisoned_dataloader_enabled()
    print("All poison tests passed.")
