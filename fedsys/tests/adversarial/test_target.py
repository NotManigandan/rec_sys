"""
tests/adversarial/test_target.py
==================================
Unit tests for fedsys.adversarial.attack.target
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import random
from typing import Dict, Tuple

import torch
import torch.nn as nn

from fedsys.adversarial.attack.target import (
    benign_segment_users,
    select_target_item,
    choose_target_genre,
    select_target_from_clean_model,
)
from fedsys.data.movielens_dataset import (
    MovieLensFederatedDataset, MovieMetadata, UserSplit,
)


# ---------------------------------------------------------------------------
# Minimal dataset fixture (reused from test_poison.py)
# ---------------------------------------------------------------------------

def _make_small_dataset(
    num_users: int = 60,
    num_items: int = 30,
    seed: int = 7,
) -> MovieLensFederatedDataset:
    rng = random.Random(seed)
    item_ids = tuple(range(num_items))
    user_ids = tuple(range(num_users))
    genres = ["Action", "Comedy", "Drama", "Horror"]
    metadata = tuple(
        MovieMetadata(i, f"Movie{i}", (genres[i % len(genres)],))
        for i in range(num_items)
    )
    splits, dominant_by_user = {}, {}
    for u in range(num_users):
        items = rng.sample(range(num_items), 7)
        splits[u] = UserSplit(
            train_ratings=tuple((it, 5.0) for it in items[:5]),
            val_item=items[5], test_item=items[6],
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
        variant="test", root=None,
        item_ids=item_ids, item_id_to_index={i: i for i in range(num_items)},
        user_ids=user_ids, user_id_to_index={u: u for u in range(num_users)},
        movie_metadata=metadata, splits_by_user=splits,
        dominant_genre_by_user=dominant_by_user,
        users_by_genre=users_by_genre, items_by_genre=items_by_genre,
        train_item_popularity=pop,
    )


# ---------------------------------------------------------------------------
# Tiny mock model
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self, num_users, num_items, dim=4):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)

    def forward(self, user_ids, item_ids, histories=None):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        return (u * v).sum(dim=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_benign_segment_users():
    ds = _make_small_dataset()
    users = benign_segment_users(ds, "Action")
    assert len(users) > 0
    # All returned users should have Action as dominant genre
    for u in users:
        assert ds.dominant_genre_by_user[u] == "Action"


def test_benign_segment_users_unknown_genre():
    ds = _make_small_dataset()
    users = benign_segment_users(ds, "Sci-Fi")
    assert users == []


def test_select_target_item_returns_valid():
    ds = _make_small_dataset()
    item = select_target_item(ds, "Action", min_popularity=1)
    genre_items = set(ds.items_by_genre["Action"])
    assert item in genre_items


def test_select_target_item_least_popular():
    ds = _make_small_dataset()
    item = select_target_item(ds, "Action", min_popularity=1)
    genre_items = ds.items_by_genre["Action"]
    pops = [ds.train_item_popularity.get(i, 0) for i in genre_items if
            ds.train_item_popularity.get(i, 0) >= 1]
    # The selected item should have the minimum popularity among qualifying items
    assert ds.train_item_popularity.get(item, 0) == min(pops)


def test_select_target_item_bad_genre():
    ds = _make_small_dataset()
    try:
        select_target_item(ds, "Sci-Fi")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_choose_target_genre():
    ds = _make_small_dataset()
    genre = choose_target_genre(ds, min_segment_users=5)
    assert genre in ds.items_by_genre
    assert len(ds.users_by_genre.get(genre, ())) >= 5


def test_select_target_from_clean_model():
    ds = _make_small_dataset()
    model = _TinyModel(ds.num_users, ds.num_items)
    item = select_target_from_clean_model(
        model=model,
        dataset=ds,
        target_genre="Action",
        k=5,
        device="cpu",
        top_n_candidates=5,
        min_popularity=1,
    )
    assert item in set(ds.items_by_genre["Action"])


if __name__ == "__main__":
    test_benign_segment_users()
    test_benign_segment_users_unknown_genre()
    test_select_target_item_returns_valid()
    test_select_target_item_least_popular()
    test_select_target_item_bad_genre()
    test_choose_target_genre()
    test_select_target_from_clean_model()
    print("All target tests passed.")
