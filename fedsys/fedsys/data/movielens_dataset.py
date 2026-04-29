"""
MovieLens dataset loader and PyTorch wrappers for federated BPR training.

This module provides two layers:

1.  ``load_movielens_dataset()``  — pure Python parsing of ml-1m / ml-10m /
    ml-25m / ml-32m archives; returns a ``MovieLensFederatedDataset`` rich
    object that is shared by both the coordinator (evaluation) and the nodes
    (training data partitions).

2.  ``BPRPairDataset``  — wraps a user shard into a PyTorch Dataset of
    (user_idx, pos_item_idx, neg_item_idx) triplets for the pairwise BPR loss.

3.  ``RankingEvalDataset``  — thin wrapper used by the coordinator to expose
    per-user (val_item / test_item) for the ``evaluate_ranking()`` evaluator.

4.  ``partition_users()``  — deterministic user shard assignment (same
    algorithm as recsys/federated/movielens.py so partitions are reproducible).

5.  Helper builders: ``build_movielens_train_dataloader`` and
    ``build_movielens_eval_dataloader``.

Supported variants
------------------
    ml-1m   — movies.dat / ratings.dat  (latin-1, "::" separator)
    ml-10m  — same format
    ml-25m  — movies.csv / ratings.csv  (utf-8)
    ml-32m  — same csv format

Download URLs
-------------
    https://files.grouplens.org/datasets/movielens/ml-1m.zip
    https://files.grouplens.org/datasets/movielens/ml-25m.zip

After unzipping place the folder next to your data root, e.g.
    data/ml-1m/movies.dat
    data/ml-1m/ratings.dat
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

MOVIELENS_VARIANTS: Dict[str, Dict[str, str]] = {
    "ml-1m":  {"folder": "ml-1m",       "format": "dat"},
    "ml-10m": {"folder": "ml-10M100K",  "format": "dat"},
    "ml-25m": {"folder": "ml-25m",       "format": "csv"},
    "ml-32m": {"folder": "ml-32m",       "format": "csv"},
}


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MovieMetadata:
    movie_id: int
    title: str
    genres: Tuple[str, ...]


@dataclass(frozen=True)
class UserSplit:
    """Train/val/test split for one user (all indices are contiguous item indices)."""
    train_ratings:  Tuple[Tuple[int, float], ...]  # (item_index, rating)
    val_item:       int                             # held-out item index for val
    test_item:      int                             # held-out item index for test
    known_items:    FrozenSet[int]                  # ALL positive item indices (train+val+test)


@dataclass(frozen=True)
class MovieLensFederatedDataset:
    """
    Fully preprocessed MovieLens dataset ready for federated training.

    All item/user references inside ``splits_by_user`` are contiguous indices
    (0 … num_items-1, 0 … num_users-1) regardless of the original raw IDs.
    """
    variant:                str
    root:                   Path
    item_ids:               Tuple[int, ...]          # raw movie IDs in sorted order
    item_id_to_index:       Dict[int, int]
    user_ids:               Tuple[int, ...]          # raw user IDs in sorted order
    user_id_to_index:       Dict[int, int]
    movie_metadata:         Tuple[MovieMetadata, ...]
    splits_by_user:         Dict[int, UserSplit]     # contiguous user_index → UserSplit
    dominant_genre_by_user: Dict[int, str]
    users_by_genre:         Dict[str, Tuple[int, ...]]
    items_by_genre:         Dict[str, Tuple[int, ...]]
    train_item_popularity:  Dict[int, int]

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    def title_for_item(self, item_index: int) -> str:
        return self.movie_metadata[item_index].title

    def genres_for_item(self, item_index: int) -> Tuple[str, ...]:
        return self.movie_metadata[item_index].genres


# ---------------------------------------------------------------------------
# Raw file readers
# ---------------------------------------------------------------------------

def _read_dat_movies(path: Path) -> Dict[int, MovieMetadata]:
    movies: Dict[int, MovieMetadata] = {}
    with path.open(encoding="latin-1") as fh:
        for line in fh:
            movie_id_str, title, genres_str = line.rstrip("\n").split("::")
            genres = tuple(g for g in genres_str.split("|") if g)
            movies[int(movie_id_str)] = MovieMetadata(int(movie_id_str), title, genres)
    return movies


def _read_csv_movies(path: Path) -> Dict[int, MovieMetadata]:
    movies: Dict[int, MovieMetadata] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            movie_id = int(row["movieId"])
            genres = tuple(g for g in row["genres"].split("|") if g)
            movies[movie_id] = MovieMetadata(movie_id, row["title"], genres)
    return movies


def _read_dat_ratings(path: Path) -> Dict[int, List[Tuple[int, float, int]]]:
    ratings: Dict[int, List[Tuple[int, float, int]]] = {}
    with path.open(encoding="latin-1") as fh:
        for line in fh:
            u, m, r, t = line.rstrip("\n").split("::")
            ratings.setdefault(int(u), []).append((int(m), float(r), int(t)))
    return ratings


def _read_csv_ratings(path: Path) -> Dict[int, List[Tuple[int, float, int]]]:
    ratings: Dict[int, List[Tuple[int, float, int]]] = {}
    with path.open(encoding="utf-8") as fh:
        fh.readline()  # skip header
        for line in fh:
            u, m, r, t = line.rstrip("\n").split(",")
            ratings.setdefault(int(u), []).append((int(m), float(r), int(t)))
    return ratings


def _dominant_genre(
    train_ratings: Sequence[Tuple[int, float]],
    meta_by_id: Dict[int, MovieMetadata],
) -> str:
    scores: Dict[str, float] = {}
    for movie_id, rating in train_ratings:
        for genre in meta_by_id[movie_id].genres or ("(no genres listed)",):
            scores[genre] = scores.get(genre, 0.0) + rating
    if not scores:
        return "(unknown)"
    return max(scores, key=lambda g: (scores[g], g))


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_movielens_dataset(
    data_root: str | Path,
    variant: str = "ml-1m",
    min_positive_rating: float = 4.0,
    min_positive_interactions: int = 3,
    max_positive_interactions: Optional[int] = None,
    show_progress: bool = True,
) -> MovieLensFederatedDataset:
    """
    Parse a MovieLens archive and return a fully preprocessed dataset object.

    Train/val/test split strategy
    ------------------------------
    For each user, sort positive interactions (rating >= min_positive_rating)
    by timestamp.  The **last two** become val_item and test_item respectively;
    all earlier interactions form the training set.  Users with fewer than
    min_positive_interactions (after filtering) are dropped.

    Parameters
    ----------
    data_root  : Directory that contains the variant folder (e.g. ``ml-1m/``).
    variant    : One of "ml-1m", "ml-10m", "ml-25m", "ml-32m".
    min_positive_rating : Interactions below this threshold are discarded.
    min_positive_interactions : Minimum required interactions per user.
    max_positive_interactions : If set, users with more interactions are dropped.
    show_progress : Print progress lines.
    """
    if variant not in MOVIELENS_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Choose from {list(MOVIELENS_VARIANTS)}"
        )
    spec = MOVIELENS_VARIANTS[variant]
    root = Path(data_root)
    candidate = root / spec["folder"]
    root = candidate if candidate.exists() else root
    fmt  = spec["format"]

    t0 = time.perf_counter()
    if show_progress:
        print(f"[movielens] Loading {variant} from {root} ...")

    if fmt == "dat":
        meta_by_id   = _read_dat_movies(root / "movies.dat")
        ratings_raw  = _read_dat_ratings(root / "ratings.dat")
    else:
        meta_by_id   = _read_csv_movies(root / "movies.csv")
        ratings_raw  = _read_csv_ratings(root / "ratings.csv")

    # ── Filter, split per user ──────────────────────────────────────────────
    filtered_users: Dict[int, UserSplit]    = {}
    dominant_by_id: Dict[int, str]          = {}
    pop_by_movie_id: Dict[int, int]         = {}
    retained_items: set                     = set()

    for user_id, ratings in ratings_raw.items():
        pos = [
            (m, r, ts) for m, r, ts in ratings
            if r >= min_positive_rating and m in meta_by_id
        ]
        if len(pos) < min_positive_interactions:
            continue
        if max_positive_interactions is not None and len(pos) > max_positive_interactions:
            continue

        pos.sort(key=lambda x: (x[2], x[0]))       # sort by (timestamp, movie_id)
        train_entries = pos[:-2]
        val_entry     = pos[-2]
        test_entry    = pos[-1]
        if not train_entries:
            continue

        train_ratings = tuple((m, r) for m, r, _ in train_entries)
        known_items   = frozenset(m for m, _, _ in pos)
        filtered_users[user_id] = UserSplit(
            train_ratings = train_ratings,
            val_item      = val_entry[0],
            test_item     = test_entry[0],
            known_items   = known_items,
        )
        dominant_by_id[user_id] = _dominant_genre(train_ratings, meta_by_id)

        for m, _ in train_ratings:
            retained_items.add(m)
            pop_by_movie_id[m] = pop_by_movie_id.get(m, 0) + 1
        retained_items.add(val_entry[0])
        retained_items.add(test_entry[0])

    if not filtered_users:
        raise ValueError("No users remained after MovieLens filtering/splitting.")

    # ── Build contiguous indices ────────────────────────────────────────────
    item_ids        = tuple(sorted(retained_items))
    item_id_to_idx  = {m: i for i, m in enumerate(item_ids)}
    user_ids        = tuple(sorted(filtered_users))
    user_id_to_idx  = {u: i for i, u in enumerate(user_ids)}
    movie_metadata  = tuple(meta_by_id[m] for m in item_ids)

    splits_by_user:     Dict[int, UserSplit] = {}
    dom_genre_by_user:  Dict[int, str]       = {}
    pop_by_item_idx:    Dict[int, int]       = {}
    users_by_genre_l:   Dict[str, List[int]] = {}
    items_by_genre_s:   Dict[str, set]       = {}

    for user_id in user_ids:
        uidx  = user_id_to_idx[user_id]
        split = filtered_users[user_id]

        mapped_train = tuple(
            (item_id_to_idx[m], r) for m, r in split.train_ratings
        )
        mapped_val   = item_id_to_idx[split.val_item]
        mapped_test  = item_id_to_idx[split.test_item]
        mapped_known = frozenset(item_id_to_idx[m] for m in split.known_items)

        splits_by_user[uidx] = UserSplit(
            train_ratings = mapped_train,
            val_item      = mapped_val,
            test_item     = mapped_test,
            known_items   = mapped_known,
        )

        genre = dominant_by_id[user_id]
        dom_genre_by_user[uidx] = genre
        users_by_genre_l.setdefault(genre, []).append(uidx)

        for iidx, _ in mapped_train:
            pop_by_item_idx[iidx] = pop_by_item_idx.get(iidx, 0) + 1

    for iidx, meta in enumerate(movie_metadata):
        for genre in meta.genres or ("(no genres listed)",):
            items_by_genre_s.setdefault(genre, set()).add(iidx)

    elapsed = time.perf_counter() - t0
    if show_progress:
        print(
            f"[movielens] Loaded in {elapsed:.1f}s: "
            f"{len(user_ids):,} users, {len(item_ids):,} items, "
            f"{len(users_by_genre_l)} genre segments"
        )

    return MovieLensFederatedDataset(
        variant                = variant,
        root                   = root,
        item_ids               = item_ids,
        item_id_to_index       = item_id_to_idx,
        user_ids               = user_ids,
        user_id_to_index       = user_id_to_idx,
        movie_metadata         = movie_metadata,
        splits_by_user         = splits_by_user,
        dominant_genre_by_user = dom_genre_by_user,
        users_by_genre         = {g: tuple(sorted(us)) for g, us in users_by_genre_l.items()},
        items_by_genre         = {g: tuple(sorted(ii)) for g, ii in items_by_genre_s.items()},
        train_item_popularity  = pop_by_item_idx,
    )


# ---------------------------------------------------------------------------
# User partitioning
# ---------------------------------------------------------------------------

def partition_users(
    num_users: int,
    num_shards: int,
    seed: int = 42,
) -> Tuple[Dict[int, Tuple[int, ...]], Dict[int, int]]:
    """
    Assign user indices to shards deterministically (LCG shuffle + round-robin).

    Returns
    -------
    shard_to_users : {shard_id: (user_index, ...)}
    user_to_shard  : {user_index: shard_id}
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    order = list(range(num_users))
    for i in range(len(order) - 1, 0, -1):
        j = (seed * 1_103_515_245 + 12_345 + i) % (i + 1)
        order[i], order[j] = order[j], order[i]
    shards: Dict[int, List[int]] = {s: [] for s in range(num_shards)}
    u2s: Dict[int, int] = {}
    for pos, uidx in enumerate(order):
        shard = pos % num_shards
        shards[shard].append(uidx)
        u2s[uidx] = shard
    return {s: tuple(us) for s, us in shards.items()}, u2s


# ---------------------------------------------------------------------------
# PyTorch Dataset: training (pairwise BPR triplets)
# ---------------------------------------------------------------------------

class BPRPairDataset(Dataset):
    """
    Produces (user_idx, pos_item_idx, neg_item_idx) triplets for BPR training.

    One triplet is created per (user, train-interaction) pair.  Negative items
    are sampled uniformly at ``__getitem__`` time so the negatives change each
    epoch (online negative sampling).

    Parameters
    ----------
    ml_dataset    : Parsed MovieLens dataset.
    user_indices  : Which user indices belong to this shard.
    rng_seed      : Seed for negative sampling RNG (reproducibility).
    """

    def __init__(
        self,
        ml_dataset: MovieLensFederatedDataset,
        user_indices: Sequence[int],
        rng_seed: Optional[int] = None,
    ) -> None:
        self._num_items = ml_dataset.num_items
        self._rng = random.Random(rng_seed)

        # Build flat list of (user_idx, pos_item_idx) and per-triple known set
        self._pairs:  List[Tuple[int, int]]          = []
        self._known:  List[FrozenSet[int]]            = []

        for uidx in user_indices:
            split = ml_dataset.splits_by_user[uidx]
            for iidx, _ in split.train_ratings:
                self._pairs.append((uidx, iidx))
                self._known.append(split.known_items)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_idx, pos_idx = self._pairs[idx]
        known = self._known[idx]

        # Online negative sampling: reject positives
        neg_idx = self._rng.randrange(self._num_items)
        while neg_idx in known:
            neg_idx = self._rng.randrange(self._num_items)

        return {
            "user_id":     torch.tensor(user_idx, dtype=torch.long),
            "pos_item_id": torch.tensor(pos_idx,  dtype=torch.long),
            "neg_item_id": torch.tensor(neg_idx,  dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_movielens_train_dataloader(
    ml_dataset: MovieLensFederatedDataset,
    shard_user_indices: Sequence[int],
    batch_size: int = 512,
    num_workers: int = 0,
    rng_seed: Optional[int] = None,
) -> DataLoader:
    """
    Build a shuffled BPR training DataLoader for one shard's users.

    Parameters
    ----------
    ml_dataset         : Loaded MovieLens dataset.
    shard_user_indices : User indices belonging to this FL partition.
    batch_size         : Mini-batch size.
    num_workers        : Parallel workers for DataLoader (0 = main thread).
    rng_seed           : Seed for negative sampling.
    """
    ds = BPRPairDataset(ml_dataset, shard_user_indices, rng_seed=rng_seed)
    print(
        f"[movielens] BPRPairDataset: {len(shard_user_indices)} users, "
        f"{len(ds):,} training triplets"
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
