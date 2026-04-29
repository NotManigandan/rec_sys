"""
fedsys.adversarial.attack.poison
=================================
Data-poisoning attack for federated BPR recommender training.

Ported and adapted from ``recsys/federated/attack.py``.

Attack model
------------
A *malicious node* injects N_synth = ceil(|shard_users| * attack_budget)
synthetic user profiles into its local training data before each local
training round.  Every synthetic profile consists of:

  * 1  target item   — the item to push (with configurable rating weight)
  * F  filler items  — other items from the same genre (hot = most popular)
  * M  neutral items — items from an unrelated genre (provides plausible cover)

Because the synthetic users share the target item as a strong positive, the
BPR loss gradient pushes the target item's embedding toward the generic
"synthetic-user direction", inflating its score for future recommendations.

Key design decision for fedsys
-------------------------------
In recsys the user embeddings are *local* to each shard (stored in
ServerState), so synthetic users can have indices ≥ num_real_users without
any shape conflict.  In fedsys both user **and** item embeddings live in the
single global BPRModel whose nn.Embedding sizes are fixed at init time.

We handle this by *reserving* a block of user indices above the real-user
range for synthetic profiles.  The coordinator calls ``poisoned_num_users()``
to get the extended user count and passes it to ModelConfig so every node
sees the same embedding-table shape.  Only the malicious node actually trains
on synthetic-user rows; clean nodes simply never index into that range.

Quick-start
-----------
::

    from fedsys.adversarial.attack.poison import (
        build_poisoned_dataloader,
        poisoned_num_users,
    )

    # At coordinator startup (to size the model correctly):
    extended_users = poisoned_num_users(dataset.num_users, cfg)

    # At the malicious node (replaces the normal BPRPairDataset):
    dl = build_poisoned_dataloader(
        dataset, shard_users, attack_cfg, model_cfg.num_users
    )
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from fedsys.data.movielens_dataset import (
    BPRPairDataset,
    MovieLensFederatedDataset,
    UserSplit,
)


# ---------------------------------------------------------------------------
# Attack configuration
# ---------------------------------------------------------------------------

@dataclass
class AttackConfig:
    """
    Parameters for the data-poisoning attack.

    Fields
    ------
    enabled          : Master on/off switch.  When False every builder returns
                       the clean dataset unchanged.
    target_item_index: Contiguous item index of the item to push.  If -1,
                       you must call ``select_target_item()`` first and fill
                       this field.
    target_genre     : Genre of the target item (used to pick filler items).
    attack_budget    : Fraction of the shard's real users to add as synthetic
                       profiles.  E.g. 0.3 -> 30 % extra synthetic users.
    num_filler_items : How many filler items (same genre as target) to include
                       in each synthetic profile.
    num_neutral_items: How many neutral items (unrelated genre) to include.
    filler_from_top  : If True pick filler from the *most* popular genre items;
                       otherwise pick uniformly at random.
    neutral_genre    : Genre to draw neutral items from (default: "Comedy").
    target_weight    : Synthetic-profile rating for the target item (relative
                       to 1.0 for filler / neutral).  Higher => stronger push.
    max_synthetic_users_per_coord: Upper bound used by the coordinator to
                       pre-allocate embedding rows.  Set to a value that is
                       >= attack_budget * max_shard_size for any shard.
    """
    enabled: bool = False

    target_item_index: int = -1
    target_genre: str = ""

    attack_budget: float = 0.30
    num_filler_items: int = 30
    num_neutral_items: int = 20

    filler_from_top: bool = True
    neutral_genre: str = "Comedy"

    target_weight: float = 1.0

    # Used by coordinator only to size the embedding table.
    max_synthetic_users_per_coord: int = 200

    # Negative-sampling seed for reproducibility.
    seed: int = 42


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def poisoned_num_users(real_num_users: int, attack_cfg: AttackConfig) -> int:
    """
    Return the *extended* user count that must be used for ModelConfig.num_users
    when the attack is active, so that synthetic user indices are within the
    embedding table.

    When the attack is disabled this just returns ``real_num_users``.
    """
    if not attack_cfg.enabled:
        return real_num_users
    return real_num_users + attack_cfg.max_synthetic_users_per_coord


# ---------------------------------------------------------------------------
# Synthetic profile builder
# ---------------------------------------------------------------------------

def _build_synthetic_profiles(
    dataset: MovieLensFederatedDataset,
    num_profiles: int,
    attack_cfg: AttackConfig,
    base_user_index: int,
) -> List[UserSplit]:
    """
    Build ``num_profiles`` synthetic UserSplit records.

    Each record has:
      - train_ratings : (target_item, high_weight) + filler + neutral tuples
      - val_item / test_item : target_item (evaluator will see high target hit)
      - known_items  : union of all items in the profile

    Parameters
    ----------
    base_user_index : First *real* synthetic user index.  Synthetic user i
                      gets global index base_user_index + i.
    """
    rng = random.Random(attack_cfg.seed)
    target_idx  = attack_cfg.target_item_index
    tgt_genre   = attack_cfg.target_genre

    # ── Filler items (same genre as target, excluding target itself) ──────────
    genre_items = list(dataset.items_by_genre.get(tgt_genre, ()))
    filler_pool = [x for x in genre_items if x != target_idx]

    if attack_cfg.filler_from_top:
        # Sort by training popularity descending
        filler_pool.sort(
            key=lambda x: dataset.train_item_popularity.get(x, 0), reverse=True
        )
        filler_items = filler_pool[: attack_cfg.num_filler_items]
    else:
        rng.shuffle(filler_pool)
        filler_items = filler_pool[: attack_cfg.num_filler_items]

    # ── Neutral items (different genre) ──────────────────────────────────────
    neutral_pool = [
        x for x in dataset.items_by_genre.get(attack_cfg.neutral_genre, ())
        if x not in set(filler_items) and x != target_idx
    ]
    rng.shuffle(neutral_pool)
    neutral_items = neutral_pool[: attack_cfg.num_neutral_items]

    # ── Assemble profiles ────────────────────────────────────────────────────
    profiles: List[UserSplit] = []
    for _ in range(num_profiles):
        train_ratings: List[Tuple[int, float]] = []

        # Target item gets a boosted "rating" to dominate the BPR loss
        train_ratings.append((target_idx, 5.0 * attack_cfg.target_weight))

        for item in filler_items:
            train_ratings.append((item, 4.0))

        for item in neutral_items:
            train_ratings.append((item, 3.5))

        known = frozenset(r[0] for r in train_ratings)
        profiles.append(
            UserSplit(
                train_ratings=tuple(train_ratings),
                val_item=target_idx,
                test_item=target_idx,
                known_items=known,
            )
        )

    return profiles


# ---------------------------------------------------------------------------
# PoisonedBPRPairDataset
# ---------------------------------------------------------------------------

class PoisonedBPRPairDataset(Dataset):
    """
    A BPR triplet dataset that combines:

    1. The *clean* shard BPRPairDataset (real users with real training data).
    2. *Synthetic* user profiles injected at the end of the user table.

    The synthetic users receive user indices in the range
    [real_num_users, real_num_users + num_profiles).

    These extra indices are valid **only if** ModelConfig.num_users was set
    to ``poisoned_num_users(real_num_users, attack_cfg)`` at coordinator
    startup.

    Parameters
    ----------
    clean_dataset     : The normal BPRPairDataset for this node.
    synthetic_profiles: Output of _build_synthetic_profiles().
    base_user_index   : First extended user index (== real_num_users).
    num_items         : Total number of items (for negative sampling).
    seed              : RNG seed for negative sampling.
    """

    def __init__(
        self,
        clean_dataset: BPRPairDataset,
        synthetic_profiles: List[UserSplit],
        base_user_index: int,
        num_items: int,
        seed: int = 42,
    ) -> None:
        self._clean = clean_dataset
        self._profiles = synthetic_profiles
        self._base_user_index = base_user_index
        self._num_items = num_items
        self._rng = random.Random(seed)

        # Pre-expand synthetic profiles into (user_idx, pos_item) pairs
        # (same strategy as BPRPairDataset._build_index)
        self._synth_pairs: List[Tuple[int, int, FrozenSet]] = []
        for i, profile in enumerate(synthetic_profiles):
            uid = base_user_index + i
            known = profile.known_items
            for item_idx, _ in profile.train_ratings:
                self._synth_pairs.append((uid, item_idx, known))

    # ── Dataset protocol ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._clean) + len(self._synth_pairs)

    def __getitem__(self, idx: int) -> Dict:
        if idx < len(self._clean):
            return self._clean[idx]
        # Synthetic sample
        uid, pos_item, known = self._synth_pairs[idx - len(self._clean)]
        neg_item = pos_item
        while neg_item == pos_item or neg_item in known:
            neg_item = self._rng.randrange(self._num_items)
        return {
            "user_id":     torch.tensor(uid, dtype=torch.long),
            "pos_item_id": torch.tensor(pos_item, dtype=torch.long),
            "neg_item_id": torch.tensor(neg_item, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Builder (public API)
# ---------------------------------------------------------------------------

def build_poisoned_dataloader(
    dataset: MovieLensFederatedDataset,
    shard_user_indices: Sequence[int],
    attack_cfg: AttackConfig,
    model_num_users: int,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for a *malicious* node.

    If ``attack_cfg.enabled`` is False this simply wraps a clean
    ``BPRPairDataset`` — identical to what a normal node would use.

    Parameters
    ----------
    dataset           : The loaded MovieLensFederatedDataset.
    shard_user_indices: The real user indices assigned to this node.
    attack_cfg        : Attack parameters.
    model_num_users   : ModelConfig.num_users (must include synthetic slots).
    batch_size        : DataLoader batch size.
    num_workers       : DataLoader worker count.
    """
    clean_ds = BPRPairDataset(
        ml_dataset=dataset,
        user_indices=list(shard_user_indices),
        rng_seed=attack_cfg.seed,
    )

    if not attack_cfg.enabled:
        return DataLoader(
            clean_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers,
        )

    if attack_cfg.target_item_index < 0:
        raise ValueError(
            "attack_cfg.target_item_index must be set (>= 0) before building "
            "a poisoned dataloader.  Call select_target_item() first."
        )
    if not attack_cfg.target_genre:
        raise ValueError(
            "attack_cfg.target_genre must be set before building a poisoned "
            "dataloader."
        )

    num_real_users = dataset.num_users
    # Synthetic user indices start right after the real user block
    base_synth_idx = num_real_users
    num_synth = math.ceil(len(shard_user_indices) * attack_cfg.attack_budget)
    num_synth = min(num_synth, attack_cfg.max_synthetic_users_per_coord)

    profiles = _build_synthetic_profiles(
        dataset=dataset,
        num_profiles=num_synth,
        attack_cfg=attack_cfg,
        base_user_index=base_synth_idx,
    )

    poisoned_ds = PoisonedBPRPairDataset(
        clean_dataset=clean_ds,
        synthetic_profiles=profiles,
        base_user_index=base_synth_idx,
        num_items=dataset.num_items,
        seed=attack_cfg.seed,
    )

    return DataLoader(
        poisoned_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
