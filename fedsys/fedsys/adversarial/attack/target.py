"""
fedsys.adversarial.attack.target
==================================
Target item / genre selection helpers for the data-poisoning attack.

Ported from ``recsys/federated/benchmark.py``.

Two strategies are provided:

1. **Simple** — ``select_target_item()``: pick the globally least popular item
   among candidates that belong to the target genre and have non-zero
   popularity (so they appear in training data at all).  Low popularity means
   the item is already hard to retrieve, making the attack more impactful.

2. **Model-based** — ``select_target_from_clean_model()``: evaluate
   the *vulnerability* of each candidate by running the current global model
   and measuring how far the target item is from appearing in the top-k for
   benign users in the target segment.  The most "promotable" item (close to
   top-k boundary but not yet there) is selected.

Automatic genre selection — ``choose_target_genre()``: score genres by
average item popularity within the segment, then pick the genre whose items
would benefit most from a push (least popular genre with enough segment
support).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from fedsys.data.movielens_dataset import MovieLensFederatedDataset


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def benign_segment_users(
    dataset: MovieLensFederatedDataset,
    target_genre: str,
) -> List[int]:
    """
    Return the list of real user indices whose *dominant genre* is
    ``target_genre``.  These are the "victim" users whose top-k lists the
    attacker wants to inject the target item into.
    """
    return list(dataset.users_by_genre.get(target_genre, ()))


def select_target_item(
    dataset: MovieLensFederatedDataset,
    target_genre: str,
    min_popularity: int = 1,
    max_popularity: Optional[int] = None,
) -> int:
    """
    Simple heuristic: pick the item with the lowest training-popularity among
    items that belong to ``target_genre`` and have at least ``min_popularity``
    interactions.

    This is the "easy target" — nearly invisible in training data, so any
    rank improvement is clearly attributable to the attack.

    Parameters
    ----------
    dataset         : Loaded MovieLensFederatedDataset.
    target_genre    : Genre to search within.
    min_popularity  : Skip items with fewer than this many training interactions.
    max_popularity  : Skip items that are already popular (optional ceiling).

    Returns
    -------
    Contiguous item index (0-based) of the chosen target.

    Raises
    ------
    ValueError : If no qualifying item is found.
    """
    genre_items = dataset.items_by_genre.get(target_genre, ())
    if not genre_items:
        raise ValueError(f"No items found for genre {target_genre!r}")

    candidates = [
        (dataset.train_item_popularity.get(i, 0), i)
        for i in genre_items
        if dataset.train_item_popularity.get(i, 0) >= min_popularity
        and (max_popularity is None
             or dataset.train_item_popularity.get(i, 0) <= max_popularity)
    ]

    if not candidates:
        raise ValueError(
            f"No item in genre {target_genre!r} satisfies "
            f"popularity in [{min_popularity}, {max_popularity}]."
        )

    # Least popular among qualifying items
    candidates.sort()
    return candidates[0][1]


def _score_item_vulnerability(
    model: nn.Module,
    item_index: int,
    segment_users: Sequence[int],
    dataset: MovieLensFederatedDataset,
    k: int,
    device: str,
    batch_size: int = 512,
) -> float:
    """
    Compute a "vulnerability score" for a single item w.r.t. a user segment.

    Score = fraction of segment users for whom the item is within the top-(2k)
    positions but not yet in top-k — i.e., it is "close" to being
    recommended.  Items with a higher score are good push targets because a
    small nudge can cross the recommendation threshold.

    A higher score means the item is more vulnerable / easy to promote.
    """
    model.eval()
    num_items = dataset.num_items
    item_t    = torch.tensor(item_index, dtype=torch.long, device=device)

    close_count = 0
    total       = 0

    with torch.no_grad():
        for start in range(0, len(segment_users), batch_size):
            batch_users = segment_users[start : start + batch_size]
            if not batch_users:
                continue

            u_t = torch.tensor(batch_users, dtype=torch.long, device=device)
            all_items_t = torch.arange(num_items, dtype=torch.long, device=device)

            # Score all items for each user in the batch
            # BPR / TwoLayer / NeuralCF signatures: forward(u, i, histories=None)
            scores_rows = []
            for uid in u_t:
                uid_rep = uid.unsqueeze(0).expand(num_items)
                sc = model(uid_rep, all_items_t)          # (num_items,)
                scores_rows.append(sc.cpu())

            scores = torch.stack(scores_rows, dim=0)      # (batch, num_items)

            # Score of the target item for each user
            target_scores = scores[:, item_index]         # (batch,)

            # How many items score higher than target for each user?
            ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1)  # (batch,)

            in_top_2k = ranks < (2 * k)
            not_top_k = ranks >= k

            close_count += int((in_top_2k & not_top_k).sum().item())
            total       += len(batch_users)

    if total == 0:
        return 0.0
    return close_count / total


def select_target_from_clean_model(
    model: nn.Module,
    dataset: MovieLensFederatedDataset,
    target_genre: str,
    k: int = 10,
    device: str = "cpu",
    top_n_candidates: int = 20,
    min_popularity: int = 1,
    max_popularity_fraction: float = 0.10,
) -> int:
    """
    Model-based target selection using vulnerability scoring.

    Algorithm (ported from recsys/federated/benchmark.py):
    1. Collect candidate items from ``target_genre`` filtered by popularity.
    2. Identify the benign segment users (dominant genre == target_genre).
    3. For each candidate item, compute the "vulnerability score" (fraction of
       segment users for whom the item is in top-2k but not yet top-k).
    4. Return the item with the highest vulnerability score.

    Parameters
    ----------
    model                  : Current global model (BPRModel or compatible).
    dataset                : Loaded MovieLensFederatedDataset.
    target_genre           : Genre to attack.
    k                      : Recommendation cutoff used for vulnerability.
    device                 : Torch device string.
    top_n_candidates       : Only evaluate the N least-popular candidates
                             (limits compute for large catalogs).
    min_popularity         : Skip items with zero training interactions.
    max_popularity_fraction: Skip items whose popularity > this fraction of
                             the total training interactions (already popular).

    Returns
    -------
    Contiguous item index of the best target.

    Raises
    ------
    ValueError : If no qualifying item is found.
    """
    total_interactions = sum(dataset.train_item_popularity.values()) or 1
    max_pop = int(max_popularity_fraction * total_interactions)

    genre_items = dataset.items_by_genre.get(target_genre, ())
    if not genre_items:
        raise ValueError(f"No items found for genre {target_genre!r}")

    candidates = [
        i for i in genre_items
        if min_popularity
            <= dataset.train_item_popularity.get(i, 0)
            <= max_pop
    ]

    if not candidates:
        raise ValueError(
            f"No item in genre {target_genre!r} satisfies popularity filter."
        )

    # Keep only the N least-popular candidates to limit GPU compute
    candidates.sort(key=lambda i: dataset.train_item_popularity.get(i, 0))
    candidates = candidates[:top_n_candidates]

    segment_users = benign_segment_users(dataset, target_genre)
    if not segment_users:
        # Fallback to simple selection if no segment users
        return select_target_item(dataset, target_genre, min_popularity, max_pop)

    model = model.to(device)
    scored: List[Tuple[float, int]] = []
    for item_index in candidates:
        score = _score_item_vulnerability(
            model=model,
            item_index=item_index,
            segment_users=segment_users,
            dataset=dataset,
            k=k,
            device=device,
        )
        scored.append((score, item_index))
        print(
            f"[target] item={item_index:5d} "
            f"pop={dataset.train_item_popularity.get(item_index, 0):6d}  "
            f"vuln={score:.4f}"
        )

    if not scored:
        raise ValueError("No candidates could be scored.")

    # Highest vulnerability score wins; break ties by lowest popularity
    scored.sort(key=lambda x: (-x[0], dataset.train_item_popularity.get(x[1], 0)))
    best_item = scored[0][1]
    print(
        f"[target] Selected target item {best_item} "
        f"(vuln={scored[0][0]:.4f}, "
        f"pop={dataset.train_item_popularity.get(best_item, 0)})"
    )
    return best_item


# ---------------------------------------------------------------------------
# Automatic genre selection
# ---------------------------------------------------------------------------

def _genre_vulnerability_score(
    dataset: MovieLensFederatedDataset,
    genre: str,
    min_segment_users: int = 10,
) -> float:
    """
    Score a genre by how "attackable" it is:

        score = 1 / (avg_item_popularity_in_genre + 1)

    Genres with low average item popularity are better attack targets.
    Genres with fewer than ``min_segment_users`` are skipped (score = -inf).
    """
    segment = dataset.users_by_genre.get(genre, ())
    if len(segment) < min_segment_users:
        return -math.inf

    items = dataset.items_by_genre.get(genre, ())
    if not items:
        return -math.inf

    avg_pop = sum(
        dataset.train_item_popularity.get(i, 0) for i in items
    ) / len(items)

    return 1.0 / (avg_pop + 1.0)


def choose_target_genre(
    dataset: MovieLensFederatedDataset,
    min_segment_users: int = 10,
    exclude_genres: Optional[Sequence[str]] = None,
) -> str:
    """
    Automatically select the genre to attack.

    Picks the genre with the highest vulnerability score:
        - Has at least ``min_segment_users`` dominant-genre users.
        - Items in the genre have the lowest average popularity
          (easiest to push up the ranking).

    Parameters
    ----------
    dataset           : Loaded MovieLensFederatedDataset.
    min_segment_users : Minimum number of segment users required.
    exclude_genres    : Genres to skip (e.g., ["(no genres listed)"]).

    Returns
    -------
    Genre string, e.g. "Horror".

    Raises
    ------
    ValueError : If no qualifying genre is found.
    """
    _exclude = set(exclude_genres or ["(no genres listed)", "(unknown)"])

    scored: List[Tuple[float, str]] = []
    for genre in dataset.items_by_genre:
        if genre in _exclude:
            continue
        score = _genre_vulnerability_score(
            dataset, genre, min_segment_users=min_segment_users
        )
        if score > -math.inf:
            scored.append((score, genre))

    if not scored:
        raise ValueError(
            "No qualifying genre found.  "
            "Try lowering min_segment_users or adjust exclude_genres."
        )

    scored.sort(key=lambda x: -x[0])
    best_genre = scored[0][1]
    print(
        f"[target] Auto-selected genre: {best_genre!r} "
        f"(score={scored[0][0]:.6f})"
    )
    return best_genre
