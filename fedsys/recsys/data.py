from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass(frozen=True)
class PreferenceSample:
    user_id: int
    positive_item_id: int
    negative_item_id: int
    weight: float


@dataclass(frozen=True)
class RatingsDataset:
    root: Path
    item_names: Sequence[str]
    user_profiles: Dict[int, str]
    actual_ratings: Dict[int, Dict[int, float]]
    reported_ratings: Dict[int, Dict[int, float]]

    @property
    def num_users(self) -> int:
        return len(self.actual_ratings)

    @property
    def num_items(self) -> int:
        return len(self.item_names)

    @property
    def user_ids(self) -> List[int]:
        return sorted(self.actual_ratings)

    def reported_items(self, user_id: int) -> List[int]:
        return sorted(self.reported_ratings[user_id])

    def unseen_items(self, user_id: int) -> List[int]:
        reported = self.reported_ratings[user_id]
        return [item_id for item_id in range(self.num_items) if item_id not in reported]

    def history_tensor(self) -> torch.Tensor:
        history = torch.zeros((self.num_users, self.num_items), dtype=torch.float32)
        for user_id, item_ratings in self.reported_ratings.items():
            for item_id, rating in item_ratings.items():
                history[user_id, item_id] = rating / 5.0
        return history

    def profile_names(self) -> List[str]:
        return sorted(set(self.user_profiles.values()))


def _load_item_names(path: Path) -> List[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    if not header or header[0] != "user_id":
        raise ValueError(f"Unexpected matrix header in {path}")
    return header[1:]


def _load_long_ratings(path: Path) -> tuple[Dict[int, Dict[int, float]], Dict[int, str]]:
    ratings: Dict[int, Dict[int, float]] = {}
    profiles: Dict[int, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"user_id", "item_id", "rating", "true_profile"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            rating = float(row["rating"])
            ratings.setdefault(user_id, {})[item_id] = rating
            profiles[user_id] = row["true_profile"]
    return ratings, profiles


def load_ratings_dataset(root: Path | str) -> RatingsDataset:
    root_path = Path(root)
    item_names = _load_item_names(root_path / "actual_matrix.csv")
    actual_ratings, actual_profiles = _load_long_ratings(root_path / "actual.csv")
    reported_ratings, reported_profiles = _load_long_ratings(root_path / "reported.csv")

    if set(actual_ratings) != set(reported_ratings):
        raise ValueError("Actual and reported user sets do not match.")

    if actual_profiles != reported_profiles:
        raise ValueError("Actual and reported profile labels do not match.")

    num_items = len(item_names)
    for user_id, item_ratings in actual_ratings.items():
        if len(item_ratings) != num_items:
            raise ValueError(f"User {user_id} in actual data does not rate all items.")
    for user_id, item_ratings in reported_ratings.items():
        if len(item_ratings) == 0:
            raise ValueError(f"User {user_id} has no reported ratings.")

    return RatingsDataset(
        root=root_path,
        item_names=item_names,
        user_profiles=actual_profiles,
        actual_ratings=actual_ratings,
        reported_ratings=reported_ratings,
    )


def build_pairwise_preferences(dataset: RatingsDataset) -> List[PreferenceSample]:
    samples: List[PreferenceSample] = []
    for user_id in dataset.user_ids:
        rated_items = dataset.reported_ratings[user_id]
        item_ids = sorted(rated_items)
        for left_index, left_item in enumerate(item_ids):
            left_rating = rated_items[left_item]
            for right_item in item_ids[left_index + 1 :]:
                right_rating = rated_items[right_item]
                if left_rating == right_rating:
                    continue
                if left_rating > right_rating:
                    positive_item, negative_item = left_item, right_item
                    weight = left_rating - right_rating
                else:
                    positive_item, negative_item = right_item, left_item
                    weight = right_rating - left_rating
                samples.append(
                    PreferenceSample(
                        user_id=user_id,
                        positive_item_id=positive_item,
                        negative_item_id=negative_item,
                        weight=float(weight),
                    )
                )
    if not samples:
        raise ValueError("No pairwise preference samples were created.")
    return samples


def build_item_popularity(dataset: RatingsDataset) -> torch.Tensor:
    item_totals = torch.zeros(dataset.num_items, dtype=torch.float32)
    item_counts = torch.zeros(dataset.num_items, dtype=torch.float32)
    for item_ratings in dataset.reported_ratings.values():
        for item_id, rating in item_ratings.items():
            item_totals[item_id] += rating
            item_counts[item_id] += 1.0
    return item_totals / item_counts.clamp_min(1.0)


def iter_profile_users(dataset: RatingsDataset, profile_name: str) -> Iterable[int]:
    for user_id, actual_profile in dataset.user_profiles.items():
        if actual_profile == profile_name:
            yield user_id
