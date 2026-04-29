from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    tqdm = None


MOVIELENS_VARIANTS = {
    "ml-1m": {"folder": "ml-1m", "format": "dat"},
    "ml-10m": {"folder": "ml-10M100K", "format": "dat"},
    "ml-25m": {"folder": "ml-25m", "format": "csv"},
    "ml-32m": {"folder": "ml-32m", "format": "csv"},
}


@dataclass(frozen=True)
class MovieMetadata:
    movie_id: int
    title: str
    genres: tuple[str, ...]


@dataclass(frozen=True)
class UserSplit:
    train_ratings: tuple[tuple[int, float], ...]
    val_item: int
    test_item: int
    known_items: frozenset[int]


@dataclass(frozen=True)
class MovieLensFederatedDataset:
    variant: str
    root: Path
    item_ids: tuple[int, ...]
    item_id_to_index: Dict[int, int]
    user_ids: tuple[int, ...]
    user_id_to_index: Dict[int, int]
    movie_metadata: tuple[MovieMetadata, ...]
    splits_by_user: Dict[int, UserSplit]
    dominant_genre_by_user: Dict[int, str]
    users_by_genre: Dict[str, tuple[int, ...]]
    items_by_genre: Dict[str, tuple[int, ...]]
    train_item_popularity: Dict[int, int]

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    def title_for_item(self, item_index: int) -> str:
        return self.movie_metadata[item_index].title

    def movie_id_for_item(self, item_index: int) -> int:
        return self.movie_metadata[item_index].movie_id

    def genres_for_item(self, item_index: int) -> tuple[str, ...]:
        return self.movie_metadata[item_index].genres


class StageProgress:
    def __init__(
        self,
        desc: str,
        enabled: bool,
        *,
        total: int | None = None,
        unit: str = "it",
        unit_scale: bool = False,
        min_interval: float = 2.0,
    ) -> None:
        self.desc = desc
        self.enabled = enabled
        self.total = total
        self.unit = unit
        self.unit_scale = unit_scale
        self.min_interval = min_interval
        self.current = 0
        self.start_time = time.perf_counter()
        self._last_print = self.start_time
        self._bar = None
        if enabled:
            print(f"{desc}...", flush=True)
        if enabled and tqdm is not None:
            self._bar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                unit_scale=unit_scale,
                dynamic_ncols=True,
            )

    def update(self, amount: int = 1) -> None:
        if amount <= 0:
            return
        self.current += amount
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.update(amount)
            return
        now = time.perf_counter()
        if now - self._last_print < self.min_interval:
            return
        elapsed = now - self.start_time
        if self.total:
            percent = 100.0 * self.current / self.total
            print(
                f"{self.desc}: {percent:5.1f}% ({self.current}/{self.total} {self.unit}) elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(f"{self.desc}: {self.current} {self.unit} elapsed={elapsed:.1f}s", flush=True)
        self._last_print = now

    def close(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        if self._bar is not None:
            self._bar.close()
        elif self.enabled:
            print(f"{self.desc}: done in {elapsed:.1f}s", flush=True)


def _resolve_variant_root(data_root: Path, variant: str) -> tuple[Path, str]:
    if variant not in MOVIELENS_VARIANTS:
        raise ValueError(f"Unsupported MovieLens variant: {variant}")
    spec = MOVIELENS_VARIANTS[variant]
    candidate_root = data_root / spec["folder"]
    root = candidate_root if candidate_root.exists() else data_root
    return root, spec["format"]


def _read_dat_movies(path: Path, show_progress: bool = False) -> Dict[int, MovieMetadata]:
    movies: Dict[int, MovieMetadata] = {}
    progress = StageProgress(
        desc=f"reading {path.name}",
        enabled=show_progress,
        total=path.stat().st_size,
        unit="B",
        unit_scale=True,
    )
    with path.open(encoding="latin-1") as handle:
        try:
            for line in handle:
                movie_id_str, title, genres_str = line.rstrip("\n").split("::")
                genres = tuple(genre for genre in genres_str.split("|") if genre)
                movies[int(movie_id_str)] = MovieMetadata(int(movie_id_str), title, genres)
                progress.update(len(line))
        finally:
            progress.close()
    return movies


def _read_csv_movies(path: Path, show_progress: bool = False) -> Dict[int, MovieMetadata]:
    movies: Dict[int, MovieMetadata] = {}
    progress = StageProgress(desc=f"reading {path.name}", enabled=show_progress, unit="rows")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        try:
            for row in reader:
                movie_id = int(row["movieId"])
                genres = tuple(genre for genre in row["genres"].split("|") if genre)
                movies[movie_id] = MovieMetadata(movie_id, row["title"], genres)
                progress.update(1)
        finally:
            progress.close()
    return movies


def _read_dat_ratings(path: Path, show_progress: bool = False) -> Dict[int, List[tuple[int, float, int]]]:
    ratings_by_user: Dict[int, List[tuple[int, float, int]]] = {}
    progress = StageProgress(
        desc=f"reading {path.name}",
        enabled=show_progress,
        total=path.stat().st_size,
        unit="B",
        unit_scale=True,
    )
    with path.open(encoding="latin-1") as handle:
        try:
            for line in handle:
                user_id_str, movie_id_str, rating_str, timestamp_str = line.rstrip("\n").split("::")
                ratings_by_user.setdefault(int(user_id_str), []).append(
                    (int(movie_id_str), float(rating_str), int(timestamp_str))
                )
                progress.update(len(line))
        finally:
            progress.close()
    return ratings_by_user


def _read_csv_ratings(path: Path, show_progress: bool = False) -> Dict[int, List[tuple[int, float, int]]]:
    ratings_by_user: Dict[int, List[tuple[int, float, int]]] = {}
    progress = StageProgress(
        desc=f"reading {path.name}",
        enabled=show_progress,
        total=path.stat().st_size,
        unit="B",
        unit_scale=True,
    )
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        progress.update(len(header))
        try:
            for line in handle:
                user_id_str, movie_id_str, rating_str, timestamp_str = line.rstrip("\n").split(",")
                ratings_by_user.setdefault(int(user_id_str), []).append(
                    (int(movie_id_str), float(rating_str), int(timestamp_str))
                )
                progress.update(len(line))
        finally:
            progress.close()
    return ratings_by_user


def _load_raw_movielens(
    root: Path,
    variant_format: str,
    show_progress: bool = False,
) -> tuple[Dict[int, MovieMetadata], Dict[int, List[tuple[int, float, int]]]]:
    if variant_format == "dat":
        movies = _read_dat_movies(root / "movies.dat", show_progress=show_progress)
        ratings = _read_dat_ratings(root / "ratings.dat", show_progress=show_progress)
    elif variant_format == "csv":
        movies = _read_csv_movies(root / "movies.csv", show_progress=show_progress)
        ratings = _read_csv_ratings(root / "ratings.csv", show_progress=show_progress)
    else:
        raise ValueError(f"Unsupported MovieLens format: {variant_format}")
    return movies, ratings


def _dominant_genre(
    train_ratings: Sequence[tuple[int, float]],
    movie_metadata_by_id: Dict[int, MovieMetadata],
) -> str:
    genre_scores: Dict[str, float] = {}
    for movie_id, rating in train_ratings:
        metadata = movie_metadata_by_id[movie_id]
        for genre in metadata.genres or ("(no genres listed)",):
            genre_scores[genre] = genre_scores.get(genre, 0.0) + rating
    if not genre_scores:
        return "(unknown)"
    return sorted(genre_scores.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


def load_movielens_dataset(
    data_root: Path | str,
    variant: str,
    min_positive_rating: float = 4.0,
    min_positive_interactions: int = 3,
    max_positive_interactions: int | None = None,
    show_progress: bool = False,
) -> MovieLensFederatedDataset:
    load_start = time.perf_counter()
    root_path, variant_format = _resolve_variant_root(Path(data_root), variant)
    movie_metadata_by_id, ratings_by_user = _load_raw_movielens(
        root_path,
        variant_format,
        show_progress=show_progress,
    )

    filtered_users: Dict[int, UserSplit] = {}
    retained_item_ids: set[int] = set()
    dominant_genre_by_user_id: Dict[int, str] = {}
    train_item_popularity_by_movie_id: Dict[int, int] = {}

    filter_progress = StageProgress(
        desc="filtering and splitting users",
        enabled=show_progress,
        total=len(ratings_by_user),
        unit="users",
    )
    try:
        for user_id, ratings in ratings_by_user.items():
            positive_ratings = [
                (movie_id, rating, timestamp)
                for movie_id, rating, timestamp in ratings
                if rating >= min_positive_rating and movie_id in movie_metadata_by_id
            ]
            if len(positive_ratings) < min_positive_interactions:
                filter_progress.update(1)
                continue
            if max_positive_interactions is not None and len(positive_ratings) > max_positive_interactions:
                filter_progress.update(1)
                continue

            positive_ratings.sort(key=lambda entry: (entry[2], entry[0]))
            train_entries = positive_ratings[:-2]
            val_entry = positive_ratings[-2]
            test_entry = positive_ratings[-1]
            if len(train_entries) == 0:
                filter_progress.update(1)
                continue

            train_ratings = tuple((movie_id, rating) for movie_id, rating, _ in train_entries)
            known_items = frozenset(movie_id for movie_id, _, _ in positive_ratings)
            filtered_users[user_id] = UserSplit(
                train_ratings=train_ratings,
                val_item=val_entry[0],
                test_item=test_entry[0],
                known_items=known_items,
            )

            dominant_genre_by_user_id[user_id] = _dominant_genre(train_ratings, movie_metadata_by_id)
            for movie_id, _ in train_ratings:
                retained_item_ids.add(movie_id)
                train_item_popularity_by_movie_id[movie_id] = train_item_popularity_by_movie_id.get(movie_id, 0) + 1
            retained_item_ids.add(val_entry[0])
            retained_item_ids.add(test_entry[0])
            filter_progress.update(1)
    finally:
        filter_progress.close()

    if not filtered_users:
        raise ValueError("No MovieLens users remained after positive-rating filtering and splitting.")

    item_ids = tuple(sorted(retained_item_ids))
    item_id_to_index = {movie_id: index for index, movie_id in enumerate(item_ids)}
    user_ids = tuple(sorted(filtered_users))
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    movie_metadata = tuple(movie_metadata_by_id[movie_id] for movie_id in item_ids)

    splits_by_user: Dict[int, UserSplit] = {}
    train_item_popularity: Dict[int, int] = {}
    users_by_genre_lists: Dict[str, List[int]] = {}
    items_by_genre_sets: Dict[str, set[int]] = {}
    dominant_genre_by_user: Dict[int, str] = {}

    for user_id in user_ids:
        split = filtered_users[user_id]
        mapped_train = tuple((item_id_to_index[movie_id], rating) for movie_id, rating in split.train_ratings)
        mapped_val = item_id_to_index[split.val_item]
        mapped_test = item_id_to_index[split.test_item]
        mapped_known = frozenset(item_id_to_index[movie_id] for movie_id in split.known_items)
        splits_by_user[user_id_to_index[user_id]] = UserSplit(
            train_ratings=mapped_train,
            val_item=mapped_val,
            test_item=mapped_test,
            known_items=mapped_known,
        )
        dominant_genre = dominant_genre_by_user_id[user_id]
        dominant_genre_by_user[user_id_to_index[user_id]] = dominant_genre
        users_by_genre_lists.setdefault(dominant_genre, []).append(user_id_to_index[user_id])
        for item_index, _ in mapped_train:
            train_item_popularity[item_index] = train_item_popularity.get(item_index, 0) + 1

    for item_index, metadata in enumerate(movie_metadata):
        item_genres = metadata.genres or ("(no genres listed)",)
        for genre in item_genres:
            items_by_genre_sets.setdefault(genre, set()).add(item_index)

    users_by_genre = {genre: tuple(sorted(users)) for genre, users in users_by_genre_lists.items()}
    items_by_genre = {genre: tuple(sorted(items)) for genre, items in items_by_genre_sets.items()}

    dataset = MovieLensFederatedDataset(
        variant=variant,
        root=root_path,
        item_ids=item_ids,
        item_id_to_index=item_id_to_index,
        user_ids=user_ids,
        user_id_to_index=user_id_to_index,
        movie_metadata=movie_metadata,
        splits_by_user=splits_by_user,
        dominant_genre_by_user=dominant_genre_by_user,
        users_by_genre=users_by_genre,
        items_by_genre=items_by_genre,
        train_item_popularity=train_item_popularity,
    )
    if show_progress:
        elapsed = time.perf_counter() - load_start
        print(
            f"prepared dataset in {elapsed:.1f}s: {dataset.num_users} users, {dataset.num_items} items, "
            f"{len(users_by_genre)} dominant-genre segments"
            + (
                ""
                if max_positive_interactions is None
                else f", max_positive_interactions={max_positive_interactions}"
            ),
            flush=True,
        )
    return dataset


def partition_users(num_users: int, num_shards: int, seed: int) -> tuple[Dict[int, tuple[int, ...]], Dict[int, int]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    ordered_users = list(range(num_users))
    # Deterministic shuffling without importing random keeps fixture expectations stable.
    for index in range(len(ordered_users) - 1, 0, -1):
        swap_index = (seed * 1103515245 + 12345 + index) % (index + 1)
        ordered_users[index], ordered_users[swap_index] = ordered_users[swap_index], ordered_users[index]
    shards: Dict[int, List[int]] = {shard_id: [] for shard_id in range(num_shards)}
    user_to_shard: Dict[int, int] = {}
    for position, user_index in enumerate(ordered_users):
        shard_id = position % num_shards
        shards[shard_id].append(user_index)
        user_to_shard[user_index] = shard_id
    return {shard_id: tuple(users) for shard_id, users in shards.items()}, user_to_shard


def benign_segment_users(
    dataset: MovieLensFederatedDataset,
    target_genre: str,
    user_to_shard: Dict[int, int],
    malicious_shard_id: int,
) -> tuple[int, ...]:
    segment_users = dataset.users_by_genre.get(target_genre, ())
    return tuple(
        user_index
        for user_index in segment_users
        if user_to_shard.get(user_index) != malicious_shard_id
    )
