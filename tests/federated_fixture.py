from __future__ import annotations

import csv
from pathlib import Path


MOVIES = [
    (1, "Action One (2000)", "Action"),
    (2, "Action Two (2000)", "Action"),
    (3, "Comedy One (2000)", "Comedy"),
    (4, "Comedy Two (2000)", "Comedy"),
    (5, "Drama One (2000)", "Drama"),
    (6, "Sci-Fi One (2000)", "Sci-Fi"),
    (7, "Action Three (2000)", "Action"),
    (8, "Comedy Three (2000)", "Comedy"),
    (9, "Action Hidden (2000)", "Action"),
    (10, "Drama Two (2000)", "Drama"),
]

RATINGS = [
    (1, 1, 5.0, 1),
    (1, 2, 4.5, 2),
    (1, 5, 4.0, 3),
    (1, 6, 4.0, 4),
    (2, 1, 5.0, 1),
    (2, 7, 4.5, 2),
    (2, 5, 4.0, 3),
    (2, 10, 4.0, 4),
    (3, 3, 5.0, 1),
    (3, 4, 4.5, 2),
    (3, 9, 5.0, 3),
    (3, 8, 4.0, 4),
    (4, 3, 5.0, 1),
    (4, 8, 4.5, 2),
    (4, 9, 4.5, 3),
    (4, 5, 4.0, 4),
    (5, 5, 5.0, 1),
    (5, 10, 4.5, 2),
    (5, 9, 5.0, 3),
    (5, 6, 4.0, 4),
    (6, 2, 5.0, 1),
    (6, 7, 4.5, 2),
    (6, 6, 4.0, 3),
    (6, 5, 4.0, 4),
]


def write_movielens_fixture(root: Path, variant: str) -> Path:
    if variant not in {"ml-1m", "ml-25m"}:
        raise ValueError(f"Unsupported test fixture variant: {variant}")

    dataset_dir = root / ("ml-1m" if variant == "ml-1m" else "ml-25m")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if variant == "ml-1m":
        with (dataset_dir / "movies.dat").open("w", encoding="latin-1") as handle:
            for movie_id, title, genres in MOVIES:
                handle.write(f"{movie_id}::{title}::{genres}\n")
        with (dataset_dir / "ratings.dat").open("w", encoding="latin-1") as handle:
            for user_id, movie_id, rating, timestamp in RATINGS:
                handle.write(f"{user_id}::{movie_id}::{rating}::{timestamp}\n")
    else:
        with (dataset_dir / "movies.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["movieId", "title", "genres"])
            writer.writerows(MOVIES)
        with (dataset_dir / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["userId", "movieId", "rating", "timestamp"])
            writer.writerows(RATINGS)

    return dataset_dir
