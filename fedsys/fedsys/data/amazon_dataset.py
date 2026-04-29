"""
Amazon Review 2023 — Video Games data loader.

Source: https://amazon-reviews-2023.github.io/
        McAuley Lab, UCSD

Download
--------
The raw JSONL file (~1 GB compressed) is fetched automatically on first use
and cached to ``~/.cache/fedsys/amazon/``.  A lightweight HuggingFace
``datasets`` integration is used as a fallback; if that fails, the file is
streamed directly from the McAuley lab CDN.

Federated partitioning
-----------------------
``partition_index`` and ``num_partitions`` are assigned by the coordinator
after registration.  The dataset is split deterministically by user_id hash
so every partition always sees the same users regardless of download order.

Data format returned by the DataLoader
---------------------------------------
Each batch is a dict with keys:
    user_id  : LongTensor (B,)  — encoded integer user index
    item_id  : LongTensor (B,)  — encoded integer item index
    label    : FloatTensor (B,) — 1.0 if rating ≥ 4, else 0.0
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".cache" / "fedsys" / "amazon"
_HF_DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
_HF_SUBSET = "raw_review_Video_Games"
_CDN_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/review_categories/Video_Games.jsonl.gz"
)

# Caps to keep memory bounded during development/testing
_MAX_REVIEWS = 2_000_000


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class AmazonVideoGamesDataset(Dataset):
    """
    Binary implicit-feedback dataset built from Amazon 2023 Video Games reviews.

    Positive labels : rating >= 4 (1.0)
    Negative labels : rating <  4 (0.0)

    Parameters
    ----------
    partition_index : int  — which shard this node owns (0-based)
    num_partitions  : int  — total number of shards
    max_reviews     : int  — cap on total rows loaded (all partitions)
    seed            : int  — for reproducibility
    cache_dir       : Path — where to cache downloaded data
    """

    def __init__(
        self,
        partition_index: int = 0,
        num_partitions: int = 1,
        max_reviews: int = _MAX_REVIEWS,
        seed: int = 42,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._partition_index = partition_index
        self._num_partitions = num_partitions
        self._cache_dir = cache_dir or _CACHE_DIR

        raw_records = self._load_or_download(max_reviews)
        self._user2idx, self._item2idx = self._build_encoders(raw_records)
        self._records = self._encode_and_partition(raw_records)

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, iid, label = self._records[idx]
        return {
            "user_id": torch.tensor(uid, dtype=torch.long),
            "item_id": torch.tensor(iid, dtype=torch.long),
            "label":   torch.tensor(label, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Vocabulary sizes (pass to ModelConfig)
    # ------------------------------------------------------------------

    @property
    def num_users(self) -> int:
        return len(self._user2idx)

    @property
    def num_items(self) -> int:
        return len(self._item2idx)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_or_download(self, max_reviews: int) -> List[dict]:
        """Return raw records as a list of dicts."""
        cache_file = self._cache_dir / "Video_Games_reviews.jsonl"
        if cache_file.exists():
            return self._read_jsonl(cache_file, max_reviews)

        print("[data] Cache miss — attempting HuggingFace download …")
        try:
            return self._load_from_hf(max_reviews)
        except Exception as hf_err:
            print(f"[data] HuggingFace load failed ({hf_err}); "
                  f"falling back to CDN download …")
            return self._download_from_cdn(cache_file, max_reviews)

    def _load_from_hf(self, max_reviews: int) -> List[dict]:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(
            _HF_DATASET_NAME,
            _HF_SUBSET,
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
        records = []
        for row in ds:
            records.append({
                "user_id": row.get("user_id") or row.get("reviewerID", ""),
                "asin":    row.get("parent_asin") or row.get("asin", ""),
                "rating":  float(row.get("rating") or row.get("overall", 3.0)),
            })
            if len(records) >= max_reviews:
                break
        return records

    def _download_from_cdn(self, dest: Path, max_reviews: int) -> List[dict]:
        import gzip
        import json
        import urllib.request

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        gz_path = self._cache_dir / "Video_Games.jsonl.gz"

        print(f"[data] Downloading from CDN to {gz_path} …")
        urllib.request.urlretrieve(_CDN_URL, gz_path, reporthook=_progress_hook)

        records = []
        with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                records.append({
                    "user_id": row.get("reviewerID", row.get("user_id", "")),
                    "asin":    row.get("asin", row.get("parent_asin", "")),
                    "rating":  float(row.get("overall", row.get("rating", 3.0))),
                })
                if len(records) >= max_reviews:
                    break

        # Persist as flat JSONL for future runs
        import json
        with open(dest, "w", encoding="utf-8") as out:
            for r in records:
                out.write(json.dumps(r) + "\n")

        return records

    @staticmethod
    def _read_jsonl(path: Path, max_reviews: int) -> List[dict]:
        import json
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if len(records) >= max_reviews:
                    break
        print(f"[data] Loaded {len(records):,} reviews from cache.")
        return records

    # ------------------------------------------------------------------
    # Encoding and partitioning
    # ------------------------------------------------------------------

    @staticmethod
    def _build_encoders(
        records: List[dict],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        user2idx: Dict[str, int] = {}
        item2idx: Dict[str, int] = {}
        for r in records:
            uid = r["user_id"]
            iid = r["asin"]
            if uid not in user2idx:
                user2idx[uid] = len(user2idx)
            if iid not in item2idx:
                item2idx[iid] = len(item2idx)
        return user2idx, item2idx

    def _encode_and_partition(
        self, records: List[dict]
    ) -> List[Tuple[int, int, float]]:
        """
        Encode raw strings to integer indices, assign records to partitions
        by hashing the user_id, and return only this node's shard.
        """
        shard: List[Tuple[int, int, float]] = []
        n = self._num_partitions
        p = self._partition_index
        for r in records:
            uid = r["user_id"]
            # Deterministic partition by user hash so each user stays in one shard
            user_hash = int(hashlib.md5(uid.encode()).hexdigest(), 16)
            if (user_hash % n) != p:
                continue
            uid_idx = self._user2idx[uid]
            iid_idx = self._item2idx[r["asin"]]
            label = 1.0 if r["rating"] >= 4.0 else 0.0
            shard.append((uid_idx, iid_idx, label))

        print(
            f"[data] Partition {p}/{n}: {len(shard):,} records "
            f"({len(shard) / max(len(records), 1) * 100:.1f}% of total)"
        )
        return shard


# ---------------------------------------------------------------------------
# Synthetic fallback for quick unit testing without downloading
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    In-memory synthetic dataset for smoke tests and CI.

    Generates random user/item interactions so the full FL pipeline can be
    exercised without network access.
    """

    def __init__(
        self,
        num_users: int = 1_000,
        num_items: int = 500,
        num_samples: int = 50_000,
        seed: int = 0,
    ) -> None:
        rng = random.Random(seed)
        self._data: List[Tuple[int, int, float]] = [
            (
                rng.randint(0, num_users - 1),
                rng.randint(0, num_items - 1),
                float(rng.random() > 0.5),
            )
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, iid, label = self._data[idx]
        return {
            "user_id": torch.tensor(uid, dtype=torch.long),
            "item_id": torch.tensor(iid, dtype=torch.long),
            "label":   torch.tensor(label, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataloader(
    partition_index: int,
    num_partitions: int,
    batch_size: int,
    use_synthetic: bool = False,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    """
    Construct the appropriate dataset and wrap it in a DataLoader.

    ``use_synthetic=True`` is useful for local development where the Amazon
    dataset is not yet downloaded.
    """
    if use_synthetic:
        dataset = SyntheticDataset(**dataset_kwargs)
    else:
        dataset = AmazonVideoGamesDataset(
            partition_index=partition_index,
            num_partitions=num_partitions,
            **dataset_kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / (1024 ** 2)
        print(f"\r[data]  {pct:.1f}%  ({mb:.1f} MB)", end="", flush=True)
