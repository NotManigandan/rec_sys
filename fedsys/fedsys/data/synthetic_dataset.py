"""
Synthetic dataset loader — completely independent of the Amazon dataset.

Reads CSV files produced by ``scripts/generate_synthetic_data.py``.

File layout expected
--------------------
    <data_dir>/
        meta.json           -- dataset metadata
        partition_0.csv     -- user_id, item_id, label
        partition_1.csv
        …

Each CSV row:
    user_id  (int)   -- integer in [0, num_users)
    item_id  (int)   -- integer in [0, num_items)
    label    (float) -- 0.0 or 1.0

The DataLoader produced here returns batches with the same dict schema as
the Amazon dataset:
    {
        "user_id":  LongTensor  (B,)
        "item_id":  LongTensor  (B,)
        "label":    FloatTensor (B,)
    }

This means the trainer, model, and aggregator are completely data-source
agnostic — they only see tensors.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticCSVDataset(Dataset):
    """
    Loads one partition of the pre-generated synthetic dataset from disk.

    Parameters
    ----------
    data_dir        : Directory containing partition_*.csv and meta.json.
    partition_index : Which CSV shard to load (0-based).
    """

    def __init__(self, data_dir: str, partition_index: int = 0) -> None:
        self._data_dir = data_dir
        self._partition_index = partition_index

        # Read metadata
        meta_path = os.path.join(data_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"meta.json not found in {data_dir}. "
                "Run scripts/generate_synthetic_data.py first."
            )
        with open(meta_path) as fh:
            self.meta: dict = json.load(fh)

        self.num_users: int = self.meta["num_users"]
        self.num_items: int = self.meta["num_items"]
        self.num_partitions: int = self.meta["num_partitions"]

        # Load the partition CSV
        csv_path = os.path.join(data_dir, f"partition_{partition_index}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Partition file not found: {csv_path}. "
                f"Available partitions: 0–{self.num_partitions - 1}"
            )

        self._records: List[Tuple[int, int, float]] = []
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._records.append((
                    int(row["user_id"]),
                    int(row["item_id"]),
                    float(row["label"]),
                ))

        print(
            f"[data] Loaded partition {partition_index}/{self.num_partitions}: "
            f"{len(self._records):,} samples from {csv_path}"
        )

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, iid, label = self._records[idx]
        return {
            "user_id": torch.tensor(uid,   dtype=torch.long),
            "item_id": torch.tensor(iid,   dtype=torch.long),
            "label":   torch.tensor(label, dtype=torch.float32),
        }


def load_meta(data_dir: str) -> dict:
    """Read the meta.json produced by generate_synthetic_data.py."""
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"meta.json not found in {data_dir}. "
            "Run: python scripts/generate_synthetic_data.py"
        )
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Plain CSV loader (for val.csv / test.csv held by the coordinator)
# ---------------------------------------------------------------------------

class PlainCSVDataset(Dataset):
    """
    Loads any single CSV file with columns user_id, item_id, label.

    Unlike SyntheticCSVDataset this class does NOT expect a meta.json or
    partition numbering — it simply reads the file directly.  Used by the
    coordinator to load val.csv and test.csv.
    """

    def __init__(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}. "
                "Run scripts/generate_synthetic_data.py first."
            )
        self._records: List[Tuple[int, int, float]] = []
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._records.append((
                    int(row["user_id"]),
                    int(row["item_id"]),
                    float(row["label"]),
                ))
        print(f"[data] Loaded {len(self._records):,} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, iid, label = self._records[idx]
        return {
            "user_id": torch.tensor(uid,   dtype=torch.long),
            "item_id": torch.tensor(iid,   dtype=torch.long),
            "label":   torch.tensor(label, dtype=torch.float32),
        }


def load_csv_dataloader(
    csv_path: str,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Wrap a single CSV file (val.csv or test.csv) in a DataLoader.

    shuffle=False so evaluation order is deterministic.
    drop_last=False so every sample is evaluated exactly once.
    """
    dataset = PlainCSVDataset(csv_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


def build_synthetic_dataloader(
    data_dir: str,
    partition_index: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience factory that wraps SyntheticCSVDataset in a DataLoader.

    Parameters
    ----------
    data_dir        : Directory with partition_*.csv and meta.json.
    partition_index : Shard to load.
    batch_size      : Mini-batch size.
    num_workers     : DataLoader worker processes.
    """
    dataset = SyntheticCSVDataset(data_dir=data_dir, partition_index=partition_index)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
