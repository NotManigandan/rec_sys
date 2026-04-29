"""
Synthetic dataset generator for the FedSys two-layer model.

Produces structured implicit-feedback data using a latent-factor model
so that the training signal is non-trivial (users have consistent
preferences; the model can actually learn something).

Data generation process
-----------------------
1. Sample latent preference vectors  U[u] ~ N(0, I)  for every user.
2. Sample latent feature vectors     V[i] ~ N(0, I)  for every item.
3. For each interaction, draw a random (user, item) pair and compute:
       p = sigmoid(U[u] . V[i] / sqrt(latent_dim))
   Label = Bernoulli(p).

The full pool is shuffled, then split before partitioning:

    Pool (num_samples)
      └─ test  (test_fraction)   → test.csv        (coordinator holds this)
      └─ val   (val_fraction)    → val.csv          (coordinator holds this)
      └─ train (remainder)       → partition_0.csv, partition_1.csv …

This guarantees val/test are unseen by any training node.

Output
------
    data/synthetic/
        meta.json          -- all generation parameters + split sizes
        val.csv            -- (user_id, item_id, label) for validation
        test.csv           -- (user_id, item_id, label) for final eval
        partition_0.csv    -- training shard for node 0
        partition_1.csv    -- training shard for node 1
        …

Usage
-----
    python scripts/generate_synthetic_data.py          # quick defaults
    python scripts/generate_synthetic_data.py \\
        --num-users 2000 --num-items 800 --num-samples 20000 \\
        --num-partitions 3 --val-fraction 0.1 --test-fraction 0.1
"""

import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate structured synthetic FL data with train/val/test splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-users",      type=int,   default=1_000,  dest="num_users")
    p.add_argument("--num-items",      type=int,   default=500,    dest="num_items")
    p.add_argument("--num-samples",    type=int,   default=10_000, dest="num_samples",
                   help="Total interactions before splitting")
    p.add_argument("--num-partitions", type=int,   default=2,      dest="num_partitions")
    p.add_argument("--val-fraction",   type=float, default=0.10,   dest="val_fraction",
                   help="Fraction of total samples held out for validation")
    p.add_argument("--test-fraction",  type=float, default=0.10,   dest="test_fraction",
                   help="Fraction of total samples held out for testing")
    p.add_argument("--latent-dim",     type=int,   default=16,     dest="latent_dim")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--out-dir",        default=DEFAULT_OUT,        dest="out_dir")
    return p.parse_args()


def dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def randn(rng: random.Random, dim: int) -> list:
    """Box-Muller normal samples using only the stdlib random module."""
    out = []
    for _ in range(dim // 2 + 1):
        u1 = rng.random() or 1e-12
        u2 = rng.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        out += [z0, z1]
    return out[:dim]


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["user_id", "item_id", "label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.val_fraction + args.test_fraction >= 1.0:
        print("[gen] ERROR: val_fraction + test_fraction must be < 1.0")
        sys.exit(1)

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    D = args.latent_dim
    scale = 1.0 / math.sqrt(D)

    print(f"[gen] Sampling latent vectors: {args.num_users} users, "
          f"{args.num_items} items, dim={D}")
    user_vecs = [randn(rng, D) for _ in range(args.num_users)]
    item_vecs = [randn(rng, D) for _ in range(args.num_items)]

    # ── Generate the full interaction pool ───────────────────────────────
    print(f"[gen] Generating {args.num_samples:,} interactions ...")
    pool: list[tuple] = []
    pos = neg = 0
    for _ in range(args.num_samples):
        u = rng.randint(0, args.num_users - 1)
        i = rng.randint(0, args.num_items - 1)
        score = dot(user_vecs[u], item_vecs[i]) * scale
        label = 1 if rng.random() < sigmoid(score) else 0
        pos += label
        neg += 1 - label
        pool.append((u, i, label))

    print(f"[gen] Label balance: {pos:,} pos  {neg:,} neg  "
          f"({pos / args.num_samples * 100:.1f}% positive)")

    # ── Shuffle, then carve out val and test before training ─────────────
    rng.shuffle(pool)
    n_test = int(len(pool) * args.test_fraction)
    n_val  = int(len(pool) * args.val_fraction)
    n_train = len(pool) - n_test - n_val

    test_rows  = pool[:n_test]
    val_rows   = pool[n_test : n_test + n_val]
    train_pool = pool[n_test + n_val :]

    print(f"[gen] Split → train={n_train:,}  val={n_val:,}  test={n_test:,}")

    # ── Write val.csv and test.csv ────────────────────────────────────────
    val_path  = os.path.join(args.out_dir, "val.csv")
    test_path = os.path.join(args.out_dir, "test.csv")
    write_csv(val_path,  val_rows)
    write_csv(test_path, test_rows)
    print(f"[gen]   val.csv   -> {len(val_rows):,} rows  ({val_path})")
    print(f"[gen]   test.csv  -> {len(test_rows):,} rows  ({test_path})")

    # ── Partition training pool by user_id % num_partitions ──────────────
    partitions: list[list[tuple]] = [[] for _ in range(args.num_partitions)]
    for row in train_pool:
        partitions[row[0] % args.num_partitions].append(row)

    for p_idx, rows in enumerate(partitions):
        rng.shuffle(rows)
        out_path = os.path.join(args.out_dir, f"partition_{p_idx}.csv")
        write_csv(out_path, rows)
        print(f"[gen]   partition_{p_idx}.csv -> {len(rows):,} rows  ({out_path})")

    # ── Write meta.json ──────────────────────────────────────────────────
    meta = {
        "num_users":      args.num_users,
        "num_items":      args.num_items,
        "num_samples":    args.num_samples,
        "num_partitions": args.num_partitions,
        "latent_dim":     args.latent_dim,
        "seed":           args.seed,
        "val_fraction":   args.val_fraction,
        "test_fraction":  args.test_fraction,
        "n_train":        n_train,
        "n_val":          n_val,
        "n_test":         n_test,
        "val_path":       "val.csv",
        "test_path":      "test.csv",
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[gen] meta.json written to {meta_path}")
    print(f"[gen] Done.")
    print(f"[gen]   coordinator: --val-data  {val_path}")
    print(f"[gen]                --test-data {test_path}")
    print(f"[gen]   nodes:       --data-dir  {args.out_dir}")


if __name__ == "__main__":
    main()
