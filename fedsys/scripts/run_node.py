"""
Training node entry point.

Usage
-----
    python scripts/run_node.py [options]

Examples
--------
    # Node 0 out of 3, using synthetic data (no download required)
    python scripts/run_node.py --node-id node-0 --synthetic

    # Node with real Amazon data, GPU
    python scripts/run_node.py --node-id node-1 --device cuda:0

    # Full custom run
    python scripts/run_node.py \\
        --node-id node-2 \\
        --coordinator localhost:50051 \\
        --device cuda \\
        --local-epochs 5 \\
        --batch-size 512 \\
        --num-rounds 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Auto-generate proto stubs if they are missing
try:
    from fedsys.generated import federated_pb2, federated_pb2_grpc  # noqa
except (ModuleNotFoundError, ImportError):
    print("[node] Proto stubs missing — generating …")
    from scripts.generate_proto import main as generate_proto
    generate_proto()
    from fedsys.generated import federated_pb2, federated_pb2_grpc  # noqa

from fedsys.config import ModelConfig, NodeConfig
from fedsys.data.synthetic_dataset import build_synthetic_dataloader, load_meta
from fedsys.node.client import FederatedNode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedSys Training Node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--node-id",        default=None,              dest="node_id")
    p.add_argument("--coordinator",    default="localhost:50051")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--local-epochs",   type=int,   default=2,     dest="local_epochs")
    p.add_argument("--batch-size",     type=int,   default=64,    dest="batch_size")
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--num-rounds",     type=int,   default=3,     dest="num_rounds")
    p.add_argument("--chunk-size-mb",  type=float, default=4.0,   dest="chunk_size_mb")
    p.add_argument("--log-dir",        default="logs",            dest="log_dir")
    # Data source (mutually exclusive)
    data_grp = p.add_mutually_exclusive_group()
    data_grp.add_argument(
        "--data-dir", default=None, dest="data_dir",
        metavar="DIR",
        help="Load pre-generated synthetic CSV partitions from DIR "
             "(run scripts/generate_synthetic_data.py first).",
    )
    data_grp.add_argument(
        "--movielens", default=None, dest="movielens_root",
        metavar="DIR",
        help="Root directory that contains the MovieLens variant folder. "
             "Use with --ml-variant and --partition.",
    )
    data_grp.add_argument(
        "--amazon", action="store_true",
        help="[legacy] Use the Amazon 2023 Video Games dataset (not recommended).",
    )
    p.add_argument("--ml-variant",   default="ml-1m", dest="ml_variant",
                   choices=["ml-1m", "ml-10m", "ml-25m", "ml-32m"],
                   help="MovieLens variant (only used with --movielens).")
    p.add_argument("--num-partitions", type=int,   default=2,     dest="num_partitions")
    p.add_argument("--partition",      type=int,   default=0)
    # Fallback in-memory synthetic options (used when neither --data-dir nor --amazon)
    p.add_argument("--syn-users",      type=int,   default=1_000, dest="syn_users")
    p.add_argument("--syn-items",      type=int,   default=500,   dest="syn_items")
    p.add_argument("--syn-samples",    type=int,   default=5_000, dest="syn_samples")
    # Model (must match coordinator exactly)
    p.add_argument("--model-type",     default="simple",
                   choices=["simple", "bpr", "neural_cf", "two_tower"],
                   dest="model_type")
    p.add_argument("--num-users",      type=int, default=1_000,   dest="num_users")
    p.add_argument("--num-items",      type=int, default=500,     dest="num_items")
    p.add_argument("--embedding-dim",  type=int, default=32,      dest="embedding_dim",
                   help="Embedding dimension. Must match the coordinator exactly.")
    p.add_argument("--hidden-dim",     type=int, default=64,      dest="hidden_dim")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    host, _, port_str = args.coordinator.partition(":")
    port = int(port_str) if port_str else 50051

    import uuid
    node_id = args.node_id or str(uuid.uuid4())[:8]

    cfg = NodeConfig(
        node_id=node_id,
        coordinator_host=host,
        coordinator_port=port,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        data_partition=args.partition,
        num_partitions=args.num_partitions,
        chunk_size_bytes=int(args.chunk_size_mb * 1024 * 1024),
        log_dir=args.log_dir,
        db_path=os.path.join(args.log_dir, f"telemetry_{node_id}.db"),
        log_file=os.path.join(args.log_dir, f"telemetry_{node_id}.jsonl"),
    )

    if args.model_type == "bpr":
        model_cfg = ModelConfig(
            model_type="bpr",
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.embedding_dim,
        )
    else:
        model_cfg = ModelConfig(
            model_type=args.model_type,
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.embedding_dim,
            mlp_hidden=[args.hidden_dim],
        )

    # ------------------------------------------------------------------ #
    # Pick data source: CSV dir > Amazon > in-memory synthetic (default) #
    # ------------------------------------------------------------------ #
    if args.data_dir:
        # Pre-generated synthetic CSV files — read model dimensions from meta.json
        meta = load_meta(args.data_dir)
        model_cfg.num_users = meta["num_users"]
        model_cfg.num_items = meta["num_items"]
        n_partitions = meta["num_partitions"]
        print(f"[node:{node_id}] dataset=csv({args.data_dir})  "
              f"users={meta['num_users']}  items={meta['num_items']}  "
              f"partitions={n_partitions}  model={model_cfg.model_type}")
        dataloader = build_synthetic_dataloader(
            data_dir=args.data_dir,
            partition_index=args.partition,
            batch_size=args.batch_size,
        )

    elif args.movielens_root:
        # MovieLens BPR training
        from fedsys.data.movielens_dataset import (
            load_movielens_dataset, partition_users,
            build_movielens_train_dataloader,
        )
        print(f"[node:{node_id}] dataset=MovieLens({args.ml_variant})  "
              f"root={args.movielens_root}  partition={args.partition}/{args.num_partitions}")
        ml_ds = load_movielens_dataset(
            args.movielens_root, args.ml_variant, show_progress=True
        )
        # Override model dimensions from the actual dataset;
        # embedding_dim stays as provided by --embedding-dim (default 32)
        model_cfg.num_users  = ml_ds.num_users
        model_cfg.num_items  = ml_ds.num_items
        model_cfg.model_type = "bpr"
        print(f"[node:{node_id}] model_cfg: users={model_cfg.num_users}  "
              f"items={model_cfg.num_items}  emb={model_cfg.embedding_dim}")

        shards, _ = partition_users(ml_ds.num_users, args.num_partitions, seed=42)
        shard_users = shards[args.partition % args.num_partitions]
        dataloader = build_movielens_train_dataloader(
            ml_ds, shard_users, batch_size=args.batch_size,
        )

    elif args.amazon:
        print("[node] --amazon is no longer supported. Use --movielens instead.")
        raise SystemExit(1)

    else:
        # Default: fast in-memory synthetic (no files needed)
        from fedsys.data.synthetic_dataset import SyntheticCSVDataset
        from torch.utils.data import DataLoader as _DL
        import torch as _t
        import random as _r

        class _InMemSynth(_t.utils.data.Dataset):
            def __init__(self, nu, ni, ns, seed=42):
                rng = _r.Random(seed)
                self._data = [
                    (rng.randrange(nu), rng.randrange(ni), float(rng.randint(0, 1)))
                    for _ in range(ns)
                ]
            def __len__(self): return len(self._data)
            def __getitem__(self, i):
                u, it, lb = self._data[i]
                return {
                    "user_id": _t.tensor(u,  dtype=_t.long),
                    "item_id": _t.tensor(it, dtype=_t.long),
                    "label":   _t.tensor(lb, dtype=_t.float32),
                }

        print(f"[node:{node_id}] dataset=in-memory-synthetic  "
              f"users={args.syn_users}  items={args.syn_items}  "
              f"samples={args.syn_samples}")
        dataloader = _DL(
            _InMemSynth(args.syn_users, args.syn_items, args.syn_samples),
            batch_size=args.batch_size, shuffle=True,
        )

    print(f"[node:{node_id}] Connecting to {cfg.coordinator_address} ...")
    FederatedNode(cfg, model_cfg, dataloader, num_rounds=args.num_rounds).run()


if __name__ == "__main__":
    main()
