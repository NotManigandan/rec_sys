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
from datetime import datetime

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

ACTIVE_RUN_FILE = os.path.join("logs", ".active_run_dir")


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
    p.add_argument("--log-dir",        default="",                dest="log_dir",
                   help="Telemetry output directory. If omitted, uses "
                        "logs/.active_run_dir from coordinator when available, "
                        "otherwise logs/run_<YYYYMMDD_HHMMSS>/")
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

    # ── Attack arguments ───────────────────────────────────────────────────
    p.add_argument("--attack", action="store_true",
                   help="Enable data-poisoning attack (malicious node mode).")
    p.add_argument("--attack-target-item", type=int, default=-1,
                   dest="attack_target_item",
                   help="Contiguous item index to push. -1 = auto select via "
                        "select_target_item().")
    p.add_argument("--attack-target-genre", default="", dest="attack_target_genre",
                   help="Genre of the target item (required when --attack is set).")
    p.add_argument("--attack-budget",  type=float, default=0.30, dest="attack_budget",
                   help="Fraction of shard users to add as synthetic profiles.")
    p.add_argument("--attack-num-filler",   type=int, default=30, dest="attack_num_filler")
    p.add_argument("--attack-num-neutral",  type=int, default=20, dest="attack_num_neutral")
    p.add_argument("--attack-neutral-genre", default="Comedy",   dest="attack_neutral_genre")
    p.add_argument("--attack-target-weight", type=float, default=1.0, dest="attack_target_weight")
    p.add_argument("--attack-max-synth",     type=int,   default=0,  dest="attack_max_synth",
                   help="Reserve this many extra user-embedding rows for synthetic profiles. "
                        "MUST be passed on EVERY node (clean and malicious) and on the "
                        "coordinator with the same value whenever any node uses --attack, "
                        "to keep embedding-table shapes consistent. Clean nodes just reserve "
                        "the rows without ever training on them.")
    p.add_argument("--attack-seed",          type=int,   default=42,  dest="attack_seed")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log_dir:
        if os.path.exists(ACTIVE_RUN_FILE):
            with open(ACTIVE_RUN_FILE, "r", encoding="utf-8") as fh:
                candidate = fh.read().strip()
            if candidate:
                args.log_dir = candidate
        if not args.log_dir:
            args.log_dir = os.path.join("logs", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

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
        # MovieLens training
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
        # embedding_dim stays as provided by --embedding-dim (default 32).
        model_cfg.num_users  = ml_ds.num_users
        model_cfg.num_items  = ml_ds.num_items
        # Keep the caller-selected model type (bpr/neural_cf/two_tower/simple).
        model_cfg.model_type = args.model_type

        # ── Synthetic-user slot reservation ───────────────────────────────
        # When any node in the experiment runs --attack, every node must carry
        # the same extended embedding table so state-dict shapes stay consistent.
        # Pass --attack-max-synth on all nodes (clean + malicious) with the same
        # value used on the coordinator.  Clean nodes only reserve the rows;
        # the malicious node's PoisonedBPRPairDataset actually fills them.
        if args.attack_max_synth > 0 and not getattr(args, "attack", False):
            model_cfg.num_users += args.attack_max_synth
            print(f"[node:{node_id}] synthetic-user reserve: +{args.attack_max_synth} "
                  f"(num_users={model_cfg.num_users})")
        print(f"[node:{node_id}] model_cfg: users={model_cfg.num_users}  "
              f"items={model_cfg.num_items}  emb={model_cfg.embedding_dim}")

        shards, _ = partition_users(ml_ds.num_users, args.num_partitions, seed=42)
        shard_users = shards[args.partition % args.num_partitions]

        if getattr(args, "attack", False):
            # ── Malicious node: build poisoned dataloader ──────────────
            from fedsys.adversarial.attack.poison import (
                AttackConfig, build_poisoned_dataloader, poisoned_num_users,
            )
            from fedsys.adversarial.attack.target import (
                select_target_item, select_target_from_clean_model,
            )
            attack_cfg = AttackConfig(
                enabled=True,
                target_item_index=args.attack_target_item,
                target_genre=args.attack_target_genre,
                attack_budget=args.attack_budget,
                num_filler_items=args.attack_num_filler,
                num_neutral_items=args.attack_num_neutral,
                neutral_genre=args.attack_neutral_genre,
                target_weight=args.attack_target_weight,
                max_synthetic_users_per_coord=args.attack_max_synth,
                seed=args.attack_seed,
            )
            if attack_cfg.target_item_index < 0 and attack_cfg.target_genre:
                attack_cfg.target_item_index = select_target_item(
                    ml_ds, attack_cfg.target_genre
                )
                print(f"[node:{node_id}] auto-selected target item: "
                      f"{attack_cfg.target_item_index}")
            elif attack_cfg.target_item_index < 0:
                raise ValueError(
                    "Provide --attack-target-item or --attack-target-genre "
                    "when --attack is enabled."
                )

            # Expand num_users to include synthetic user slots
            model_cfg.num_users = poisoned_num_users(ml_ds.num_users, attack_cfg)
            print(
                f"[node:{node_id}] ATTACK mode  target={attack_cfg.target_item_index}  "
                f"genre={attack_cfg.target_genre!r}  "
                f"budget={attack_cfg.attack_budget}  "
                f"num_users extended to {model_cfg.num_users}"
            )
            dataloader = build_poisoned_dataloader(
                dataset=ml_ds,
                shard_user_indices=shard_users,
                attack_cfg=attack_cfg,
                model_num_users=model_cfg.num_users,
                batch_size=args.batch_size,
            )
        else:
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
