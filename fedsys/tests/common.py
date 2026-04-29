from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data", "synthetic")
META_PATH = os.path.join(DATA_DIR, "meta.json")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    ("grpc.keepalive_time_ms", 120_000),
    ("grpc.keepalive_timeout_ms", 20_000),
    ("grpc.keepalive_permit_without_calls", False),
]


def ensure_synthetic_data() -> dict:
    if not (os.path.exists(VAL_CSV) and os.path.exists(TEST_CSV)):
        subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "scripts", "generate_synthetic_data.py"),
                "--num-users",
                "1000",
                "--num-items",
                "500",
                "--num-samples",
                "10000",
                "--num-partitions",
                "2",
                "--val-fraction",
                "0.10",
                "--test-fraction",
                "0.10",
            ],
            check=True,
        )
    with open(META_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def run_synthetic_integration(model_type: str, port: int) -> None:
    meta = ensure_synthetic_data()
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)

    from fedsys.config import CoordinatorConfig, ModelConfig, NodeConfig
    from fedsys.coordinator.server import serve
    from fedsys.data.synthetic_dataset import build_synthetic_dataloader
    from fedsys.node.client import FederatedNode

    model_cfg = ModelConfig(
        model_type=model_type,
        num_users=meta["num_users"],
        num_items=meta["num_items"],
        embedding_dim=16,
        mlp_hidden=[64],
    )
    coord_cfg = CoordinatorConfig(
        host="127.0.0.1",
        port=port,
        total_nodes=2,
        min_nodes=2,
        num_rounds=3,
        round_timeout_seconds=30,
        checkpoint_dir="checkpoints",
        val_data_path=VAL_CSV,
        test_data_path=TEST_CSV,
        log_dir="logs",
        db_path=f"logs/{model_type}_synthetic_coord.db",
        log_file=f"logs/{model_type}_synthetic_coord.jsonl",
        grpc_options=TEST_GRPC_OPTIONS,
    )
    threading.Thread(target=serve, args=(coord_cfg, model_cfg), daemon=True).start()
    time.sleep(0.8)

    errors: list[str] = []

    def run_node(node_id: str, partition: int) -> None:
        try:
            cfg = NodeConfig(
                node_id=node_id,
                coordinator_host="127.0.0.1",
                coordinator_port=port,
                device="cpu",
                local_epochs=2,
                batch_size=64,
                log_dir="logs",
                db_path=f"logs/{node_id}.db",
                log_file=f"logs/{node_id}.jsonl",
                grpc_options=TEST_GRPC_OPTIONS,
            )
            dl = build_synthetic_dataloader(
                data_dir=DATA_DIR, partition_index=partition, batch_size=64
            )
            FederatedNode(cfg, model_cfg, dl, num_rounds=3).run()
        except Exception as exc:
            errors.append(f"{node_id}: {exc}")

    t0 = threading.Thread(target=run_node, args=(f"{model_type}-syn-node-0", 0))
    t1 = threading.Thread(target=run_node, args=(f"{model_type}-syn-node-1", 1))
    t0.start()
    t1.start()
    t0.join(timeout=180)
    t1.join(timeout=180)
    if errors:
        raise RuntimeError(" | ".join(errors))
    print(f"PASSED: synthetic | {model_type} | 1 coordinator + 2 nodes")


def run_movielens_integration(model_type: str, port: int, ml_data_root: str = "data/", ml_variant: str = "ml-1m") -> None:
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)

    from fedsys.config import CoordinatorConfig, ModelConfig, NodeConfig
    from fedsys.coordinator.server import serve
    from fedsys.data.movielens_dataset import (
        build_movielens_train_dataloader,
        load_movielens_dataset,
        partition_users,
    )
    from fedsys.node.client import FederatedNode

    ml_ds = load_movielens_dataset(ml_data_root, ml_variant, show_progress=True)
    shards, _ = partition_users(ml_ds.num_users, 2, seed=42)

    model_cfg = ModelConfig(
        model_type=model_type,
        num_users=ml_ds.num_users,
        num_items=ml_ds.num_items,
        embedding_dim=32,
        mlp_hidden=[64],
    )
    coord_cfg = CoordinatorConfig(
        host="127.0.0.1",
        port=port,
        total_nodes=2,
        min_nodes=2,
        num_rounds=3,
        round_timeout_seconds=120,
        checkpoint_dir="checkpoints",
        ml_data_root=ml_data_root,
        ml_variant=ml_variant,
        log_dir="logs",
        db_path=f"logs/{model_type}_movielens_coord.db",
        log_file=f"logs/{model_type}_movielens_coord.jsonl",
        grpc_options=TEST_GRPC_OPTIONS,
    )
    threading.Thread(target=serve, args=(coord_cfg, model_cfg), daemon=True).start()
    time.sleep(1.5)

    errors: list[str] = []

    def run_node(node_id: str, shard_idx: int) -> None:
        try:
            cfg = NodeConfig(
                node_id=node_id,
                coordinator_host="127.0.0.1",
                coordinator_port=port,
                device="cpu",
                local_epochs=1,
                batch_size=512,
                log_dir="logs",
                db_path=f"logs/{node_id}.db",
                log_file=f"logs/{node_id}.jsonl",
                grpc_options=TEST_GRPC_OPTIONS,
            )
            dl = build_movielens_train_dataloader(ml_ds, shards[shard_idx], batch_size=512)
            FederatedNode(cfg, model_cfg, dl, num_rounds=3).run()
        except Exception as exc:
            errors.append(f"{node_id}: {exc}")

    t0 = threading.Thread(target=run_node, args=(f"{model_type}-ml-node-0", 0))
    t1 = threading.Thread(target=run_node, args=(f"{model_type}-ml-node-1", 1))
    t0.start()
    t1.start()
    t0.join(timeout=360)
    t1.join(timeout=360)
    if errors:
        raise RuntimeError(" | ".join(errors))
    print(f"PASSED: movielens | {model_type} | 1 coordinator + 2 nodes")
