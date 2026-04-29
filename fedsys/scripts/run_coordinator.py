"""
Coordinator (server) entry point.

Usage
-----
    python scripts/run_coordinator.py [options]

Examples
--------
    # 3-node experiment, wait for all 3 before aggregating
    python scripts/run_coordinator.py --total-nodes 3 --min-nodes 3 --num-rounds 20

    # 5-node experiment, aggregate when 4 have reported (1 straggler tolerance)
    python scripts/run_coordinator.py --total-nodes 5 --min-nodes 4 --round-timeout 60
"""

import argparse
import os
import sys

# Ensure the repo root is on sys.path regardless of where the script is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Auto-generate proto stubs if they are missing
from scripts.generate_proto import main as generate_proto
from fedsys.generated import federated_pb2  # noqa: F401 – triggers auto-gen check
try:
    from fedsys.generated import federated_pb2_grpc  # noqa: F401
except ModuleNotFoundError:
    print("[coordinator] Proto stubs missing — generating …")
    generate_proto()

from fedsys.config import CoordinatorConfig, ModelConfig
from fedsys.coordinator.server import serve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedSys Coordinator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",           default="0.0.0.0")
    p.add_argument("--port",           type=int,   default=50051)
    p.add_argument("--total-nodes",    type=int,   default=2,   dest="total_nodes")
    p.add_argument("--min-nodes",      type=int,   default=2,   dest="min_nodes")
    p.add_argument("--num-rounds",     type=int,   default=3,   dest="num_rounds")
    p.add_argument("--round-timeout",  type=float, default=60,  dest="round_timeout")
    p.add_argument("--chunk-size-mb",  type=float, default=4.0, dest="chunk_size_mb")
    p.add_argument("--log-dir",        default="logs",          dest="log_dir")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   dest="checkpoint_dir",
                   help="Directory to save model checkpoints (set to '' to disable)")
    p.add_argument("--val-data",  default="", dest="val_data_path",
                   metavar="VAL_CSV",
                   help="Path to val.csv (from generate_synthetic_data.py). "
                        "Used to track best model per round.")
    p.add_argument("--test-data", default="", dest="test_data_path",
                   metavar="TEST_CSV",
                   help="Path to test.csv. Evaluated once after all rounds complete.")
    # Model type
    p.add_argument("--model-type",     default="simple", choices=["simple", "ncf"],
                   dest="model_type",
                   help="'simple'=two-layer test model, 'ncf'=full ~300M param model")
    # Model size knobs (only relevant for --model-type ncf)
    p.add_argument("--num-users",      type=int, default=1_000,   dest="num_users")
    p.add_argument("--num-items",      type=int, default=500,     dest="num_items")
    p.add_argument("--embedding-dim",  type=int, default=16,      dest="embedding_dim")
    p.add_argument("--hidden-dim",     type=int, default=64,      dest="hidden_dim")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CoordinatorConfig(
        host=args.host,
        port=args.port,
        total_nodes=args.total_nodes,
        min_nodes=args.min_nodes,
        num_rounds=args.num_rounds,
        round_timeout_seconds=args.round_timeout,
        chunk_size_bytes=int(args.chunk_size_mb * 1024 * 1024),
        checkpoint_dir=args.checkpoint_dir,
        val_data_path=args.val_data_path,
        test_data_path=args.test_data_path,
        log_dir=args.log_dir,
        db_path=os.path.join(args.log_dir, "telemetry.db"),
        log_file=os.path.join(args.log_dir, "telemetry.jsonl"),
    )

    if args.model_type == "ncf":
        from fedsys.config import large_model_config
        model_cfg = large_model_config()
    else:
        model_cfg = ModelConfig(
            model_type="simple",
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.embedding_dim,
            mlp_hidden=[args.hidden_dim],
        )

    print(f"[coordinator] model={model_cfg.model_type}  "
          f"users={model_cfg.num_users}  items={model_cfg.num_items}  "
          f"emb={model_cfg.embedding_dim}  hidden={model_cfg.mlp_hidden}")
    print(f"[coordinator] Starting  host={cfg.host}:{cfg.port}  "
          f"N={cfg.total_nodes}  K={cfg.min_nodes}  rounds={cfg.num_rounds}")
    serve(cfg, model_cfg)


if __name__ == "__main__":
    main()
