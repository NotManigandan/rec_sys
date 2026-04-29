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
import subprocess
import sys
from datetime import datetime

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

ACTIVE_RUN_FILE = os.path.join("logs", ".active_run_dir")


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
    p.add_argument("--log-dir",        default="",              dest="log_dir",
                   help="Telemetry output directory. If omitted, uses "
                        "logs/run_<YYYYMMDD_HHMMSS>/ and marks it active for nodes.")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   dest="checkpoint_dir",
                   help="Directory to save model checkpoints (set to '' to disable)")
    p.add_argument("--auto-analyze", action="store_true", default=True, dest="auto_analyze",
                   help="Run analysis/analyze_logs.py automatically after coordinator exits.")
    p.add_argument("--no-auto-analyze", action="store_false", dest="auto_analyze",
                   help="Disable automatic log analysis after run completion.")
    p.add_argument("--val-data",  default="", dest="val_data_path",
                   metavar="VAL_CSV",
                   help="Path to val.csv (synthetic data). "
                        "Used to track best model per round.")
    p.add_argument("--test-data", default="", dest="test_data_path",
                   metavar="TEST_CSV",
                   help="Path to test.csv (synthetic data). Evaluated once after all rounds.")
    # MovieLens ranking evaluation
    p.add_argument("--ml-data-root", default="", dest="ml_data_root",
                   metavar="DIR",
                   help="Root directory containing the MovieLens variant folder "
                        "(e.g. data/ containing data/ml-1m/). "
                        "When set, enables BPR ranking evaluation (Hit@K, NDCG@K) "
                        "and overrides --val-data / --test-data.")
    p.add_argument("--ml-variant",   default="ml-1m", dest="ml_variant",
                   choices=["ml-1m", "ml-10m", "ml-25m", "ml-32m"],
                   help="MovieLens dataset variant.")

    # ── Defense arguments ──────────────────────────────────────────────────
    p.add_argument("--defense",
                   default="none",
                   choices=["none", "clip_mean", "clip_trimmed_mean",
                            "focus_clip_mean", "focus_clip_trimmed_mean"],
                   dest="defense_method",
                   help="Robust aggregation method. 'none'=plain FedAvg.")
    p.add_argument("--defense-clip-thresh", type=float, default=5.0,
                   dest="defense_clip_thresh",
                   help="L2 norm clip threshold (Theta) for defense methods.")
    p.add_argument("--defense-trim-frac",   type=float, default=0.10,
                   dest="defense_trim_frac",
                   help="Trimmed-mean fraction removed from each tail.")
    p.add_argument("--defense-focus-k-frac", type=float, default=0.05,
                   dest="defense_focus_k_frac",
                   help="Top-item fraction used to compute focus score.")

    # ── Adversarial evaluation arguments ──────────────────────────────────
    p.add_argument("--adv-target-item", type=int, default=-1,
                   dest="adv_target_item",
                   help="Contiguous item index to track as attack target. "
                        "-1 disables adversarial evaluation.")
    p.add_argument("--adv-target-genre", default="", dest="adv_target_genre",
                   help="Genre of the attack target item.")
    p.add_argument("--attack-max-synth", type=int, default=0,
                   dest="attack_max_synth",
                   help="Reserve this many synthetic user rows in the global "
                        "embedding table (needed when any node runs --attack).")

    # Model type
    p.add_argument("--model-type",     default="simple",
                   choices=["simple", "bpr", "neural_cf", "two_tower"],
                   dest="model_type",
                   help="'simple'=two-layer baseline, 'bpr'=BPR-MF, "
                        "'neural_cf'=MLP recommender, 'two_tower'=dual-tower scorer")
    # Model size knobs (only relevant for --model-type ncf)
    p.add_argument("--num-users",      type=int, default=1_000,   dest="num_users")
    p.add_argument("--num-items",      type=int, default=500,     dest="num_items")
    p.add_argument("--embedding-dim",  type=int, default=32,      dest="embedding_dim",
                   help="Embedding dimension. Default 32 works well for BPR; "
                        "use 16 for the simple model.")
    p.add_argument("--hidden-dim",     type=int, default=64,      dest="hidden_dim")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log_dir:
        args.log_dir = os.path.join("logs", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs("logs", exist_ok=True)
    with open(ACTIVE_RUN_FILE, "w", encoding="utf-8") as fh:
        fh.write(args.log_dir)

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
        ml_data_root=args.ml_data_root,
        ml_variant=args.ml_variant,
        # Defense
        defense_method=args.defense_method,
        defense_clip_thresh=args.defense_clip_thresh,
        defense_trim_frac=args.defense_trim_frac,
        defense_focus_k_frac=args.defense_focus_k_frac,
        # Adversarial eval
        adv_target_item=args.adv_target_item,
        adv_target_genre=args.adv_target_genre,
        log_dir=args.log_dir,
        db_path=os.path.join(args.log_dir, "telemetry.db"),
        log_file=os.path.join(args.log_dir, "telemetry.jsonl"),
    )

    # When --ml-data-root is given, always load dataset dimensions for all model types.
    # This avoids coordinator/node shape mismatches on MovieLens for non-BPR models.
    if args.ml_data_root:
        from fedsys.data.movielens_dataset import load_movielens_dataset
        print(f"[coordinator] Pre-loading {args.ml_variant} to determine model dimensions ...")
        _ml_ds = load_movielens_dataset(
            args.ml_data_root, args.ml_variant, show_progress=False
        )
        cfg_num_users = _ml_ds.num_users
        cfg_num_items = _ml_ds.num_items
        print(f"[coordinator] Dataset: {cfg_num_users} users, {cfg_num_items} items")
    else:
        cfg_num_users = args.num_users
        cfg_num_items = args.num_items

    model_cfg = ModelConfig(
        model_type=args.model_type,
        num_users=cfg_num_users,
        num_items=cfg_num_items,
        embedding_dim=args.embedding_dim,
        mlp_hidden=[args.hidden_dim],
    )

    if args.attack_max_synth > 0:
        model_cfg.num_users += args.attack_max_synth
        print(f"[coordinator] attack reserve: +{args.attack_max_synth} synthetic users "
              f"(num_users={model_cfg.num_users})")

    print(f"[coordinator] model={model_cfg.model_type}  "
          f"users={model_cfg.num_users}  items={model_cfg.num_items}  "
          f"emb={model_cfg.embedding_dim}  hidden={model_cfg.mlp_hidden}")
    print(f"[coordinator] Starting  host={cfg.host}:{cfg.port}  "
          f"N={cfg.total_nodes}  K={cfg.min_nodes}  rounds={cfg.num_rounds}")
    serve(cfg, model_cfg)

    if args.auto_analyze:
        try:
            print(f"[coordinator] Auto-analysis on {args.log_dir} ...")
            subprocess.run(
                [sys.executable, "analysis/analyze_logs.py", "--log-dir", args.log_dir],
                check=True,
            )
        except Exception as exc:
            print(f"[coordinator] Auto-analysis failed: {exc}")


if __name__ == "__main__":
    main()
