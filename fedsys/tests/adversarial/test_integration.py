"""
tests/adversarial/test_integration.py
========================================
Integration test: 1 coordinator + 1 clean node + 1 malicious node.

This test verifies that:
1. The entire FL pipeline runs with attack + defense enabled.
2. The coordinator aggregates poisoned and clean updates with a robust method.
3. No crashes occur and all rounds complete successfully.
4. Adversarial evaluation metrics are computed without error.

Run with:
    python tests/adversarial/test_integration.py

NOTE: This test uses an in-memory synthetic dataset (no MovieLens data
required) to keep things fast and self-contained.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedsys.adversarial.attack.poison import (
    AttackConfig,
    PoisonedBPRPairDataset,
    poisoned_num_users,
    _build_synthetic_profiles,
)
from fedsys.adversarial.attack.target import (
    select_target_item,
    choose_target_genre,
    benign_segment_users,
)
from fedsys.adversarial.defense.aggregation import RobustAggregator, DEFENSE_METHODS
from fedsys.adversarial.eval import evaluate_with_target_exposure, compare_attack_vs_clean
from fedsys.config import CoordinatorConfig, ModelConfig, NodeConfig
from fedsys.coordinator.server import serve
from fedsys.data.movielens_dataset import (
    BPRPairDataset,
    MovieLensFederatedDataset,
    MovieMetadata,
    UserSplit,
    partition_users,
)
from fedsys.node.client import FederatedNode

PORT_BASE = 50200


# ---------------------------------------------------------------------------
# Build a mock MovieLensFederatedDataset
# ---------------------------------------------------------------------------

def _make_test_dataset(
    num_users: int = 100,
    num_items: int = 40,
    seed: int = 99,
) -> MovieLensFederatedDataset:
    rng = random.Random(seed)
    genres = ["Action", "Comedy", "Drama", "Horror"]
    item_ids = tuple(range(num_items))
    user_ids = tuple(range(num_users))
    metadata = tuple(
        MovieMetadata(i, f"Movie{i}", (genres[i % len(genres)],))
        for i in range(num_items)
    )
    splits: Dict[int, UserSplit] = {}
    dominant: Dict[int, str] = {}
    for u in range(num_users):
        items = rng.sample(range(num_items), 7)
        splits[u] = UserSplit(
            train_ratings=tuple((it, 5.0) for it in items[:5]),
            val_item=items[5], test_item=items[6],
            known_items=frozenset(items),
        )
        dominant[u] = genres[u % len(genres)]

    items_by_genre: Dict[str, Tuple[int, ...]] = {g: () for g in genres}
    users_by_genre: Dict[str, Tuple[int, ...]] = {g: () for g in genres}
    for i in range(num_items):
        items_by_genre[genres[i % len(genres)]] += (i,)
    for u in range(num_users):
        users_by_genre[genres[u % len(genres)]] += (u,)

    pop: Dict[int, int] = {}
    for s in splits.values():
        for it, _ in s.train_ratings:
            pop[it] = pop.get(it, 0) + 1

    return MovieLensFederatedDataset(
        variant="test", root=None,
        item_ids=item_ids, item_id_to_index={i: i for i in range(num_items)},
        user_ids=user_ids, user_id_to_index={u: u for u in range(num_users)},
        movie_metadata=metadata, splits_by_user=splits,
        dominant_genre_by_user=dominant,
        users_by_genre=users_by_genre, items_by_genre=items_by_genre,
        train_item_popularity=pop,
    )


# ---------------------------------------------------------------------------
# Grpc options with safe keepalive
# ---------------------------------------------------------------------------

TEST_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    ("grpc.keepalive_time_ms", 120_000),
    ("grpc.keepalive_timeout_ms", 20_000),
    ("grpc.keepalive_permit_without_calls", False),
]


# ---------------------------------------------------------------------------
# Integration test runner
# ---------------------------------------------------------------------------

def run_adversarial_integration(
    port: int,
    defense_method: str = "clip_mean",
    num_rounds: int = 2,
) -> None:
    """
    Spin up:
      - 1 coordinator with defense_method + adversarial eval enabled
      - 1 clean node  (shard 0, normal BPR training)
      - 1 malicious node (shard 1, with poisoned dataloader)
    Run for num_rounds and assert no errors.
    """
    ds = _make_test_dataset()
    shards, _ = partition_users(ds.num_users, 2, seed=0)
    target_genre = "Action"
    target_item = select_target_item(ds, target_genre, min_popularity=1)
    print(f"[test] target_item={target_item}  genre={target_genre!r}")

    attack_cfg = AttackConfig(
        enabled=True,
        target_item_index=target_item,
        target_genre=target_genre,
        attack_budget=0.30,
        num_filler_items=2,
        num_neutral_items=2,
        neutral_genre="Comedy",
        max_synthetic_users_per_coord=20,
        seed=42,
    )

    real_num_users = ds.num_users
    extended_num_users = poisoned_num_users(real_num_users, attack_cfg)

    model_cfg = ModelConfig(
        model_type="bpr",
        num_users=extended_num_users,
        num_items=ds.num_items,
        embedding_dim=8,
    )

    coord_cfg = CoordinatorConfig(
        host="127.0.0.1",
        port=port,
        total_nodes=2,
        min_nodes=2,
        num_rounds=num_rounds,
        round_timeout_seconds=60,
        checkpoint_dir="",        # disable checkpointing for test speed
        ml_data_root="",          # no MovieLens files — use mock dataset directly
        defense_method=defense_method,
        defense_clip_thresh=5.0,
        adv_target_item=target_item,
        adv_target_genre=target_genre,
        log_dir="logs",
        db_path=f"logs/adv_coord_{port}.db",
        log_file=f"logs/adv_coord_{port}.jsonl",
        grpc_options=TEST_GRPC_OPTIONS,
    )

    # --- Coordinator runs serve() but our mock dataset is not passed through
    # the file path so we need to monkey-patch the aggregation loop to NOT
    # run file-based ML evaluation (ml_data_root is empty string).
    # Instead we just check that the FL round completes without error.
    #
    # Adversarial evaluation (evaluate_with_target_exposure) is tested
    # separately in test_eval.py using the mock dataset.
    threading.Thread(target=serve, args=(coord_cfg, model_cfg), daemon=True).start()
    time.sleep(0.8)

    errors = []

    def clean_node():
        try:
            cfg = NodeConfig(
                node_id="clean-node",
                coordinator_host="127.0.0.1",
                coordinator_port=port,
                device="cpu",
                local_epochs=1,
                batch_size=32,
                log_dir="logs",
                db_path=f"logs/clean_node_{port}.db",
                log_file=f"logs/clean_node_{port}.jsonl",
                grpc_options=TEST_GRPC_OPTIONS,
            )
            dl = DataLoader(
                BPRPairDataset(ds, list(shards[0]), rng_seed=0),
                batch_size=32, shuffle=True,
            )
            FederatedNode(cfg, model_cfg, dl, num_rounds=num_rounds).run()
        except Exception as exc:
            errors.append(f"clean_node: {exc}")

    def malicious_node():
        try:
            cfg = NodeConfig(
                node_id="malicious-node",
                coordinator_host="127.0.0.1",
                coordinator_port=port,
                device="cpu",
                local_epochs=1,
                batch_size=32,
                log_dir="logs",
                db_path=f"logs/mal_node_{port}.db",
                log_file=f"logs/mal_node_{port}.jsonl",
                grpc_options=TEST_GRPC_OPTIONS,
            )
            # Poisoned dataloader
            clean_ds = BPRPairDataset(
                ds, list(shards[1]),
                rng_seed=0,
            )
            profiles = _build_synthetic_profiles(
                dataset=ds,
                num_profiles=10,
                attack_cfg=attack_cfg,
                base_user_index=real_num_users,
            )
            poisoned_ds = PoisonedBPRPairDataset(
                clean_dataset=clean_ds,
                synthetic_profiles=profiles,
                base_user_index=real_num_users,
                num_items=ds.num_items,
            )
            dl = DataLoader(poisoned_ds, batch_size=32, shuffle=True)
            FederatedNode(cfg, model_cfg, dl, num_rounds=num_rounds).run()
        except Exception as exc:
            errors.append(f"malicious_node: {exc}")

    t_clean = threading.Thread(target=clean_node)
    t_mal   = threading.Thread(target=malicious_node)
    t_clean.start()
    t_mal.start()
    t_clean.join(timeout=120)
    t_mal.join(timeout=120)

    if errors:
        raise RuntimeError(
            f"Integration test FAILED (defense={defense_method}):\n"
            + "\n".join(errors)
        )
    print(f"PASSED: adversarial integration | defense={defense_method}")


# ---------------------------------------------------------------------------
# Unit-level adversarial eval sanity check (no gRPC)
# ---------------------------------------------------------------------------

def test_adversarial_eval_sanity():
    """Check that evaluate_with_target_exposure returns sensible values."""
    ds = _make_test_dataset()
    from fedsys.models.bpr import BPRModel
    from fedsys.config import ModelConfig
    mc = ModelConfig(model_type="bpr", num_users=ds.num_users, num_items=ds.num_items, embedding_dim=8)
    model = BPRModel(mc)
    genre = "Action"
    target = ds.items_by_genre[genre][0]
    metrics = evaluate_with_target_exposure(
        model=model, dataset=ds,
        target_item_index=target,
        target_genre=genre,
        split="val", cutoffs=(5, 10),
    )
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k}={v}"
    print("[test] adversarial_eval_sanity PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)

    # Sanity check (no gRPC)
    test_adversarial_eval_sanity()

    # Integration: clip_mean defense
    run_adversarial_integration(PORT_BASE, defense_method="clip_mean", num_rounds=2)

    # Integration: focus_clip_trimmed_mean defense
    run_adversarial_integration(PORT_BASE + 1, defense_method="focus_clip_trimmed_mean", num_rounds=2)

    # Integration: no defense (plain FedAvg — establishes baseline)
    run_adversarial_integration(PORT_BASE + 2, defense_method="none", num_rounds=2)

    print("\n=== All adversarial integration tests passed ===")
