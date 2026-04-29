"""
Central configuration dataclasses
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

@dataclass
class CoordinatorConfig:
    host: str = "0.0.0.0"
    port: int = 50051

    # K-out-of-N synchronous aggregation
    min_nodes: int = 2          # K: wait for at least this many updates
    total_nodes: int = 2        # N: total expected nodes (used for partitioning)

    # How long the coordinator waits before timing out a round and dropping stragglers
    round_timeout_seconds: float = 120.0

    num_rounds: int = 10

    # Streaming chunk size in bytes (4 MB default)
    chunk_size_bytes: int = 4 * 1024 * 1024

    # Checkpointing — global model saved to disk after every completed round.
    # Files written:
    #   checkpoints/model_epoch_{N}.pt  — after each round
    #   checkpoints/model_best.pt       — round with lowest val loss (if val set given)
    #   checkpoints/model_final.pt      — always the last completed round
    # Set to "" to disable all saving.
    checkpoint_dir: str = "checkpoints"

    # Paths to held-out evaluation CSVs (produced by generate_synthetic_data.py).
    # Set to "" to skip validation / test evaluation.
    val_data_path:  str = ""
    test_data_path: str = ""

    # Telemetry
    log_dir: str = "logs"
    db_path: str = "logs/telemetry.db"
    log_file: str = "logs/telemetry.jsonl"

    # gRPC channel options — unlimited message sizes
    grpc_options: List = field(default_factory=lambda: [
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
        ("grpc.keepalive_time_ms", 10_000),
        ("grpc.keepalive_timeout_ms", 5_000),
        ("grpc.keepalive_permit_without_calls", True),
    ])


# ---------------------------------------------------------------------------
# Training Node
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    coordinator_host: str = "localhost"
    coordinator_port: int = 50051

    local_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3

    # "cuda", "cuda:0", or "cpu"
    device: str = "cpu"

    # Assigned by coordinator after registration
    data_partition: int = 0
    num_partitions: int = 3

    chunk_size_bytes: int = 4 * 1024 * 1024

    # Telemetry
    log_dir: str = "logs"
    db_path: str = "logs/telemetry.db"
    log_file: str = "logs/telemetry.jsonl"

    grpc_options: List = field(default_factory=lambda: [
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
        ("grpc.keepalive_time_ms", 10_000),
        ("grpc.keepalive_timeout_ms", 5_000),
        ("grpc.keepalive_permit_without_calls", True),
    ])

    @property
    def coordinator_address(self) -> str:
        return f"{self.coordinator_host}:{self.coordinator_port}"


# ---------------------------------------------------------------------------
# Model (shared shape so coordinator and nodes agree on architecture)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Shared model shape used by both coordinator and training nodes.

    model_type : "simple"  -> TwoLayerModel  (fast, for testing)
                 "ncf"     -> NCFRecommender (~300 M params, production)

    Simple defaults (model_type="simple"):
        user_embedding : 1 000 × 16  =  16 K
        item_embedding :   500 × 16  =   8 K
        FC layers      : 32→64→1     =  ~2 K
        Total                        ~  26 K params

    NCF defaults (model_type="ncf"):
        user_embedding : 1 000 000 × 128  = 128 M
        item_embedding :   500 000 × 128  =  64 M
        MLP backbone                      ~ 108 M
        Total                             ~ 300 M
    """
    # "simple" is the default so tests run instantly on CPU
    model_type: str = "simple"

    num_users: int = 1_000
    num_items: int = 500
    embedding_dim: int = 16
    # Hidden layer widths; for TwoLayerModel only the first value is used
    mlp_hidden: List[int] = field(default_factory=lambda: [64])


def large_model_config() -> "ModelConfig":
    """Return a ModelConfig pre-set for the full ~300 M parameter NCF."""
    return ModelConfig(
        model_type="ncf",
        num_users=1_000_000,
        num_items=500_000,
        embedding_dim=128,
        mlp_hidden=[256, 4096, 8192, 8192, 4096, 2048, 1024],
    )


def ensure_log_dir(cfg: CoordinatorConfig | NodeConfig) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
