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

    # MovieLens ranking evaluation (used when model_type="bpr").
    # Set ml_data_root to the directory containing the variant folder.
    # When set, overrides val_data_path / test_data_path for evaluation.
    ml_data_root: str = ""
    ml_variant:   str = "ml-1m"

    # ── Defense configuration ──────────────────────────────────────────────
    # Set defense_method to one of:
    #   "none"                    — plain FedAvg (default)
    #   "clip_mean"               — L2 norm-clip + mean
    #   "clip_trimmed_mean"       — L2 norm-clip + trimmed mean
    #   "focus_clip_mean"         — focus-score + clip + mean
    #   "focus_clip_trimmed_mean" — focus-score + clip + trimmed mean
    defense_method:       str   = "none"
    defense_clip_thresh:  float = 5.0    # L2 norm clip threshold (Theta)
    defense_trim_frac:    float = 0.10   # trimmed-mean fraction from each tail
    defense_focus_k_frac: float = 0.05   # fraction of top items for focus score

    # ── Adversarial evaluation ─────────────────────────────────────────────
    # When set, the coordinator also logs target-exposure metrics every round.
    # target_item_index = -1 means disabled.
    adv_target_item:  int = -1
    adv_target_genre: str = ""

    # Telemetry
    log_dir: str = "logs"
    db_path: str = "logs/telemetry.db"
    log_file: str = "logs/telemetry.jsonl"

    # gRPC channel options — unlimited message sizes
    grpc_options: List = field(default_factory=lambda: [
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
        # Keepalive: use values aligned with tests to avoid ENHANCE_YOUR_CALM
        # "too_many_pings" during long-idle gaps (e.g., MovieLens loading).
        ("grpc.keepalive_time_ms", 120_000),
        ("grpc.keepalive_timeout_ms", 20_000),
        ("grpc.keepalive_permit_without_calls", False),
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

    # MovieLens data source (used when model_type="bpr")
    ml_data_root: str = ""
    ml_variant:   str = "ml-1m"

    # ── Attack configuration ───────────────────────────────────────────────
    # Set attack_enabled=True on a *malicious* node to inject synthetic
    # user profiles.  All other attack_* fields control the profile shape.
    attack_enabled:         bool  = False
    attack_target_item:     int   = -1    # contiguous item index to push
    attack_target_genre:    str   = ""
    attack_budget:          float = 0.30
    attack_num_filler:      int   = 30
    attack_num_neutral:     int   = 20
    attack_neutral_genre:   str   = "Comedy"
    attack_target_weight:   float = 1.0
    attack_max_synth_users: int   = 200
    attack_seed:            int   = 42

    # Telemetry
    log_dir: str = "logs"
    db_path: str = "logs/telemetry.db"
    log_file: str = "logs/telemetry.jsonl"

    grpc_options: List = field(default_factory=lambda: [
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
        # Keepalive: use values aligned with tests to avoid ENHANCE_YOUR_CALM
        # "too_many_pings" during long-idle gaps.
        ("grpc.keepalive_time_ms", 120_000),
        ("grpc.keepalive_timeout_ms", 20_000),
        ("grpc.keepalive_permit_without_calls", False),
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

    model_type : "simple"  -> TwoLayerModel  (fast synthetic-data testing)
                 "bpr"     -> BPRModel        (BPR-MF for MovieLens)

    Simple defaults (model_type="simple", synthetic data):
        user_embedding : 1 000 × 16  =  16 K
        item_embedding :   500 × 16  =   8 K
        FC layers      : 32→64→1     =  ~2 K
        Total                        ~  26 K params

    BPR defaults (model_type="bpr", ml-1m):
        user_embedding : 6 040 × 32  = 193 K
        item_embedding : 3 706 × 32  = 119 K
        user_bias      : 6 040 × 1   =   6 K
        item_bias      : 3 706 × 1   =   4 K
        Total                        ~ 321 K params
    """
    # "simple" is the default so tests run instantly on CPU
    model_type: str = "simple"

    num_users: int = 1_000
    num_items: int = 500
    embedding_dim: int = 16
    # Hidden layer widths; used by TwoLayerModel (only first value) — ignored by BPR
    mlp_hidden: List[int] = field(default_factory=lambda: [64])


def ensure_log_dir(cfg: CoordinatorConfig | NodeConfig) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
