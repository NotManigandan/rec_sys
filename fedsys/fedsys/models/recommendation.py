"""
Recommendation model implementations.

Two variants are provided:

  TwoLayerModel  (model_type="simple")
      Lightweight two-FC-layer NCF for fast local testing.
      Default config: ~10 K parameters, trains in milliseconds on CPU.

  NCFRecommender (model_type="ncf")
      Deep MLP NCF targeting ~300 M parameters for production FL experiments.

Both classes share the same forward signature so the rest of the framework
(trainer, aggregator, client, server) is completely model-agnostic.

Use build_model(cfg) as the single factory consumed by server.py and client.py.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from fedsys.config import ModelConfig


# ---------------------------------------------------------------------------
# Simple two-layer model  (default for testing)
# ---------------------------------------------------------------------------

class TwoLayerModel(nn.Module):
    """
    Minimal NCF: embed → concat → Linear → ReLU → Linear → logit.

    Inputs
    ------
    user_ids : LongTensor (B,)
    item_ids : LongTensor (B,)

    Output
    ------
    logits : FloatTensor (B, 1)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(cfg.num_users, cfg.embedding_dim)
        self.item_embedding = nn.Embedding(cfg.num_items, cfg.embedding_dim)

        hidden = cfg.mlp_hidden[0] if cfg.mlp_hidden else 64
        in_dim = 2 * cfg.embedding_dim

        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            self.user_embedding(user_ids),
            self.item_embedding(item_ids),
        ], dim=-1)
        return self.fc2(torch.relu(self.fc1(x)))

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self) -> str:
        return _fmt_params(self.param_count)


# ---------------------------------------------------------------------------
# Full NCF model  (for production / large-scale experiments)
# ---------------------------------------------------------------------------

class NCFRecommender(nn.Module):
    """
    Deep MLP Neural Collaborative Filtering recommender targeting ~300 M params.

    Inputs
    ------
    user_ids : LongTensor (batch,)
    item_ids : LongTensor (batch,)

    Output
    ------
    logits : FloatTensor (batch, 1)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.user_embedding = nn.Embedding(cfg.num_users, cfg.embedding_dim)
        self.item_embedding = nn.Embedding(cfg.num_items, cfg.embedding_dim)

        in_dim = 2 * cfg.embedding_dim
        layers: List[nn.Module] = []
        for hidden in cfg.mlp_hidden:
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            self.user_embedding(user_ids),
            self.item_embedding(item_ids),
        ], dim=-1)
        return self.mlp(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self) -> str:
        return _fmt_params(self.param_count)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> nn.Module:
    """
    Construct the correct model class from ModelConfig.model_type.

      "simple"  ->  TwoLayerModel   (fast, for testing)
      "ncf"     ->  NCFRecommender  (full scale, ~300 M params)
    """
    if cfg.model_type == "simple":
        return TwoLayerModel(cfg)
    if cfg.model_type == "ncf":
        return NCFRecommender(cfg)
    raise ValueError(f"Unknown model_type={cfg.model_type!r}. Use 'simple' or 'ncf'.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} M"
    if n >= 1_000:
        return f"{n / 1e3:.1f} K"
    return str(n)
