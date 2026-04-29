"""Recommendation model factory."""

from __future__ import annotations

import torch.nn as nn
from fedsys.config import ModelConfig
from fedsys.models.bpr import BPRModel
from fedsys.models.neural_cf import NeuralCFModel
from fedsys.models.simple import TwoLayerModel
from fedsys.models.two_tower import TwoTowerModel


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> nn.Module:
    """
    Construct the correct model class from ModelConfig.model_type.

      "simple"     -> TwoLayerModel   (fast synthetic baseline)
      "bpr"        -> BPRModel        (BPR-MF for MovieLens)
      "neural_cf"  -> NeuralCFModel   (MLP recommender)
      "two_tower"  -> TwoTowerModel   (history/user tower + item tower)
    """
    if cfg.model_type == "simple":
        m = TwoLayerModel(cfg)
    elif cfg.model_type == "bpr":
        m = BPRModel(cfg)
    elif cfg.model_type == "neural_cf":
        m = NeuralCFModel(cfg)
    elif cfg.model_type == "two_tower":
        m = TwoTowerModel(cfg)
    else:
        raise ValueError(
            f"Unknown model_type={cfg.model_type!r}. "
            "Use 'simple', 'bpr', 'neural_cf', or 'two_tower'."
        )
    n = sum(p.numel() for p in m.parameters())
    print(f"[model] {cfg.model_type}  params={_fmt_params(n)}  "
          f"users={cfg.num_users}  items={cfg.num_items}  emb={cfg.embedding_dim}")
    return m


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
