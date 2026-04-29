from __future__ import annotations

import torch
from torch import nn

from fedsys.config import ModelConfig


class TwoTowerModel(nn.Module):
    """
    Two-tower recommender.

    If `histories` is provided (dense multi-hot item vector), it uses a history
    tower. Otherwise it falls back to a learned user embedding so existing
    dataloaders continue to work.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden_dim = cfg.mlp_hidden[0] if cfg.mlp_hidden else 64
        self.user_tower = nn.Sequential(
            nn.Linear(cfg.num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cfg.embedding_dim),
        )
        self.user_embedding = nn.Embedding(cfg.num_users, cfg.embedding_dim)
        self.item_embedding = nn.Embedding(cfg.num_items, cfg.embedding_dim)
        self.item_bias = nn.Embedding(cfg.num_items, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.user_tower:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        histories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if histories is not None:
            user_vector = self.user_tower(histories.float())
        else:
            user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)
        return (user_vector * item_vector).sum(dim=-1) + self.item_bias(item_ids).squeeze(-1)
