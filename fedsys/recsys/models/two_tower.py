from __future__ import annotations

import torch
from torch import nn

from .base import RecommenderModel


class TwoTowerHistoryModel(RecommenderModel):
    def __init__(self, num_items: int, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.user_tower:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        del user_ids
        user_vector = self.user_tower(histories)
        item_vector = self.item_embedding(item_ids)
        return (user_vector * item_vector).sum(dim=-1) + self.item_bias(item_ids).squeeze(-1)
