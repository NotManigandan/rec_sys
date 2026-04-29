from __future__ import annotations

import torch
from torch import nn

from .base import RecommenderModel


class NeuralCollaborativeFiltering(RecommenderModel):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        del histories
        user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)
        return self.mlp(torch.cat([user_vector, item_vector], dim=-1)).squeeze(-1)
