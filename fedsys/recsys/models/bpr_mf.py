from __future__ import annotations

import torch
from torch import nn

from .base import RecommenderModel


class BPRMatrixFactorization(RecommenderModel):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        del histories
        user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)
        interaction = (user_vector * item_vector).sum(dim=-1)
        return interaction + self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1)
