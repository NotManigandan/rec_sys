from __future__ import annotations

import torch
from torch import nn

from fedsys.config import ModelConfig


class NeuralCFModel(nn.Module):
    """MLP-based Neural Collaborative Filtering scorer."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden_dim = cfg.mlp_hidden[0] if cfg.mlp_hidden else 64

        self.user_embedding = nn.Embedding(cfg.num_users, cfg.embedding_dim)
        self.item_embedding = nn.Embedding(cfg.num_items, cfg.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 8), 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        histories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del histories
        user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)
        return self.mlp(torch.cat([user_vector, item_vector], dim=-1)).squeeze(-1)
