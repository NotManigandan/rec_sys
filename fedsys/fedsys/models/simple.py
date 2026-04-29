from __future__ import annotations

import torch
import torch.nn as nn

from fedsys.config import ModelConfig


class TwoLayerModel(nn.Module):
    """
    Minimal NCF: embed -> concat -> Linear -> ReLU -> Linear -> logit.
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

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        histories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del histories
        x = torch.cat([
            self.user_embedding(user_ids),
            self.item_embedding(item_ids),
        ], dim=-1)
        return self.fc2(torch.relu(self.fc1(x)))

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self) -> str:
        n = self.param_count
        if n >= 1_000_000_000:
            return f"{n / 1e9:.2f} B"
        if n >= 1_000_000:
            return f"{n / 1e6:.1f} M"
        if n >= 1_000:
            return f"{n / 1e3:.1f} K"
        return str(n)
