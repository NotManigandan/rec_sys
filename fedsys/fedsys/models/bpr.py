"""
Bayesian Personalised Ranking Matrix Factorisation (BPR-MF).

Architecture
------------
    score(u, i) = < e_u, e_i >  +  b_u  +  b_i

where e_u and e_i are learnable user/item embedding vectors and b_u, b_i are
scalar biases.  The pairwise BPR ranking loss optimised during local training is:

    L = -E[ log σ( score(u, i+) - score(u, i-) ) ]

The model is deliberately lightweight (no MLP on top of embeddings) so that
large numbers of users and items still produce compact weight tensors that
transfer cheaply over gRPC.

Parameter count example (default config):
    num_users=6 040, num_items=3 706, embedding_dim=32
    user_embedding: 6 040 × 32 = 193 280
    item_embedding: 3 706 × 32 = 118 592
    user_bias     : 6 040 × 1  =   6 040
    item_bias     : 3 706 × 1  =   3 706
    Total                       ~ 321 K params
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fedsys.config import ModelConfig


class BPRModel(nn.Module):
    """
    BPR Matrix Factorisation model.

    Inputs
    ------
    user_ids : LongTensor (B,)   — integer user indices in [0, num_users)
    item_ids : LongTensor (B,)   — integer item indices in [0, num_items)

    Output
    ------
    scores : FloatTensor (B,)    — real-valued relevance score per pair
                                   (NOT passed through sigmoid here — the BPR
                                    loss and ranking code work directly on raw
                                    scores)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.embedding_dim
        self.user_embedding = nn.Embedding(cfg.num_users, D)
        self.item_embedding = nn.Embedding(cfg.num_items, D)
        self.user_bias      = nn.Embedding(cfg.num_users, 1)
        self.item_bias      = nn.Embedding(cfg.num_items, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        histories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw scores — shape (B,)."""
        del histories
        u = self.user_embedding(user_ids)              # (B, D)
        v = self.item_embedding(item_ids)              # (B, D)
        dot = (u * v).sum(dim=-1)                      # (B,)
        b_u = self.user_bias(user_ids).squeeze(-1)     # (B,)
        b_i = self.item_bias(item_ids).squeeze(-1)     # (B,)
        return dot + b_u + b_i                         # (B,)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
