from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class RecommenderModel(nn.Module, ABC):
    @abstractmethod
    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
