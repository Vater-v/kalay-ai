"""Small MLP networks for the toy-game ladder.

Structure mirrors the production split: `backbone` (weight decay) + `head`
(no weight decay, keeps NeuRD logit thresholds unpenalized — SPEC 5.6).
The transformer upgrade for HUNL replaces these behind the same interface.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs))  # raw logits [..., n_actions]


class ValueMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs)).squeeze(-1)  # [...]
