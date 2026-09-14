from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    """Shared homogeneous actor for all UAV agents."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(obs)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e8)
        return logits


class Critic(nn.Module):
    """Centralized critic that consumes the global state."""

    def __init__(self, global_state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
