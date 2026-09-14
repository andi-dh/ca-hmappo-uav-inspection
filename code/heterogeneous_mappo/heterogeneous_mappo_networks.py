from __future__ import annotations

import torch
from torch import nn


class HeterogeneousActor(nn.Module):
    """Shared encoder with UAV-type-specific action heads."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.fixedwing_head = nn.Linear(hidden_dim, action_dim)
        self.quadrotor_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(obs)
        logits_fixed = self.fixedwing_head(features)
        logits_quad = self.quadrotor_head(features)
        # Observation index 5 is uav_type_code: fixed-wing=0, quadrotor=1.
        is_quadrotor = (obs[:, 5] > 0.5).unsqueeze(-1)
        logits = torch.where(is_quadrotor, logits_quad, logits_fixed)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e8)
        return logits
