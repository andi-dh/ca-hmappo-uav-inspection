from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class MAPPOBuffer:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    obs: list[np.ndarray] = field(default_factory=list)
    global_states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    action_log_probs: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    dones: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        value: float,
        action_masks: np.ndarray,
    ) -> None:
        self.obs.append(obs.copy())
        self.global_states.append(global_state.copy())
        self.actions.append(actions.copy())
        self.action_log_probs.append(action_log_probs.copy())
        self.rewards.append(rewards.copy())
        self.dones.append(dones.copy())
        self.values.append(float(value))
        self.action_masks.append(action_masks.copy())

    def clear(self) -> None:
        self.obs.clear()
        self.global_states.clear()
        self.actions.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_masks.clear()

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray([float(np.mean(r)) for r in self.rewards], dtype=np.float32)
        dones = np.asarray([bool(np.any(d)) for d in self.dones], dtype=bool)
        values = np.asarray(self.values + [float(last_value)], dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            non_terminal = 0.0 if dones[step] else 1.0
            delta = rewards[step] + self.gamma * values[step + 1] * non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        return returns.astype(np.float32), advantages.astype(np.float32)

    def as_tensors(self, device: torch.device, last_value: float = 0.0) -> dict[str, torch.Tensor]:
        returns, advantages = self.compute_returns_and_advantages(last_value=last_value)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n_agents = self.obs[0].shape[0]
        return {
            "obs": torch.as_tensor(np.concatenate(self.obs, axis=0), dtype=torch.float32, device=device),
            "global_states": torch.as_tensor(np.asarray(self.global_states), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.concatenate(self.actions, axis=0), dtype=torch.long, device=device),
            "old_log_probs": torch.as_tensor(np.concatenate(self.action_log_probs, axis=0), dtype=torch.float32, device=device),
            "action_masks": torch.as_tensor(np.concatenate(self.action_masks, axis=0), dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(np.repeat(advantages, n_agents), dtype=torch.float32, device=device),
        }
