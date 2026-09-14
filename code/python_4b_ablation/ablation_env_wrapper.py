from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ["python_3a_homogeneous_mappo", "python_3c_capability_aware_mappo"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mappo_env_wrapper import MAPPOEnvWrapper  # noqa: E402
from capability_aware_env_wrapper import CapabilityAwareMAPPOEnvWrapper  # noqa: E402


class AblationMAPPOEnvWrapper(CapabilityAwareMAPPOEnvWrapper):
    """CA-HMAPPO wrapper with switchable ablation components."""

    def __init__(
        self,
        scenario: int = 1,
        seed: int | None = None,
        use_guidance: bool = True,
        use_capability_reward: bool = True,
        use_safety_shaping: bool = True,
    ) -> None:
        self.use_guidance = bool(use_guidance)
        self.use_capability_reward = bool(use_capability_reward)
        self.use_safety_shaping = bool(use_safety_shaping)
        super().__init__(scenario=scenario, seed=seed)
        if not self.use_guidance:
            self.obs_dim = self.base_obs_dim
            self.global_state_dim = self.obs_dim * self.n_agents + 4

    def step(self, actions: Iterable[int] | Dict[str, int]) -> Dict[str, np.ndarray | dict | bool]:
        action_dict = self._to_action_dict(actions)
        previous_distances = self._quadrotor_candidate_distances()
        previous_vectors = self._quadrotor_candidate_vectors()
        previous_coverage = int(self.env.coverage_map.sum())
        previous_detections = int(self.env.target_detected.sum())
        previous_heading = float(self.env.uavs["fixedwing_0"].heading)
        previous_collision_count = int(self.env.collision_count)
        previous_near_miss_count = int(self.env.near_miss_count)
        previous_inspected = self.env.target_inspected.copy()

        data = MAPPOEnvWrapper.step(self, action_dict)
        data = self._augment_output(data)

        shaped = np.asarray(data["reward_n"], dtype=np.float32).copy()
        if self.use_capability_reward:
            current_distances = self._quadrotor_candidate_distances()
            shaped += self._fixedwing_capability_reward(previous_coverage, previous_detections, previous_heading)
            shaped += self._quadrotor_capability_reward(action_dict, previous_distances, current_distances, previous_vectors)
            shaped += self._quadrotor_inspection_credit(action_dict, previous_inspected)
        if self.use_safety_shaping:
            shaped -= self._safety_shaping(previous_collision_count, previous_near_miss_count)
        data["reward_n"] = shaped.astype(np.float32)
        return data

    def get_action_masks(self) -> np.ndarray:
        masks = MAPPOEnvWrapper.get_action_masks(self)
        if not self.use_safety_shaping:
            return masks.astype(np.float32)
        for idx, agent_id in enumerate(self.agents):
            if not agent_id.startswith("quadrotor"):
                continue
            original = masks[idx].copy()
            for action in [0, 1, 2, 3, 4, 10]:
                if masks[idx, action] > 0 and not self._is_quadrotor_action_safe(agent_id, action):
                    masks[idx, action] = 0.0
            if masks[idx].sum() <= 0:
                masks[idx] = original
        return masks.astype(np.float32)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.use_guidance:
            return obs.astype(np.float32)
        return super()._augment_obs(obs)

    def _near_neighbor_penalty(self, agent_id: str) -> float:
        if not self.use_safety_shaping:
            return 0.0
        return super()._near_neighbor_penalty(agent_id)

    def _is_quadrotor_action_safe(self, agent_id: str, action: int) -> bool:
        uav = self.env.uavs[agent_id]
        next_position = uav.position().copy()
        if action == 0:
            next_position[1] += uav.speed
        elif action == 1:
            next_position[1] -= uav.speed
        elif action == 2:
            next_position[0] += uav.speed
        elif action == 3:
            next_position[0] -= uav.speed
        elif action == 10:
            delta = self.env.base - uav.position()
            norm = np.linalg.norm(delta)
            if norm > 1e-9:
                next_position += delta / norm * min(uav.speed, norm)
        next_position = np.clip(next_position, 0.0, self.env.area_size)
        for other_id, other in self.env.uavs.items():
            if other_id == agent_id or other.uav_type != "quadrotor":
                continue
            if np.linalg.norm(next_position - other.position()) <= self.env.near_miss_radius:
                return False
        return True


def ablation_name(use_guidance: bool, use_capability_reward: bool, use_safety_shaping: bool) -> str:
    if use_guidance and use_capability_reward and use_safety_shaping:
        return "full_ca_hmappo"
    if not use_guidance and use_capability_reward and use_safety_shaping:
        return "without_guidance"
    if use_guidance and not use_capability_reward and use_safety_shaping:
        return "without_capability_reward"
    if use_guidance and use_capability_reward and not use_safety_shaping:
        return "without_safety_shaping"
    return f"guidance_{int(use_guidance)}_capability_{int(use_capability_reward)}_safety_{int(use_safety_shaping)}"
