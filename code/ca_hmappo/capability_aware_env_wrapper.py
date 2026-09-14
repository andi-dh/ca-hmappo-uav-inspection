from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

THREE_A_DIR = Path(__file__).resolve().parents[1] / "homogeneous_mappo"
if str(THREE_A_DIR) not in sys.path:
    sys.path.insert(0, str(THREE_A_DIR))

from mappo_env_wrapper import MAPPOEnvWrapper  # noqa: E402


class CapabilityAwareMAPPOEnvWrapper(MAPPOEnvWrapper):
    """MAPPO wrapper with capability-aware reward shaping.

    The underlying environment and metrics stay unchanged. This wrapper only
    modifies training rewards to provide denser learning signals for UAV roles.
    """

    def __init__(self, scenario: int = 0, seed: int | None = None) -> None:
        super().__init__(scenario=scenario, seed=seed)
        self.base_obs_dim = self.obs_dim
        self.obs_dim = self.base_obs_dim + 2
        self.global_state_dim = self.obs_dim * self.n_agents + 4

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray | dict | bool]:
        data = super().reset(seed=seed)
        return self._augment_output(data)

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

        data = super().step(action_dict)
        data = self._augment_output(data)

        current_distances = self._quadrotor_candidate_distances()
        shaped = np.asarray(data["reward_n"], dtype=np.float32).copy()
        shaped += self._fixedwing_capability_reward(previous_coverage, previous_detections, previous_heading)
        shaped += self._quadrotor_capability_reward(action_dict, previous_distances, current_distances, previous_vectors)
        shaped += self._quadrotor_inspection_credit(action_dict, previous_inspected)
        shaped -= self._safety_shaping(previous_collision_count, previous_near_miss_count)
        data["reward_n"] = shaped.astype(np.float32)
        data["info"]["base_reward_mean"] = float(np.mean(data["reward_n"]))
        return data

    def get_global_state(self) -> np.ndarray:
        obs = self._augment_obs(np.stack([self.env.observe(agent_id) for agent_id in self.agents]).astype(np.float32))
        metrics = self.env.compute_metrics()
        extras = np.array(
            [
                metrics["coverage_ratio"],
                metrics["target_detection_rate"],
                metrics["inspection_success_rate"],
                self.env.steps / max(1, self.env.max_steps),
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs.reshape(-1), extras]).astype(np.float32)

    def get_action_masks(self) -> np.ndarray:
        masks = super().get_action_masks()
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

    def _augment_output(self, data: Dict[str, np.ndarray | dict | bool]) -> Dict[str, np.ndarray | dict | bool]:
        data["obs_n"] = self._augment_obs(np.asarray(data["obs_n"], dtype=np.float32))
        data["state_global"] = self.get_global_state()
        return data

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        guidance = np.zeros((self.n_agents, 2), dtype=np.float32)
        assignments = self._quadrotor_candidate_assignments()
        if not assignments:
            return np.concatenate([obs, guidance], axis=1).astype(np.float32)
        for idx, agent_id in enumerate(self.agents):
            if not agent_id.startswith("quadrotor"):
                continue
            if agent_id not in assignments:
                continue
            uav_position = self.env.uavs[agent_id].position()
            target = assignments[agent_id]
            delta = (target - uav_position) / max(1.0, self.env.area_size)
            guidance[idx] = delta.astype(np.float32)
        return np.concatenate([obs, guidance], axis=1).astype(np.float32)

    def _fixedwing_capability_reward(self, previous_coverage: int, previous_detections: int, previous_heading: float) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        fixedwing = self.env.uavs["fixedwing_0"]
        coverage_gain = max(0, int(self.env.coverage_map.sum()) - previous_coverage)
        detection_gain = max(0, int(self.env.target_detected.sum()) - previous_detections)
        heading_change = abs(float(fixedwing.heading) - previous_heading)
        redundant_sweep = 1.0 if coverage_gain == 0 else 0.0
        rewards[0] += 0.07 * coverage_gain
        rewards[0] += 2.0 * detection_gain
        rewards[0] -= 0.02 * redundant_sweep
        rewards[0] -= 0.05 * float(heading_change > np.deg2rad(30.0))
        return rewards

    def _quadrotor_capability_reward(
        self,
        actions: Dict[str, int],
        previous_distances: Dict[str, float],
        current_distances: Dict[str, float],
        previous_vectors: Dict[str, np.ndarray],
    ) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        for idx, agent_id in enumerate(self.agents):
            if not agent_id.startswith("quadrotor"):
                continue
            uav = self.env.uavs[agent_id]
            prev_distance = previous_distances.get(agent_id, self.env.area_size)
            current_distance = current_distances.get(agent_id, self.env.area_size)
            vector = previous_vectors.get(agent_id, np.zeros(2, dtype=np.float64))
            action = int(actions.get(agent_id, 4))
            approach_gain = max(0.0, prev_distance - current_distance)
            retreat_loss = max(0.0, current_distance - prev_distance)
            in_range = current_distance <= uav.sensor_range
            has_candidate = prev_distance < self.env.area_size
            rewards[idx] += 0.08 * approach_gain
            rewards[idx] -= 0.03 * retreat_loss
            rewards[idx] += self._directional_action_reward(action, vector, in_range, has_candidate)
            if in_range:
                rewards[idx] += 0.3
            if action == 4:
                rewards[idx] -= 0.05
            if action == 4 and in_range:
                rewards[idx] += 12.0
            elif action == 4 and not in_range:
                rewards[idx] -= 3.0
            elif in_range:
                rewards[idx] -= 0.2
            if action == 10 and has_candidate and not in_range:
                rewards[idx] -= 1.0
            if action == 4 and len(self.env.candidate_targets) == 0:
                rewards[idx] -= 0.2
            rewards[idx] -= self._near_neighbor_penalty(agent_id)
            rewards[idx] -= 0.005
        return rewards

    def _quadrotor_inspection_credit(self, actions: Dict[str, int], previous_inspected: np.ndarray) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        newly_inspected = np.where((~previous_inspected) & self.env.target_inspected)[0]
        if len(newly_inspected) == 0:
            return rewards
        for target_idx in newly_inspected:
            target = self.env.targets[target_idx]
            best_agent_idx = None
            best_distance = float("inf")
            for idx, agent_id in enumerate(self.agents):
                if not agent_id.startswith("quadrotor") or int(actions.get(agent_id, -1)) != 4:
                    continue
                distance = float(np.linalg.norm(self.env.uavs[agent_id].position() - target))
                if distance < best_distance:
                    best_distance = distance
                    best_agent_idx = idx
            if best_agent_idx is not None:
                rewards[best_agent_idx] += 25.0
        return rewards

    def _near_neighbor_penalty(self, agent_id: str) -> float:
        uav = self.env.uavs[agent_id]
        distances = [
            float(np.linalg.norm(uav.position() - other.position()))
            for other_id, other in self.env.uavs.items()
            if other_id != agent_id and other.uav_type == "quadrotor"
        ]
        if not distances:
            return 0.0
        nearest = min(distances)
        if nearest >= self.env.near_miss_radius:
            return 0.0
        return 0.5 * (1.0 - nearest / max(1.0, self.env.near_miss_radius))

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

    def _directional_action_reward(self, action: int, vector: np.ndarray, in_range: bool, has_candidate: bool) -> float:
        if not has_candidate or in_range:
            return 0.0
        dx, dy = float(vector[0]), float(vector[1])
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        desired_actions = []
        if abs(dx) >= abs(dy):
            desired_actions.append(2 if dx > 0 else 3)
            desired_actions.append(0 if dy > 0 else 1)
        else:
            desired_actions.append(0 if dy > 0 else 1)
            desired_actions.append(2 if dx > 0 else 3)
        if action == desired_actions[0]:
            return 1.5
        if len(desired_actions) > 1 and action == desired_actions[1]:
            return 0.5
        if action in {0, 1, 2, 3}:
            return -0.3
        return 0.0

    def _safety_shaping(self, previous_collision_count: int, previous_near_miss_count: int) -> np.ndarray:
        collision_gain = max(0, int(self.env.collision_count) - previous_collision_count)
        near_miss_gain = max(0, int(self.env.near_miss_count) - previous_near_miss_count)
        penalty = 25.0 * collision_gain + 3.0 * near_miss_gain
        if penalty <= 0:
            return np.zeros(self.n_agents, dtype=np.float32)
        return np.full(self.n_agents, penalty, dtype=np.float32)

    def _quadrotor_candidate_distances(self) -> Dict[str, float]:
        distances: Dict[str, float] = {}
        assignments = self._quadrotor_candidate_assignments()
        for agent_id in self.agents:
            if not agent_id.startswith("quadrotor"):
                continue
            if agent_id not in assignments:
                distances[agent_id] = self.env.area_size
                continue
            uav_position = self.env.uavs[agent_id].position()
            distances[agent_id] = float(np.linalg.norm(assignments[agent_id] - uav_position))
        return distances

    def _quadrotor_candidate_vectors(self) -> Dict[str, np.ndarray]:
        vectors: Dict[str, np.ndarray] = {}
        assignments = self._quadrotor_candidate_assignments()
        for agent_id in self.agents:
            if not agent_id.startswith("quadrotor"):
                continue
            if agent_id not in assignments:
                vectors[agent_id] = np.zeros(2, dtype=np.float64)
                continue
            uav_position = self.env.uavs[agent_id].position()
            vectors[agent_id] = assignments[agent_id] - uav_position
        return vectors

    def _quadrotor_candidate_assignments(self) -> Dict[str, np.ndarray]:
        candidates = list(self._active_candidates())
        assignments: Dict[str, np.ndarray] = {}
        for agent_id in [agent for agent in self.agents if agent.startswith("quadrotor")]:
            if not candidates:
                break
            uav_position = self.env.uavs[agent_id].position()
            distances = [float(np.linalg.norm(candidate - uav_position)) for candidate in candidates]
            selected_idx = int(np.argmin(distances))
            assignments[agent_id] = candidates.pop(selected_idx)
        return assignments

    def _active_candidates(self) -> np.ndarray:
        detected_targets = self.env.targets[self.env.target_detected & ~self.env.target_inspected]
        if len(detected_targets) > 0:
            return np.asarray(detected_targets, dtype=np.float64)
        candidates = []
        for candidate in self.env.candidate_targets:
            candidate_array = np.array(candidate, dtype=np.float64)
            if self._candidate_already_inspected(candidate_array):
                continue
            candidates.append(candidate_array)
        if not candidates:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(candidates, dtype=np.float64)

    def _candidate_already_inspected(self, candidate: np.ndarray) -> bool:
        inspected_targets = self.env.targets[self.env.target_inspected]
        if len(inspected_targets) == 0:
            return False
        return bool(np.min(np.linalg.norm(inspected_targets - candidate, axis=1)) <= self.env.cell_size)
