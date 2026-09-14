from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ["python_3c_capability_aware_mappo", "python_rev_reviewer_baselines"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from assignment_aware_env_wrapper import TemporalHoverMaskMAPPOEnvWrapper  # noqa: E402


class TemporalRewardRedesignMAPPOEnvWrapper(TemporalHoverMaskMAPPOEnvWrapper):
    """Temporal hover mask with reward weights retuned for from-scratch training."""

    def _fixedwing_capability_reward(self, previous_coverage: int, previous_detections: int, previous_heading: float) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        fixedwing_indices = [idx for idx, agent_id in enumerate(self.agents) if agent_id.startswith("fixedwing")]
        if not fixedwing_indices:
            return rewards
        coverage_gain = max(0, int(self.env.coverage_map.sum()) - previous_coverage)
        detection_gain = max(0, int(self.env.target_detected.sum()) - previous_detections)
        for idx in fixedwing_indices:
            rewards[idx] += 0.12 * coverage_gain / len(fixedwing_indices)
            rewards[idx] += 4.0 * detection_gain / len(fixedwing_indices)
            rewards[idx] -= 0.01
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
            action = int(actions.get(agent_id, 10))
            approach_gain = max(0.0, prev_distance - current_distance)
            retreat_loss = max(0.0, current_distance - prev_distance)
            has_candidate = prev_distance < self.env.area_size
            in_range = current_distance <= uav.sensor_range
            rewards[idx] += 0.12 * approach_gain
            rewards[idx] -= 0.04 * retreat_loss
            rewards[idx] += 1.2 * self._directional_action_reward(action, vector, in_range, has_candidate)
            if has_candidate and in_range:
                rewards[idx] += 0.8
                rewards[idx] += 18.0 if action == 4 else -0.8
            elif action == 10 and has_candidate:
                rewards[idx] -= 1.2
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
                rewards[best_agent_idx] += 40.0
        return rewards
