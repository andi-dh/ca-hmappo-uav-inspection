from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

BASELINE_DIR = Path(__file__).resolve().parents[1] / "simulation_environment"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from heterogeneous_uav_env import HeterogeneousUAVEnv  # noqa: E402


class MAPPOEnvWrapper:
    """MAPPO-friendly wrapper around the custom heterogeneous UAV simulator."""

    def __init__(self, scenario: int = 0, seed: int | None = None) -> None:
        self.scenario = scenario
        self.env = self._make_env(scenario=scenario, seed=seed)
        self.agents = list(self.env.agents)
        self.n_agents = len(self.agents)
        self.obs_dim = 12
        self.action_dim = 11
        self.global_state_dim = self.obs_dim * self.n_agents + 4

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        observations = self.env.reset(seed=seed)
        if self.scenario == 0:
            self._set_easy_targets()
        return self._format_output(observations)

    def step(self, actions: Iterable[int] | Dict[str, int]) -> Dict[str, np.ndarray | dict | bool]:
        action_dict = self._to_action_dict(actions)
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        done = bool(all(terminations.values()) or all(truncations.values()))
        output = self._format_output(observations)
        output["reward_n"] = np.array([rewards[agent_id] for agent_id in self.agents], dtype=np.float32)
        output["done_n"] = np.array([done] * self.n_agents, dtype=bool)
        output["done"] = done
        output["info"] = infos["__all__"]
        return output

    def get_global_state(self) -> np.ndarray:
        obs = np.concatenate([self.env.observe(agent_id) for agent_id in self.agents]).astype(np.float32)
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
        return np.concatenate([obs, extras]).astype(np.float32)

    def get_action_masks(self) -> np.ndarray:
        return np.stack([self.env.get_action_mask(agent_id) for agent_id in self.agents]).astype(np.float32)

    def compute_metrics(self) -> dict:
        return self.env.compute_metrics()

    def close(self) -> None:
        self.env.close()

    def _format_output(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | dict | bool]:
        return {
            "obs_n": np.stack([observations[agent_id] for agent_id in self.agents]).astype(np.float32),
            "state_global": self.get_global_state(),
            "action_mask_n": self.get_action_masks(),
            "reward_n": np.zeros(self.n_agents, dtype=np.float32),
            "done_n": np.zeros(self.n_agents, dtype=bool),
            "done": False,
            "info": self.env.compute_metrics(),
        }

    def _to_action_dict(self, actions: Iterable[int] | Dict[str, int]) -> Dict[str, int]:
        if isinstance(actions, dict):
            return {agent_id: int(actions[agent_id]) for agent_id in self.agents}
        return {agent_id: int(action) for agent_id, action in zip(self.agents, actions)}

    def _make_env(self, scenario: int, seed: int | None) -> HeterogeneousUAVEnv:
        if scenario == 0:
            env = HeterogeneousUAVEnv(
                area_size=1000.0,
                grid_size=20,
                n_quadrotors=1,
                n_targets=2,
                max_steps=700,
                seed=seed,
            )
            env.quadrotor_sensor_range = 100.0
            env.p_detect_fixedwing = 1.0
            env.p_false_alarm_fixedwing = 0.0
            env.p_inspect_quadrotor = 0.95
            env.p_false_alarm_quadrotor = 0.0
            return env
        return HeterogeneousUAVEnv(
            area_size=2000.0,
            grid_size=40,
            n_quadrotors=2,
            n_targets=8,
            max_steps=1000,
            seed=seed,
        )

    def _set_easy_targets(self) -> None:
        self.env.targets = np.array([[120.0, 75.0], [220.0, 75.0]], dtype=np.float64)
        self.env.target_detected = np.zeros(self.env.n_targets, dtype=bool)
        self.env.target_inspected = np.zeros(self.env.n_targets, dtype=bool)
        self.env.candidate_targets = []
