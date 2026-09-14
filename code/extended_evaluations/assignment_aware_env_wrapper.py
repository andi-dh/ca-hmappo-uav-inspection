from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ["simulation_environment", "homogeneous_mappo", "ca_hmappo"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from heterogeneous_uav_env import HeterogeneousUAVEnv, UAVState  # noqa: E402
from capability_aware_env_wrapper import CapabilityAwareMAPPOEnvWrapper  # noqa: E402


class ScalableHeterogeneousUAVEnv(HeterogeneousUAVEnv):
    """Diagnostic environment variant with multiple fixed-wing scouts."""

    def __init__(self, n_fixedwings: int = 2, **kwargs) -> None:
        self.n_fixedwings = int(n_fixedwings)
        super().__init__(**kwargs)
        self.agents = [f"fixedwing_{i}" for i in range(self.n_fixedwings)] + [f"quadrotor_{i}" for i in range(self.n_quadrotors)]

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        scalable_agents = list(self.agents)
        self.agents = ["fixedwing_0"] + [f"quadrotor_{i}" for i in range(self.n_quadrotors)]
        super().reset(seed=seed)
        self.agents = scalable_agents
        self.uavs = {}
        for i in range(self.n_fixedwings):
            agent_id = f"fixedwing_{i}"
            x = 0.0 if i % 2 == 0 else self.area_size
            y = min((i + 1) * self.near_miss_radius * 4.0, self.area_size)
            heading = 0.0 if i % 2 == 0 else np.pi
            self.uavs[agent_id] = UAVState(
                agent_id=agent_id,
                uav_type="fixedwing",
                x=x,
                y=y,
                heading=heading,
                speed=self.fixedwing_speed,
                battery=100.0,
                sensor_range=self.fixedwing_sensor_range,
                altitude_layer=2,
            )
        for i in range(self.n_quadrotors):
            agent_id = f"quadrotor_{i}"
            offset = (i + 1) * self.near_miss_radius * 1.5
            self.uavs[agent_id] = UAVState(
                agent_id=agent_id,
                uav_type="quadrotor",
                x=0.0,
                y=min(offset, self.area_size),
                heading=0.0,
                speed=self.quadrotor_speed,
                battery=100.0,
                sensor_range=self.quadrotor_sensor_range,
                altitude_layer=1,
            )
        for uav in self.uavs.values():
            uav.record()
        self.update_coverage()
        return {agent_id: self.observe(agent_id) for agent_id in self.agents}

    def detect_targets(self) -> None:
        for fixedwing in [uav for uav in self.uavs.values() if uav.uav_type == "fixedwing"]:
            distances = np.linalg.norm(self.targets - fixedwing.position(), axis=1)
            for idx, distance in enumerate(distances):
                if self.target_detected[idx] or distance > fixedwing.sensor_range:
                    continue
                if self.rng.random() <= self.p_detect_fixedwing:
                    self.target_detected[idx] = True
                    candidate = tuple(self.targets[idx].tolist())
                    self._add_candidate(candidate)
                    self.detections_count += 1
                    gx, gy = self._position_to_grid(*candidate)
                    self.target_probability_map[gx, gy] = 0.9
            if self.rng.random() <= self.p_false_alarm_fixedwing * 0.05:
                false_candidate = self._sample_near(fixedwing.position(), fixedwing.sensor_range)
                self._add_candidate(tuple(false_candidate.tolist()))
                gx, gy = self._position_to_grid(*false_candidate)
                self.target_probability_map[gx, gy] = max(self.target_probability_map[gx, gy], 0.7)

    def compute_metrics(self) -> Dict[str, float | int | str]:
        metrics = super().compute_metrics()
        fixedwing_batteries = [uav.battery for uav in self.uavs.values() if uav.uav_type == "fixedwing"]
        metrics["fixedwing_mean_final_battery"] = float(np.mean(fixedwing_batteries)) if fixedwing_batteries else 0.0
        metrics["n_fixedwings"] = self.n_fixedwings
        metrics["n_quadrotors"] = self.n_quadrotors
        metrics["n_targets"] = self.n_targets
        return metrics


class ScalableCapabilityAwareMAPPOEnvWrapper(CapabilityAwareMAPPOEnvWrapper):
    """Capability-aware wrapper extended to the larger Scenario 2 team."""

    def _make_env(self, scenario: int, seed: int | None) -> HeterogeneousUAVEnv:
        if scenario != 2:
            return super()._make_env(scenario=scenario, seed=seed)
        env = ScalableHeterogeneousUAVEnv(
            area_size=2400.0,
            grid_size=48,
            n_fixedwings=2,
            n_quadrotors=2,
            n_targets=10,
            max_steps=1200,
            seed=seed,
        )
        env.p_detect_fixedwing = 0.75
        env.p_false_alarm_fixedwing = 0.15
        env.p_inspect_quadrotor = 0.90
        return env

    def _fixedwing_capability_reward(self, previous_coverage: int, previous_detections: int, previous_heading: float) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        fixedwing_indices = [idx for idx, agent_id in enumerate(self.agents) if agent_id.startswith("fixedwing")]
        if not fixedwing_indices:
            return rewards
        coverage_gain = max(0, int(self.env.coverage_map.sum()) - previous_coverage)
        detection_gain = max(0, int(self.env.target_detected.sum()) - previous_detections)
        for idx in fixedwing_indices:
            rewards[idx] += 0.07 * coverage_gain / len(fixedwing_indices)
            rewards[idx] += 2.0 * detection_gain / len(fixedwing_indices)
            rewards[idx] -= 0.005
        return rewards


class TemporalHoverMaskMAPPOEnvWrapper(ScalableCapabilityAwareMAPPOEnvWrapper):
    """CA-HMAPPO variant that only masks premature hover-inspect actions.

    Unlike AssignmentAwareMAPPOEnvWrapper, this wrapper does not replace movement
    actions, force hover commits, or impose cooldowns. It preserves the learned
    policy and only removes hover-inspect from the action mask until a quadrotor is
    within inspection range of its soft assigned candidate.
    """

    def get_action_masks(self) -> np.ndarray:
        masks = super().get_action_masks()
        assignments = self._quadrotor_candidate_assignments()
        for idx, agent_id in enumerate(self.agents):
            if not agent_id.startswith("quadrotor"):
                continue
            target = assignments.get(agent_id)
            if target is None:
                masks[idx, 4] = 0.0
                continue
            uav = self.env.uavs[agent_id]
            if float(np.linalg.norm(target - uav.position())) > uav.sensor_range:
                masks[idx, 4] = 0.0
            if masks[idx].sum() <= 0:
                masks[idx, 10] = 1.0
        return masks.astype(np.float32)


class AssignmentAwareMAPPOEnvWrapper(CapabilityAwareMAPPOEnvWrapper):
    """Capability-aware wrapper with explicit candidate assignment and hover timing.

    The baseline CA-HMAPPO gives quadrotors guidance but still lets the policy hover
    too early. This diagnostic variant makes the candidate coordination layer
    explicit: candidates are assigned to quadrotors by a global minimum-distance
    matching and hover-inspect is only exposed when the assigned target is in range.
    """

    def __init__(
        self,
        scenario: int = 1,
        seed: int | None = None,
        hover_commit_steps: int = 2,
        hover_cooldown_steps: int = 6,
    ) -> None:
        self.hover_commit_steps = int(hover_commit_steps)
        self.hover_cooldown_steps = int(hover_cooldown_steps)
        self._hover_commit: Dict[str, int] = {}
        self._hover_cooldown: Dict[str, int] = {}
        super().__init__(scenario=scenario, seed=seed)

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray | dict | bool]:
        self._hover_commit = {}
        self._hover_cooldown = {}
        data = super().reset(seed=seed)
        for agent_id in self.agents:
            if agent_id.startswith("quadrotor"):
                self._hover_commit[agent_id] = 0
                self._hover_cooldown[agent_id] = 0
        return data

    def step(self, actions: Iterable[int] | Dict[str, int]) -> Dict[str, np.ndarray | dict | bool]:
        action_dict = self._to_action_dict(actions)
        action_dict = self._apply_hover_timing(action_dict)
        data = super().step(action_dict)
        self._tick_hover_timers(action_dict)
        data["action_mask_n"] = self.get_action_masks()
        data["state_global"] = self.get_global_state()
        return data

    def get_action_masks(self) -> np.ndarray:
        masks = super().get_action_masks()
        assignments = self._quadrotor_candidate_assignments()
        for idx, agent_id in enumerate(self.agents):
            if not agent_id.startswith("quadrotor"):
                continue
            if self._must_commit_hover(agent_id, assignments):
                masks[idx] = 0.0
                masks[idx, 4] = 1.0
                continue
            if not self._can_validly_hover(agent_id, assignments):
                masks[idx, 4] = 0.0
            if masks[idx].sum() <= 0:
                masks[idx, 10] = 1.0
        return masks.astype(np.float32)

    def _make_env(self, scenario: int, seed: int | None) -> HeterogeneousUAVEnv:
        if scenario != 2:
            return super()._make_env(scenario=scenario, seed=seed)
        env = ScalableHeterogeneousUAVEnv(
            area_size=2400.0,
            grid_size=48,
            n_fixedwings=2,
            n_quadrotors=2,
            n_targets=10,
            max_steps=1200,
            seed=seed,
        )
        env.p_detect_fixedwing = 0.75
        env.p_false_alarm_fixedwing = 0.15
        env.p_inspect_quadrotor = 0.90
        return env

    def _fixedwing_capability_reward(self, previous_coverage: int, previous_detections: int, previous_heading: float) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        coverage_gain = max(0, int(self.env.coverage_map.sum()) - previous_coverage)
        detection_gain = max(0, int(self.env.target_detected.sum()) - previous_detections)
        fixedwing_indices = [idx for idx, agent_id in enumerate(self.agents) if agent_id.startswith("fixedwing")]
        if not fixedwing_indices:
            return rewards
        for idx in fixedwing_indices:
            rewards[idx] += 0.07 * coverage_gain / len(fixedwing_indices)
            rewards[idx] += 2.0 * detection_gain / len(fixedwing_indices)
            rewards[idx] -= 0.005
        return rewards

    def _quadrotor_candidate_assignments(self) -> Dict[str, np.ndarray]:
        candidates = list(self._active_candidates())
        quadrotors = [agent_id for agent_id in self.agents if agent_id.startswith("quadrotor")]
        if not candidates or not quadrotors:
            return {}
        if len(candidates) >= len(quadrotors):
            candidate_orders = itertools.permutations(range(len(candidates)), len(quadrotors))
        else:
            candidate_orders = itertools.permutations(range(len(candidates)), len(candidates))
        best_order: tuple[int, ...] | None = None
        best_cost = float("inf")
        for order in candidate_orders:
            cost = 0.0
            for agent_id, candidate_idx in zip(quadrotors, order):
                uav_position = self.env.uavs[agent_id].position()
                cost += float(np.linalg.norm(candidates[candidate_idx] - uav_position))
            if cost < best_cost:
                best_cost = cost
                best_order = tuple(order)
        if best_order is None:
            return {}
        return {agent_id: candidates[candidate_idx] for agent_id, candidate_idx in zip(quadrotors, best_order)}

    def _apply_hover_timing(self, actions: Dict[str, int]) -> Dict[str, int]:
        assignments = self._quadrotor_candidate_assignments()
        adjusted = dict(actions)
        for agent_id in [agent for agent in self.agents if agent.startswith("quadrotor")]:
            if self._must_commit_hover(agent_id, assignments):
                adjusted[agent_id] = 4
                continue
            if int(adjusted.get(agent_id, 10)) == 4 and not self._can_validly_hover(agent_id, assignments):
                adjusted[agent_id] = self._move_toward_assignment(agent_id, assignments)
        return adjusted

    def _tick_hover_timers(self, actions: Dict[str, int]) -> None:
        assignments = self._quadrotor_candidate_assignments()
        for agent_id in [agent for agent in self.agents if agent.startswith("quadrotor")]:
            self._hover_commit[agent_id] = max(0, self._hover_commit.get(agent_id, 0) - 1)
            self._hover_cooldown[agent_id] = max(0, self._hover_cooldown.get(agent_id, 0) - 1)
            if int(actions.get(agent_id, -1)) == 4 and self._can_validly_hover(agent_id, assignments):
                self._hover_commit[agent_id] = self.hover_commit_steps
                self._hover_cooldown[agent_id] = self.hover_cooldown_steps

    def _must_commit_hover(self, agent_id: str, assignments: Dict[str, np.ndarray]) -> bool:
        return self._hover_commit.get(agent_id, 0) > 0 and self._can_validly_hover(agent_id, assignments)

    def _can_validly_hover(self, agent_id: str, assignments: Dict[str, np.ndarray]) -> bool:
        if self._hover_cooldown.get(agent_id, 0) > 0 and self._hover_commit.get(agent_id, 0) <= 0:
            return False
        target = assignments.get(agent_id)
        if target is None:
            return False
        uav = self.env.uavs[agent_id]
        return float(np.linalg.norm(target - uav.position())) <= uav.sensor_range

    def _move_toward_assignment(self, agent_id: str, assignments: Dict[str, np.ndarray]) -> int:
        target = assignments.get(agent_id)
        if target is None:
            return 10
        delta = target - self.env.uavs[agent_id].position()
        if abs(float(delta[0])) >= abs(float(delta[1])):
            return 2 if delta[0] > 0 else 3
        return 0 if delta[1] > 0 else 1


class GuidedAssignmentMAPPOEnvWrapper(AssignmentAwareMAPPOEnvWrapper):
    """Hybrid learned/controller variant for stress-testing mission completion.

    The learned policy still controls fixed-wing scouting. Quadrotor actions are
    replaced by an explicit assignment-and-inspection controller once candidates
    exist, so this variant tests whether the remaining failure is primarily the
    learned quadrotor assignment/timing layer.
    """

    def _apply_hover_timing(self, actions: Dict[str, int]) -> Dict[str, int]:
        assignments = self._quadrotor_candidate_assignments()
        adjusted = dict(actions)
        for agent_id in [agent for agent in self.agents if agent.startswith("quadrotor")]:
            if agent_id not in assignments:
                adjusted[agent_id] = 10
                continue
            if self._can_validly_hover(agent_id, assignments):
                adjusted[agent_id] = 4
            else:
                adjusted[agent_id] = self._move_toward_assignment(agent_id, assignments)
        return adjusted
