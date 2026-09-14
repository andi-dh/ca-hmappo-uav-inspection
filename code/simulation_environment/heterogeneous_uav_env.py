from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


Position = Tuple[float, float]


UNIFIED_ACTIONS = {
    0: "move_north",
    1: "move_south",
    2: "move_east",
    3: "move_west",
    4: "hover_inspect",
    5: "turn_left",
    6: "straight",
    7: "turn_right",
    8: "orbit_candidate",
    9: "change_altitude",
    10: "return_to_base",
}

FIXEDWING_VALID_ACTIONS = {5, 6, 7, 8, 9, 10}
QUADROTOR_VALID_ACTIONS = {0, 1, 2, 3, 4, 10}


@dataclass
class UAVState:
    agent_id: str
    uav_type: str
    x: float
    y: float
    heading: float = 0.0
    speed: float = 0.0
    battery: float = 100.0
    sensor_range: float = 50.0
    altitude_layer: int = 1
    trajectory: List[Position] = field(default_factory=list)

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def record(self) -> None:
        self.trajectory.append((float(self.x), float(self.y)))


class HeterogeneousUAVEnv:
    """Custom Python Gymnasium/PettingZoo-style simulator.

    The environment follows a parallel multi-agent API style:
    reset() returns observations, and step(actions) accepts a dict of actions.
    """

    metadata = {"name": "heterogeneous_uav_coverage_inspection_v0"}

    def __init__(
        self,
        area_size: float = 2000.0,
        grid_size: int = 40,
        n_quadrotors: int = 2,
        n_targets: int = 8,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        self.area_size = float(area_size)
        self.grid_size = int(grid_size)
        self.cell_size = self.area_size / self.grid_size
        self.n_quadrotors = int(n_quadrotors)
        self.n_targets = int(n_targets)
        self.max_steps = int(max_steps)
        self.base = np.array([0.0, 0.0], dtype=np.float64)
        self.communication_radius = 800.0
        self.collision_radius = 20.0
        self.near_miss_radius = 50.0

        self.fixedwing_speed = 30.0
        self.quadrotor_speed = 15.0
        self.fixedwing_sensor_range = 150.0
        self.quadrotor_sensor_range = 75.0
        self.p_detect_fixedwing = 0.70
        self.p_false_alarm_fixedwing = 0.20
        self.p_inspect_quadrotor = 0.90
        self.p_false_alarm_quadrotor = 0.05
        self.low_battery_threshold = 15.0

        self.fixedwing_move_cost = 0.05
        self.fixedwing_turn_cost = 0.10
        self.quadrotor_move_cost = 0.10
        self.quadrotor_hover_inspect_cost = 0.25

        self.agents = ["fixedwing_0"] + [f"quadrotor_{i}" for i in range(self.n_quadrotors)]
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.uavs: Dict[str, UAVState] = {}
        self.targets: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self.target_detected: np.ndarray = np.zeros(0, dtype=bool)
        self.target_inspected: np.ndarray = np.zeros(0, dtype=bool)
        self.candidate_targets: List[Position] = []
        self.coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.target_probability_map = np.full((self.grid_size, self.grid_size), 0.5, dtype=np.float64)
        self.inspection_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.steps = 0
        self.done = False
        self.termination_reason = ""
        self.total_energy_consumption = 0.0
        self.collision_count = 0
        self.near_miss_count = 0
        self.communication_loss_count = 0
        self.invalid_action_count = 0
        self.detections_count = 0
        self.inspections_count = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed

        self.steps = 0
        self.done = False
        self.termination_reason = ""
        self.total_energy_consumption = 0.0
        self.collision_count = 0
        self.near_miss_count = 0
        self.communication_loss_count = 0
        self.invalid_action_count = 0
        self.detections_count = 0
        self.inspections_count = 0
        self.candidate_targets = []
        self.coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.target_probability_map = np.full((self.grid_size, self.grid_size), 0.5, dtype=np.float64)
        self.inspection_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        self.targets = self.rng.uniform(
            low=self.cell_size * 2,
            high=self.area_size - self.cell_size * 2,
            size=(self.n_targets, 2),
        )
        self.target_detected = np.zeros(self.n_targets, dtype=bool)
        self.target_inspected = np.zeros(self.n_targets, dtype=bool)

        self.uavs = {
            "fixedwing_0": UAVState(
                agent_id="fixedwing_0",
                uav_type="fixedwing",
                x=0.0,
                y=0.0,
                heading=0.0,
                speed=self.fixedwing_speed,
                battery=100.0,
                sensor_range=self.fixedwing_sensor_range,
                altitude_layer=2,
            )
        }
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

    def step(self, actions: Dict[str, int]):
        if self.done:
            raise RuntimeError("Cannot call step() after episode termination. Call reset() first.")

        self.steps += 1
        previous_coverage = int(self.coverage_map.sum())
        previous_detections = int(self.target_detected.sum())
        previous_inspections = int(self.target_inspected.sum())
        previous_energy = float(self.total_energy_consumption)
        previous_collision_count = int(self.collision_count)
        previous_near_miss_count = int(self.near_miss_count)
        previous_communication_loss_count = int(self.communication_loss_count)
        previous_invalid_action_count = int(self.invalid_action_count)

        for agent_id in self.agents:
            if self.uavs[agent_id].battery <= 0.0:
                continue
            action = int(actions.get(agent_id, 6 if agent_id.startswith("fixedwing") else 4))
            mask = self.get_action_mask(agent_id)
            if action < 0 or action >= len(mask) or mask[action] == 0:
                self.invalid_action_count += 1
                action = self._fallback_action(agent_id)
            self._apply_action(agent_id, action)

        self._clip_positions()
        self.update_coverage()
        self.detect_targets()
        self.inspect_targets(actions)
        self._update_safety_counts()
        self._update_communication_loss()

        reward = self.compute_reward(
            previous_coverage=previous_coverage,
            previous_detections=previous_detections,
            previous_inspections=previous_inspections,
            previous_energy=previous_energy,
            previous_collision_count=previous_collision_count,
            previous_near_miss_count=previous_near_miss_count,
            previous_communication_loss_count=previous_communication_loss_count,
            previous_invalid_action_count=previous_invalid_action_count,
        )
        self.check_termination()
        observations = {agent_id: self.observe(agent_id) for agent_id in self.agents}
        rewards = {agent_id: reward for agent_id in self.agents}
        terminations = {agent_id: self.done for agent_id in self.agents}
        truncations = {agent_id: self.steps >= self.max_steps for agent_id in self.agents}
        infos = {agent_id: {"action_mask": self.get_action_mask(agent_id)} for agent_id in self.agents}
        infos["__all__"] = self.compute_metrics()
        return observations, rewards, terminations, truncations, infos

    def observe(self, agent_id: str) -> np.ndarray:
        uav = self.uavs[agent_id]
        nearest_candidate = self._distance_to_nearest_candidate(uav.position())
        nearest_neighbor = self._distance_to_nearest_neighbor(agent_id)
        communication_status = self._has_communication(agent_id)
        local_cov = self._local_patch_mean(self.coverage_map.astype(float), uav.x, uav.y)
        local_prob = self._local_patch_mean(self.target_probability_map, uav.x, uav.y)
        uav_type_code = 0.0 if uav.uav_type == "fixedwing" else 1.0
        return np.array(
            [
                uav.x / self.area_size,
                uav.y / self.area_size,
                np.cos(uav.heading),
                np.sin(uav.heading),
                uav.battery / 100.0,
                uav_type_code,
                local_cov,
                local_prob,
                nearest_candidate / self.area_size,
                np.linalg.norm(uav.position() - self.base) / self.area_size,
                nearest_neighbor / self.area_size,
                float(communication_status),
            ],
            dtype=np.float32,
        )

    def get_action_mask(self, agent_id: str) -> np.ndarray:
        uav = self.uavs[agent_id]
        mask = np.zeros(len(UNIFIED_ACTIONS), dtype=np.int8)
        if uav.battery <= 0.0:
            return mask
        valid_actions = FIXEDWING_VALID_ACTIONS if uav.uav_type == "fixedwing" else QUADROTOR_VALID_ACTIONS
        for action in valid_actions:
            mask[action] = 1
        return mask

    def update_coverage(self) -> None:
        for uav in self.uavs.values():
            cells = self._cells_within_range(uav.x, uav.y, uav.sensor_range)
            for gx, gy in cells:
                self.coverage_map[gx, gy] = True

    def detect_targets(self) -> None:
        fixedwing = self.uavs["fixedwing_0"]
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

    def inspect_targets(self, actions: Dict[str, int]) -> None:
        for agent_id in self.agents:
            if not agent_id.startswith("quadrotor"):
                continue
            action = int(actions.get(agent_id, 4))
            if action != 4:
                continue
            uav = self.uavs[agent_id]
            distances = np.linalg.norm(self.targets - uav.position(), axis=1)
            candidate_indices = np.where((distances <= uav.sensor_range) & (~self.target_inspected))[0]
            target_idx = int(candidate_indices[np.argmin(distances[candidate_indices])]) if len(candidate_indices) else -1
            if target_idx >= 0 and distances[target_idx] <= uav.sensor_range:
                self._consume_battery(uav, self.quadrotor_hover_inspect_cost)
                if self.rng.random() <= self.p_inspect_quadrotor:
                    self.target_inspected[target_idx] = True
                    self.inspections_count += 1
                    gx, gy = self._position_to_grid(*self.targets[target_idx])
                    self.inspection_map[gx, gy] = True
                    self.target_probability_map[gx, gy] = 1.0
            elif self.rng.random() <= self.p_false_alarm_quadrotor:
                self._consume_battery(uav, self.quadrotor_hover_inspect_cost)
                gx, gy = self._position_to_grid(uav.x, uav.y)
                self.inspection_map[gx, gy] = True

    def compute_reward(
        self,
        previous_coverage: int,
        previous_detections: int,
        previous_inspections: int,
        previous_energy: float,
        previous_collision_count: int,
        previous_near_miss_count: int,
        previous_communication_loss_count: int,
        previous_invalid_action_count: int,
    ) -> float:
        coverage_gain = int(self.coverage_map.sum()) - previous_coverage
        detection_gain = int(self.target_detected.sum()) - previous_detections
        inspection_gain = int(self.target_inspected.sum()) - previous_inspections
        energy_step = float(self.total_energy_consumption - previous_energy)
        collision_step = int(self.collision_count - previous_collision_count)
        near_miss_step = int(self.near_miss_count - previous_near_miss_count)
        communication_loss_step = int(self.communication_loss_count - previous_communication_loss_count)
        invalid_action_step = int(self.invalid_action_count - previous_invalid_action_count)
        reward = 0.1 * coverage_gain
        reward += 25.0 * inspection_gain
        reward += 3.0 * detection_gain
        reward -= 0.05 * energy_step
        reward -= 5.0 * collision_step
        reward -= 0.5 * near_miss_step
        reward -= 0.02 * communication_loss_step
        reward -= 2.0 * invalid_action_step
        reward -= 0.01
        return float(reward)

    def check_termination(self) -> bool:
        if bool(np.all(self.target_inspected)):
            self.done = True
            self.termination_reason = "all_targets_inspected"
        elif self.steps >= self.max_steps:
            self.done = True
            self.termination_reason = "max_steps_reached"
        elif all(np.linalg.norm(uav.position() - self.base) <= self.cell_size for uav in self.uavs.values() if uav.battery > 0):
            if self.steps > 10 and any(uav.battery < self.low_battery_threshold for uav in self.uavs.values()):
                self.done = True
                self.termination_reason = "all_available_uavs_returned"
        elif all(uav.battery <= 0.0 for uav in self.uavs.values()):
            self.done = True
            self.termination_reason = "all_batteries_depleted"
        elif self.collision_count > 0:
            self.done = True
            self.termination_reason = "collision"
        return self.done

    def compute_metrics(self) -> Dict[str, float | int | str]:
        coverage_ratio = float(self.coverage_map.mean())
        inspection_success_rate = float(self.target_inspected.mean()) if self.n_targets else 0.0
        target_detection_rate = float(self.target_detected.mean()) if self.n_targets else 0.0
        quadrotor_batteries = [uav.battery for uav in self.uavs.values() if uav.uav_type == "quadrotor"]
        redundant_denominator = max(1, self.steps * len(self.agents))
        return {
            "coverage_ratio": coverage_ratio,
            "inspection_success_rate": inspection_success_rate,
            "target_detection_rate": target_detection_rate,
            "mission_success": int(bool(np.all(self.target_inspected))),
            "mission_time": self.steps,
            "total_energy_consumption": float(self.total_energy_consumption),
            "fixedwing_final_battery": float(self.uavs["fixedwing_0"].battery),
            "quadrotor_mean_final_battery": float(np.mean(quadrotor_batteries)) if quadrotor_batteries else 0.0,
            "quadrotor_min_final_battery": float(np.min(quadrotor_batteries)) if quadrotor_batteries else 0.0,
            "collision_count": int(self.collision_count),
            "near_miss_count": int(self.near_miss_count),
            "communication_loss_count": int(self.communication_loss_count),
            "invalid_action_count": int(self.invalid_action_count),
            "detected_targets": int(self.target_detected.sum()),
            "inspected_targets": int(self.target_inspected.sum()),
            "candidate_targets": int(len(self.candidate_targets)),
            "redundant_coverage_proxy": float(max(0, redundant_denominator - self.coverage_map.sum()) / redundant_denominator),
            "termination_reason": self.termination_reason,
        }

    def render(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(
            self.coverage_map.T,
            origin="lower",
            extent=[0, self.area_size, 0, self.area_size],
            cmap="Blues",
            alpha=0.45,
        )
        ax.scatter(self.targets[:, 0], self.targets[:, 1], c="gold", marker="*", s=120, label="True targets")
        inspected = self.targets[self.target_inspected]
        if len(inspected):
            ax.scatter(inspected[:, 0], inspected[:, 1], facecolors="none", edgecolors="green", s=180, label="Inspected")
        for uav in self.uavs.values():
            traj = np.array(uav.trajectory)
            if len(traj):
                ax.plot(traj[:, 0], traj[:, 1], label=uav.agent_id)
                ax.scatter(traj[-1, 0], traj[-1, 1], s=45)
        ax.scatter([0], [0], c="black", marker="s", label="Base")
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_title("Heterogeneous UAV Coverage-Inspection Simulation")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper right")
        return fig, ax

    def close(self) -> None:
        return None

    def _apply_action(self, agent_id: str, action: int) -> None:
        uav = self.uavs[agent_id]
        if uav.battery <= 0.0:
            return
        if uav.uav_type == "fixedwing":
            self._apply_fixedwing_action(uav, action)
        else:
            self._apply_quadrotor_action(uav, action)
        uav.record()

    def _apply_fixedwing_action(self, uav: UAVState, action: int) -> None:
        if action == 5:
            uav.heading += np.deg2rad(25.0)
            self._consume_battery(uav, self.fixedwing_turn_cost)
        elif action == 7:
            uav.heading -= np.deg2rad(25.0)
            self._consume_battery(uav, self.fixedwing_turn_cost)
        elif action == 8:
            target = self._nearest_candidate_position(uav.position())
            if target is not None:
                direction = np.arctan2(target[1] - uav.y, target[0] - uav.x)
                uav.heading = 0.85 * uav.heading + 0.15 * direction
        elif action == 9:
            uav.altitude_layer = 1 if uav.altitude_layer == 2 else 2
        elif action == 10:
            uav.heading = np.arctan2(self.base[1] - uav.y, self.base[0] - uav.x)

        uav.x += np.cos(uav.heading) * uav.speed
        uav.y += np.sin(uav.heading) * uav.speed
        self._consume_battery(uav, self.fixedwing_move_cost)

    def _apply_quadrotor_action(self, uav: UAVState, action: int) -> None:
        if action == 0:
            uav.y += uav.speed
            self._consume_battery(uav, self.quadrotor_move_cost)
        elif action == 1:
            uav.y -= uav.speed
            self._consume_battery(uav, self.quadrotor_move_cost)
        elif action == 2:
            uav.x += uav.speed
            self._consume_battery(uav, self.quadrotor_move_cost)
        elif action == 3:
            uav.x -= uav.speed
            self._consume_battery(uav, self.quadrotor_move_cost)
        elif action == 10:
            delta = self.base - uav.position()
            norm = np.linalg.norm(delta)
            if norm > 1e-9:
                step = min(uav.speed, norm)
                uav.x += delta[0] / norm * step
                uav.y += delta[1] / norm * step
            self._consume_battery(uav, self.quadrotor_move_cost)

    def _fallback_action(self, agent_id: str) -> int:
        return 6 if self.uavs[agent_id].uav_type == "fixedwing" else 10

    def _consume_battery(self, uav: UAVState, amount: float) -> None:
        spent = min(uav.battery, amount)
        uav.battery -= spent
        self.total_energy_consumption += spent

    def _clip_positions(self) -> None:
        for uav in self.uavs.values():
            uav.x = float(np.clip(uav.x, 0.0, self.area_size))
            uav.y = float(np.clip(uav.y, 0.0, self.area_size))

    def _position_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(np.clip(x / self.cell_size, 0, self.grid_size - 1))
        gy = int(np.clip(y / self.cell_size, 0, self.grid_size - 1))
        return gx, gy

    def _cells_within_range(self, x: float, y: float, radius: float) -> List[Tuple[int, int]]:
        gx, gy = self._position_to_grid(x, y)
        cell_radius = int(np.ceil(radius / self.cell_size))
        cells = []
        for ix in range(max(0, gx - cell_radius), min(self.grid_size, gx + cell_radius + 1)):
            for iy in range(max(0, gy - cell_radius), min(self.grid_size, gy + cell_radius + 1)):
                cx = (ix + 0.5) * self.cell_size
                cy = (iy + 0.5) * self.cell_size
                if np.hypot(cx - x, cy - y) <= radius:
                    cells.append((ix, iy))
        return cells

    def _local_patch_mean(self, data: np.ndarray, x: float, y: float, patch_radius: int = 2) -> float:
        gx, gy = self._position_to_grid(x, y)
        patch = data[
            max(0, gx - patch_radius) : min(self.grid_size, gx + patch_radius + 1),
            max(0, gy - patch_radius) : min(self.grid_size, gy + patch_radius + 1),
        ]
        return float(np.mean(patch)) if patch.size else 0.0

    def _nearest_candidate_position(self, position: np.ndarray) -> Optional[np.ndarray]:
        if not self.candidate_targets:
            return None
        candidates = np.array(self.candidate_targets, dtype=np.float64)
        return candidates[int(np.argmin(np.linalg.norm(candidates - position, axis=1)))]

    def _distance_to_nearest_candidate(self, position: np.ndarray) -> float:
        candidate = self._nearest_candidate_position(position)
        if candidate is None:
            return self.area_size
        return float(np.linalg.norm(candidate - position))

    def _distance_to_nearest_neighbor(self, agent_id: str) -> float:
        position = self.uavs[agent_id].position()
        distances = [np.linalg.norm(position - uav.position()) for other_id, uav in self.uavs.items() if other_id != agent_id]
        return float(min(distances)) if distances else self.area_size

    def _has_communication(self, agent_id: str) -> bool:
        position = self.uavs[agent_id].position()
        if np.linalg.norm(position - self.base) <= self.communication_radius:
            return True
        return any(
            np.linalg.norm(position - other.position()) <= self.communication_radius
            for other_id, other in self.uavs.items()
            if other_id != agent_id
        )

    def _update_safety_counts(self) -> None:
        ids = list(self.uavs)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if self.uavs[ids[i]].altitude_layer != self.uavs[ids[j]].altitude_layer:
                    continue
                if (
                    np.linalg.norm(self.uavs[ids[i]].position() - self.base) <= self.cell_size
                    and np.linalg.norm(self.uavs[ids[j]].position() - self.base) <= self.cell_size
                ):
                    continue
                distance = np.linalg.norm(self.uavs[ids[i]].position() - self.uavs[ids[j]].position())
                if distance <= self.collision_radius:
                    self.collision_count += 1
                elif distance <= self.near_miss_radius:
                    self.near_miss_count += 1

    def _update_communication_loss(self) -> None:
        for agent_id in self.agents:
            if not self._has_communication(agent_id):
                self.communication_loss_count += 1

    def _sample_near(self, position: np.ndarray, radius: float) -> np.ndarray:
        angle = self.rng.uniform(0, 2 * np.pi)
        distance = self.rng.uniform(0, radius)
        sample = position + np.array([np.cos(angle), np.sin(angle)]) * distance
        return np.clip(sample, 0.0, self.area_size)

    def _add_candidate(self, candidate: Position) -> None:
        candidate_array = np.array(candidate, dtype=np.float64)
        for existing in self.candidate_targets:
            if np.linalg.norm(candidate_array - np.array(existing, dtype=np.float64)) <= self.cell_size:
                return
        self.candidate_targets.append((float(candidate[0]), float(candidate[1])))


class RuleBasedHeterogeneousPolicy:
    """Rule-based heterogeneous swarm baseline."""

    def __init__(self, env: HeterogeneousUAVEnv) -> None:
        self.env = env
        self.fixedwing_direction = 1
        self.lane_spacing = env.fixedwing_sensor_range * 1.5
        self.assigned_targets: Dict[str, Optional[Position]] = {agent_id: None for agent_id in env.agents if agent_id.startswith("quadrotor")}

    def act(self) -> Dict[str, int]:
        actions: Dict[str, int] = {"fixedwing_0": self._fixedwing_action()}
        self._assign_quadrotor_targets()
        for agent_id in self.assigned_targets:
            actions[agent_id] = self._quadrotor_action(agent_id)
        return actions

    def _fixedwing_action(self) -> int:
        fixedwing = self.env.uavs["fixedwing_0"]
        if fixedwing.battery <= self.env.low_battery_threshold:
            return 10
        if self.env.candidate_targets and self.env.rng.random() < 0.15:
            return 8
        next_x = fixedwing.x + np.cos(fixedwing.heading) * fixedwing.speed
        next_y = fixedwing.y + np.sin(fixedwing.heading) * fixedwing.speed
        if next_x >= self.env.area_size - self.env.cell_size or next_y >= self.env.area_size - self.env.cell_size:
            return 5
        if next_x <= self.env.cell_size or next_y <= self.env.cell_size:
            return 7
        return 6

    def _assign_quadrotor_targets(self) -> None:
        assigned = {target for target in self.assigned_targets.values() if target is not None}
        available = [
            target
            for target in self.env.candidate_targets
            if target not in assigned and not self._candidate_already_inspected(target)
        ]
        for agent_id, current in list(self.assigned_targets.items()):
            if current is not None:
                continue
            if not available:
                continue
            uav = self.env.uavs[agent_id]
            candidates = np.array(available, dtype=np.float64)
            idx = int(np.argmin(np.linalg.norm(candidates - uav.position(), axis=1)))
            target = tuple(candidates[idx].tolist())
            self.assigned_targets[agent_id] = target
            available.remove(target)

    def _quadrotor_action(self, agent_id: str) -> int:
        uav = self.env.uavs[agent_id]
        if uav.battery <= self.env.low_battery_threshold:
            return 10
        avoidance_action = self._quadrotor_collision_avoidance(agent_id)
        if avoidance_action is not None:
            return avoidance_action
        target = self.assigned_targets.get(agent_id)
        if target is None:
            return 4
        target_position = np.array(target, dtype=np.float64)
        distance = np.linalg.norm(target_position - uav.position())
        if distance <= uav.sensor_range:
            self.assigned_targets[agent_id] = None
            return 4
        dx = target_position[0] - uav.x
        dy = target_position[1] - uav.y
        if abs(dx) >= abs(dy):
            desired_action = 2 if dx > 0 else 3
        else:
            desired_action = 0 if dy > 0 else 1
        return self._safe_quadrotor_action(agent_id, desired_action)

    def _candidate_already_inspected(self, candidate: Position) -> bool:
        inspected_targets = self.env.targets[self.env.target_inspected]
        if len(inspected_targets) == 0:
            return False
        distances = np.linalg.norm(inspected_targets - np.array(candidate, dtype=np.float64), axis=1)
        return bool(np.min(distances) <= self.env.cell_size)

    def _quadrotor_collision_avoidance(self, agent_id: str) -> Optional[int]:
        uav = self.env.uavs[agent_id]
        for other_id, other in self.env.uavs.items():
            if other_id == agent_id or other.uav_type != "quadrotor":
                continue
            distance = np.linalg.norm(uav.position() - other.position())
            if distance > self.env.near_miss_radius:
                continue
            if uav.x <= other.x and uav.x > self.env.cell_size:
                return 3
            if uav.x > other.x and uav.x < self.env.area_size - self.env.cell_size:
                return 2
            if uav.y <= other.y and uav.y > self.env.cell_size:
                return 1
            return 0
        return None

    def _safe_quadrotor_action(self, agent_id: str, desired_action: int) -> int:
        for action in [desired_action, 0, 1, 2, 3, 4, 10]:
            if self._is_quadrotor_action_safe(agent_id, action):
                return action
        return 4

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
                step = min(uav.speed, norm)
                next_position += delta / norm * step
        next_position = np.clip(next_position, 0.0, self.env.area_size)
        for other_id, other in self.env.uavs.items():
            if other_id == agent_id or other.uav_type != "quadrotor":
                continue
            if np.linalg.norm(next_position - other.position()) <= self.env.near_miss_radius:
                return False
        return True


class RuleBasedHeterogeneousPolicyV2(RuleBasedHeterogeneousPolicy):
    """Stronger rule-based baseline for fairer comparison.

    Fixed-wing focuses on coverage and rarely orbits. Quadrotors use direct vector
    movement to assigned candidates and retry hover inspection for several steps.
    """

    def __init__(self, env: HeterogeneousUAVEnv, inspect_retry_steps: int = 5) -> None:
        super().__init__(env)
        self.inspect_retry_steps = inspect_retry_steps
        self.inspect_attempts: Dict[str, int] = {agent_id: 0 for agent_id in self.assigned_targets}

    def _fixedwing_action(self) -> int:
        fixedwing = self.env.uavs["fixedwing_0"]
        if fixedwing.battery <= self.env.low_battery_threshold:
            return 10
        if self.env.candidate_targets and self.env.rng.random() < 0.03:
            return 8
        next_x = fixedwing.x + np.cos(fixedwing.heading) * fixedwing.speed
        next_y = fixedwing.y + np.sin(fixedwing.heading) * fixedwing.speed
        if next_x >= self.env.area_size - self.env.cell_size or next_y >= self.env.area_size - self.env.cell_size:
            return 5
        if next_x <= self.env.cell_size or next_y <= self.env.cell_size:
            return 7
        return 6

    def _assign_quadrotor_targets(self) -> None:
        assigned = {target for target in self.assigned_targets.values() if target is not None}
        available = [
            target
            for target in self.env.candidate_targets
            if target not in assigned and not self._candidate_already_inspected(target)
        ]
        for agent_id, current in list(self.assigned_targets.items()):
            if current is not None or not available:
                continue
            uav = self.env.uavs[agent_id]
            candidates = np.array(available, dtype=np.float64)
            distances = np.linalg.norm(candidates - uav.position(), axis=1)
            idx = int(np.argmin(distances))
            target = tuple(candidates[idx].tolist())
            self.assigned_targets[agent_id] = target
            self.inspect_attempts[agent_id] = 0
            available.remove(target)

    def _quadrotor_action(self, agent_id: str) -> int:
        uav = self.env.uavs[agent_id]
        if uav.battery <= self.env.low_battery_threshold:
            return 10
        avoidance_action = self._quadrotor_collision_avoidance(agent_id)
        if avoidance_action is not None:
            return avoidance_action
        target = self.assigned_targets.get(agent_id)
        if target is None:
            return 4

        target_position = np.array(target, dtype=np.float64)
        distance = np.linalg.norm(target_position - uav.position())
        if distance <= uav.sensor_range:
            self.inspect_attempts[agent_id] += 1
            if self.inspect_attempts[agent_id] > self.inspect_retry_steps:
                self.assigned_targets[agent_id] = None
                self.inspect_attempts[agent_id] = 0
            return 4

        self.inspect_attempts[agent_id] = 0
        self._move_quadrotor_direct(uav, target_position)
        return 4

    def _move_quadrotor_direct(self, uav: UAVState, target_position: np.ndarray) -> None:
        if uav.battery <= 0.0:
            return
        delta = target_position - uav.position()
        norm = np.linalg.norm(delta)
        if norm <= 1e-9:
            return
        step = min(uav.speed, norm)
        uav.x += delta[0] / norm * step
        uav.y += delta[1] / norm * step
        uav.x = float(np.clip(uav.x, 0.0, self.env.area_size))
        uav.y = float(np.clip(uav.y, 0.0, self.env.area_size))
        self.env._consume_battery(uav, self.env.quadrotor_move_cost)
        uav.record()
