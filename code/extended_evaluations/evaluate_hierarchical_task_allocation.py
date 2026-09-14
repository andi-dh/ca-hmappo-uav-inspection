from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from assignment_aware_env_wrapper import AssignmentAwareMAPPOEnvWrapper


NUMERIC_COLS = [
    "coverage_ratio",
    "inspection_success_rate",
    "target_detection_rate",
    "mission_success",
    "mission_time",
    "total_energy_consumption",
    "collision_count",
    "near_miss_count",
    "communication_loss_count",
    "invalid_action_count",
    "detected_targets",
    "inspected_targets",
    "candidate_targets",
    "quadrotor_hover_inspect_count",
    "quadrotor_valid_hover_inspect_count",
    "quadrotor_invalid_hover_inspect_count",
    "mean_distance_to_candidate",
    "time_in_candidate_range",
]


class HierarchicalTaskAllocationPolicy:
    """High-level structured assignment with low-level hand-coded execution."""

    def __init__(self, env: AssignmentAwareMAPPOEnvWrapper, lane_spacing_factor: float = 1.55, waypoint_threshold_factor: float = 0.45) -> None:
        self.env = env
        self.lane_spacing_factor = float(lane_spacing_factor)
        self.waypoint_threshold_factor = float(waypoint_threshold_factor)
        self.fixedwing_direction: dict[str, int] = {agent_id: 1 for agent_id in env.agents if agent_id.startswith("fixedwing")}
        self.fixedwing_waypoint_idx: dict[str, int] = {agent_id: 0 for agent_id in env.agents if agent_id.startswith("fixedwing")}

    def act(self) -> list[int]:
        assignments = self.env._quadrotor_candidate_assignments()
        actions = []
        for agent_id in self.env.agents:
            if agent_id.startswith("fixedwing"):
                actions.append(self._fixedwing_action(agent_id))
            elif agent_id in assignments:
                actions.append(self._quadrotor_action(agent_id, assignments[agent_id]))
            else:
                actions.append(self._quadrotor_patrol_action(agent_id))
        return actions

    def _fixedwing_action(self, agent_id: str) -> int:
        uav = self.env.env.uavs[agent_id]
        if uav.battery <= self.env.env.low_battery_threshold:
            return 10
        if len([agent for agent in self.env.agents if agent.startswith("fixedwing")]) > 1:
            next_x = uav.x + np.cos(uav.heading) * uav.speed
            next_y = uav.y + np.sin(uav.heading) * uav.speed
            margin = self.env.env.fixedwing_sensor_range * 0.5
            if next_x < margin or next_x > self.env.env.area_size - margin or next_y < margin or next_y > self.env.env.area_size - margin:
                return 7 if self.fixedwing_direction.get(agent_id, 1) > 0 else 5
            if int(self.env.env.steps) % 45 == 0:
                self.fixedwing_direction[agent_id] = -self.fixedwing_direction.get(agent_id, 1)
                return 7 if self.fixedwing_direction[agent_id] > 0 else 5
            return 6
        waypoint = self._fixedwing_waypoint(agent_id)
        if np.linalg.norm(waypoint - uav.position()) <= self.env.env.fixedwing_sensor_range * self.waypoint_threshold_factor:
            self.fixedwing_waypoint_idx[agent_id] = self.fixedwing_waypoint_idx.get(agent_id, 0) + 1
            waypoint = self._fixedwing_waypoint(agent_id)
        desired_heading = float(np.arctan2(waypoint[1] - uav.y, waypoint[0] - uav.x))
        uav.heading = desired_heading
        return 6

    def _fixedwing_waypoint(self, agent_id: str) -> np.ndarray:
        area = self.env.env.area_size
        margin = self.env.env.fixedwing_sensor_range
        lane_spacing = self.env.env.fixedwing_sensor_range * self.lane_spacing_factor
        y_values = list(np.arange(margin, area - margin + 1e-6, lane_spacing))
        if y_values[-1] < area - margin:
            y_values.append(area - margin)
        fixedwings = [agent for agent in self.env.agents if agent.startswith("fixedwing")]
        if len(fixedwings) > 1:
            agent_idx = fixedwings.index(agent_id)
            y_values = y_values[agent_idx::len(fixedwings)] or y_values
        lanes = []
        for lane_idx, y in enumerate(y_values):
            x = area - margin if lane_idx % 2 == 0 else margin
            lanes.append((x, float(y)))
        lanes.append((area * 0.5, area * 0.5))
        idx = self.fixedwing_waypoint_idx.get(agent_id, 0) % len(lanes)
        return np.array(lanes[idx], dtype=np.float64)

    def _quadrotor_action(self, agent_id: str, target: np.ndarray) -> int:
        uav = self.env.env.uavs[agent_id]
        distance = float(np.linalg.norm(target - uav.position()))
        if distance <= uav.sensor_range:
            return 4
        delta = target - uav.position()
        if abs(float(delta[0])) >= abs(float(delta[1])):
            desired_action = 2 if delta[0] > 0 else 3
        else:
            desired_action = 0 if delta[1] > 0 else 1
        return self._safe_quadrotor_action(agent_id, desired_action)

    def _quadrotor_patrol_action(self, agent_id: str) -> int:
        uav = self.env.env.uavs[agent_id]
        idx = int(agent_id.rsplit("_", 1)[-1]) if "_" in agent_id else 0
        target = np.array(
            [
                self.env.env.area_size * (0.25 + 0.5 * (idx % 2)),
                self.env.env.area_size * (0.35 + 0.2 * (idx // 2)),
            ],
            dtype=np.float64,
        )
        delta = target - uav.position()
        if np.linalg.norm(delta) <= uav.sensor_range:
            return 4 if len(self.env.env.candidate_targets) > 0 else 10
        if abs(float(delta[0])) >= abs(float(delta[1])):
            desired_action = 2 if delta[0] > 0 else 3
        else:
            desired_action = 0 if delta[1] > 0 else 1
        return self._safe_quadrotor_action(agent_id, desired_action)

    def _safe_quadrotor_action(self, agent_id: str, desired_action: int) -> int:
        for action in [desired_action, 0, 1, 2, 3, 10, 4]:
            if self._is_quadrotor_action_safe(agent_id, action):
                return action
        return 10

    def _is_quadrotor_action_safe(self, agent_id: str, action: int) -> bool:
        uav = self.env.env.uavs[agent_id]
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
            delta = self.env.env.base - uav.position()
            norm = np.linalg.norm(delta)
            if norm > 1e-9:
                next_position += delta / norm * min(uav.speed, norm)
        next_position = np.clip(next_position, 0.0, self.env.env.area_size)
        for other_id, other in self.env.env.uavs.items():
            if other_id == agent_id or not other_id.startswith("quadrotor"):
                continue
            if np.linalg.norm(next_position - other.position()) <= self.env.env.collision_radius * 1.5:
                return False
        return True


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        env = AssignmentAwareMAPPOEnvWrapper(scenario=args.scenario, seed=seed, hover_commit_steps=args.hover_commit_steps, hover_cooldown_steps=args.hover_cooldown_steps)
        policy = HierarchicalTaskAllocationPolicy(env, lane_spacing_factor=args.lane_spacing_factor, waypoint_threshold_factor=args.waypoint_threshold_factor)
        env.reset(seed=seed)
        diagnostics = _new_diagnostics()
        while True:
            actions = policy.act()
            _update_diagnostics(env, actions, diagnostics)
            data = env.step(actions)
            if bool(data["done"]):
                break
        metrics = dict(env.compute_metrics())
        metrics.update(_finalize_diagnostics(diagnostics))
        metrics["scenario"] = args.scenario
        metrics["seed"] = seed
        metrics["baseline"] = "hierarchical_task_allocation"
        metrics["evaluation_mode"] = "rule"
        rows.append(metrics)
    prefix = f"scenario_{args.scenario}_hierarchical_task_allocation"
    summary_file = output_dir / f"{prefix}_summary.csv"
    mean_std_file = output_dir / f"{prefix}_mean_std.csv"
    mean_std = _compute_mean_std(rows)
    _write_rows_csv(rows, summary_file)
    _write_rows_csv([mean_std], mean_std_file)
    print(f"Hierarchical task-allocation evaluation completed: {len(rows)} episodes")
    print(f"Saved summary: {summary_file}")
    print(f"Saved mean/std: {mean_std_file}")


def _compute_mean_std(rows: list[dict]) -> dict:
    result = {}
    for col in NUMERIC_COLS:
        values = [float(row.get(col, 0.0)) for row in rows]
        result[f"{col}_mean"] = mean(values) if values else 0.0
        result[f"{col}_std"] = stdev(values) if len(values) > 1 else 0.0
    return result


def _new_diagnostics() -> dict:
    return {"quadrotor_hover_inspect_count": 0, "quadrotor_valid_hover_inspect_count": 0, "quadrotor_invalid_hover_inspect_count": 0, "distance_sum": 0.0, "distance_count": 0, "time_in_candidate_range": 0}


def _update_diagnostics(env: AssignmentAwareMAPPOEnvWrapper, actions: list[int], diagnostics: dict) -> None:
    assignments = env._quadrotor_candidate_assignments()
    for idx, agent_id in enumerate(env.agents):
        if not agent_id.startswith("quadrotor") or agent_id not in assignments:
            continue
        uav = env.env.uavs[agent_id]
        distance = float(np.linalg.norm(assignments[agent_id] - uav.position()))
        in_range = distance <= uav.sensor_range
        diagnostics["distance_sum"] += distance
        diagnostics["distance_count"] += 1
        diagnostics["time_in_candidate_range"] += int(in_range)
        if int(actions[idx]) == 4:
            diagnostics["quadrotor_hover_inspect_count"] += 1
            diagnostics["quadrotor_valid_hover_inspect_count"] += int(in_range)
            diagnostics["quadrotor_invalid_hover_inspect_count"] += int(not in_range)


def _finalize_diagnostics(diagnostics: dict) -> dict:
    return {
        "quadrotor_hover_inspect_count": int(diagnostics["quadrotor_hover_inspect_count"]),
        "quadrotor_valid_hover_inspect_count": int(diagnostics["quadrotor_valid_hover_inspect_count"]),
        "quadrotor_invalid_hover_inspect_count": int(diagnostics["quadrotor_invalid_hover_inspect_count"]),
        "mean_distance_to_candidate": float(diagnostics["distance_sum"] / max(1, int(diagnostics["distance_count"]))),
        "time_in_candidate_range": int(diagnostics["time_in_candidate_range"]),
    }


def _write_rows_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hierarchical task-allocation baseline.")
    parser.add_argument("--scenario", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--hover-commit-steps", type=int, default=2)
    parser.add_argument("--hover-cooldown-steps", type=int, default=6)
    parser.add_argument("--lane-spacing-factor", type=float, default=1.55)
    parser.add_argument("--waypoint-threshold-factor", type=float, default=0.45)
    parser.add_argument("--output-dir", type=str, default="outputs/extended_evaluations")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
