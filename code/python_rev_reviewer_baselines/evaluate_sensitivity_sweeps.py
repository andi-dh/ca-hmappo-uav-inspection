from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from assignment_aware_env_wrapper import AssignmentAwareMAPPOEnvWrapper
from evaluate_hierarchical_task_allocation import (
    HierarchicalTaskAllocationPolicy,
    _finalize_diagnostics,
    _new_diagnostics,
    _update_diagnostics,
)


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
    "quadrotor_invalid_hover_inspect_count",
]


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comm_radii = [float(value) for value in args.communication_radii.split(",")]
    distributions = [value.strip() for value in args.target_distributions.split(",") if value.strip()]
    rows = []
    for target_distribution in distributions:
        for communication_radius in comm_radii:
            for seed in range(args.base_seed, args.base_seed + args.num_seeds):
                env = AssignmentAwareMAPPOEnvWrapper(
                    scenario=1,
                    seed=seed,
                    hover_commit_steps=args.hover_commit_steps,
                    hover_cooldown_steps=args.hover_cooldown_steps,
                )
                env.reset(seed=seed)
                env.env.communication_radius = communication_radius
                _apply_target_distribution(env, target_distribution, seed)
                policy = _make_policy(env, args.communication_mode)
                diagnostics = _new_diagnostics()
                while True:
                    actions = policy.act()
                    if args.communication_mode == "hard":
                        _update_hard_diagnostics(env, actions, diagnostics, policy.last_assignments)
                    else:
                        _update_diagnostics(env, actions, diagnostics)
                    data = env.step(actions)
                    if bool(data["done"]):
                        break
                metrics = dict(env.compute_metrics())
                metrics.update(_finalize_diagnostics(diagnostics))
                metrics["scenario"] = 1
                metrics["seed"] = seed
                metrics["target_distribution"] = target_distribution
                metrics["communication_radius"] = communication_radius
                metrics["communication_mode"] = args.communication_mode
                metrics["baseline"] = "hierarchical_task_allocation"
                rows.append(metrics)

    suffix = "" if args.communication_mode == "diagnostic" else f"_{args.communication_mode}"
    summary_file = output_dir / f"scenario_1_sensitivity_sweep{suffix}_summary.csv"
    mean_std_file = output_dir / f"scenario_1_sensitivity_sweep{suffix}_mean_std.csv"
    _write_rows_csv(rows, summary_file)
    _write_rows_csv(_compute_group_mean_std(rows), mean_std_file)
    print(f"Sensitivity sweep completed: {len(rows)} episodes")
    print(f"Saved summary: {summary_file}")
    print(f"Saved mean/std: {mean_std_file}")


class HardCommunicationHierarchicalPolicy(HierarchicalTaskAllocationPolicy):
    """Assignment policy where disconnected quadrotors do not receive candidates."""

    def __init__(self, env: AssignmentAwareMAPPOEnvWrapper) -> None:
        super().__init__(env)
        self.last_assignments: dict[str, np.ndarray] = {}

    def act(self) -> list[int]:
        assignments = self._hard_quadrotor_candidate_assignments()
        self.last_assignments = dict(assignments)
        actions = []
        for agent_id in self.env.agents:
            if agent_id.startswith("fixedwing"):
                actions.append(self._fixedwing_action(agent_id))
            elif agent_id in assignments:
                actions.append(self._quadrotor_action(agent_id, assignments[agent_id]))
            elif self._quadrotor_connected(agent_id):
                actions.append(self._quadrotor_patrol_action(agent_id))
            else:
                actions.append(10)
        return actions

    def _hard_quadrotor_candidate_assignments(self) -> dict[str, np.ndarray]:
        candidates = list(self.env._active_candidates())
        quadrotors = [agent_id for agent_id in self.env.agents if agent_id.startswith("quadrotor") and self._quadrotor_connected(agent_id)]
        assignments: dict[str, np.ndarray] = {}
        for agent_id in quadrotors:
            if not candidates:
                break
            uav_position = self.env.env.uavs[agent_id].position()
            distances = [float(np.linalg.norm(candidate - uav_position)) for candidate in candidates]
            selected_idx = int(np.argmin(distances))
            assignments[agent_id] = candidates.pop(selected_idx)
        return assignments

    def _quadrotor_connected(self, agent_id: str) -> bool:
        uav = self.env.env.uavs[agent_id]
        radius = float(self.env.env.communication_radius)
        if float(np.linalg.norm(uav.position() - self.env.env.base)) <= radius:
            return True
        for other in self.env.env.uavs.values():
            if other.uav_type == "fixedwing" and float(np.linalg.norm(uav.position() - other.position())) <= radius:
                return True
        return False


def _make_policy(env: AssignmentAwareMAPPOEnvWrapper, communication_mode: str):
    if communication_mode == "hard":
        return HardCommunicationHierarchicalPolicy(env)
    return HierarchicalTaskAllocationPolicy(env)


def _update_hard_diagnostics(env: AssignmentAwareMAPPOEnvWrapper, actions: list[int], diagnostics: dict, assignments: dict[str, np.ndarray]) -> None:
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


def _apply_target_distribution(env: AssignmentAwareMAPPOEnvWrapper, distribution: str, seed: int) -> None:
    rng = np.random.default_rng(seed + 10_000)
    mission = env.env
    margin = mission.cell_size * 2.0
    if distribution == "uniform":
        return
    if distribution == "clustered":
        centers = np.array(
            [
                [mission.area_size * 0.35, mission.area_size * 0.35],
                [mission.area_size * 0.70, mission.area_size * 0.70],
            ],
            dtype=np.float64,
        )
        targets = []
        for idx in range(mission.n_targets):
            center = centers[idx % len(centers)]
            target = center + rng.normal(0.0, mission.fixedwing_sensor_range * 0.55, size=2)
            targets.append(np.clip(target, margin, mission.area_size - margin))
        mission.targets = np.asarray(targets, dtype=np.float64)
    elif distribution == "edge":
        targets = []
        for idx in range(mission.n_targets):
            side = idx % 4
            if side == 0:
                target = [rng.uniform(margin, mission.area_size - margin), margin]
            elif side == 1:
                target = [mission.area_size - margin, rng.uniform(margin, mission.area_size - margin)]
            elif side == 2:
                target = [rng.uniform(margin, mission.area_size - margin), mission.area_size - margin]
            else:
                target = [margin, rng.uniform(margin, mission.area_size - margin)]
            jitter = rng.normal(0.0, mission.cell_size, size=2)
            targets.append(np.clip(np.asarray(target) + jitter, margin, mission.area_size - margin))
        mission.targets = np.asarray(targets, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported target distribution: {distribution}")
    mission.target_detected = np.zeros(mission.n_targets, dtype=bool)
    mission.target_inspected = np.zeros(mission.n_targets, dtype=bool)
    mission.candidate_targets = []
    mission.target_probability_map = np.full((mission.grid_size, mission.grid_size), 0.5, dtype=np.float64)
    mission.inspection_map = np.zeros((mission.grid_size, mission.grid_size), dtype=bool)
    mission.detections_count = 0
    mission.inspections_count = 0


def _compute_group_mean_std(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float, str], list[dict]] = {}
    for row in rows:
        key = (str(row["target_distribution"]), float(row["communication_radius"]), str(row.get("communication_mode", "diagnostic")))
        grouped.setdefault(key, []).append(row)
    output = []
    for (target_distribution, communication_radius, communication_mode), group in sorted(grouped.items()):
        result = {
            "target_distribution": target_distribution,
            "communication_radius": communication_radius,
            "communication_mode": communication_mode,
            "episodes": len(group),
        }
        for col in NUMERIC_COLS:
            values = [float(row.get(col, 0.0)) for row in group]
            result[f"{col}_mean"] = mean(values) if values else 0.0
            result[f"{col}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def _write_rows_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate communication and target-distribution sensitivity.")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--communication-radii", type=str, default="400,800,1200")
    parser.add_argument("--communication-mode", type=str, default="diagnostic", choices=["diagnostic", "hard"])
    parser.add_argument("--target-distributions", type=str, default="uniform,clustered,edge")
    parser.add_argument("--hover-commit-steps", type=int, default=2)
    parser.add_argument("--hover-cooldown-steps", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="outputs/rev_reviewer_baselines")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
