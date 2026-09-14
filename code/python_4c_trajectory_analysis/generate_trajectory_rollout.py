from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
for folder in ["python_3a_homogeneous_mappo", "python_3b_heterogeneous_mappo", "python_3c_capability_aware_mappo", "python_baseline"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from capability_aware_env_wrapper import CapabilityAwareMAPPOEnvWrapper  # noqa: E402
from heterogeneous_uav_env import UNIFIED_ACTIONS  # noqa: E402


def generate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.model, map_location=device)
    actor = HeterogeneousActor(checkpoint["obs_dim"], checkpoint["action_dim"], hidden_dim=args.hidden_dim).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = CapabilityAwareMAPPOEnvWrapper(scenario=args.scenario, seed=args.seed)
    data = env.reset(seed=args.seed)
    rollout_rows: list[dict] = []
    event_rows: list[dict] = []
    diagnostics = _new_diagnostics()
    previous_detected = env.env.target_detected.copy()
    previous_inspected = env.env.target_inspected.copy()
    previous_collision = int(env.env.collision_count)
    previous_near_miss = int(env.env.near_miss_count)

    _record_targets(env, output_dir / f"targets_ca_hmappo_seed_{args.seed}.csv")
    for step in range(env.env.max_steps + 1):
        obs = torch.as_tensor(data["obs_n"], dtype=torch.float32, device=device)
        masks = torch.as_tensor(data["action_mask_n"], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = actor(obs, masks)
            actions = torch.argmax(logits, dim=-1).cpu().numpy() if args.deterministic else Categorical(logits=logits).sample().cpu().numpy()
        _record_rollout_step(env, step, actions, rollout_rows, diagnostics)
        if bool(data["done"]):
            break
        data = env.step(actions)
        _record_events(env, step + 1, previous_detected, previous_inspected, previous_collision, previous_near_miss, actions, event_rows)
        previous_detected = env.env.target_detected.copy()
        previous_inspected = env.env.target_inspected.copy()
        previous_collision = int(env.env.collision_count)
        previous_near_miss = int(env.env.near_miss_count)
        if bool(data["done"]):
            _record_rollout_step(env, step + 1, np.full(env.n_agents, -1), rollout_rows, diagnostics)
            break

    prefix = f"ca_hmappo_seed_{args.seed}"
    _write_rows_csv(rollout_rows, output_dir / f"rollout_{prefix}.csv")
    _write_rows_csv(event_rows, output_dir / f"events_{prefix}.csv")
    _write_rows_csv([_finalize_diagnostics(env, diagnostics)], output_dir / f"action_diagnostics_{prefix}.csv")
    np.save(output_dir / f"coverage_map_{prefix}.npy", env.env.coverage_map.astype(np.uint8))
    print(f"Saved rollout outputs to {output_dir}")


def _record_targets(env: CapabilityAwareMAPPOEnvWrapper, path: Path) -> None:
    rows = []
    for idx, target in enumerate(env.env.targets):
        rows.append({"target_id": idx, "x": float(target[0]), "y": float(target[1])})
    _write_rows_csv(rows, path)


def _record_rollout_step(env: CapabilityAwareMAPPOEnvWrapper, step: int, actions, rows: list[dict], diagnostics: dict) -> None:
    assignments = env._quadrotor_candidate_assignments()
    metrics = env.compute_metrics()
    for idx, agent_id in enumerate(env.agents):
        uav = env.env.uavs[agent_id]
        action = int(actions[idx]) if idx < len(actions) else -1
        distance = ""
        in_range = ""
        if agent_id in assignments:
            distance_value = float(np.linalg.norm(assignments[agent_id] - uav.position()))
            distance = distance_value
            in_range_value = distance_value <= uav.sensor_range
            in_range = int(in_range_value)
            diagnostics["distance_sum"] += distance_value
            diagnostics["distance_count"] += 1
            if in_range_value:
                diagnostics["time_in_candidate_range"] += 1
        if agent_id.startswith("quadrotor") and action == 4:
            diagnostics["quadrotor_hover_inspect_count"] += 1
            if in_range == 1:
                diagnostics["quadrotor_valid_hover_inspect_count"] += 1
            else:
                diagnostics["quadrotor_invalid_hover_inspect_count"] += 1
        rows.append(
            {
                "step": step,
                "agent_id": agent_id,
                "uav_type": uav.uav_type,
                "x": float(uav.x),
                "y": float(uav.y),
                "action": action,
                "action_name": UNIFIED_ACTIONS.get(action, "terminal"),
                "battery": float(uav.battery),
                "nearest_candidate_distance": distance,
                "in_candidate_range": in_range,
                "coverage_ratio": metrics["coverage_ratio"],
                "target_detection_rate": metrics["target_detection_rate"],
                "inspection_success_rate": metrics["inspection_success_rate"],
                "collision_count": metrics["collision_count"],
            }
        )


def _record_events(
    env: CapabilityAwareMAPPOEnvWrapper,
    step: int,
    previous_detected: np.ndarray,
    previous_inspected: np.ndarray,
    previous_collision: int,
    previous_near_miss: int,
    actions,
    rows: list[dict],
) -> None:
    for target_id in np.where((~previous_detected) & env.env.target_detected)[0]:
        target = env.env.targets[target_id]
        rows.append({"step": step, "event_type": "target_detected", "target_id": int(target_id), "x": float(target[0]), "y": float(target[1]), "agent_id": "fixedwing_0"})
    for target_id in np.where((~previous_inspected) & env.env.target_inspected)[0]:
        target = env.env.targets[target_id]
        inspector = _nearest_hovering_quadrotor(env, target, actions)
        rows.append({"step": step, "event_type": "target_inspected", "target_id": int(target_id), "x": float(target[0]), "y": float(target[1]), "agent_id": inspector})
    for _ in range(max(0, int(env.env.collision_count) - previous_collision)):
        rows.append({"step": step, "event_type": "collision", "target_id": "", "x": "", "y": "", "agent_id": ""})
    for _ in range(max(0, int(env.env.near_miss_count) - previous_near_miss)):
        rows.append({"step": step, "event_type": "near_miss", "target_id": "", "x": "", "y": "", "agent_id": ""})


def _nearest_hovering_quadrotor(env: CapabilityAwareMAPPOEnvWrapper, target: np.ndarray, actions) -> str:
    best_agent = ""
    best_distance = float("inf")
    for idx, agent_id in enumerate(env.agents):
        if not agent_id.startswith("quadrotor") or int(actions[idx]) != 4:
            continue
        distance = float(np.linalg.norm(env.env.uavs[agent_id].position() - target))
        if distance < best_distance:
            best_distance = distance
            best_agent = agent_id
    return best_agent


def _new_diagnostics() -> dict:
    return {"quadrotor_hover_inspect_count": 0, "quadrotor_valid_hover_inspect_count": 0, "quadrotor_invalid_hover_inspect_count": 0, "distance_sum": 0.0, "distance_count": 0, "time_in_candidate_range": 0}


def _finalize_diagnostics(env: CapabilityAwareMAPPOEnvWrapper, diagnostics: dict) -> dict:
    metrics = env.compute_metrics()
    distance_count = max(1, int(diagnostics["distance_count"]))
    return {
        "coverage_ratio": metrics["coverage_ratio"],
        "target_detection_rate": metrics["target_detection_rate"],
        "inspection_success_rate": metrics["inspection_success_rate"],
        "mission_success": metrics["mission_success"],
        "mission_time": metrics["mission_time"],
        "total_energy_consumption": metrics["total_energy_consumption"],
        "collision_count": metrics["collision_count"],
        "near_miss_count": metrics["near_miss_count"],
        "detected_targets": metrics["detected_targets"],
        "inspected_targets": metrics["inspected_targets"],
        "termination_reason": metrics["termination_reason"],
        "quadrotor_hover_inspect_count": int(diagnostics["quadrotor_hover_inspect_count"]),
        "quadrotor_valid_hover_inspect_count": int(diagnostics["quadrotor_valid_hover_inspect_count"]),
        "quadrotor_invalid_hover_inspect_count": int(diagnostics["quadrotor_invalid_hover_inspect_count"]),
        "mean_distance_to_candidate": float(diagnostics["distance_sum"] / distance_count),
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
    parser = argparse.ArgumentParser(description="Generate CA-HMAPPO trajectory rollout logs.")
    parser.add_argument("--scenario", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--model", type=str, default="models/3c_capability_aware_mappo/capability_aware_mappo_scenario_1.pt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs/4c_trajectory_analysis")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
