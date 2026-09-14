from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
for folder in ["heterogeneous_mappo", "ca_hmappo"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from capability_aware_env_wrapper import CapabilityAwareMAPPOEnvWrapper  # noqa: E402
from assignment_aware_env_wrapper import AssignmentAwareMAPPOEnvWrapper, GuidedAssignmentMAPPOEnvWrapper, ScalableCapabilityAwareMAPPOEnvWrapper, TemporalHoverMaskMAPPOEnvWrapper  # noqa: E402


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


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    env_mode = args.env_mode or checkpoint.get("env_mode", "capability_aware")
    algo = args.algo or checkpoint.get("algo", "happo")
    scenario = args.scenario if args.scenario is not None else int(checkpoint.get("scenario", 1))
    actor = HeterogeneousActor(checkpoint["obs_dim"], checkpoint["action_dim"], hidden_dim=args.hidden_dim).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    rows = []
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        torch.manual_seed(seed)
        env = _make_env(env_mode, scenario, seed, args.hover_commit_steps, args.hover_cooldown_steps)
        data = env.reset(seed=seed)
        diagnostics = _new_diagnostics()
        while True:
            obs = torch.as_tensor(data["obs_n"], dtype=torch.float32, device=device)
            masks = torch.as_tensor(data["action_mask_n"], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = actor(obs, masks)
                actions = torch.argmax(logits, dim=-1).cpu().numpy() if args.deterministic else Categorical(logits=logits).sample().cpu().numpy()
            _update_diagnostics(env, actions, diagnostics)
            data = env.step(actions)
            if bool(data["done"]):
                break
        metrics = dict(env.compute_metrics())
        metrics.update(_finalize_diagnostics(diagnostics))
        metrics["scenario"] = scenario
        metrics["seed"] = seed
        metrics["baseline"] = f"{algo}_{env_mode}"
        metrics["evaluation_mode"] = "deterministic" if args.deterministic else "sampling"
        rows.append(metrics)
    mode = "deterministic" if args.deterministic else "sampling"
    prefix = f"scenario_{scenario}_{algo}_{env_mode}_{mode}"
    summary_file = output_dir / f"{prefix}_summary.csv"
    mean_std_file = output_dir / f"{prefix}_mean_std.csv"
    mean_std = _compute_mean_std(rows)
    _write_rows_csv(rows, summary_file)
    _write_rows_csv([mean_std], mean_std_file)
    print(f"{algo.upper()} evaluation completed: {len(rows)} episodes")
    print(f"Saved summary: {summary_file}")
    print(f"Saved mean/std: {mean_std_file}")
    for key, value in mean_std.items():
        print(f"{key}: {value}")


def _make_env(env_mode: str, scenario: int, seed: int, hover_commit_steps: int, hover_cooldown_steps: int):
    if env_mode == "assignment_aware":
        return AssignmentAwareMAPPOEnvWrapper(scenario=scenario, seed=seed, hover_commit_steps=hover_commit_steps, hover_cooldown_steps=hover_cooldown_steps)
    if env_mode == "guided_assignment":
        return GuidedAssignmentMAPPOEnvWrapper(scenario=scenario, seed=seed, hover_commit_steps=hover_commit_steps, hover_cooldown_steps=hover_cooldown_steps)
    if env_mode == "temporal_hover_mask":
        return TemporalHoverMaskMAPPOEnvWrapper(scenario=scenario, seed=seed)
    if env_mode == "scalable_capability_aware":
        return ScalableCapabilityAwareMAPPOEnvWrapper(scenario=scenario, seed=seed)
    return CapabilityAwareMAPPOEnvWrapper(scenario=scenario, seed=seed)


def _compute_mean_std(rows: list[dict]) -> dict:
    result = {}
    for col in NUMERIC_COLS:
        values = [float(row.get(col, 0.0)) for row in rows]
        result[f"{col}_mean"] = mean(values) if values else 0.0
        result[f"{col}_std"] = stdev(values) if len(values) > 1 else 0.0
    return result


def _new_diagnostics() -> dict:
    return {"quadrotor_hover_inspect_count": 0, "quadrotor_valid_hover_inspect_count": 0, "quadrotor_invalid_hover_inspect_count": 0, "distance_sum": 0.0, "distance_count": 0, "time_in_candidate_range": 0}


def _update_diagnostics(env, actions, diagnostics: dict) -> None:
    assignments = env._quadrotor_candidate_assignments() if hasattr(env, "_quadrotor_candidate_assignments") else {}
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
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HAPPO-inspired sequential-update baseline.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--scenario", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--env-mode", type=str, default="", choices=["", "capability_aware", "assignment_aware", "guided_assignment", "temporal_hover_mask", "scalable_capability_aware"])
    parser.add_argument("--algo", type=str, default="", choices=["", "happo", "hatrpo", "assignment_constrained_ca_hmappo"])
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--hover-commit-steps", type=int, default=2)
    parser.add_argument("--hover-cooldown-steps", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="outputs/extended_evaluations")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
