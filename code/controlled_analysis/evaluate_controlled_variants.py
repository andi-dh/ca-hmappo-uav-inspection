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
for folder in ["homogeneous_mappo", "heterogeneous_mappo"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from controlled_variant_env_wrapper import AblationMAPPOEnvWrapper, ablation_name  # noqa: E402


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.model, map_location=args.device)
    use_guidance = bool(checkpoint.get("use_guidance", args.use_guidance))
    use_capability_reward = bool(checkpoint.get("use_capability_reward", args.use_capability_reward))
    use_safety_shaping = bool(checkpoint.get("use_safety_shaping", args.use_safety_shaping))
    name = checkpoint.get("ablation", ablation_name(use_guidance, use_capability_reward, use_safety_shaping))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    print(f"Ablation: {name}")

    actor = HeterogeneousActor(checkpoint["obs_dim"], checkpoint["action_dim"], hidden_dim=args.hidden_dim).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    rows = []
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        env = AblationMAPPOEnvWrapper(
            scenario=args.scenario,
            seed=seed,
            use_guidance=use_guidance,
            use_capability_reward=use_capability_reward,
            use_safety_shaping=use_safety_shaping,
        )
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
        metrics["scenario"] = args.scenario
        metrics["seed"] = seed
        metrics["ablation"] = name
        metrics["use_guidance"] = int(use_guidance)
        metrics["use_capability_reward"] = int(use_capability_reward)
        metrics["use_safety_shaping"] = int(use_safety_shaping)
        metrics["evaluation_mode"] = "deterministic" if args.deterministic else "sampling"
        rows.append(metrics)

    mode = "deterministic" if args.deterministic else "sampling"
    summary_file = output_dir / f"ablation_{name}_{mode}_summary.csv"
    mean_std_file = output_dir / f"ablation_{name}_{mode}_mean_std.csv"
    mean_std = _compute_mean_std(rows)
    mean_std["ablation"] = name
    mean_std["use_guidance"] = int(use_guidance)
    mean_std["use_capability_reward"] = int(use_capability_reward)
    mean_std["use_safety_shaping"] = int(use_safety_shaping)
    _write_rows_csv(rows, summary_file)
    _write_rows_csv([mean_std], mean_std_file)
    print(f"Saved summary: {summary_file}")
    print(f"Saved mean/std: {mean_std_file}")
    for key, value in mean_std.items():
        print(f"{key}: {value}")


def _compute_mean_std(rows: list[dict]) -> dict:
    numeric_cols = [
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
    result = {}
    for col in numeric_cols:
        values = [float(row[col]) for row in rows]
        result[f"{col}_mean"] = mean(values) if values else 0.0
        result[f"{col}_std"] = stdev(values) if len(values) > 1 else 0.0
    return result


def _new_diagnostics() -> dict:
    return {"quadrotor_hover_inspect_count": 0, "quadrotor_valid_hover_inspect_count": 0, "quadrotor_invalid_hover_inspect_count": 0, "distance_sum": 0.0, "distance_count": 0, "time_in_candidate_range": 0}


def _update_diagnostics(env: AblationMAPPOEnvWrapper, actions, diagnostics: dict) -> None:
    assignments = env._quadrotor_candidate_assignments()
    for idx, agent_id in enumerate(env.agents):
        if not agent_id.startswith("quadrotor") or agent_id not in assignments:
            continue
        uav = env.env.uavs[agent_id]
        distance = float(np.linalg.norm(assignments[agent_id] - uav.position()))
        in_range = distance <= uav.sensor_range
        diagnostics["distance_sum"] += distance
        diagnostics["distance_count"] += 1
        if in_range:
            diagnostics["time_in_candidate_range"] += 1
        if int(actions[idx]) == 4:
            diagnostics["quadrotor_hover_inspect_count"] += 1
            if in_range:
                diagnostics["quadrotor_valid_hover_inspect_count"] += 1
            else:
                diagnostics["quadrotor_invalid_hover_inspect_count"] += 1


def _finalize_diagnostics(diagnostics: dict) -> dict:
    distance_count = max(1, int(diagnostics["distance_count"]))
    return {
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
    parser = argparse.ArgumentParser(description="Evaluate CA-HMAPPO ablation variant.")
    parser.add_argument("--scenario", type=int, default=1, choices=[0, 1])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--use-guidance", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-capability-reward", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-safety-shaping", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs/controlled_analysis")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
