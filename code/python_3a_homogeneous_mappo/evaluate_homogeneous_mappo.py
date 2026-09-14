from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.distributions import Categorical

from mappo_env_wrapper import MAPPOEnvWrapper
from mappo_networks import Actor


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.model, map_location=device)

    env = MAPPOEnvWrapper(scenario=args.scenario, seed=args.base_seed)
    actor = Actor(checkpoint["obs_dim"], checkpoint["action_dim"], hidden_dim=args.hidden_dim).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    rows = []
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        data = env.reset(seed=seed)
        while True:
            obs = torch.as_tensor(data["obs_n"], dtype=torch.float32, device=device)
            masks = torch.as_tensor(data["action_mask_n"], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = actor(obs, masks)
                if args.deterministic:
                    actions = torch.argmax(logits, dim=-1).cpu().numpy()
                else:
                    actions = Categorical(logits=logits).sample().cpu().numpy()
            data = env.step(actions)
            if bool(data["done"]):
                break
        metrics = dict(env.compute_metrics())
        metrics["scenario"] = args.scenario
        metrics["seed"] = seed
        metrics["baseline"] = "homogeneous_mappo"
        metrics["evaluation_mode"] = "deterministic" if args.deterministic else "sampling"
        rows.append(metrics)

    evaluation_mode = "deterministic" if args.deterministic else "sampling"
    summary_file = output_dir / f"scenario_{args.scenario}_homogeneous_mappo_{evaluation_mode}_summary.csv"
    _write_rows_csv(rows, summary_file)
    mean_std = _compute_mean_std(rows)
    mean_std_file = output_dir / f"scenario_{args.scenario}_homogeneous_mappo_{evaluation_mode}_mean_std.csv"
    _write_rows_csv([mean_std], mean_std_file)
    print("Homogeneous MAPPO evaluation completed")
    print(f"episodes: {len(rows)}")
    for key, value in mean_std.items():
        print(f"{key}: {value}")
    print(f"Saved summary: {summary_file}")
    print(f"Saved mean/std: {mean_std_file}")


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
    ]
    result = {}
    for col in numeric_cols:
        values = [float(row[col]) for row in rows]
        result[f"{col}_mean"] = mean(values) if values else 0.0
        result[f"{col}_std"] = stdev(values) if len(values) > 1 else 0.0
    return result


def _write_rows_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Homogeneous MAPPO baseline.")
    parser.add_argument("--scenario", type=int, default=0, choices=[0, 1])
    parser.add_argument("--model", type=str, default="models/3a_homogeneous_mappo/homogeneous_mappo_scenario_0.pt")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/3a_homogeneous_mappo")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions. By default, sample from the policy.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
