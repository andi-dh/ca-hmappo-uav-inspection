from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from mappo_env_wrapper import MAPPOEnvWrapper


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(args.base_seed)

    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        env = MAPPOEnvWrapper(scenario=args.scenario, seed=seed)
        data = env.reset(seed=seed)
        while True:
            actions = _sample_valid_actions(data["action_mask_n"], rng)
            data = env.step(actions)
            if bool(data["done"]):
                break
        metrics = dict(env.compute_metrics())
        metrics["scenario"] = args.scenario
        metrics["seed"] = seed
        metrics["baseline"] = "random_masked"
        rows.append(metrics)

    summary_file = output_dir / f"scenario_{args.scenario}_random_masked_summary.csv"
    mean_std_file = output_dir / f"scenario_{args.scenario}_random_masked_mean_std.csv"
    mean_std = _compute_mean_std(rows)
    _write_rows_csv(rows, summary_file)
    _write_rows_csv([mean_std], mean_std_file)
    print("Random masked baseline evaluation completed")
    print(f"episodes: {len(rows)}")
    for key, value in mean_std.items():
        print(f"{key}: {value}")
    print(f"Saved summary: {summary_file}")
    print(f"Saved mean/std: {mean_std_file}")


def _sample_valid_actions(action_masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    actions = []
    for mask in action_masks:
        valid_actions = np.flatnonzero(mask > 0)
        if len(valid_actions) == 0:
            actions.append(0)
        else:
            actions.append(int(rng.choice(valid_actions)))
    return np.asarray(actions, dtype=np.int64)


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
    parser = argparse.ArgumentParser(description="Evaluate random masked baseline.")
    parser.add_argument("--scenario", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/3a_homogeneous_mappo")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
