from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEY_METRICS = [
    "coverage_ratio_mean",
    "inspection_success_rate_mean",
    "target_detection_rate_mean",
    "mission_success_mean",
    "mission_time_mean",
    "total_energy_consumption_mean",
    "collision_count_mean",
    "invalid_action_count_mean",
]


def compare(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline_dir)
    mappo_dir = Path(args.mappo_dir)
    output_file = mappo_dir / f"scenario_{args.scenario}_3a_baseline_comparison_{args.mappo_eval_mode}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(_read_rule_based_rows(baseline_dir / "rule_based_v2_mean_std_table.csv", args.scenario))
    rows.append(_read_single_row(mappo_dir / f"scenario_{args.scenario}_random_masked_mean_std.csv", "random_masked", args.scenario))
    rows.append(
        _read_single_row(
            mappo_dir / f"scenario_{args.scenario}_homogeneous_mappo_{args.mappo_eval_mode}_mean_std.csv",
            f"homogeneous_mappo_{args.mappo_eval_mode}",
            args.scenario,
        )
    )

    rows = [row for row in rows if row]
    with output_file.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["baseline", "scenario"] + KEY_METRICS
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("3A baseline comparison saved")
    print(output_file)
    for row in rows:
        print(row)


def _read_rule_based_rows(path: Path, scenario: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(float(row.get("scenario", -1))) == scenario:
                return [_select_metrics(row, "rule_based_v2", scenario)]
    return []


def _read_single_row(path: Path, baseline: str, scenario: int) -> dict:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    return _select_metrics(rows[0], baseline, scenario)


def _select_metrics(row: dict, baseline: str, scenario: int) -> dict:
    selected = {"baseline": baseline, "scenario": scenario}
    for metric in KEY_METRICS:
        selected[metric] = row.get(metric, "")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 3A baselines.")
    parser.add_argument("--scenario", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mappo-eval-mode", type=str, default="deterministic", choices=["deterministic", "sampling"])
    parser.add_argument("--baseline-dir", type=str, default="outputs/rule_based_baselines")
    parser.add_argument("--mappo-dir", type=str, default="outputs/homogeneous_mappo")
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
