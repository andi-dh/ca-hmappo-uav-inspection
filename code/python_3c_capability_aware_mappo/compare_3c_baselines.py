from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEY_METRICS = ["coverage_ratio_mean", "inspection_success_rate_mean", "target_detection_rate_mean", "mission_success_mean", "mission_time_mean", "total_energy_consumption_mean", "collision_count_mean", "invalid_action_count_mean"]


def compare(args: argparse.Namespace) -> None:
    rows = []
    rows.append(_read_rule_based(Path(args.baseline_dir) / "rule_based_v2_mean_std_table.csv", args.scenario))
    rows.append(_read_mean_std(Path(args.three_a_dir) / f"scenario_{args.scenario}_random_masked_mean_std.csv", "random_masked", args.scenario))
    rows.append(_read_mean_std(Path(args.three_a_dir) / f"scenario_{args.scenario}_homogeneous_mappo_{args.eval_mode}_mean_std.csv", f"homogeneous_mappo_{args.eval_mode}", args.scenario))
    rows.append(_read_mean_std(Path(args.three_b_dir) / f"scenario_{args.scenario}_heterogeneous_mappo_{args.eval_mode}_mean_std.csv", f"heterogeneous_mappo_{args.eval_mode}", args.scenario))
    rows.append(_read_mean_std(Path(args.three_c_dir) / f"scenario_{args.scenario}_capability_aware_mappo_{args.eval_mode}_mean_std.csv", f"capability_aware_mappo_{args.eval_mode}", args.scenario))
    rows = [row for row in rows if row]
    output_file = Path(args.three_c_dir) / f"scenario_{args.scenario}_3c_baseline_comparison_{args.eval_mode}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["baseline", "scenario"] + KEY_METRICS)
        writer.writeheader()
        writer.writerows(rows)
    print("3C baseline comparison saved")
    print(output_file)
    for row in rows:
        print(row)


def _read_rule_based(path: Path, scenario: int) -> dict:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if int(float(row.get("scenario", -1))) == scenario:
                return _select(row, "rule_based_v2", scenario)
    return {}


def _read_mean_std(path: Path, baseline: str, scenario: int) -> dict:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return _select(rows[0], baseline, scenario) if rows else {}


def _select(row: dict, baseline: str, scenario: int) -> dict:
    selected = {"baseline": baseline, "scenario": scenario}
    for metric in KEY_METRICS:
        selected[metric] = row.get(metric, "")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 3C baselines.")
    parser.add_argument("--scenario", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eval-mode", type=str, default="deterministic", choices=["deterministic", "sampling"])
    parser.add_argument("--baseline-dir", type=str, default="outputs/baseline")
    parser.add_argument("--three-a-dir", type=str, default="outputs/3a_homogeneous_mappo")
    parser.add_argument("--three-b-dir", type=str, default="outputs/3b_heterogeneous_mappo")
    parser.add_argument("--three-c-dir", type=str, default="outputs/3c_capability_aware_mappo")
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
