from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from heterogeneous_uav_env import HeterogeneousUAVEnv, RuleBasedHeterogeneousPolicy, RuleBasedHeterogeneousPolicyV2
from visualize_simulation import save_simulation_plot


def run_episode(
    scenario: int = 1,
    seed: int = 42,
    output_dir: str = "outputs/baseline",
    save_artifacts: bool = True,
    baseline: str = "v1",
) -> dict:
    scenario_name = "scenario_0_easy" if scenario == 0 else "scenario_1_document"
    if scenario == 0:
        env = HeterogeneousUAVEnv(
            area_size=1000.0,
            grid_size=20,
            n_quadrotors=1,
            n_targets=2,
            max_steps=700,
            seed=seed,
        )
        env.quadrotor_sensor_range = 100.0
        env.p_detect_fixedwing = 1.0
        env.p_false_alarm_fixedwing = 0.0
        env.p_inspect_quadrotor = 0.95
        env.p_false_alarm_quadrotor = 0.0
    else:
        env = HeterogeneousUAVEnv(
            area_size=2000.0,
            grid_size=40,
            n_quadrotors=2,
            n_targets=8,
            max_steps=1000,
            seed=seed,
        )

    env.reset(seed=seed)
    if scenario == 0:
        env.targets = np.array([[120.0, 75.0], [220.0, 75.0]], dtype=np.float64)
        env.target_detected = np.zeros(env.n_targets, dtype=bool)
        env.target_inspected = np.zeros(env.n_targets, dtype=bool)
    if baseline == "v2":
        policy = RuleBasedHeterogeneousPolicyV2(env)
    else:
        policy = RuleBasedHeterogeneousPolicy(env)
    history = []

    while not env.done:
        actions = policy.act()
        _, rewards, _, _, infos = env.step(actions)
        metrics = dict(infos["__all__"])
        metrics["reward"] = float(next(iter(rewards.values())))
        history.append(metrics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = env.compute_metrics()
    metrics["scenario"] = scenario
    metrics["scenario_name"] = scenario_name
    metrics["seed"] = seed
    metrics["baseline"] = baseline
    if save_artifacts:
        metrics_file = output_path / f"scenario_{scenario}_{baseline}_seed_{seed}_metrics.csv"
        trajectory_file = output_path / f"scenario_{scenario}_{baseline}_seed_{seed}_trajectories.csv"
        plot_file = output_path / f"scenario_{scenario}_{baseline}_seed_{seed}_plot.png"

        _write_rows_csv(history, metrics_file)
        _save_trajectories(env, trajectory_file)
        try:
            save_simulation_plot(env, plot_file)
            metrics["plot_error"] = ""
        except Exception as exc:
            metrics["plot_error"] = f"plot skipped: {exc}"

        metrics["metrics_file"] = str(metrics_file)
        metrics["trajectory_file"] = str(trajectory_file)
        metrics["plot_file"] = str(plot_file) if not metrics["plot_error"] else ""
    return metrics


def run_multi_seed(
    scenarios: list[int],
    num_seeds: int = 10,
    output_dir: str = "outputs/baseline",
    base_seed: int = 0,
    baseline: str = "v1",
) -> tuple[list[dict], list[dict]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in scenarios:
        for seed in range(base_seed, base_seed + num_seeds):
            rows.append(
                run_episode(
                    scenario=scenario,
                    seed=seed,
                    output_dir=output_dir,
                    save_artifacts=False,
                    baseline=baseline,
                )
            )

    summary = rows
    numeric_cols = [
        "coverage_ratio",
        "inspection_success_rate",
        "target_detection_rate",
        "mission_success",
        "mission_time",
        "total_energy_consumption",
        "fixedwing_final_battery",
        "quadrotor_mean_final_battery",
        "quadrotor_min_final_battery",
        "collision_count",
        "near_miss_count",
        "communication_loss_count",
        "invalid_action_count",
        "detected_targets",
        "inspected_targets",
        "candidate_targets",
        "redundant_coverage_proxy",
    ]
    mean_std = []
    for scenario in scenarios:
        scenario_rows = [row for row in summary if row["scenario"] == scenario]
        stats_row = {"scenario": scenario}
        for metric in numeric_cols:
            values = [float(row[metric]) for row in scenario_rows]
            stats_row[f"{metric}_mean"] = mean(values) if values else 0.0
            stats_row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        mean_std.append(stats_row)

    for scenario in scenarios:
        scenario_summary = [row for row in summary if row["scenario"] == scenario]
        _write_rows_csv(scenario_summary, output_path / f"scenario_{scenario}_rule_based_{baseline}_summary.csv")
    _write_rows_csv(mean_std, output_path / f"rule_based_{baseline}_mean_std_table.csv")
    return summary, mean_std


def _save_trajectories(env: HeterogeneousUAVEnv, path: Path) -> None:
    rows = []
    for agent_id, uav in env.uavs.items():
        for step, (x, y) in enumerate(uav.trajectory):
            rows.append(
                {
                    "agent_id": agent_id,
                    "uav_type": uav.uav_type,
                    "step": step,
                    "x": x,
                    "y": y,
                }
            )
    _write_rows_csv(rows, path)


def _write_rows_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    widths = {column: max(len(column), *(len(str(row[column])) for row in rows)) for column in columns}
    header = "  ".join(column.rjust(widths[column]) for column in columns)
    lines = [header]
    for row in rows:
        lines.append("  ".join(str(row[column]).rjust(widths[column]) for column in columns))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rule-based heterogeneous UAV baseline.")
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        choices=[0, 1],
        help="0=easy mission-success calibration, 1=document configuration",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument("--baseline", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--multi-seed", action="store_true", help="Run multi-seed validation instead of one episode")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--all-scenarios", action="store_true", help="Run scenario 0 and 1 in multi-seed mode")
    args = parser.parse_args()

    if args.multi_seed:
        scenarios = [0, 1] if args.all_scenarios else [args.scenario]
        summary, mean_std = run_multi_seed(
            scenarios=scenarios,
            num_seeds=args.num_seeds,
            output_dir=args.output_dir,
            base_seed=args.base_seed,
            baseline=args.baseline,
        )
        print("Rule-based multi-seed validation completed")
        print(f"episodes: {len(summary)}")
        print(_format_table(mean_std))
        return

    metrics = run_episode(scenario=args.scenario, seed=args.seed, output_dir=args.output_dir, baseline=args.baseline)
    print("Rule-based heterogeneous baseline completed")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
