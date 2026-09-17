"""Summarize held-out generalization results with proper statistical aggregation.

Statistical Aggregation
-----------------------
Unit of statistical analysis: n=5 (five independently trained policies)

For each training seed k and condition c:
  1. Average over 100 episodes (20 env seeds × 5 action seeds) → mean_{k,c}
  2. Compute mean and SD across the 5 checkpoint means → final statistics

This ensures the statistical unit is the independently trained policy, not
the individual episode.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

CONDITIONS = [
    "nominal",
    "clustered_targets",
    "edge_targets",
    "sensing_moderate",
    "sensing_severe",
    "hard_comm_800",
    "hard_comm_400",
    "slower_motion",
    "faster_motion",
    "scenario_2",
]

METRIC_COLS = [
    "coverage_ratio",
    "target_detection_rate",
    "inspection_success_rate",
    "mission_success",
    "collision_count",
    "near_miss_count",
    "quadrotor_valid_hover_inspect_count",
    "quadrotor_invalid_hover_inspect_count",
]


def summarize(args: argparse.Namespace) -> None:
    """Compute checkpoint-level and condition-level statistics."""
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw results
    with input_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    
    print(f"Loaded {len(all_rows)} raw episodes from {input_file}")

    # Group by (train_seed, condition)
    checkpoint_groups: dict[tuple[int, str], list[dict]] = {}
    for row in all_rows:
        key = (int(row["train_seed"]), str(row["condition"]))
        checkpoint_groups.setdefault(key, []).append(row)
    
    # Step 1: Compute per-checkpoint means (average over 100 episodes)
    checkpoint_summary = []
    for (train_seed, condition), episodes in sorted(checkpoint_groups.items()):
        row = {
            "train_seed": train_seed,
            "condition": condition,
            "episodes": len(episodes),
        }
        for col in METRIC_COLS:
            values = [float(ep.get(col, 0.0)) for ep in episodes]
            row[col] = mean(values) if values else 0.0
        checkpoint_summary.append(row)
    
    checkpoint_file = output_dir / "checkpoint_summary.csv"
    _write_csv(checkpoint_summary, checkpoint_file)
    print(f"Saved checkpoint-level summary (n={len(checkpoint_summary)}): {checkpoint_file}")

    # Step 2: Compute condition-level statistics (mean ± SD across 5 checkpoints)
    condition_summary = []
    for condition in CONDITIONS:
        ckpt_rows = [r for r in checkpoint_summary if r["condition"] == condition]
        if len(ckpt_rows) != 5:
            print(f"WARNING: Expected 5 checkpoints for {condition}, found {len(ckpt_rows)}")
        
        row = {
            "condition": condition,
            "n_checkpoints": len(ckpt_rows),
        }
        for col in METRIC_COLS:
            values = [float(r[col]) for r in ckpt_rows]
            row[f"{col}_mean"] = round(mean(values), 4) if values else 0.0
            row[f"{col}_std"] = round(stdev(values), 4) if len(values) > 1 else 0.0
        condition_summary.append(row)
    
    condition_file = output_dir / "condition_summary.csv"
    _write_csv(condition_summary, condition_file)
    print(f"Saved condition-level summary (n=5): {condition_file}")

    # Step 3: Build manuscript table with formatted mean ± SD
    table_rows = []
    for row in condition_summary:
        table_row = {
            "Condition": _format_condition_name(row["condition"]),
            "Coverage": _format_mean_std(row["coverage_ratio_mean"], row["coverage_ratio_std"]),
            "Detection": _format_mean_std(row["target_detection_rate_mean"], row["target_detection_rate_std"]),
            "Inspection": _format_mean_std(row["inspection_success_rate_mean"], row["inspection_success_rate_std"]),
            "Mission Success": _format_mean_std(row["mission_success_mean"], row["mission_success_std"]),
            "Collision": _format_mean_std(row["collision_count_mean"], row["collision_count_std"]),
            "Near Miss": _format_mean_std(row["near_miss_count_mean"], row["near_miss_count_std"]),
        }
        table_rows.append(table_row)
    
    table_file = output_dir / "heldout_generalization_table.csv"
    _write_csv(table_rows, table_file)
    print(f"Saved manuscript table: {table_file}")

    # Step 4: Compute performance retention relative to nominal
    nominal_row = next((r for r in condition_summary if r["condition"] == "nominal"), None)
    if nominal_row:
        nominal_inspection = nominal_row["inspection_success_rate_mean"]
        print(f"\n{'='*60}")
        print(f"Performance Retention (relative to nominal)")
        print(f"Nominal inspection rate: {nominal_inspection:.4f}")
        print(f"{'='*60}")
        for row in condition_summary:
            if row["condition"] == "nominal":
                continue
            retention = 100.0 * row["inspection_success_rate_mean"] / nominal_inspection if nominal_inspection > 0 else 0.0
            print(f"{row['condition']:25s}  {row['inspection_success_rate_mean']:.4f}  ({retention:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("Summary complete!")
    print(f"{'='*60}")


def _format_condition_name(name: str) -> str:
    """Format condition name for manuscript table."""
    mapping = {
        "nominal": "Nominal",
        "clustered_targets": "Clustered targets",
        "edge_targets": "Edge-biased targets",
        "sensing_moderate": "Sensing 0.60/0.80",
        "sensing_severe": "Sensing 0.50/0.70",
        "hard_comm_800": "Hard comm. 800 m",
        "hard_comm_400": "Hard comm. 400 m",
        "slower_motion": "Motion ×0.8",
        "faster_motion": "Motion ×1.2",
        "scenario_2": "Scenario 2",
    }
    return mapping.get(name, name)


def _format_mean_std(mean_val: float, std_val: float) -> str:
    """Format mean ± std for table."""
    return f"{mean_val:.3f} ± {std_val:.3f}"


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write rows to CSV file."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize held-out generalization results with proper n=5 statistics."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="outputs/heldout_generalization/raw_generalization_results.csv",
        help="Input raw results CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/heldout_generalization",
        help="Output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
