from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def evaluate_checkpoints(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(model_dir.glob(f"capability_aware_mappo_scenario_{args.scenario}_update_*.pt"), key=_checkpoint_update)
    final_model = model_dir / f"capability_aware_mappo_scenario_{args.scenario}.pt"
    if final_model.exists():
        checkpoints.append(final_model)
    rows = []
    for checkpoint in checkpoints:
        update = _checkpoint_update(checkpoint)
        mode_args = ["--deterministic"] if args.deterministic else []
        command = [
            sys.executable,
            str(Path(__file__).with_name("evaluate_capability_aware_mappo.py")),
            "--scenario",
            str(args.scenario),
            "--num-seeds",
            str(args.num_seeds),
            "--base-seed",
            str(args.base_seed),
            "--model",
            str(checkpoint),
            "--device",
            args.device,
            "--output-dir",
            str(output_dir / "checkpoint_eval_tmp"),
            *mode_args,
        ]
        subprocess.run(command, check=True)
        mode = "deterministic" if args.deterministic else "sampling"
        mean_std_file = output_dir / "checkpoint_eval_tmp" / f"scenario_{args.scenario}_capability_aware_mappo_{mode}_mean_std.csv"
        row = _read_one(mean_std_file)
        row["checkpoint"] = str(checkpoint)
        row["update"] = update
        row["eval_mode"] = mode
        rows.append(row)
    rows.sort(key=lambda row: _score(row), reverse=True)
    summary_file = output_dir / f"scenario_{args.scenario}_checkpoint_selection_{'deterministic' if args.deterministic else 'sampling'}.csv"
    _write_rows_csv(rows, summary_file)
    if rows:
        print("Best checkpoint:")
        print(rows[0])
    print(f"Saved checkpoint selection: {summary_file}")


def _checkpoint_update(path: Path) -> int:
    stem = path.stem
    if "_update_" not in stem:
        return 10**9
    return int(stem.rsplit("_update_", 1)[1])


def _score(row: dict) -> tuple[float, float, float, float]:
    return (
        float(row.get("inspection_success_rate_mean", 0.0)),
        -float(row.get("collision_count_mean", 0.0)),
        float(row.get("coverage_ratio_mean", 0.0)),
        float(row.get("target_detection_rate_mean", 0.0)),
    )


def _read_one(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return dict(rows[0]) if rows else {}


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
    parser = argparse.ArgumentParser(description="Evaluate and rank 3C checkpoints.")
    parser.add_argument("--scenario", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-dir", type=str, default="models/ca_hmappo")
    parser.add_argument("--output-dir", type=str, default="outputs/ca_hmappo")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_checkpoints(parse_args())
