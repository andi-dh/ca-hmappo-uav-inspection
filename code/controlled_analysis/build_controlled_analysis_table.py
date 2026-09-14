from __future__ import annotations

import argparse
import csv
from pathlib import Path


ABLATIONS = [
    ("Full CA-HMAPPO", "full_ca_hmappo", "Yes", "Yes", "Yes"),
    ("w/o Guidance", "without_guidance", "No", "Yes", "Yes"),
    ("w/o Capability Reward", "without_capability_reward", "Yes", "No", "Yes"),
    ("w/o Safety Shaping", "without_safety_shaping", "Yes", "Yes", "No"),
]


def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows = []
    for method, name, guidance, capability_reward, safety_shaping in ABLATIONS:
        file = output_dir / f"ablation_{name}_sampling_mean_std.csv"
        if not file.exists() and name == "full_ca_hmappo":
            file = Path(args.full_mean_std)
        if not file.exists():
            rows.append(_empty_row(method, guidance, capability_reward, safety_shaping, "missing"))
            continue
        data = _read_one(file)
        rows.append(
            {
                "method": method,
                "guidance": guidance,
                "capability_reward": capability_reward,
                "safety_shaping": safety_shaping,
                "coverage_ratio": _fmt(data, "coverage_ratio"),
                "target_detection_rate": _fmt(data, "target_detection_rate"),
                "inspection_success_rate": _fmt(data, "inspection_success_rate"),
                "collision_count": _fmt(data, "collision_count"),
                "near_miss_count": _fmt(data, "near_miss_count"),
                "total_energy_consumption": _fmt(data, "total_energy_consumption"),
                "status": "available",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(rows, output_dir / "ablation_comparison_scenario_1_sampling.csv")
    _write_latex(rows, output_dir / "ablation_table_for_paper.tex")


def _read_one(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows[0] if rows else {}


def _fmt(data: dict, key: str) -> str:
    mean = float(data.get(f"{key}_mean", 0.0))
    std = float(data.get(f"{key}_std", 0.0))
    if key in {"total_energy_consumption"}:
        return f"{mean:.2f} +/- {std:.2f}"
    return f"{mean:.3f} +/- {std:.3f}"


def _empty_row(method: str, guidance: str, capability_reward: str, safety_shaping: str, status: str) -> dict:
    return {
        "method": method,
        "guidance": guidance,
        "capability_reward": capability_reward,
        "safety_shaping": safety_shaping,
        "coverage_ratio": "",
        "target_detection_rate": "",
        "inspection_success_rate": "",
        "collision_count": "",
        "near_miss_count": "",
        "total_energy_consumption": "",
        "status": status,
    }


def _write_rows_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on Scenario 1 under sampling evaluation. Values are reported as mean $\pm$ standard deviation.}",
        r"\label{tab:ablation_scenario_1}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Method & Guidance & Capability Reward & Safety Shaping & Coverage $\uparrow$ & Inspection $\uparrow$ & Collision $\downarrow$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {row['guidance']} & {row['capability_reward']} & {row['safety_shaping']} & "
            f"{row['coverage_ratio'].replace('+/-', r'\pm')} & {row['inspection_success_rate'].replace('+/-', r'\pm')} & {row['collision_count'].replace('+/-', r'\pm')} \\\\" 
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build controlled ablation comparison table.")
    parser.add_argument("--output-dir", type=str, default="outputs/controlled_analysis")
    parser.add_argument("--full-mean-std", type=str, default="outputs/ca_hmappo/scenario_1_capability_aware_mappo_sampling_mean_std.csv")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
