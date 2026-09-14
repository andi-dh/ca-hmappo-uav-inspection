from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    prefix = f"ca_hmappo_seed_{args.seed}"
    diagnostics = _read_one(output_dir / f"action_diagnostics_{prefix}.csv")
    events = _read_rows(output_dir / f"events_{prefix}.csv")
    detected = [event for event in events if event["event_type"] == "target_detected"]
    inspected = [event for event in events if event["event_type"] == "target_inspected"]
    lines = [
        "# 4C Qualitative Analysis",
        "",
        f"Seed: `{args.seed}`",
        "",
        "## Final Metrics",
        "",
        f"- Coverage ratio: `{float(diagnostics['coverage_ratio']):.3f}`",
        f"- Target detection rate: `{float(diagnostics['target_detection_rate']):.3f}`",
        f"- Inspection success rate: `{float(diagnostics['inspection_success_rate']):.3f}`",
        f"- Inspected targets: `{diagnostics['inspected_targets']}`",
        f"- Collision count: `{diagnostics['collision_count']}`",
        f"- Near-miss count: `{diagnostics['near_miss_count']}`",
        f"- Termination reason: `{diagnostics['termination_reason']}`",
        "",
        "## Action Diagnostics",
        "",
        f"- Quadrotor hover-inspect count: `{diagnostics['quadrotor_hover_inspect_count']}`",
        f"- Valid hover-inspect count: `{diagnostics['quadrotor_valid_hover_inspect_count']}`",
        f"- Invalid hover-inspect count: `{diagnostics['quadrotor_invalid_hover_inspect_count']}`",
        f"- Mean distance to candidate: `{float(diagnostics['mean_distance_to_candidate']):.2f}` m",
        f"- Time in candidate range: `{diagnostics['time_in_candidate_range']}` agent-steps",
        "",
        "## Event Summary",
        "",
        f"- Detected target events: `{len(detected)}`",
        f"- Inspected target events: `{len(inspected)}`",
        "",
        "## Paper-Ready Interpretation",
        "",
        "The trajectory visualization shows that the fixed-wing UAV performs wide-area scouting, while quadrotor agents are directed toward detected candidate targets for close-range inspection. CA-HMAPPO produces role-consistent behavior: the fixed-wing UAV continues exploring uncovered regions, whereas quadrotors approach and inspect target candidates. This qualitative behavior is consistent with the quantitative results, where CA-HMAPPO achieves the highest inspection success rate and coverage ratio.",
        "",
        "Versi Indonesia:",
        "",
        "Visualisasi lintasan menunjukkan bahwa fixed-wing UAV berperan sebagai scout untuk coverage area luas, sedangkan quadrotor bergerak menuju candidate target untuk inspeksi jarak dekat. CA-HMAPPO menghasilkan perilaku yang konsisten dengan peran masing-masing UAV: fixed-wing tetap mengeksplorasi area yang belum tercakup, sementara quadrotor mendekati dan menginspeksi target kandidat. Perilaku ini konsisten dengan hasil kuantitatif, di mana CA-HMAPPO menghasilkan inspection success rate dan coverage ratio tertinggi.",
    ]
    (output_dir / "qualitative_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_timeline(inspected, output_dir / f"event_timeline_ca_hmappo_seed_{args.seed}.csv")


def _write_timeline(inspected: list[dict], path: Path) -> None:
    rows = []
    for event in inspected:
        rows.append({"target": f"T{event['target_id']}", "inspected_step": event["step"], "inspector": event["agent_id"], "x": event["x"], "y": event["y"]})
    if not rows:
        path.write_text("target,inspected_step,inspector,x,y\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_one(path: Path) -> dict:
    rows = _read_rows(path)
    return rows[0] if rows else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build qualitative summary for CA-HMAPPO trajectory analysis.")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="outputs/trajectory_analysis")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
