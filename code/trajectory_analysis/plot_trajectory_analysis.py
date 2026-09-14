from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    prefix = f"ca_hmappo_seed_{args.seed}"
    rollout = _read_rows(output_dir / f"rollout_{prefix}.csv")
    events = _read_rows(output_dir / f"events_{prefix}.csv")
    targets = _read_rows(output_dir / f"targets_{prefix}.csv")
    coverage = np.load(output_dir / f"coverage_map_{prefix}.npy")
    _plot_map(rollout, events, targets, coverage, output_dir / f"trajectory_ca_hmappo_seed_{args.seed}.png", grayscale=False)
    _plot_map(rollout, events, targets, coverage, output_dir / f"trajectory_ca_hmappo_seed_{args.seed}.pdf", grayscale=False)
    _plot_map(rollout, events, targets, coverage, output_dir / f"figure_trajectory_ca_hmappo_seed_{args.seed}_grayscale.pdf", grayscale=True)
    _plot_timeline(events, output_dir / f"event_timeline_ca_hmappo_seed_{args.seed}.png")
    _plot_distance(rollout, output_dir / f"distance_to_candidate_ca_hmappo_seed_{args.seed}.png")


def _plot_map(rollout: list[dict], events: list[dict], targets: list[dict], coverage: np.ndarray, path: Path, grayscale: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = "Greys" if grayscale else "Blues"
    ax.imshow(coverage.T, origin="lower", extent=[0, 2000, 0, 2000], cmap=cmap, alpha=0.28)
    colors = {"fixedwing_0": "black" if grayscale else "tab:blue", "quadrotor_0": "dimgray" if grayscale else "tab:orange", "quadrotor_1": "gray" if grayscale else "tab:green"}
    for agent_id in sorted({row["agent_id"] for row in rollout}):
        rows = [row for row in rollout if row["agent_id"] == agent_id]
        x = [float(row["x"]) for row in rows]
        y = [float(row["y"]) for row in rows]
        ax.plot(x, y, linewidth=1.6, color=colors.get(agent_id, "black"), label=agent_id)
        ax.scatter([x[0]], [y[0]], marker="o", s=20, color=colors.get(agent_id, "black"))
        ax.scatter([x[-1]], [y[-1]], marker="x", s=35, color=colors.get(agent_id, "black"))
    inspected_ids = {int(event["target_id"]) for event in events if event["event_type"] == "target_inspected" and event["target_id"] != ""}
    detected_ids = {int(event["target_id"]) for event in events if event["event_type"] == "target_detected" and event["target_id"] != ""}
    for target in targets:
        target_id = int(target["target_id"])
        x = float(target["x"])
        y = float(target["y"])
        if target_id in inspected_ids:
            ax.scatter(x, y, marker="*", s=150, facecolors="none", edgecolors="black" if grayscale else "green", linewidths=1.8, label="inspected target" if target_id == min(inspected_ids) else None)
        elif target_id in detected_ids:
            ax.scatter(x, y, marker="^", s=70, color="gray" if grayscale else "gold", label="detected target" if target_id == min(detected_ids) else None)
        else:
            ax.scatter(x, y, marker=".", s=55, color="lightgray" if grayscale else "red", label="undetected target" if target_id == 0 else None)
    ax.scatter(0, 0, marker="s", s=80, color="black", label="base")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("CA-HMAPPO trajectory, Scenario 1, seed 16")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_timeline(events: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    y_map = {"target_detected": 1, "target_inspected": 2, "collision": 3, "near_miss": 4}
    for event in events:
        event_type = event["event_type"]
        if event_type not in y_map:
            continue
        ax.scatter(float(event["step"]), y_map[event_type], s=35, label=event_type)
    ax.set_yticks(list(y_map.values()), list(y_map.keys()))
    ax.set_xlabel("step")
    ax.set_title("Event timeline")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_distance(rollout: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for agent_id in ["quadrotor_0", "quadrotor_1"]:
        rows = [row for row in rollout if row["agent_id"] == agent_id and row["nearest_candidate_distance"] != ""]
        ax.plot([float(row["step"]) for row in rows], [float(row["nearest_candidate_distance"]) for row in rows], label=agent_id)
    ax.axhline(75.0, color="black", linestyle="--", linewidth=1.0, label="sensor range")
    ax.set_xlabel("step")
    ax.set_ylabel("distance to assigned candidate (m)")
    ax.set_title("Quadrotor distance to candidate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CA-HMAPPO trajectory analysis.")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="outputs/trajectory_analysis")
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
