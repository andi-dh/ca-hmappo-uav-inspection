"""Zero-shot held-out generalization evaluation for CA-HMAPPO.

Evaluates all 5 independently trained CA-HMAPPO checkpoints (seeds 0-4) on
10 evaluation conditions (one nominal reference and nine held-out shifts)
without retraining, fine-tuning, reward retuning, or checkpoint reselection.

Total episodes: 5 training seeds x 10 conditions x 20 env seeds x 5 action seeds = 5000

Conditions
----------
1.  nominal              : Scenario 1 baseline
2.  clustered_targets    : Clustered target distribution
3.  edge_targets         : Edge-biased target distribution
4.  sensing_moderate     : p_detect=0.60, p_inspect=0.80
5.  sensing_severe       : p_detect=0.50, p_inspect=0.70
6.  hard_comm_800        : communication_radius=800 m, assignment withheld when disconnected
7.  hard_comm_400        : communication_radius=400 m, assignment withheld when disconnected
8.  slower_motion        : FW 24 m/step, Quad 12 m/step (x0.8)
9.  faster_motion        : FW 36 m/step, Quad 18 m/step (x1.2)
10. scenario_2           : 2 FW + 2 Quad, 10 targets, 2400x2400 m

Localization-noise and missed-report perturbations are intentionally excluded
here: they are covered separately by the candidate-information robustness
evaluation (`code/ca_hmappo/evaluate_candidate_robustness.py`), which
perturbs the reported candidate coordinate rather than the held-out
environment configuration.

Seed Protocol (matches the training-seed robustness protocol in
`code/controlled_analysis/evaluate_robustness.py`)
-------------------------------------------------------------------------
environment seeds = 0, 1, ..., 19   (controls target/candidate stochasticity)
action seeds      = 100, 101, ..., 104  (controls stochastic action sampling)

Exactly one `env.reset(seed=env_seed)` call is made per episode. The action
seed only reseeds `random`/`numpy`/`torch` global RNGs used for action
sampling; it never reseeds the environment's own RNG.

Statistical Aggregation
-----------------------
Unit of statistical analysis: n=5 (five independently trained policies)
For each checkpoint, average over 100 episodes (20 env seeds x 5 action seeds).
Then compute mean +/- SD across the 5 checkpoint means.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
for folder in ["code/homogeneous_mappo", "code/heterogeneous_mappo", "code/ca_hmappo", "code/extended_evaluations"]:
    p = ROOT / folder
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from assignment_aware_env_wrapper import ScalableCapabilityAwareMAPPOEnvWrapper  # noqa: E402


def set_seeds(seed: int) -> None:
    """Reseed action-sampling RNGs only. Never touches the environment's own RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Condition Wrapper
# ==============================================================================

class HeldOutConditionWrapper(ScalableCapabilityAwareMAPPOEnvWrapper):
    """Applies held-out condition transformations to environment parameters."""

    def __init__(
        self,
        scenario: int = 1,
        seed: int | None = None,
        target_distribution: str = "uniform",
        p_detect_fw: float | None = None,
        p_inspect_quad: float | None = None,
        communication_radius: float | None = None,
        motion_scale: float = 1.0,
        hard_communication: bool = False,
    ) -> None:
        super().__init__(scenario=scenario, seed=seed)
        self.target_distribution = target_distribution
        self.hard_communication = bool(hard_communication)

        # Override sensing parameters if specified
        if p_detect_fw is not None:
            self.env.p_detect_fixedwing = float(p_detect_fw)
        if p_inspect_quad is not None:
            self.env.p_inspect_quadrotor = float(p_inspect_quad)

        # Override communication radius if specified
        if communication_radius is not None:
            self.env.communication_radius = float(communication_radius)

        # Override motion speeds if motion_scale != 1.0
        if motion_scale != 1.0:
            self.env.fixedwing_speed *= motion_scale
            self.env.quadrotor_speed *= motion_scale
            # Update existing UAV speeds if already created
            for uav in self.env.uavs.values():
                if uav.uav_type == "fixedwing":
                    uav.speed = self.env.fixedwing_speed
                else:
                    uav.speed = self.env.quadrotor_speed

    def reset(self, seed: int | None = None):
        data = super().reset(seed=seed)
        # Apply target distribution after reset, keyed on the same env seed
        # so each of the 20 environment seeds gets a distinct layout.
        self._apply_target_distribution(seed if seed is not None else 0)
        return data

    def _apply_target_distribution(self, seed: int) -> None:
        """Apply target distribution: uniform, clustered, or edge."""
        if self.target_distribution == "uniform":
            return  # already set by reset()

        rng = np.random.default_rng(seed + 10_000)
        mission = self.env
        margin = mission.cell_size * 2.0

        if self.target_distribution == "clustered":
            centers = np.array(
                [
                    [mission.area_size * 0.35, mission.area_size * 0.35],
                    [mission.area_size * 0.70, mission.area_size * 0.70],
                ],
                dtype=np.float64,
            )
            targets = []
            for idx in range(mission.n_targets):
                center = centers[idx % len(centers)]
                target = center + rng.normal(0.0, mission.fixedwing_sensor_range * 0.55, size=2)
                targets.append(np.clip(target, margin, mission.area_size - margin))
            mission.targets = np.asarray(targets, dtype=np.float64)

        elif self.target_distribution == "edge":
            targets = []
            for idx in range(mission.n_targets):
                side = idx % 4
                if side == 0:
                    target = [rng.uniform(margin, mission.area_size - margin), margin]
                elif side == 1:
                    target = [mission.area_size - margin, rng.uniform(margin, mission.area_size - margin)]
                elif side == 2:
                    target = [rng.uniform(margin, mission.area_size - margin), mission.area_size - margin]
                else:
                    target = [margin, rng.uniform(margin, mission.area_size - margin)]
                jitter = rng.normal(0.0, mission.cell_size, size=2)
                targets.append(np.clip(np.asarray(target) + jitter, margin, mission.area_size - margin))
            mission.targets = np.asarray(targets, dtype=np.float64)

        else:
            raise ValueError(f"Unsupported target distribution: {self.target_distribution}")

        # Reset detection/inspection state
        mission.target_detected = np.zeros(mission.n_targets, dtype=bool)
        mission.target_inspected = np.zeros(mission.n_targets, dtype=bool)
        mission.candidate_targets = []
        mission.target_probability_map = np.full((mission.grid_size, mission.grid_size), 0.5, dtype=np.float64)
        mission.inspection_map = np.zeros((mission.grid_size, mission.grid_size), dtype=bool)
        mission.detections_count = 0
        mission.inspections_count = 0

    def _quadrotor_connected(self, agent_id: str) -> bool:
        """True if the quadrotor can reach base or any fixed-wing within communication_radius."""
        uav = self.env.uavs[agent_id]
        radius = float(self.env.communication_radius)
        if float(np.linalg.norm(uav.position() - self.env.base)) <= radius:
            return True
        for other in self.env.uavs.values():
            if other.uav_type == "fixedwing" and float(np.linalg.norm(uav.position() - other.position())) <= radius:
                return True
        return False

    def _quadrotor_candidate_assignments(self) -> Dict[str, np.ndarray]:
        """Under hard communication, disconnected quadrotors receive no assignment (zero guidance)."""
        if not self.hard_communication:
            return super()._quadrotor_candidate_assignments()
        candidates = list(self._active_candidates())
        assignments: Dict[str, np.ndarray] = {}
        connected_quadrotors = [
            agent_id for agent_id in self.agents
            if agent_id.startswith("quadrotor") and self._quadrotor_connected(agent_id)
        ]
        for agent_id in connected_quadrotors:
            if not candidates:
                break
            uav_position = self.env.uavs[agent_id].position()
            distances = [float(np.linalg.norm(candidate - uav_position)) for candidate in candidates]
            selected_idx = int(np.argmin(distances))
            assignments[agent_id] = candidates.pop(selected_idx)
        return assignments


# ==============================================================================
# Conditions
# ==============================================================================

CONDITIONS = [
    {
        "name": "nominal",
        "scenario": 1,
        "target_distribution": "uniform",
        "p_detect_fw": None,
        "p_inspect_quad": None,
        "communication_radius": None,
        "motion_scale": 1.0,
        "hard_communication": False,
    },
    {
        "name": "clustered_targets",
        "scenario": 1,
        "target_distribution": "clustered",
        "p_detect_fw": None,
        "p_inspect_quad": None,
        "communication_radius": None,
        "motion_scale": 1.0,
        "hard_communication": False,
    },
    {
        "name": "edge_targets",
        "scenario": 1,
        "target_distribution": "edge",
        "p_detect_fw": None,
        "p_inspect_quad": None,
        "communication_radius": None,
        "motion_scale": 1.0,
        "hard_communication": False,
    },
    {
        "name": "sensing_moderate",
        "scenario": 1,
        "target_distribution": "uniform",
        "p_detect_fw": 0.60,
        "p_inspect_quad": 0.80,
        "communication_radius": None,
        "motion_scale": 1.0,
        "hard_communication": False,
    },
    {
        "name": "sensing_severe",
        "scenario": 1,
        "target_distribution": "uniform",
        "p_detect_fw": 0.50,
        "p_inspect_quad": 0.70,
        "communication_radius": None,
        "motion_scale": 1.0,
        "hard_communication": False,
    },
    {
        "name": "hard_comm_800",
        "scenario": 1,
        "target_distribution": "uniform",
        "p_detect_fw": None,
        "p_inspect_quad": None,
        "communication_radius": 800.0,
        "motion_scale": 1.0,
        "hard_communication": True,
    },
    {
        "name": "hard_comm_400",
        "scenario": 1,
        "target_distribution": "uniform",
        "p_detect_fw": None,
        "p_inspect_quad": None,
        "communication_radius": 400.0,
        "motion_scale": 1.0,
        "hard_communication": True,
    },
    {
        "name": "slower_motion",
        "scenario": 1,
        "target_distribution": "uniform",
        "p_detect_fw": None,
        "p_inspect_quad": None,
        "communication_radius": None,
        "motion_scale": 0.8,
        "hard_communication": False,
    },
    {
        "name": "faster_motion",
        "scenario": 1,
        "target_distribution": "uniform",
        "p_detect_fw": None,
        "p_inspect_quad": None,
        "communication_radius": None,
        "motion_scale": 1.2,
        "hard_communication": False,
    },
    {
        "name": "scenario_2",
        "scenario": 2,
        "target_distribution": "uniform",
        "p_detect_fw": None,
        "p_inspect_quad": None,
        "communication_radius": None,
        "motion_scale": 1.0,
        "hard_communication": False,
    },
]

NUM_CONDITIONS = len(CONDITIONS)


METRIC_COLS = [
    "coverage_ratio",
    "target_detection_rate",
    "inspection_success_rate",
    "mission_success",
    "collision_count",
    "near_miss_count",
    "communication_loss_count",
    "invalid_action_count",
    "total_energy_consumption",
    "mission_time",
    "detected_targets",
    "inspected_targets",
    "candidate_targets",
    "quadrotor_hover_inspect_count",
    "quadrotor_valid_hover_inspect_count",
    "quadrotor_invalid_hover_inspect_count",
]


# ==============================================================================
# Episode Evaluation
# ==============================================================================

def run_episode(actor, env, device: str, deterministic: bool, env_seed: int) -> dict:
    """Run a single evaluation episode with exactly one environment reset."""
    data = env.reset(seed=env_seed)
    diagnostics = {
        "quadrotor_hover_inspect_count": 0,
        "quadrotor_valid_hover_inspect_count": 0,
        "quadrotor_invalid_hover_inspect_count": 0,
    }

    while True:
        obs = torch.as_tensor(data["obs_n"], dtype=torch.float32, device=device)
        masks = torch.as_tensor(data["action_mask_n"], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = actor(obs, masks)
            if deterministic:
                actions = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                actions = Categorical(logits=logits).sample().cpu().numpy()

        # Track hover-inspect diagnostics
        assignments = env._quadrotor_candidate_assignments()
        for idx, agent_id in enumerate(env.agents):
            if not agent_id.startswith("quadrotor"):
                continue
            if int(actions[idx]) != 4:
                continue
            diagnostics["quadrotor_hover_inspect_count"] += 1
            if agent_id in assignments:
                uav = env.env.uavs[agent_id]
                dist = float(np.linalg.norm(assignments[agent_id] - uav.position()))
                if dist <= uav.sensor_range:
                    diagnostics["quadrotor_valid_hover_inspect_count"] += 1
                else:
                    diagnostics["quadrotor_invalid_hover_inspect_count"] += 1
            else:
                diagnostics["quadrotor_invalid_hover_inspect_count"] += 1

        data = env.step(actions)
        if bool(data["done"]):
            break

    metrics = dict(env.compute_metrics())
    metrics.update(diagnostics)
    return metrics


# ==============================================================================
# Main Evaluation
# ==============================================================================

def evaluate(args: argparse.Namespace) -> None:
    """Run full held-out generalization evaluation."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # Load all 5 training seed checkpoints
    checkpoints = []
    for train_seed in range(5):
        ckpt_path = ROOT / f"models/training_seed_robustness/seed_{train_seed}/full_ca_hmappo/full_ca_hmappo_scenario_1.pt"
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device)
        actor = HeterogeneousActor(
            checkpoint["obs_dim"], checkpoint["action_dim"], hidden_dim=args.hidden_dim
        ).to(device)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        checkpoints.append((train_seed, actor))
        print(f"Loaded checkpoint for training seed {train_seed}")

    if len(checkpoints) != 5:
        print(f"ERROR: Expected 5 checkpoints, found {len(checkpoints)}. Aborting.")
        return

    action_seeds = list(range(100, 100 + args.num_action_seeds))

    print(f"\n{'='*80}")
    print(f"Starting held-out generalization evaluation")
    print(f"  5 training seeds x {NUM_CONDITIONS} conditions x {args.num_env_seeds} env seeds x {args.num_action_seeds} action seeds")
    print(f"  = {5 * NUM_CONDITIONS * args.num_env_seeds * args.num_action_seeds} total episodes")
    print(f"  Environment seeds: 0..{args.num_env_seeds - 1}")
    print(f"  Action seeds: {action_seeds[0]}..{action_seeds[-1]}")
    print(f"  Deterministic: {args.deterministic}")
    print(f"{'='*80}\n")

    all_rows: list[dict] = []

    for cond in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Condition: {cond['name']}")
        print(f"{'='*60}")

        for train_seed, actor in checkpoints:
            print(f"\n  Training seed {train_seed}:")

            for env_seed in range(args.num_env_seeds):
                for action_seed in action_seeds:
                    # Action seed reseeds only the action-sampling RNGs.
                    set_seeds(action_seed)

                    # Create environment with condition parameters; the
                    # environment's own RNG stream is governed solely by
                    # env_seed, and reset(seed=env_seed) is called exactly
                    # once inside run_episode().
                    env = HeldOutConditionWrapper(
                        scenario=cond["scenario"],
                        seed=env_seed,
                        target_distribution=cond["target_distribution"],
                        p_detect_fw=cond["p_detect_fw"],
                        p_inspect_quad=cond["p_inspect_quad"],
                        communication_radius=cond["communication_radius"],
                        motion_scale=cond["motion_scale"],
                        hard_communication=cond["hard_communication"],
                    )

                    # Run episode
                    metrics = run_episode(actor, env, device, args.deterministic, env_seed)

                    # Save row
                    row: dict = {
                        "train_seed": train_seed,
                        "condition": cond["name"],
                        "environment_seed": env_seed,
                        "action_seed": action_seed,
                    }
                    for col in METRIC_COLS:
                        row[col] = metrics.get(col, 0)
                    all_rows.append(row)

                # Progress report per env seed
                if (env_seed + 1) % 5 == 0:
                    print(f"    Completed {env_seed + 1}/{args.num_env_seeds} environment seeds")

    # Save raw results
    raw_file = output_dir / "raw_generalization_results.csv"
    _write_csv(all_rows, raw_file)
    print(f"\n{'='*80}")
    print(f"Saved raw results ({len(all_rows)} episodes): {raw_file}")
    print(f"{'='*80}\n")


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
        description="Zero-shot held-out generalization evaluation for CA-HMAPPO."
    )
    parser.add_argument("--num-env-seeds", type=int, default=20, help="Number of environment seeds per condition (0..N-1)")
    parser.add_argument("--num-action-seeds", type=int, default=5, help="Number of action-sampling seeds per env seed (100..100+N-1)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for actor network")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs/heldout_generalization"), help="Output directory")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic (argmax) action selection")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
