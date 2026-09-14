from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ["python_3b_heterogeneous_mappo", "python_3c_capability_aware_mappo", "python_rev_reviewer_baselines"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from assignment_aware_env_wrapper import AssignmentAwareMAPPOEnvWrapper  # noqa: E402
from evaluate_hierarchical_task_allocation import HierarchicalTaskAllocationPolicy  # noqa: E402
from evaluate_happo import _compute_mean_std, _finalize_diagnostics, _new_diagnostics, _update_diagnostics  # noqa: E402


class RuleBasedQuadrotorInspector:
    """
    Deterministic rule-based quadrotor inspector under structured sweep framework.
    
    Uses the same candidate assignment from AssignmentAwareMAPPOEnvWrapper as the
    sweep-assisted learned policy, so the only difference is the quadrotor decision logic.
    
    Decision logic:
    - Get assignment from wrapper (same minimum-distance matching as learned variant)
    - If in valid inspection range → hover-inspect (action 4)
    - Otherwise → move toward assigned candidate using discrete movement actions
    - No assignment → idle (action 10)
    """
    
    def __init__(self, env: AssignmentAwareMAPPOEnvWrapper):
        self.env = env
    
    def act(self, agent_id: str, action_mask: np.ndarray) -> int:
        """
        Returns discrete action for one quadrotor agent using wrapper's assignment.
        """
        # Get assignment from wrapper (same as sweep-assisted)
        assignments = self.env._quadrotor_candidate_assignments()
        target = assignments.get(agent_id)
        
        if target is None:
            # No assignment: idle
            if action_mask[10] > 0:
                return 10
            # Fallback to first valid
            valid = np.where(action_mask > 0)[0]
            return int(valid[0]) if len(valid) > 0 else 0
        
        # Has assignment: check if in range for hover-inspect
        uav = self.env.env.uavs[agent_id]
        distance = float(np.linalg.norm(target - uav.position()))
        
        # If in valid inspection range and hover-inspect is valid → hover-inspect
        # Wrapper controls action 4 availability based on timing logic
        if distance <= uav.sensor_range and action_mask[4] > 0:
            return 4  # hover-inspect
        
        # Not in range: move toward assignment
        # Use same logic as wrapper's _move_toward_assignment
        delta = target - uav.position()
        if abs(float(delta[0])) >= abs(float(delta[1])):
            preferred = 2 if delta[0] > 0 else 3  # E or W
        else:
            preferred = 0 if delta[1] > 0 else 1  # N or S
        
        # Try preferred direction
        if action_mask[preferred] > 0:
            return preferred
        
        # Fallback: try all movement actions
        for action in [0, 1, 2, 3]:
            if action_mask[action] > 0:
                return action
        
        # Ultimate fallback
        valid = np.where(action_mask > 0)[0]
        return int(valid[0]) if len(valid) > 0 else 0


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        env = AssignmentAwareMAPPOEnvWrapper(
            scenario=args.scenario,
            seed=seed,
            hover_commit_steps=args.hover_commit_steps,
            hover_cooldown_steps=args.hover_cooldown_steps,
        )
        data = env.reset(seed=seed)
        
        # Structured sweep policy for fixed-wing (same as sweep-assisted)
        sweep_policy = HierarchicalTaskAllocationPolicy(
            env,
            lane_spacing_factor=args.lane_spacing_factor,
            waypoint_threshold_factor=args.waypoint_threshold_factor,
        )
        
        # Rule-based inspector for quadrotors
        quad_policy = RuleBasedQuadrotorInspector(env)
        
        diagnostics = _new_diagnostics()
        
        while True:
            mask_n = data["action_mask_n"]
            
            # Get structured sweep actions for fixed-wing
            sweep_actions = np.asarray(sweep_policy.act(), dtype=np.int64)
            
            # Construct final action array
            actions = np.zeros(len(env.agents), dtype=np.int64)
            for idx, agent_id in enumerate(env.agents):
                if agent_id.startswith("fixedwing"):
                    actions[idx] = sweep_actions[idx]
                else:
                    # Quadrotor: use rule-based inspector (same assignment as sweep-assisted)
                    actions[idx] = quad_policy.act(agent_id, mask_n[idx])
            
            _update_diagnostics(env, actions, diagnostics)
            data = env.step(actions)
            
            if bool(data["done"]):
                break
        
        metrics = dict(env.compute_metrics())
        metrics.update(_finalize_diagnostics(diagnostics))
        metrics["scenario"] = args.scenario
        metrics["seed"] = seed
        metrics["baseline"] = args.label
        rows.append(metrics)
    
    prefix = f"scenario_{args.scenario}_{args.label}"
    summary_file = output_dir / f"{prefix}_summary.csv"
    mean_std_file = output_dir / f"{prefix}_mean_std.csv"
    
    mean_std = _compute_mean_std(rows)
    write_rows(rows, summary_file)
    write_rows([mean_std], mean_std_file)
    
    print(f"Structured sweep + rule-based quadrotor evaluation completed: {len(rows)} episodes")
    print(f"Saved summary: {summary_file}")
    print(f"Saved mean/std: {mean_std_file}")
    for key, value in mean_std.items():
        print(f"{key}: {value}")


def write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate structured sweep + rule-based quadrotor inspector.")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--hover-commit-steps", type=int, default=2)
    parser.add_argument("--hover-cooldown-steps", type=int, default=6)
    parser.add_argument("--lane-spacing-factor", type=float, default=1.1)
    parser.add_argument("--waypoint-threshold-factor", type=float, default=0.35)
    parser.add_argument("--label", type=str, default="sweep_rulebased_quad")
    parser.add_argument("--output-dir", type=str, default="outputs/sweep_rulebased_quad")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
