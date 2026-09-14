from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
for folder in ["heterogeneous_mappo", "ca_hmappo", "extended_evaluations"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from assignment_aware_env_wrapper import AssignmentAwareMAPPOEnvWrapper  # noqa: E402
from evaluate_hierarchical_task_allocation import HierarchicalTaskAllocationPolicy  # noqa: E402
from evaluate_happo import NUMERIC_COLS, _compute_mean_std, _finalize_diagnostics, _new_diagnostics, _update_diagnostics  # noqa: E402


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    actor = HeterogeneousActor(checkpoint["obs_dim"], checkpoint["action_dim"], hidden_dim=args.hidden_dim).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    rows = []
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        torch.manual_seed(seed)
        env = AssignmentAwareMAPPOEnvWrapper(
            scenario=args.scenario,
            seed=seed,
            hover_commit_steps=args.hover_commit_steps,
            hover_cooldown_steps=args.hover_cooldown_steps,
        )
        data = env.reset(seed=seed)
        sweep_policy = HierarchicalTaskAllocationPolicy(
            env,
            lane_spacing_factor=args.lane_spacing_factor,
            waypoint_threshold_factor=args.waypoint_threshold_factor,
        )
        diagnostics = _new_diagnostics()
        while True:
            obs = torch.as_tensor(data["obs_n"], dtype=torch.float32, device=device)
            masks = torch.as_tensor(data["action_mask_n"], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = actor(obs, masks)
                actor_actions = torch.argmax(logits, dim=-1).cpu().numpy() if args.deterministic else Categorical(logits=logits).sample().cpu().numpy()
            sweep_actions = np.asarray(sweep_policy.act(), dtype=np.int64)
            actions = actor_actions.copy()
            for idx, agent_id in enumerate(env.agents):
                if agent_id.startswith("fixedwing"):
                    actions[idx] = sweep_actions[idx]
            if args.action_noise_prob > 0.0:
                actions = apply_action_noise(env, actions, args.action_noise_prob)
            _update_diagnostics(env, actions, diagnostics)
            data = env.step(actions)
            if bool(data["done"]):
                break
        metrics = dict(env.compute_metrics())
        metrics.update(_finalize_diagnostics(diagnostics))
        metrics["scenario"] = args.scenario
        metrics["seed"] = seed
        metrics["baseline"] = args.label
        metrics["evaluation_mode"] = "deterministic" if args.deterministic else "sampling"
        rows.append(metrics)
    prefix = f"scenario_{args.scenario}_{args.label}_{'deterministic' if args.deterministic else 'sampling'}"
    summary_file = output_dir / f"{prefix}_summary.csv"
    mean_std_file = output_dir / f"{prefix}_mean_std.csv"
    mean_std = _compute_mean_std(rows)
    write_rows(rows, summary_file)
    write_rows([mean_std], mean_std_file)
    print(f"Sweep-assisted actor evaluation completed: {len(rows)} episodes")
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
    parser = argparse.ArgumentParser(description="Evaluate learned actor with structured fixed-wing sweep assistance.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--hover-commit-steps", type=int, default=2)
    parser.add_argument("--hover-cooldown-steps", type=int, default=6)
    parser.add_argument("--lane-spacing-factor", type=float, default=1.1)
    parser.add_argument("--waypoint-threshold-factor", type=float, default=0.35)
    parser.add_argument("--action-noise-prob", type=float, default=0.0)
    parser.add_argument("--label", type=str, default="sweep_assisted_actor")
    parser.add_argument("--output-dir", type=str, default="outputs/sweep_assisted_actor")
    return parser.parse_args()


def apply_action_noise(env: AssignmentAwareMAPPOEnvWrapper, actions: np.ndarray, probability: float) -> np.ndarray:
    noisy = actions.copy()
    for idx, agent_id in enumerate(env.agents):
        if env.env.rng.random() > probability:
            continue
        action = int(noisy[idx])
        if agent_id.startswith("fixedwing"):
            alternatives = [5, 6, 7]
        else:
            alternatives = [0, 1, 2, 3, 10]
            if action == 4:
                alternatives = [4, 10]
        mask = env.get_action_masks()[idx]
        valid = [candidate for candidate in alternatives if candidate < len(mask) and mask[candidate] > 0]
        if valid:
            noisy[idx] = int(env.env.rng.choice(valid))
    return noisy


if __name__ == "__main__":
    evaluate(parse_args())
