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
for folder in ["python_3b_heterogeneous_mappo", "python_3c_capability_aware_mappo"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from temporal_reward_env_wrapper import TemporalRewardRedesignMAPPOEnvWrapper  # noqa: E402


COLS = ["coverage_ratio", "target_detection_rate", "inspection_success_rate", "mission_success", "collision_count", "invalid_action_count", "quadrotor_hover_inspect_count", "quadrotor_valid_hover_inspect_count", "quadrotor_invalid_hover_inspect_count"]


def evaluate(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    actor = HeterogeneousActor(ckpt["obs_dim"], ckpt["action_dim"], hidden_dim=args.hidden_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    rows = []
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        torch.manual_seed(seed)
        env = TemporalRewardRedesignMAPPOEnvWrapper(scenario=args.scenario, seed=seed)
        data = env.reset(seed=seed)
        diag = {"hover": 0, "valid": 0, "invalid": 0}
        while True:
            obs = torch.as_tensor(data["obs_n"], dtype=torch.float32, device=device)
            masks = torch.as_tensor(data["action_mask_n"], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = actor(obs, masks)
                actions = torch.argmax(logits, dim=-1).cpu().numpy() if args.deterministic else Categorical(logits=logits).sample().cpu().numpy()
            update_diag(env, actions, diag)
            data = env.step(actions)
            if bool(data["done"]):
                break
        row = dict(env.compute_metrics())
        row.update({"seed": seed, "quadrotor_hover_inspect_count": diag["hover"], "quadrotor_valid_hover_inspect_count": diag["valid"], "quadrotor_invalid_hover_inspect_count": diag["invalid"]})
        rows.append(row)
    summary = {f"{col}_mean": mean(float(r.get(col, 0.0)) for r in rows) for col in COLS}
    summary.update({f"{col}_std": stdev(float(r.get(col, 0.0)) for r in rows) if len(rows) > 1 else 0.0 for col in COLS})
    prefix = f"scenario_{args.scenario}_temporal_reward_redesign_{'deterministic' if args.deterministic else 'sampling'}"
    write_rows(rows, out / f"{prefix}_summary.csv")
    write_rows([summary], out / f"{prefix}_mean_std.csv")
    for key, value in summary.items():
        print(f"{key}: {value}")


def update_diag(env, actions, diag: dict) -> None:
    assignments = env._quadrotor_candidate_assignments()
    for idx, agent_id in enumerate(env.agents):
        if not agent_id.startswith("quadrotor") or int(actions[idx]) != 4:
            continue
        diag["hover"] += 1
        target = assignments.get(agent_id)
        if target is not None and float(np.linalg.norm(target - env.env.uavs[agent_id].position())) <= env.env.uavs[agent_id].sensor_range:
            diag["valid"] += 1
        else:
            diag["invalid"] += 1


def write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/temporal_reward_redesign")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
