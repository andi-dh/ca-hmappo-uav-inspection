from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for folder in ["python_3a_homogeneous_mappo", "python_3b_heterogeneous_mappo", "python_3c_capability_aware_mappo", "python_rev_reviewer_baselines"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mappo_networks import Critic  # noqa: E402
from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from assignment_aware_env_wrapper import AssignmentAwareMAPPOEnvWrapper  # noqa: E402
from evaluate_hierarchical_task_allocation import HierarchicalTaskAllocationPolicy  # noqa: E402


def collect_demonstrations(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, AssignmentAwareMAPPOEnvWrapper]:
    obs_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    env = AssignmentAwareMAPPOEnvWrapper(
        scenario=args.scenario,
        seed=args.seed,
        hover_commit_steps=args.hover_commit_steps,
        hover_cooldown_steps=args.hover_cooldown_steps,
    )
    for episode in range(args.episodes):
        seed = args.seed + episode
        data = env.reset(seed=seed)
        policy = HierarchicalTaskAllocationPolicy(
            env,
            lane_spacing_factor=args.lane_spacing_factor,
            waypoint_threshold_factor=args.waypoint_threshold_factor,
        )
        while True:
            actions = np.asarray(policy.act(), dtype=np.int64)
            obs_rows.append(np.asarray(data["obs_n"], dtype=np.float32))
            mask_rows.append(np.asarray(data["action_mask_n"], dtype=np.float32))
            action_rows.append(actions)
            data = env.step(actions)
            if bool(data["done"]):
                break
    obs = np.concatenate(obs_rows, axis=0)
    masks = np.concatenate(mask_rows, axis=0)
    actions = np.concatenate(action_rows, axis=0)
    return obs, masks, actions, env


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    obs, masks, actions, env = collect_demonstrations(args)
    actor = HeterogeneousActor(env.obs_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    critic = Critic(env.global_state_dim, hidden_dim=args.hidden_dim).to(device)
    if args.init_model:
        checkpoint = torch.load(args.init_model, map_location=device)
        actor.load_state_dict(checkpoint["actor"])
        if "critic" in checkpoint:
            critic.load_state_dict(checkpoint["critic"])
        print(f"Loaded initial model: {args.init_model}")

    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(masks, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    n = obs_t.shape[0]
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        total_loss = 0.0
        total_correct = 0
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            logits = actor(obs_t[idx], masks_t[idx])
            loss = F.cross_entropy(logits, actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item()) * idx.numel()
            total_correct += int((logits.argmax(dim=-1) == actions_t[idx]).sum().item())
        row = {"epoch": epoch, "loss": total_loss / n, "accuracy": total_correct / n, "samples": n}
        rows.append(row)
        if epoch == 1 or epoch % args.log_interval == 0:
            print(f"epoch={epoch} loss={row['loss']:.4f} accuracy={row['accuracy']:.4f} samples={n}")
        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            save(actor, critic, env, args, model_dir / f"bc_structured_assignment_scenario_{args.scenario}_epoch_{epoch}.pt")
    save(actor, critic, env, args, model_dir / f"bc_structured_assignment_scenario_{args.scenario}.pt")
    write_rows(rows, output_dir / f"training_curve_bc_structured_assignment_scenario_{args.scenario}.csv")


def save(actor: HeterogeneousActor, critic: Critic, env: AssignmentAwareMAPPOEnvWrapper, args: argparse.Namespace, path: Path) -> None:
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "obs_dim": env.obs_dim,
            "action_dim": env.action_dim,
            "global_state_dim": env.global_state_dim,
            "scenario": args.scenario,
            "env_mode": "assignment_aware",
            "algo": "bc_structured_ca_hmappo",
        },
        path,
    )


def write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior cloning warm-start from structured assignment demonstrations.")
    parser.add_argument("--scenario", type=int, default=1, choices=[1])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--init-model", type=str, default="")
    parser.add_argument("--hover-commit-steps", type=int, default=2)
    parser.add_argument("--hover-cooldown-steps", type=int, default=6)
    parser.add_argument("--lane-spacing-factor", type=float, default=1.1)
    parser.add_argument("--waypoint-threshold-factor", type=float, default=0.35)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs/bc_structured_assignment")
    parser.add_argument("--model-dir", type=str, default="models/bc_structured_assignment")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
