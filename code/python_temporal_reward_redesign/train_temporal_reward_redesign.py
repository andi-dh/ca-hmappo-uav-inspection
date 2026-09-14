from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for folder in ["python_3a_homogeneous_mappo", "python_3b_heterogeneous_mappo", "python_3c_capability_aware_mappo"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mappo_buffer import MAPPOBuffer  # noqa: E402
from mappo_networks import Critic  # noqa: E402
from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from temporal_reward_env_wrapper import TemporalRewardRedesignMAPPOEnvWrapper  # noqa: E402


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = TemporalRewardRedesignMAPPOEnvWrapper(scenario=args.scenario, seed=args.seed)
    actor = HeterogeneousActor(env.obs_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    critic = Critic(env.global_state_dim, hidden_dim=args.hidden_dim).to(device)
    if args.init_model:
        checkpoint = torch.load(args.init_model, map_location=device)
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        print(f"Loaded initial model: {args.init_model}")
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate)
    buffer = MAPPOBuffer(gamma=args.gamma, gae_lambda=args.gae_lambda)
    data = env.reset(seed=args.seed)
    episode_count = 0
    rows: list[dict] = []
    for update in range(1, args.num_updates + 1):
        buffer.clear()
        rollout_reward = 0.0
        for _ in range(args.rollout_steps):
            obs = data["obs_n"]
            state = data["state_global"]
            masks = data["action_mask_n"]
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                masks_t = torch.as_tensor(masks, dtype=torch.float32, device=device)
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                dist = Categorical(logits=actor(obs_t, masks_t))
                actions_t = dist.sample()
                log_probs_t = dist.log_prob(actions_t)
                value_t = critic(state_t)
            step_data = env.step(actions_t.cpu().numpy())
            buffer.add(obs, state, actions_t.cpu().numpy(), log_probs_t.cpu().numpy(), step_data["reward_n"], step_data["done_n"], float(value_t.item()), masks)
            rollout_reward += float(np.mean(step_data["reward_n"]))
            data = step_data
            if bool(step_data["done"]):
                episode_count += 1
                data = env.reset(seed=args.seed + episode_count)
        with torch.no_grad():
            last_state = torch.as_tensor(data["state_global"], dtype=torch.float32, device=device).unsqueeze(0)
            last_value = float(critic(last_state).item())
        batch = buffer.as_tensors(device=device, last_value=last_value)
        loss_metrics = update_actor_critic(actor, critic, optimizer, batch, args)
        metrics = env.compute_metrics()
        row = {
            "update": update,
            "episode_count": episode_count,
            "rollout_reward_mean": rollout_reward / max(1, args.rollout_steps),
            "coverage_ratio": metrics["coverage_ratio"],
            "target_detection_rate": metrics["target_detection_rate"],
            "inspection_success_rate": metrics["inspection_success_rate"],
            "mission_success": metrics["mission_success"],
            "collision_count": metrics["collision_count"],
            "invalid_action_count": metrics["invalid_action_count"],
            **loss_metrics,
        }
        rows.append(row)
        if update == 1 or update % args.log_interval == 0:
            print(f"update={update} reward={row['rollout_reward_mean']:.3f} coverage={row['coverage_ratio']:.3f} detection={row['target_detection_rate']:.3f} inspection={row['inspection_success_rate']:.3f} success={row['mission_success']} collision={row['collision_count']:.2f}")
        if args.checkpoint_interval > 0 and update % args.checkpoint_interval == 0:
            save_model(actor, critic, env, args, model_dir / f"temporal_reward_redesign_scenario_{args.scenario}_update_{update}.pt")
    write_rows(rows, output_dir / f"training_curve_temporal_reward_redesign_scenario_{args.scenario}.csv")
    save_model(actor, critic, env, args, model_dir / f"temporal_reward_redesign_scenario_{args.scenario}.pt")


def update_actor_critic(actor, critic, optimizer, batch: dict, args: argparse.Namespace) -> dict[str, float]:
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(args.update_epochs):
        dist = Categorical(logits=actor(batch["obs"], batch["action_masks"]))
        new_log_probs = dist.log_prob(batch["actions"])
        entropy = dist.entropy().mean()
        ratio = torch.exp(new_log_probs - batch["old_log_probs"])
        policy_loss = -torch.min(ratio * batch["advantages"], torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * batch["advantages"]).mean()
        value_loss = F.mse_loss(critic(batch["global_states"]), batch["returns"])
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
        optimizer.step()
        metrics = {"policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item()), "entropy": float(entropy.item())}
    return metrics


def save_model(actor, critic, env, args: argparse.Namespace, path: Path) -> None:
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "obs_dim": env.obs_dim, "action_dim": env.action_dim, "global_state_dim": env.global_state_dim, "scenario": args.scenario, "env_mode": "temporal_reward_redesign"}, path)


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
    parser.add_argument("--scenario", type=int, default=1, choices=[1])
    parser.add_argument("--num-updates", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--init-model", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/temporal_reward_redesign")
    parser.add_argument("--model-dir", type=str, default="models/temporal_reward_redesign")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
