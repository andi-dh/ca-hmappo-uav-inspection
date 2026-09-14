from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for folder in ["python_3a_homogeneous_mappo", "python_3b_heterogeneous_mappo"]:
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mappo_buffer import MAPPOBuffer  # noqa: E402
from mappo_networks import Critic  # noqa: E402
from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from ablation_env_wrapper import AblationMAPPOEnvWrapper, ablation_name  # noqa: E402


def train(args: argparse.Namespace) -> None:
    use_guidance = bool(args.use_guidance)
    use_capability_reward = bool(args.use_capability_reward)
    use_safety_shaping = bool(args.use_safety_shaping)
    name = ablation_name(use_guidance, use_capability_reward, use_safety_shaping)
    output_dir = Path(args.output_dir) / name
    model_dir = Path(args.model_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    print(f"Ablation: {name}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = AblationMAPPOEnvWrapper(
        scenario=args.scenario,
        seed=args.seed,
        use_guidance=use_guidance,
        use_capability_reward=use_capability_reward,
        use_safety_shaping=use_safety_shaping,
    )
    actor = HeterogeneousActor(env.obs_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    critic = Critic(env.global_state_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate)
    buffer = MAPPOBuffer(gamma=args.gamma, gae_lambda=args.gae_lambda)
    data = env.reset(seed=args.seed)
    episode_count = 0
    curve_rows: list[dict] = []

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
            last_state_t = torch.as_tensor(data["state_global"], dtype=torch.float32, device=device).unsqueeze(0)
            last_value = float(critic(last_state_t).item())
        batch = buffer.as_tensors(device=device, last_value=last_value)
        loss_metrics = _update_actor_critic(actor, critic, optimizer, batch, args)
        metrics = env.compute_metrics()
        row = {
            "update": update,
            "ablation": name,
            "scenario": args.scenario,
            "rollout_reward_mean": rollout_reward / max(1, args.rollout_steps),
            "episode_count": episode_count,
            "coverage_ratio": metrics["coverage_ratio"],
            "inspection_success_rate": metrics["inspection_success_rate"],
            "target_detection_rate": metrics["target_detection_rate"],
            "mission_success": metrics["mission_success"],
            "mission_time": metrics["mission_time"],
            "collision_count": metrics["collision_count"],
            "invalid_action_count": metrics["invalid_action_count"],
            **loss_metrics,
        }
        curve_rows.append(row)
        if update % args.log_interval == 0 or update == 1:
            print(
                f"update={update} reward={row['rollout_reward_mean']:.3f} coverage={row['coverage_ratio']:.3f} "
                f"inspection={row['inspection_success_rate']:.3f} collision={row['collision_count']} "
                f"entropy={row['entropy']:.3f}"
            )
        if args.checkpoint_interval > 0 and update % args.checkpoint_interval == 0:
            _save_model(actor, critic, env, args.scenario, model_dir / f"{name}_scenario_{args.scenario}_update_{update}.pt", name)

    _write_rows_csv(curve_rows, output_dir / f"training_curve_{name}_scenario_{args.scenario}.csv")
    _save_training_curve(curve_rows, output_dir / f"training_curve_{name}_scenario_{args.scenario}.png")
    _save_model(actor, critic, env, args.scenario, model_dir / f"{name}_scenario_{args.scenario}.pt", name)


def _save_model(actor: HeterogeneousActor, critic: Critic, env: AblationMAPPOEnvWrapper, scenario: int, path: Path, ablation: str) -> None:
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "obs_dim": env.obs_dim,
            "action_dim": env.action_dim,
            "global_state_dim": env.global_state_dim,
            "scenario": scenario,
            "ablation": ablation,
            "use_guidance": env.use_guidance,
            "use_capability_reward": env.use_capability_reward,
            "use_safety_shaping": env.use_safety_shaping,
        },
        path,
    )


def _update_actor_critic(actor: HeterogeneousActor, critic: Critic, optimizer: torch.optim.Optimizer, batch: dict, args: argparse.Namespace) -> dict[str, float]:
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0, "mean_action_probability": 0.0}
    for _ in range(args.update_epochs):
        dist = Categorical(logits=actor(batch["obs"], batch["action_masks"]))
        new_log_probs = dist.log_prob(batch["actions"])
        entropy = dist.entropy().mean()
        ratio = torch.exp(new_log_probs - batch["old_log_probs"])
        approx_kl = (batch["old_log_probs"] - new_log_probs).mean()
        clip_fraction = ((ratio - 1.0).abs() > args.clip_ratio).float().mean()
        mean_action_probability = torch.exp(new_log_probs).mean()
        policy_loss = -torch.min(ratio * batch["advantages"], torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * batch["advantages"]).mean()
        value_loss = F.mse_loss(critic(batch["global_states"]), batch["returns"])
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
        optimizer.step()
        metrics = {
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "clip_fraction": float(clip_fraction.detach().cpu().item()),
            "mean_action_probability": float(mean_action_probability.detach().cpu().item()),
        }
    return metrics


def _write_rows_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_training_curve(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    updates = [row["update"] for row in rows]
    rewards = [row["rollout_reward_mean"] for row in rows]
    inspections = [row["inspection_success_rate"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(updates, rewards, color="tab:blue")
    ax1.set_xlabel("update")
    ax1.set_ylabel("rollout reward", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(updates, inspections, color="tab:green")
    ax2.set_ylabel("inspection success", color="tab:green")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CA-HMAPPO ablation variant.")
    parser.add_argument("--scenario", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-guidance", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-capability-reward", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-safety-shaping", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num-updates", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--output-dir", type=str, default="outputs/4b_ablation")
    parser.add_argument("--model-dir", type=str, default="models/4b_ablation")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
