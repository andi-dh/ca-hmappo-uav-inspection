from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
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
from capability_aware_env_wrapper import CapabilityAwareMAPPOEnvWrapper  # noqa: E402
from assignment_aware_env_wrapper import AssignmentAwareMAPPOEnvWrapper  # noqa: E402


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Using device: {device}")

    env = _make_env(args.env_mode, args.scenario, args.seed, args.hover_commit_steps, args.hover_cooldown_steps)
    actor = HeterogeneousActor(env.obs_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    critic = Critic(env.global_state_dim, hidden_dim=args.hidden_dim).to(device)
    if args.init_model:
        checkpoint = torch.load(args.init_model, map_location=device)
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        print(f"Loaded initial model: {args.init_model}")

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.value_learning_rate)
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
        loss_metrics = _update_hatrpo(actor, critic, critic_optimizer, batch, env.n_agents, args)
        metrics = env.compute_metrics()
        row = {
            "update": update,
            "scenario": args.scenario,
            "env_mode": args.env_mode,
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
        if update == 1 or update % args.log_interval == 0:
            print(
                f"update={update} reward={row['rollout_reward_mean']:.3f} coverage={row['coverage_ratio']:.3f} "
                f"inspection={row['inspection_success_rate']:.3f} success={row['mission_success']} "
                f"policy_loss={row['policy_loss']:.3f} value_loss={row['value_loss']:.3f} kl={row['approx_kl']:.5f}"
            )
        if args.checkpoint_interval > 0 and update % args.checkpoint_interval == 0:
            _save_model(actor, critic, env, args, model_dir / f"{_model_prefix(args)}_update_{update}.pt")

    curve_file = output_dir / f"training_curve_{_model_prefix(args)}.csv"
    _write_rows_csv(curve_rows, curve_file)
    model_file = model_dir / f"{_model_prefix(args)}.pt"
    _save_model(actor, critic, env, args, model_file)
    print(f"Saved training curve: {curve_file}")
    print(f"Saved model: {model_file}")


def _update_hatrpo(
    actor: HeterogeneousActor,
    critic: Critic,
    critic_optimizer: torch.optim.Optimizer,
    batch: dict,
    n_agents: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "accepted_fraction": 0.0, "step_norm": 0.0}
    total = batch["obs"].shape[0]
    agent_ids = torch.arange(total, device=batch["obs"].device) % n_agents
    accepted = 0
    attempted = 0

    for _ in range(args.update_epochs):
        order = torch.randperm(n_agents, device=batch["obs"].device).tolist()
        for agent_idx in order:
            selected = agent_ids == int(agent_idx)
            if int(selected.sum().item()) == 0:
                continue
            update_metrics = _trpo_agent_step(actor, batch, selected, args)
            attempted += 1
            accepted += int(update_metrics["accepted"])
            metrics.update(update_metrics)

        for _ in range(args.value_epochs):
            value_loss = F.mse_loss(critic(batch["global_states"]), batch["returns"])
            critic_optimizer.zero_grad()
            (args.value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            critic_optimizer.step()

        with torch.no_grad():
            dist_all = Categorical(logits=actor(batch["obs"], batch["action_masks"]))
            new_log_probs_all = dist_all.log_prob(batch["actions"])
            metrics["value_loss"] = float(F.mse_loss(critic(batch["global_states"]), batch["returns"]).detach().cpu().item())
            metrics["entropy"] = float(dist_all.entropy().mean().detach().cpu().item())
            metrics["approx_kl"] = float((batch["old_log_probs"] - new_log_probs_all).mean().detach().cpu().item())
    metrics["accepted_fraction"] = accepted / max(1, attempted)
    return metrics


def _trpo_agent_step(actor: HeterogeneousActor, batch: dict, selected: torch.Tensor, args: argparse.Namespace) -> dict[str, float]:
    obs = batch["obs"][selected]
    masks = batch["action_masks"][selected]
    actions = batch["actions"][selected]
    old_log_probs = batch["old_log_probs"][selected]
    advantages = batch["advantages"][selected]

    with torch.no_grad():
        old_logits = actor(obs, masks)
        old_dist = Categorical(logits=old_logits)
        old_probs = old_dist.probs.detach()
        old_log_all = torch.log(old_probs.clamp_min(1e-8))
        old_objective = _surrogate_objective(actor, obs, masks, actions, old_log_probs, advantages)

    objective = _surrogate_objective(actor, obs, masks, actions, old_log_probs, advantages)
    grads = torch.autograd.grad(objective, actor.parameters(), allow_unused=True)
    flat_grad = _flat_grad(grads, actor).detach()
    if not torch.isfinite(flat_grad).all() or float(flat_grad.norm().item()) <= 1e-12:
        return {"policy_loss": float(-objective.detach().cpu().item()), "approx_kl": 0.0, "accepted": 0.0, "step_norm": 0.0}

    def hvp(vector: torch.Tensor) -> torch.Tensor:
        return _fisher_vector_product(actor, obs, masks, old_probs, old_log_all, vector, args.damping)

    step_dir = _conjugate_gradient(hvp, flat_grad, args.cg_iters, args.cg_tol)
    hvp_step = hvp(step_dir)
    shs = 0.5 * torch.dot(step_dir, hvp_step)
    if not torch.isfinite(shs) or float(shs.item()) <= 1e-12:
        return {"policy_loss": float(-objective.detach().cpu().item()), "approx_kl": 0.0, "accepted": 0.0, "step_norm": 0.0}
    full_step = step_dir * torch.sqrt(torch.as_tensor(args.max_kl, device=step_dir.device) / shs)
    old_params = _flat_params(actor).detach()

    accepted = False
    final_kl = 0.0
    final_objective = old_objective
    for backtrack in range(args.backtrack_iters):
        step_frac = args.backtrack_coef**backtrack
        new_params = old_params + step_frac * full_step
        _set_flat_params(actor, new_params)
        with torch.no_grad():
            new_objective = _surrogate_objective(actor, obs, masks, actions, old_log_probs, advantages)
            new_kl = _kl_from_old(actor, obs, masks, old_probs, old_log_all)
        objective_gain = float((new_objective - old_objective).detach().cpu().item())
        final_kl = float(new_kl.detach().cpu().item())
        final_objective = new_objective
        if objective_gain >= 0.0 and final_kl <= args.max_kl and np.isfinite(final_kl):
            accepted = True
            break
    if not accepted:
        _set_flat_params(actor, old_params)
        final_objective = old_objective

    return {
        "policy_loss": float(-final_objective.detach().cpu().item()),
        "approx_kl": final_kl,
        "accepted": float(accepted),
        "step_norm": float(full_step.norm().detach().cpu().item()),
    }


def _surrogate_objective(actor: HeterogeneousActor, obs: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    dist = Categorical(logits=actor(obs, masks))
    ratio = torch.exp(dist.log_prob(actions) - old_log_probs)
    return (ratio * advantages).mean()


def _kl_from_old(actor: HeterogeneousActor, obs: torch.Tensor, masks: torch.Tensor, old_probs: torch.Tensor, old_log_all: torch.Tensor) -> torch.Tensor:
    new_dist = Categorical(logits=actor(obs, masks))
    new_log_all = torch.log(new_dist.probs.clamp_min(1e-8))
    return (old_probs * (old_log_all - new_log_all)).sum(dim=-1).mean()


def _fisher_vector_product(
    actor: HeterogeneousActor,
    obs: torch.Tensor,
    masks: torch.Tensor,
    old_probs: torch.Tensor,
    old_log_all: torch.Tensor,
    vector: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    kl = _kl_from_old(actor, obs, masks, old_probs, old_log_all)
    grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True, allow_unused=True)
    flat_grad_kl = _flat_grad(grads, actor)
    grad_vector = torch.dot(flat_grad_kl, vector)
    hvp = torch.autograd.grad(grad_vector, actor.parameters(), allow_unused=True)
    return _flat_grad(hvp, actor).detach() + damping * vector


def _conjugate_gradient(hvp, b: torch.Tensor, cg_iters: int, residual_tol: float) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
        hvp_p = hvp(p)
        alpha = rdotr / (torch.dot(p, hvp_p) + 1e-8)
        x += alpha * p
        r -= alpha * hvp_p
        new_rdotr = torch.dot(r, r)
        if float(new_rdotr.item()) < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def _flat_grad(grads: tuple[torch.Tensor | None, ...], module: nn.Module) -> torch.Tensor:
    parts = []
    for grad, param in zip(grads, module.parameters()):
        if grad is None:
            parts.append(torch.zeros_like(param).reshape(-1))
        else:
            parts.append(grad.contiguous().reshape(-1))
    return torch.cat(parts)


def _flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in module.parameters()])


def _set_flat_params(module: nn.Module, flat_params: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for param in module.parameters():
            numel = param.numel()
            param.copy_(flat_params[offset : offset + numel].view_as(param))
            offset += numel


def _make_env(env_mode: str, scenario: int, seed: int, hover_commit_steps: int, hover_cooldown_steps: int):
    if env_mode == "assignment_aware":
        return AssignmentAwareMAPPOEnvWrapper(scenario=scenario, seed=seed, hover_commit_steps=hover_commit_steps, hover_cooldown_steps=hover_cooldown_steps)
    return CapabilityAwareMAPPOEnvWrapper(scenario=scenario, seed=seed)


def _model_prefix(args: argparse.Namespace) -> str:
    return f"hatrpo_{args.env_mode}_scenario_{args.scenario}"


def _save_model(actor: HeterogeneousActor, critic: Critic, env, args: argparse.Namespace, path: Path) -> None:
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "obs_dim": env.obs_dim,
            "action_dim": env.action_dim,
            "global_state_dim": env.global_state_dim,
            "scenario": args.scenario,
            "env_mode": args.env_mode,
            "algo": "hatrpo",
        },
        path,
    )


def _write_rows_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reviewer HATRPO baselines.")
    parser.add_argument("--scenario", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--env-mode", type=str, default="capability_aware", choices=["capability_aware", "assignment_aware"])
    parser.add_argument("--num-updates", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--value-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--cg-iters", type=int, default=10)
    parser.add_argument("--cg-tol", type=float, default=1e-10)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--backtrack-iters", type=int, default=10)
    parser.add_argument("--backtrack-coef", type=float, default=0.8)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--value-epochs", type=int, default=2)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--init-model", type=str, default="")
    parser.add_argument("--hover-commit-steps", type=int, default=2)
    parser.add_argument("--hover-cooldown-steps", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="outputs/rev_reviewer_baselines")
    parser.add_argument("--model-dir", type=str, default="models/rev_reviewer_baselines")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
