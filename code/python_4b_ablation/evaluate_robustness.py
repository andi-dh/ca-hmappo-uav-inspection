"""
Robustness evaluation: 5 train seeds x 2 methods x 20 env_seeds x 5 action_seeds (stochastic)
                     + 5 train seeds x 2 methods x 20 env_seeds x 1 (deterministic)
Total: 1000 stochastic + 200 deterministic = 1200 episodes

Columns: method, train_seed, env_seed, action_seed, eval_mode,
         coverage, detection, inspection, mission_success, collision, near_miss, invalid_hover
"""

import argparse
import csv
import pathlib
import sys
import random
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ablation_env_wrapper import AblationMAPPOEnvWrapper
from train_ablation_mappo import HeterogeneousActor  # noqa: F401 – needed for model


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_actor(ckpt_path: pathlib.Path, device: str) -> HeterogeneousActor:
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = HeterogeneousActor(
        obs_dim=data["obs_dim"],
        action_dim=data["action_dim"],
        hidden_dim=128,
    ).to(device)
    actor.load_state_dict(data["actor"])
    actor.eval()
    return actor


def run_episode(env, actor, deterministic: bool, device: str) -> dict:
    """Run one episode. Env uses dict-based API (MAPPOEnvWrapper style).
    Actor returns raw logits tensor of shape (n_agents, action_dim).
    """
    data = env.reset()
    done = False
    info = {}
    while not done:
        obs = data["obs_n"]  # (n_agents, obs_dim)
        obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
        masks_np = env.get_action_masks() if hasattr(env, "get_action_masks") else None
        masks_t = (torch.tensor(masks_np, dtype=torch.float32, device=device)
                   if masks_np is not None else None)

        with torch.no_grad():
            logits = actor(obs_t, masks_t)  # (n_agents, action_dim)

        if masks_t is not None:
            logits = logits + (1 - masks_t) * (-1e9)

        if deterministic:
            actions = logits.argmax(dim=-1).tolist()
        else:
            actions = torch.distributions.Categorical(logits=logits).sample().tolist()

        data = env.step([int(a) for a in actions])
        done = bool(data.get("done", False))
        last_data = data

    info = last_data.get("info", last_data)  # nested dict or flat dict
    return {
        "coverage":        float(info.get("coverage_ratio", 0.0)),
        "detection":       float(info.get("target_detection_rate", 0.0)),
        "inspection":      float(info.get("inspection_success_rate", 0.0)),
        "mission_success": float(info.get("mission_success", 0.0)),
        "collision":       int(info.get("collision_count", 0)),
        "near_miss":       int(info.get("near_miss_count", 0)),
        "invalid_hover":   int(info.get("invalid_action_count", 0)),
    }


def get_action_masks_from_env(env):
    """Try to get action masks; return None if not supported."""
    try:
        return env.get_action_masks()
    except AttributeError:
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str,
                        default="models/4b_ablation/robustness_v2")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/robustness_eval")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--n-env-seeds", type=int, default=20)
    parser.add_argument("--n-action-seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_base = pathlib.Path(args.model_base)
    out_dir    = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv    = out_dir / "robustness_raw.csv"

    env_seeds    = list(range(args.n_env_seeds))          # 0..19
    action_seeds = list(range(100, 100 + args.n_action_seeds))  # 100..104

    METHODS = {
        "full_ca_hmappo":                ("full_ca_hmappo",
                                          True, True, True),
        "heterogeneous_heads_only":      ("guidance_0_capability_0_safety_0",
                                          False, False, False),
    }
    TRAIN_SEEDS = [0, 1, 2, 3, 4]

    fieldnames = ["method", "train_seed", "env_seed", "action_seed", "eval_mode",
                  "coverage", "detection", "inspection", "mission_success",
                  "collision", "near_miss", "invalid_hover"]

    total_done = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for method_key, (folder_name, use_g, use_cr, use_s) in METHODS.items():
            for train_seed in TRAIN_SEEDS:
                ckpt = model_base / f"seed_{train_seed}" / folder_name / f"{folder_name}_scenario_{args.scenario}.pt"
                if not ckpt.exists():
                    print(f"  MISSING: {ckpt}")
                    continue
                actor = load_actor(ckpt, args.device)
                print(f"\n[{method_key}  train_seed={train_seed}]  {ckpt.name}")

                for env_seed in env_seeds:
                    # ── stochastic ──────────────────────────────────────────
                    for act_seed in action_seeds:
                        set_seeds(act_seed)
                        env = AblationMAPPOEnvWrapper(
                            scenario=args.scenario,
                            seed=env_seed,
                            use_guidance=use_g,
                            use_capability_reward=use_cr,
                            use_safety_shaping=use_s,
                        )
                        metrics = run_episode(env, actor, deterministic=False,
                                              device=args.device)
                        env.close()
                        row = dict(method=method_key, train_seed=train_seed,
                                   env_seed=env_seed, action_seed=act_seed,
                                   eval_mode="stochastic", **metrics)
                        writer.writerow(row)
                        f.flush()
                        total_done += 1

                    # ── deterministic ───────────────────────────────────────
                    set_seeds(0)
                    env = AblationMAPPOEnvWrapper(
                        scenario=args.scenario,
                        seed=env_seed,
                        use_guidance=use_g,
                        use_capability_reward=use_cr,
                        use_safety_shaping=use_s,
                    )
                    metrics = run_episode(env, actor, deterministic=True,
                                          device=args.device)
                    env.close()
                    row = dict(method=method_key, train_seed=train_seed,
                               env_seed=env_seed, action_seed="NA",
                               eval_mode="deterministic", **metrics)
                    writer.writerow(row)
                    f.flush()
                    total_done += 1

                print(f"  env_seeds done={len(env_seeds)}  total_episodes={total_done}")

    print(f"\nAll done. {total_done} episodes. CSV: {out_csv}")


if __name__ == "__main__":
    main()
