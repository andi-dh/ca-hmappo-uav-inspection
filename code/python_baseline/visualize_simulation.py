from __future__ import annotations

from pathlib import Path

from heterogeneous_uav_env import HeterogeneousUAVEnv


def save_simulation_plot(env: HeterogeneousUAVEnv, output_file: str | Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = env.render()
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    fig.clf()


def main() -> None:
    env = HeterogeneousUAVEnv(seed=42)
    env.reset(seed=42)
    save_simulation_plot(env, "outputs/baseline/initial_environment.png")
    print("Saved outputs/baseline/initial_environment.png")


if __name__ == "__main__":
    main()
