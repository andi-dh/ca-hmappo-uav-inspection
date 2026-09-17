"""Zero-shot candidate-information robustness evaluation for CA-HMAPPO.

6 conditions x 20 seeds = 120 evaluation episodes.
No retraining; uses the original CA-HMAPPO checkpoint.

Conditions
----------
clean            : sigma_loc=0,   p_miss=0.00
loc_25           : sigma_loc=25,  p_miss=0.00
loc_50           : sigma_loc=50,  p_miss=0.00
loc_100          : sigma_loc=100, p_miss=0.00
miss_10          : sigma_loc=0,   p_miss=0.10
miss_20          : sigma_loc=0,   p_miss=0.20
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
for folder in ["homogeneous_mappo", "heterogeneous_mappo"]:
    p = ROOT / folder
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from heterogeneous_mappo_networks import HeterogeneousActor  # noqa: E402
from capability_aware_env_wrapper import CapabilityAwareMAPPOEnvWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Perturbed wrapper
# ---------------------------------------------------------------------------

class CandidateRobustnessWrapper(CapabilityAwareMAPPOEnvWrapper):
    """Overrides _add_candidate to inject localization noise and report drops.

    Parameters
    ----------
    sigma_loc : float
        Std-dev (metres) of Gaussian localization noise added once per
        candidate report.  0 = clean.
    p_miss : float
        Probability that a successfully detected candidate report is
        suppressed before entering the active-candidate table.
    """

    def __init__(
        self,
        scenario: int = 1,
        seed: int | None = None,
        sigma_loc: float = 0.0,
        p_miss: float = 0.0,
    ) -> None:
        super().__init__(scenario=scenario, seed=seed)
        self.sigma_loc = float(sigma_loc)
        self.p_miss = float(p_miss)
        # track sensor detections separately from candidate insertions
        self._sensor_detections: int = 0
        # per-target reported coordinate for true (non-false-alarm) detections;
        # a target index is absent here iff its report was dropped (missed report)
        self._reported_targets: Dict[int, tuple[float, float]] = {}

    def reset(self, seed: int | None = None):
        self._sensor_detections = 0
        self._reported_targets = {}
        data = super().reset(seed=seed)
        return data

    # ------------------------------------------------------------------
    # Monkey-patch the base env's detect_targets so we can distinguish a
    # true-target report (perturbed by sigma_loc/p_miss, tracked per target
    # index in self._reported_targets) from a false-alarm report (passed
    # through to candidate_targets unperturbed, exactly as in the original
    # environment). This preserves the original priority order: true
    # detected-but-uninspected targets take precedence over false alarms.
    # ------------------------------------------------------------------

    def step(self, actions):
        original_add = self.env._add_candidate
        original_detect_targets = self.env.detect_targets
        rng = self.env.rng
        area_size = float(self.env.area_size)
        sigma = self.sigma_loc
        p_miss = self.p_miss
        sensor_counter = [self._sensor_detections]

        def patched_detect_targets() -> None:
            fixedwing = self.env.uavs["fixedwing_0"]
            distances = np.linalg.norm(self.env.targets - fixedwing.position(), axis=1)
            for idx, distance in enumerate(distances):
                if self.env.target_detected[idx] or distance > fixedwing.sensor_range:
                    continue
                if rng.random() > self.env.p_detect_fixedwing:
                    continue
                # True sensor detection fires regardless of what happens to
                # its report afterward.
                self.env.target_detected[idx] = True
                self.env.detections_count += 1
                sensor_counter[0] += 1
                true_coord = self.env.targets[idx]
                gx, gy = self.env._position_to_grid(*true_coord)
                self.env.target_probability_map[gx, gy] = 0.9
                # Missed-report suppression: sensor detected it, but no
                # report ever reaches the candidate table -> zero guidance.
                if p_miss > 0.0 and rng.random() < p_miss:
                    continue
                reported = np.array(true_coord, dtype=np.float64)
                if sigma > 0.0:
                    noise = rng.normal(0.0, sigma, size=2)
                    reported = np.clip(reported + noise, 0.0, area_size)
                self._reported_targets[idx] = (float(reported[0]), float(reported[1]))
                original_add((float(reported[0]), float(reported[1])))

            # False alarms are unperturbed secondary candidates, exactly as
            # in the original environment.
            if rng.random() <= self.env.p_false_alarm_fixedwing * 0.05:
                false_candidate = self.env._sample_near(fixedwing.position(), fixedwing.sensor_range)
                sensor_counter[0] += 1
                original_add(tuple(false_candidate.tolist()))
                gx, gy = self.env._position_to_grid(*false_candidate)
                self.env.target_probability_map[gx, gy] = max(self.env.target_probability_map[gx, gy], 0.7)

        self.env.detect_targets = patched_detect_targets
        data = super().step(actions)
        self.env.detect_targets = original_detect_targets
        self._sensor_detections = sensor_counter[0]
        return data

    def compute_metrics(self) -> dict:
        m = super().compute_metrics()
        m["sensor_detections"] = self._sensor_detections
        return m

    def _active_candidates(self) -> np.ndarray:
        """Guidance must come from the *reported* coordinate, and must keep
        the original priority order: true detected-but-uninspected targets
        take precedence over false-alarm candidates.

        `clean` (sigma_loc=0, p_miss=0) reports the exact true coordinate for
        every detected target, so this reproduces the original
        ground-truth-first behavior. `loc_*` shifts the reported coordinate
        by a fixed Gaussian error sampled once at report creation. `miss_*`
        leaves the target's index out of `_reported_targets` entirely, so it
        is unavailable for guidance even though `target_detected` is True.
        False alarms in `candidate_targets` are only used as a fallback when
        no true report is currently available, matching the original
        (unperturbed) environment's behavior.
        """
        primary = [
            np.array(coord, dtype=np.float64)
            for idx, coord in self._reported_targets.items()
            if self.env.target_detected[idx] and not self.env.target_inspected[idx]
        ]
        if primary:
            return np.asarray(primary, dtype=np.float64)

        secondary = []
        for candidate in self.env.candidate_targets:
            candidate_array = np.array(candidate, dtype=np.float64)
            if self._candidate_already_inspected(candidate_array):
                continue
            secondary.append(candidate_array)
        if not secondary:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(secondary, dtype=np.float64)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

CONDITIONS = [
    {"name": "clean",   "sigma_loc": 0,   "p_miss": 0.00},
    {"name": "loc_25",  "sigma_loc": 25,  "p_miss": 0.00},
    {"name": "loc_50",  "sigma_loc": 50,  "p_miss": 0.00},
    {"name": "loc_100", "sigma_loc": 100, "p_miss": 0.00},
    {"name": "miss_10", "sigma_loc": 0,   "p_miss": 0.10},
    {"name": "miss_20", "sigma_loc": 0,   "p_miss": 0.20},
]

METRIC_COLS = [
    "coverage_ratio",
    "target_detection_rate",
    "inspection_success_rate",
    "mission_success",
    "collision_count",
    "near_miss_count",
    "quadrotor_hover_inspect_count",
    "quadrotor_valid_hover_inspect_count",
    "quadrotor_invalid_hover_inspect_count",
    "detected_targets",
    "inspected_targets",
    "sensor_detections",
]


def run_episode(actor, env, device: str, deterministic: bool, seed: int) -> dict:
    data = env.reset(seed=seed)  # the only reset call for this episode
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


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    checkpoint = torch.load(args.model, map_location=device)
    actor = HeterogeneousActor(
        checkpoint["obs_dim"], checkpoint["action_dim"], hidden_dim=args.hidden_dim
    ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    seeds = list(range(args.base_seed, args.base_seed + args.num_seeds))
    all_rows: list[dict] = []

    for cond in CONDITIONS:
        print(f"\n--- Condition: {cond['name']} (sigma_loc={cond['sigma_loc']}, p_miss={cond['p_miss']}) ---")
        for seed in seeds:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            env = CandidateRobustnessWrapper(
                scenario=args.scenario,
                seed=seed,
                sigma_loc=cond["sigma_loc"],
                p_miss=cond["p_miss"],
            )
            metrics = run_episode(actor, env, device, args.deterministic, seed)
            row: dict = {
                "condition": cond["name"],
                "sigma_loc": cond["sigma_loc"],
                "p_miss": cond["p_miss"],
                "seed": seed,
            }
            for col in METRIC_COLS:
                row[col] = metrics.get(col, 0)
            all_rows.append(row)
            ms = float(row["mission_success"])
            insp = float(row["inspection_success_rate"])
            det = float(row["target_detection_rate"])
            print(f"  seed={seed}  mission_success={ms:.3f}  inspection={insp:.3f}  detection={det:.3f}")

    # Save all rows
    all_file = output_dir / "candidate_robustness_all_seeds.csv"
    _write_csv(all_rows, all_file)
    print(f"\nSaved all rows: {all_file}")

    # Summary per condition
    summary_rows: list[dict] = []
    for cond in CONDITIONS:
        crows = [r for r in all_rows if r["condition"] == cond["name"]]
        row: dict = {
            "condition": cond["name"],
            "sigma_loc": cond["sigma_loc"],
            "p_miss": cond["p_miss"],
            "n": len(crows),
        }
        for col in METRIC_COLS:
            vals = [float(r[col]) for r in crows]
            row[f"{col}_mean"] = round(mean(vals), 4) if vals else 0.0
            row[f"{col}_std"] = round(stdev(vals), 4) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    summary_file = output_dir / "candidate_robustness_summary.csv"
    _write_csv(summary_rows, summary_file)
    print(f"Saved summary: {summary_file}")

    # Print summary table
    print("\n=== Summary ===")
    cols_print = [
        "condition", "sigma_loc", "p_miss",
        "mission_success_mean", "mission_success_std",
        "inspection_success_rate_mean", "inspection_success_rate_std",
        "target_detection_rate_mean",
        "collision_count_mean",
        "quadrotor_invalid_hover_inspect_count_mean",
    ]
    header = "  ".join(f"{c:<42}" for c in cols_print)
    print(header)
    for row in summary_rows:
        line = "  ".join(f"{str(row.get(c, '')):<42}" for c in cols_print)
        print(line)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot candidate-information robustness evaluation."
    )
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        default=str(
            ROOT / "models/ca_hmappo/capability_aware_mappo_scenario_1.pt"
        ),
    )
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs/candidate_robustness"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
