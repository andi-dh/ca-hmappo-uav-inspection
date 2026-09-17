# Held-Out Generalization Evaluation

Zero-shot generalization evaluation of independently trained CA-HMAPPO policies across 10 evaluation conditions (one nominal reference and nine held-out shifts) without retraining, fine-tuning, reward retuning, or checkpoint reselection.

## Evaluation Design

**Principle:** Load checkpoint -> change environment condition -> evaluate (no gradient updates)

**Total episodes:** 5 training seeds x 10 conditions x 20 environment seeds x 5 action seeds = **5000 episodes**

**Statistical unit:** n=5 (five independently trained policies)

**Seed protocol** (matches `code/controlled_analysis/evaluate_robustness.py`, the training-seed robustness protocol used for Section VI-K):

- Environment seeds: `0, 1, ..., 19` — control target layout and all environment stochasticity (via the environment's own `np.random.default_rng(env_seed)`).
- Action seeds: `100, 101, ..., 104` — reseed only the global `random`/`numpy`/`torch` RNGs used for stochastic action sampling; they never touch the environment's RNG.
- Exactly **one** `env.reset(seed=env_seed)` call is made per episode, inside `run_episode()`. There is no reset outside `run_episode()` and no second reset with `seed=None` inside it.

This matters because a second `reset(seed=None)` call after the first reset re-advances the environment's RNG from wherever the first reset left it, rather than reproducing a deterministic function of `env_seed` alone — and for the `clustered`/`edge` target-distribution conditions, the target-layout seed was previously computed from the reset call's `seed` argument, which was `None` on that second call and therefore silently defaulted to `0` for every episode (so all 20 "different" environment seeds reused the exact same clustered/edge layout). Removing the duplicate reset fixes both problems at once. With the corrected protocol, the nominal condition's inspection rate (0.142) reproduces the manuscript's Section VI-K value almost exactly.

## Conditions

| No. | Condition | Description |
|-----|-----------|-------------|
| 1 | Nominal | Scenario 1 baseline |
| 2 | Clustered targets | Clustered target distribution |
| 3 | Edge-biased targets | Edge-biased target distribution |
| 4 | Sensing moderate | p_detect=0.60, p_inspect=0.80 |
| 5 | Sensing severe | p_detect=0.50, p_inspect=0.70 |
| 6 | Hard comm. 800 m | communication_radius=800 m; assignment withheld when disconnected |
| 7 | Hard comm. 400 m | communication_radius=400 m; assignment withheld when disconnected |
| 8 | Slower motion | FW 24 m/step, Quad 12 m/step (x0.8) |
| 9 | Faster motion | FW 36 m/step, Quad 18 m/step (x1.2) |
| 10 | Scenario 2 | 2 FW + 2 Quad, 10 targets, 2400x2400 m |

Localization-noise and missed-report perturbations are **intentionally excluded** here to avoid duplicating the candidate-information robustness evaluation (`code/ca_hmappo/evaluate_candidate_robustness.py`, see its README/results for that dedicated experiment), which perturbs the reported candidate coordinate directly rather than the held-out environment configuration.

The Scenario 2 condition uses `ScalableCapabilityAwareMAPPOEnvWrapper` (from `code/extended_evaluations/assignment_aware_env_wrapper.py`), which supports multiple fixed-wing agents and the larger-team environment configuration. `HeldOutConditionWrapper` inherits from this class (not the plain `CapabilityAwareMAPPOEnvWrapper`, which only distinguishes Scenario 0 from Scenario 1 and would silently fall back to the Scenario 1 configuration for `scenario=2`).

**Hard communication mechanics:** `HeldOutConditionWrapper._quadrotor_connected(agent_id)` returns `True` only if the quadrotor is within `communication_radius` of base or of any fixed-wing UAV. `_quadrotor_candidate_assignments()` is overridden so that, under `hard_communication=True`, only connected quadrotors are eligible for a candidate assignment — a disconnected quadrotor receives zero guidance (matching the guidance vector's existing "no assignment" zero-fill behavior), directly testing the learned policy's dependence on the guidance signal rather than just changing a radius value that has no behavioral effect.

## Run Evaluation

**Full evaluation (5000 episodes, ~30-60 min on GPU):**

```bash
python code/heldout_generalization/evaluate_heldout_generalization.py \
  --num-env-seeds 20 \
  --num-action-seeds 5 \
  --device cuda \
  --output-dir outputs/heldout_generalization
```

**Quick test:**

```bash
python code/heldout_generalization/evaluate_heldout_generalization.py \
  --num-env-seeds 2 \
  --num-action-seeds 2 \
  --device cuda \
  --output-dir outputs/heldout_generalization_test
```

## Generate Summary Tables

```bash
python code/heldout_generalization/summarize_generalization.py \
  --input-file outputs/heldout_generalization/raw_generalization_results.csv \
  --output-dir outputs/heldout_generalization
```

## Outputs

```
outputs/heldout_generalization/
├── raw_generalization_results.csv          # All 5000 episodes
├── checkpoint_summary.csv                  # Per-checkpoint means (5x10=50 rows)
├── condition_summary.csv                   # Mean +/- SD across 5 checkpoints (10 rows)
└── heldout_generalization_table.csv        # Formatted manuscript table
```

## Statistical Aggregation

1. **Per-checkpoint mean:** Average over 100 episodes (20 env x 5 action seeds)
2. **Condition statistics:** Mean +/- SD across 5 checkpoint means (n=5)

This ensures the statistical unit is the independently trained policy, not individual episodes.

## Checkpoints Used

Evaluation uses all 5 independently trained CA-HMAPPO checkpoints from training-seed robustness analysis:

```
models/training_seed_robustness/
├── seed_0/full_ca_hmappo/full_ca_hmappo_scenario_1.pt
├── seed_1/full_ca_hmappo/full_ca_hmappo_scenario_1.pt
├── seed_2/full_ca_hmappo/full_ca_hmappo_scenario_1.pt
├── seed_3/full_ca_hmappo/full_ca_hmappo_scenario_1.pt
└── seed_4/full_ca_hmappo/full_ca_hmappo_scenario_1.pt
```

No checkpoint selection or retraining is performed.

## Results (Completed)

Full 5000-episode evaluation has been run to completion with the corrected protocol described above. Manuscript-ready table (mean +/- SD, n=5):

| Condition | Coverage | Detection | Inspection | Mission Success | Collision | Near Miss |
|---|---|---|---|---|---|---|
| Nominal | 0.318 ± 0.289 | 0.307 ± 0.287 | 0.142 ± 0.188 | 0.000 ± 0.000 | 0.320 ± 0.076 | 4.452 ± 0.759 |
| Clustered targets | 0.303 ± 0.273 | 0.385 ± 0.372 | 0.227 ± 0.278 | 0.046 ± 0.082 | 0.340 ± 0.090 | 6.206 ± 1.118 |
| Edge-biased targets | 0.339 ± 0.302 | 0.275 ± 0.240 | 0.099 ± 0.116 | 0.000 ± 0.000 | 0.280 ± 0.043 | 4.810 ± 0.574 |
| Sensing 0.60/0.80 | 0.322 ± 0.294 | 0.305 ± 0.286 | 0.138 ± 0.188 | 0.000 ± 0.000 | 0.304 ± 0.081 | 4.368 ± 0.786 |
| Sensing 0.50/0.70 | 0.318 ± 0.288 | 0.293 ± 0.272 | 0.126 ± 0.175 | 0.000 ± 0.000 | 0.322 ± 0.084 | 4.408 ± 0.812 |
| Hard comm. 800 m | 0.326 ± 0.293 | 0.316 ± 0.293 | 0.097 ± 0.130 | 0.000 ± 0.000 | 0.312 ± 0.083 | 4.650 ± 0.601 |
| Hard comm. 400 m | 0.327 ± 0.293 | 0.307 ± 0.283 | 0.043 ± 0.058 | 0.000 ± 0.000 | 0.316 ± 0.068 | 5.284 ± 0.743 |
| Motion ×0.8 | 0.313 ± 0.278 | 0.300 ± 0.274 | 0.121 ± 0.149 | 0.000 ± 0.000 | 0.238 ± 0.155 | 5.130 ± 0.767 |
| Motion ×1.2 | 0.390 ± 0.349 | 0.373 ± 0.342 | 0.164 ± 0.205 | 0.000 ± 0.000 | 0.340 ± 0.110 | 3.726 ± 0.443 |
| Scenario 2 | 0.446 ± 0.302 | 0.439 ± 0.308 | 0.107 ± 0.138 | 0.000 ± 0.000 | 0.514 ± 0.156 | 7.152 ± 3.804 |

Performance retention relative to nominal inspection rate (0.142):

| Condition | Retention |
|---|---|
| Clustered targets | 159.9% |
| Edge-biased targets | 69.7% |
| Sensing 0.60/0.80 | 97.3% |
| Sensing 0.50/0.70 | 88.7% |
| Hard comm. 800 m | 68.3% |
| Hard comm. 400 m | 30.0% |
| Motion ×0.8 | 85.3% |
| Motion ×1.2 | 115.7% |
| Scenario 2 | 75.2% |

**Interpretation:** The independently trained policies retain most of their nominal inspection performance under moderate sensing degradation and motion-scale changes (85–97% retention), and improve under clustered target distributions and faster motion (which likely make targets easier to reach and re-detect). Generalization degrades more substantially under edge-biased target distributions (69.7%) and the larger-team Scenario 2 configuration (75.2%, with collision rate rising from 0.320 to 0.514), and degrades most severely under hard communication constraints — 68.3% retention at 800 m dropping to 30.0% at 400 m — confirming that the learned policies rely materially on the candidate-guidance signal and lose most of their inspection capability once a meaningful fraction of quadrotors are cut off from it. This is consistent with partial zero-shot generalization: robust under moderate distribution and sensing shifts, but limited under communication constraints and out-of-distribution team scale.
