# CA-HMAPPO

Code, trained models, and experimental results for:

**Capability-Aware Heterogeneous MAPPO for Cooperative Coverage and Inspection Using Fixed-Wing and Quadrotor UAVs**

This repository contains the implementation and experimental artifacts for CA-HMAPPO, a task-specific heterogeneous multi-agent reinforcement learning framework for cooperative fixed-wing UAV coverage and quadrotor target inspection.

The repository includes training and evaluation code, trained checkpoints, raw and processed experimental results, and scripts used to reproduce the main tables and figures reported in the manuscript.

## Repository Structure

```
ca-hmappo/
├── code/
│   ├── simulation_environment/    # UAV mission simulator and base environment
│   ├── homogeneous_mappo/         # Homogeneous MAPPO baseline
│   ├── heterogeneous_mappo/       # Heterogeneous MAPPO baseline
│   ├── ca_hmappo/                 # CA-HMAPPO training and evaluation
│   ├── controlled_analysis/       # Ablation experiments (G, CR, S factors)
│   ├── trajectory_analysis/       # Single-episode trajectory diagnostics
│   ├── temporal_constrained_variant/  # Temporal reward redesign
│   ├── extended_evaluations/      # Additional baselines and sensitivity tests
│   └── heldout_generalization/    # Zero-shot generalization across 10 evaluation conditions
├── models/
│   ├── homogeneous_mappo/
│   ├── heterogeneous_mappo/
│   ├── ca_hmappo/                 # Original terminal 3000-update checkpoint
│   ├── controlled_analysis/       # Ablation checkpoints
│   ├── training_seed_robustness/  # Independent training seeds 0–4
│   ├── temporal_constrained_variant/
│   └── extended_evaluations/
├── outputs/
│   ├── main_benchmark/            # Primary evaluation results
│   ├── rule_based_baselines/
│   ├── homogeneous_mappo/
│   ├── heterogeneous_mappo/
│   ├── ca_hmappo/
│   ├── controlled_analysis/
│   ├── training_seed_robustness/
│   ├── candidate_robustness/      # Localization noise and missed-report tests
│   ├── temporal_constrained_variant/
│   ├── extended_evaluations/
│   ├── hybrid_sweep/              # Structured sweep diagnostic
│   ├── trajectory_analysis/
│   └── heldout_generalization/    # Zero-shot generalization results
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Tested environment:
- Python 3.11.16
- NumPy 2.4.6
- PyTorch 2.14.0+cu126
- Matplotlib 3.11.2, pandas 3.0.5, SciPy 1.17.1

GPU is recommended but not required for evaluation.

## Main CA-HMAPPO Checkpoint

The original Scenario 1 results reported in the manuscript use the terminal checkpoint from the 3000-update CA-HMAPPO training run:

```
models/ca_hmappo/capability_aware_mappo_scenario_1.pt
```

This checkpoint is tensor-identical to `models/ca_hmappo/capability_aware_mappo_scenario_1_update_3000.pt`.

**The reported checkpoint was not selected using a best-checkpoint criterion.**

## Running the Main Evaluation

Evaluate the main CA-HMAPPO checkpoint on 20 environment seeds:

```bash
python code/ca_hmappo/evaluate_capability_aware_mappo.py \
  --scenario 1 --model models/ca_hmappo/capability_aware_mappo_scenario_1.pt \
  --num-seeds 20 --base-seed 0 --device cuda \
  --output-dir outputs/ca_hmappo
```

Generate detailed diagnostics for seed 16:

```bash
python code/trajectory_analysis/generate_trajectory_rollout.py \
  --scenario 1 --seed 16 --model models/ca_hmappo/capability_aware_mappo_scenario_1.pt \
  --device cuda --output-dir outputs/trajectory_analysis

python code/trajectory_analysis/plot_trajectory_analysis.py \
  --seed 16 --output-dir outputs/trajectory_analysis

python code/trajectory_analysis/build_qualitative_summary.py \
  --seed 16 --output-dir outputs/trajectory_analysis
```

## Controlled Contribution Analysis

The controlled ablation experiments test the individual contributions of candidate guidance (G), capability-aware reward (CR), and safety mechanisms (S). Trained checkpoints are available under `models/controlled_analysis/`. Evaluation outputs are under `outputs/controlled_analysis/`.

Evaluate individual controlled variants using:

```bash
python code/controlled_analysis/evaluate_controlled_variants.py \
  --scenario 1 --model models/controlled_analysis/[checkpoint_path] \
  --num-seeds 20 --device cuda \
  --output-dir outputs/controlled_analysis
```

Replace `[checkpoint_path]` with the specific variant checkpoint (e.g., G0_CR0_S0, G1_CR1_S1, etc.).

## Training-Seed Robustness Evaluation

Independent training-seed robustness experiments use training seeds 0–4. Checkpoints are stored under `models/training_seed_robustness/seed_0` through `seed_4`. Aggregated and raw evaluation results are provided under `outputs/training_seed_robustness/`. Each checkpoint is evaluated on 20 environment seeds with 5 action-sampling seeds per environment seed (100 stochastic episodes per checkpoint).

## Candidate-Information Robustness

Zero-shot robustness under localization noise and missed-report conditions is evaluated using the original checkpoint without retraining. Guidance is computed from the *reported* candidate coordinate (after noise/drop is applied), not from the ground-truth detected-target coordinate — this matters because the environment marks a target as "detected" independently of whether its candidate report was later corrupted or dropped, so a naive implementation could let guidance silently ignore the perturbation being tested. Results are under `outputs/candidate_robustness/`.

| Condition | sigma_loc (m) | p_miss | Inspection | Detection | Collision |
|---|---|---|---|---|---|
| Clean | 0 | 0.00 | 0.300 +/- 0.200 | 0.675 +/- 0.196 | 0.25 +/- 0.44 |
| Localization 25 m | 25 | 0.00 | 0.275 +/- 0.189 | 0.575 +/- 0.183 | 0.30 +/- 0.47 |
| Localization 50 m | 50 | 0.00 | 0.213 +/- 0.152 | 0.588 +/- 0.219 | 0.25 +/- 0.44 |
| Localization 100 m | 100 | 0.00 | 0.138 +/- 0.121 | 0.619 +/- 0.235 | 0.30 +/- 0.47 |
| Missed reports 10% | 0 | 0.10 | 0.275 +/- 0.155 | 0.631 +/- 0.197 | 0.25 +/- 0.44 |
| Missed reports 20% | 0 | 0.20 | 0.225 +/- 0.175 | 0.563 +/- 0.288 | 0.50 +/- 0.51 |

(20 seeds per condition; mean +/- SD across seeds.) Inspection success now degrades monotonically with localization error and missed-report rate, as expected once guidance is actually corrupted by the perturbation.

## Held-Out Generalization Evaluation

Zero-shot generalization evaluation of all 5 independently trained CA-HMAPPO policies across 10 evaluation conditions (one nominal reference and nine held-out shifts) without retraining, fine-tuning, or checkpoint reselection. Conditions vary target distribution, sensing reliability, communication availability, motion scale, and team composition (including a larger-team Scenario 2 variant with 2 fixed-wing + 2 quadrotor UAVs, 10 targets, 2400x2400 m area). Localization-noise and missed-report perturbations are intentionally not repeated here since they are already covered by Candidate-Information Robustness above.

**Total evaluation:** 5 training seeds x 10 conditions x 20 environment seeds x 5 action seeds = 5000 episodes (completed; raw and summary results are included under `outputs/heldout_generalization/`).

**Seed protocol** (matches the training-seed robustness protocol used elsewhere in this repository): environment seeds 0-19 control target/candidate stochasticity; action seeds 100-104 control stochastic action sampling only. Exactly one `env.reset(seed=env_seed)` call is made per episode. Under this protocol, the nominal condition reproduces the manuscript's Section VI-K inspection rate (0.142) almost exactly.

**Hard communication** (`hard_comm_800`, `hard_comm_400`) withholds candidate assignment/guidance from any quadrotor that cannot reach base or a fixed-wing within `communication_radius` — it does not merely shrink the radius while still handing out guidance, which would leave the condition indistinguishable from nominal.

**Unit of statistical analysis:** n=5 (the five independently trained checkpoints). For each checkpoint, metrics are averaged over 100 episodes (20 environment seeds x 5 action seeds) per condition, then mean +/- SD is computed across the 5 checkpoint-level means.

| Condition | Coverage | Detection | Inspection | Mission Success | Collision | Near Miss |
|---|---|---|---|---|---|---|
| Nominal | 0.318 +/- 0.289 | 0.307 +/- 0.287 | 0.142 +/- 0.188 | 0.000 +/- 0.000 | 0.320 +/- 0.076 | 4.452 +/- 0.759 |
| Clustered targets | 0.303 +/- 0.273 | 0.385 +/- 0.372 | 0.227 +/- 0.278 | 0.046 +/- 0.082 | 0.340 +/- 0.090 | 6.206 +/- 1.118 |
| Edge-biased targets | 0.339 +/- 0.302 | 0.275 +/- 0.240 | 0.099 +/- 0.116 | 0.000 +/- 0.000 | 0.280 +/- 0.043 | 4.810 +/- 0.574 |
| Sensing 0.60/0.80 | 0.322 +/- 0.294 | 0.305 +/- 0.286 | 0.138 +/- 0.188 | 0.000 +/- 0.000 | 0.304 +/- 0.081 | 4.368 +/- 0.786 |
| Sensing 0.50/0.70 | 0.318 +/- 0.288 | 0.293 +/- 0.272 | 0.126 +/- 0.175 | 0.000 +/- 0.000 | 0.322 +/- 0.084 | 4.408 +/- 0.812 |
| Hard comm. 800 m | 0.326 +/- 0.293 | 0.316 +/- 0.293 | 0.097 +/- 0.130 | 0.000 +/- 0.000 | 0.312 +/- 0.083 | 4.650 +/- 0.601 |
| Hard comm. 400 m | 0.327 +/- 0.293 | 0.307 +/- 0.283 | 0.043 +/- 0.058 | 0.000 +/- 0.000 | 0.316 +/- 0.068 | 5.284 +/- 0.743 |
| Motion x0.8 | 0.313 +/- 0.278 | 0.300 +/- 0.274 | 0.121 +/- 0.149 | 0.000 +/- 0.000 | 0.238 +/- 0.155 | 5.130 +/- 0.767 |
| Motion x1.2 | 0.390 +/- 0.349 | 0.373 +/- 0.342 | 0.164 +/- 0.205 | 0.000 +/- 0.000 | 0.340 +/- 0.110 | 3.726 +/- 0.443 |
| Scenario 2 | 0.446 +/- 0.302 | 0.439 +/- 0.308 | 0.107 +/- 0.138 | 0.000 +/- 0.000 | 0.514 +/- 0.156 | 7.152 +/- 3.804 |

Reproduce the full evaluation:

```bash
python code/heldout_generalization/evaluate_heldout_generalization.py \
  --num-env-seeds 20 --num-action-seeds 5 --device cuda \
  --output-dir outputs/heldout_generalization
```

Generate summary statistics (n=5):

```bash
python code/heldout_generalization/summarize_generalization.py \
  --input-file outputs/heldout_generalization/raw_generalization_results.csv \
  --output-dir outputs/heldout_generalization
```

See `code/heldout_generalization/README.md` for detailed condition descriptions and statistical aggregation methodology.

## Hybrid Sweep and Rule-Based Inspector Control

The hybrid sweep and matched rule-based quadrotor inspector controls are implemented in `code/extended_evaluations/`, particularly `evaluate_sweep_assisted_actor.py` and `evaluate_sweep_rulebased_quad.py`. Evaluation outputs are under `outputs/hybrid_sweep/`.

## Reproducibility Notes

The original Scenario 1 checkpoint is verified as the terminal checkpoint at training update 3000. The exact training seed used for this historical checkpoint was not preserved in the original experiment log and is therefore not asserted here.

Independent robustness experiments reported in the manuscript use explicitly recorded training seeds 0–4.

Evaluation seeds are fully controlled by the evaluation scripts. The reported evaluation protocol uses matched environment seeds and controlled action-sampling seeds for all stochastic rollouts.

## Software Environment

The code and results were generated using:
- Python 3.11.16
- NumPy 2.4.6
- PyTorch 2.14.0 with CUDA 12.6
- Matplotlib 3.11.2
- pandas 3.0.5
- SciPy 1.17.1

A frozen `requirements.txt` is provided. Note that the environment was tested with NumPy 2.x; earlier environment specifications using NumPy <2 are outdated.

## Citation

If you use this code or results in your research, please cite:

```
[Citation will be added upon publication]
```

## Contact

For questions regarding the implementation or experimental results, please contact the corresponding author.
