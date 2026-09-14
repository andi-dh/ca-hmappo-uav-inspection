# CA-HMAPPO

Code, trained models, and experimental results for:

**Capability-Aware Heterogeneous MAPPO for Cooperative Coverage and Inspection Using Fixed-Wing and Quadrotor UAVs**

This repository contains the implementation and experimental artifacts for CA-HMAPPO, a task-specific heterogeneous multi-agent reinforcement learning framework for cooperative fixed-wing UAV coverage and quadrotor target inspection.

The repository includes training and evaluation code, trained checkpoints, raw and processed experimental results, and scripts used to reproduce the main tables and figures reported in the manuscript.

## Repository Structure

- `code/` — simulator, training/evaluation scripts, ablations, baselines, and trajectory analysis
- `models/` — trained checkpoints for CA-HMAPPO and comparison methods
- `outputs/` — evaluation CSV files, diagnostics, and processed results
- `requirements.txt` — Python package versions
- `.gitignore` — standard Python/LaTeX ignore patterns

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
models/3c_capability_aware_mappo/capability_aware_mappo_scenario_1.pt
```

This checkpoint is tensor-identical to `models/3c_capability_aware_mappo/capability_aware_mappo_scenario_1_update_3000.pt`.

**The reported checkpoint was not selected using a best-checkpoint criterion.**

Checkpoints under `models/3c_capability_aware_mappo_safety_ft/` were produced during safety-focused fine-tuning experiments and are retained for reference, but are not the baseline checkpoint used in the main reported results.

## Running the Main Evaluation

Evaluate the main CA-HMAPPO checkpoint on 20 environment seeds:

```bash
python code/python_3c_capability_aware_mappo/evaluate_capability_aware_mappo.py \
  --scenario 1 --model models/3c_capability_aware_mappo/capability_aware_mappo_scenario_1.pt \
  --num-seeds 20 --base-seed 0 --device cuda \
  --output-dir outputs/3c_capability_aware_mappo
```

Generate detailed diagnostics for seed 16:

```bash
python code/python_4c_trajectory_analysis/generate_trajectory_rollout.py \
  --scenario 1 --seed 16 --model models/3c_capability_aware_mappo/capability_aware_mappo_scenario_1.pt \
  --device cuda --output-dir outputs/4c_trajectory_analysis

python code/python_4c_trajectory_analysis/plot_trajectory_analysis.py \
  --seed 16 --output-dir outputs/4c_trajectory_analysis

python code/python_4c_trajectory_analysis/build_qualitative_summary.py \
  --seed 16 --output-dir outputs/4c_trajectory_analysis
```

## Controlled Contribution Analysis

The controlled ablation experiments test the individual contributions of candidate guidance (G), capability-aware reward (CR), and safety mechanisms (S). Trained checkpoints are available under `models/4b_ablation/`. Evaluation outputs are under `outputs/4b_ablation/`.

Evaluate individual ablation checkpoints using:

```bash
python code/python_4b_ablation/evaluate_ablation_mappo.py \
  --scenario 1 --model models/4b_ablation/[checkpoint_path] \
  --num-seeds 20 --device cuda \
  --output-dir outputs/4b_ablation
```

Replace `[checkpoint_path]` with the specific ablation variant checkpoint (e.g., G0_CR0_S0, G1_CR1_S1, etc.).

## Training-Seed Robustness Evaluation

Independent training-seed robustness experiments use training seeds 0–4. Checkpoints are stored under `models/4b_ablation/robustness_v2/seed_0` through `seed_4`. Aggregated and raw evaluation results are provided under `outputs/robustness_eval/`. Each checkpoint is evaluated on 20 environment seeds with 5 action-sampling seeds per environment seed (100 stochastic episodes per checkpoint).

## Candidate-Information Robustness

Zero-shot robustness under localization noise and missed-report conditions is evaluated using the original checkpoint without retraining. Results are under `outputs/candidate_robustness/`.

## Hybrid Sweep and Rule-Based Inspector Control

The hybrid sweep and matched rule-based quadrotor inspector controls are implemented in `code/python_rev_reviewer_baselines/`, particularly `evaluate_sweep_assisted_actor.py` and `evaluate_sweep_rulebased_quad.py`. Their comparison outputs are included with the reviewer-baseline results.

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
