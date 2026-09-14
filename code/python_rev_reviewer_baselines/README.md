# Reviewer Baselines

This folder contains reviewer-requested additions. It intentionally does not replace the earlier `python_3*` experiment code.

## Components

- `train_happo.py`: HAPPO-style sequential actor update using the existing heterogeneous actor and centralized critic.
- `evaluate_happo.py`: multi-seed evaluator for HAPPO and assignment-aware HAPPO.
- `assignment_aware_env_wrapper.py`: explicit candidate-to-quadrotor assignment plus hover-inspect commit/cooldown timing.
- `evaluate_hierarchical_task_allocation.py`: hierarchical/task-allocation baseline with high-level structured assignment and low-level rule execution.
- `compare_reviewer_baselines.py`: merges existing CSVs with reviewer additions.

## Literature Inputs

- HAPPO/HATRPO theory: Kuba et al., trust-region policy optimization in MARL.
- Multi-UAV HAPPO/task allocation: `Multi-UAV_Task_Allocation_and_Trajectory_Tracking_Based_on_HAPPO-AGRN.pdf`.
- Sequential HAPPO implementation details: `Task_Offloading_in_UAV-Assisted_Mobile_Cloud-Edge_Computing_Networks_An_AoP-Aware_HAPPO_Approach.pdf`.
- Hierarchical MARL baseline design: `Optimal_Frequency_Reuse_and_Power_Control_in_Multi-UAV_Wireless_Networks_Hierarchical_Multi-Agent_Reinforcement_Learning_Perspective.pdf`.
- Dynamic task assignment and scalability motivation: `Digital_Twin_Assisted_Dynamic_Task_Assignment_for_Multi-UAV_Systems_Using_Multi-Agent_Reinforcement_Learning.pdf`.
- Cooperative task assignment constraints: `A_Method_of_Multi-UAV_Cooperative_Task_Assignment_.pdf`.

## Example Commands

Smoke test HAPPO:

```bash
python python_rev_reviewer_baselines/train_happo.py --scenario 1 --env-mode capability_aware --num-updates 2 --rollout-steps 32 --device cpu
```

Train assignment-aware variant:

```bash
python python_rev_reviewer_baselines/train_happo.py --scenario 1 --env-mode assignment_aware --num-updates 1000 --rollout-steps 256
```

Evaluate hierarchical baseline:

```bash
python python_rev_reviewer_baselines/evaluate_hierarchical_task_allocation.py --scenario 1 --num-seeds 10
```

Scalability point, 2 fixed-wing + 2 quadrotor + 10 targets:

```bash
python python_rev_reviewer_baselines/evaluate_hierarchical_task_allocation.py --scenario 2 --num-seeds 10
```
