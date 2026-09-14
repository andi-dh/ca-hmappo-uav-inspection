# 4B Ablation Study

This folder stores Scenario 1 sampling ablation outputs for CA-HMAPPO.

Planned variants:

- `full_ca_hmappo`: guidance + capability-aware reward + safety shaping.
- `without_guidance`: removes the candidate guidance vector from observation.
- `without_capability_reward`: removes fixed-wing/quadrotor role-aware reward shaping and inspection credit.
- `without_safety_shaping`: removes additional collision/near-miss/near-neighbor shaping and safety-aware action masking.

Main commands:

```bash
python python_4b_ablation/train_ablation_mappo.py --scenario 1 --use-guidance 0 --use-capability-reward 1 --use-safety-shaping 1 --num-updates 1000 --checkpoint-interval 250 --device cuda
python python_4b_ablation/train_ablation_mappo.py --scenario 1 --use-guidance 1 --use-capability-reward 0 --use-safety-shaping 1 --num-updates 1000 --checkpoint-interval 250 --device cuda
python python_4b_ablation/train_ablation_mappo.py --scenario 1 --use-guidance 1 --use-capability-reward 1 --use-safety-shaping 0 --num-updates 1000 --checkpoint-interval 250 --device cuda
```

Evaluation example:

```bash
python python_4b_ablation/evaluate_ablation_mappo.py --scenario 1 --num-seeds 10 --model models/4b_ablation/without_guidance/without_guidance_scenario_1.pt --device cuda
```

Build table:

```bash
python python_4b_ablation/build_ablation_table.py --output-dir outputs/4b_ablation
```
