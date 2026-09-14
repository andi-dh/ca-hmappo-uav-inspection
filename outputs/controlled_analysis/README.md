# Controlled Contribution and Ablation Analysis

This folder stores Scenario 1 sampling ablation outputs for CA-HMAPPO.

## Planned Variants

- `full_ca_hmappo`: guidance + capability-aware reward + safety shaping.
- `without_guidance`: removes the candidate guidance vector from observation.
- `without_capability_reward`: removes fixed-wing/quadrotor role-aware reward shaping and inspection credit.
- `without_safety_shaping`: removes additional collision/near-miss/near-neighbor shaping and safety-aware action masking.

## Training Commands

```bash
python code/controlled_analysis/train_controlled_variants.py \
  --scenario 1 --use-guidance 0 --use-capability-reward 1 --use-safety-shaping 1 \
  --num-updates 1000 --checkpoint-interval 250 --device cuda

python code/controlled_analysis/train_controlled_variants.py \
  --scenario 1 --use-guidance 1 --use-capability-reward 0 --use-safety-shaping 1 \
  --num-updates 1000 --checkpoint-interval 250 --device cuda

python code/controlled_analysis/train_controlled_variants.py \
  --scenario 1 --use-guidance 1 --use-capability-reward 1 --use-safety-shaping 0 \
  --num-updates 1000 --checkpoint-interval 250 --device cuda
```

## Evaluation Example

```bash
python code/controlled_analysis/evaluate_controlled_variants.py \
  --scenario 1 --num-seeds 10 \
  --model models/controlled_analysis/without_guidance/without_guidance_scenario_1.pt \
  --device cuda
```

## Build Table

```bash
python code/controlled_analysis/build_controlled_analysis_table.py \
  --output-dir outputs/controlled_analysis
```
