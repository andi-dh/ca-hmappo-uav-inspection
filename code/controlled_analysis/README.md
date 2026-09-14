# Controlled Contribution and Ablation Analysis

This folder contains the controlled component-ablation experiments used to evaluate candidate guidance (G), capability-aware reward (CR), and safety mechanisms (S).

## Files

- `controlled_variant_env_wrapper.py`: switchable wrapper with `use_guidance`, `use_capability_reward`, and `use_safety_shaping`.
- `train_controlled_variants.py`: trains one ablation variant.
- `evaluate_controlled_variants.py`: evaluates one trained ablation model.
- `build_controlled_analysis_table.py`: builds ablation comparison table from available evaluation results.
- `evaluate_robustness.py`: evaluates training-seed robustness across independent runs.

## Example Training Commands

Train individual ablation variants:

```bash
python code/controlled_analysis/train_controlled_variants.py \
  --scenario 1 --use-guidance 1 --use-capability-reward 0 --use-safety-shaping 1 \
  --num-updates 1000 --checkpoint-interval 250 --device cuda

python code/controlled_analysis/train_controlled_variants.py \
  --scenario 1 --use-guidance 0 --use-capability-reward 1 --use-safety-shaping 1 \
  --num-updates 1000 --checkpoint-interval 250 --device cuda

python code/controlled_analysis/train_controlled_variants.py \
  --scenario 1 --use-guidance 1 --use-capability-reward 1 --use-safety-shaping 0 \
  --num-updates 1000 --checkpoint-interval 250 --device cuda
```

## Evaluation

Evaluate a trained variant:

```bash
python code/controlled_analysis/evaluate_controlled_variants.py \
  --scenario 1 --model models/controlled_analysis/[checkpoint_path] \
  --num-seeds 20 --device cuda --output-dir outputs/controlled_analysis
```
