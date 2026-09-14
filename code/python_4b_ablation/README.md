# Python 4B Ablation

Scripts for CA-HMAPPO ablation experiments.

Files:

- `ablation_env_wrapper.py`: switchable wrapper with `use_guidance`, `use_capability_reward`, and `use_safety_shaping`.
- `train_ablation_mappo.py`: trains one ablation variant.
- `evaluate_ablation_mappo.py`: evaluates one trained ablation model.
- `build_ablation_table.py`: builds `outputs/4b_ablation/ablation_comparison_scenario_1_sampling.csv` and LaTeX table from available mean/std files.

Recommended first run order:

```bash
python python_4b_ablation/train_ablation_mappo.py --scenario 1 --use-guidance 1 --use-capability-reward 0 --use-safety-shaping 1 --num-updates 1000 --checkpoint-interval 250 --device cuda
python python_4b_ablation/train_ablation_mappo.py --scenario 1 --use-guidance 0 --use-capability-reward 1 --use-safety-shaping 1 --num-updates 1000 --checkpoint-interval 250 --device cuda
python python_4b_ablation/train_ablation_mappo.py --scenario 1 --use-guidance 1 --use-capability-reward 1 --use-safety-shaping 0 --num-updates 1000 --checkpoint-interval 250 --device cuda
```
