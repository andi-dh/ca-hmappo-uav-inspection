# Trajectory Analysis

Scripts for qualitative CA-HMAPPO behavior analysis.

## Generate Rollout

```bash
python code/trajectory_analysis/generate_trajectory_rollout.py \
  --scenario 1 --seed 16 --model models/ca_hmappo/capability_aware_mappo_scenario_1.pt \
  --device cuda --output-dir outputs/trajectory_analysis
```

## Plot Figures

```bash
python code/trajectory_analysis/plot_trajectory_analysis.py \
  --seed 16 --output-dir outputs/trajectory_analysis
```

## Build Qualitative Summary

```bash
python code/trajectory_analysis/build_qualitative_summary.py \
  --seed 16 --output-dir outputs/trajectory_analysis
```
