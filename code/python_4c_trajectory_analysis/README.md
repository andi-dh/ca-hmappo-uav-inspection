# Python 4C Trajectory Analysis

Scripts for qualitative CA-HMAPPO behavior analysis.

Generate rollout:

```bash
python python_4c_trajectory_analysis/generate_trajectory_rollout.py --scenario 1 --seed 16 --model models/3c_capability_aware_mappo/capability_aware_mappo_scenario_1.pt --device cuda --output-dir outputs/4c_trajectory_analysis
```

Plot figures:

```bash
python python_4c_trajectory_analysis/plot_trajectory_analysis.py --seed 16 --output-dir outputs/4c_trajectory_analysis
```

Build qualitative summary:

```bash
python python_4c_trajectory_analysis/build_qualitative_summary.py --seed 16 --output-dir outputs/4c_trajectory_analysis
```
