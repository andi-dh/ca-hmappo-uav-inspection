# 4C Trajectory Analysis

This folder contains qualitative trajectory analysis for the final CA-HMAPPO policy on Scenario 1.

Representative seed:

- `seed = 16`
- `coverage_ratio = 0.786875`
- `target_detection_rate = 1.0`
- `inspection_success_rate = 1.0`
- `inspected_targets = 8 / 8`
- `collision_count = 0`
- `termination_reason = all_targets_inspected`

Diagnostic limitation:

- `quadrotor_hover_inspect_count = 444`
- `quadrotor_valid_hover_inspect_count = 196`
- `quadrotor_invalid_hover_inspect_count = 248`
- `mean_distance_to_candidate = 321.96 m`
- `time_in_candidate_range = 421 agent-steps`

Although all targets are inspected with zero collision, invalid hover-inspect actions remain frequent. This should be discussed as a limitation related to action efficiency and inspection timing.

Main files:

- `trajectory_ca_hmappo_seed_16.png`: color trajectory figure.
- `trajectory_ca_hmappo_seed_16.pdf`: color trajectory figure for paper.
- `figure_trajectory_ca_hmappo_seed_16_grayscale.pdf`: grayscale paper-ready figure.
- `event_timeline_ca_hmappo_seed_16.csv`: inspected-target timeline.
- `event_timeline_ca_hmappo_seed_16.png`: event timeline plot.
- `distance_to_candidate_ca_hmappo_seed_16.png`: quadrotor distance-to-candidate diagnostic.
- `rollout_ca_hmappo_seed_16.csv`: per-step per-agent rollout log.
- `events_ca_hmappo_seed_16.csv`: detection, inspection, collision, and near-miss event log.
- `action_diagnostics_ca_hmappo_seed_16.csv`: final action diagnostics.
- `qualitative_analysis.md`: paper-ready qualitative interpretation.

Scripts:

- `../../trajectory_analysis/generate_trajectory_rollout.py`
- `../../trajectory_analysis/plot_trajectory_analysis.py`
- `../../trajectory_analysis/build_qualitative_summary.py`
