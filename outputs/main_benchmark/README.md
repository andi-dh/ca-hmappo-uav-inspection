# Main Benchmark Results

This folder contains the final Scenario 1 sampling comparison for the manuscript Results and Discussion section.

## Files

- `final_comparison_scenario_1_sampling.csv`: final comparison with method metadata and mean metrics.
- `final_comparison_scenario_1_sampling_mean_std.csv`: paper-ready mean ± std table.
- `final_comparison_table_for_paper.tex`: LaTeX table for manuscript insertion.
- `interpretation_results.md`: English and Indonesian interpretation paragraphs.

## Primary Proposed Method

**CA-HMAPPO**

The reported Scenario 1 benchmark uses the terminal checkpoint at training update 3000:
```
../../models/ca_hmappo/capability_aware_mappo_scenario_1.pt
```

The checkpoint was not selected using a best-checkpoint criterion.

## Source Outputs

- `../ca_hmappo/scenario_1_3c_baseline_comparison_sampling.csv`
- `../ca_hmappo/scenario_1_capability_aware_mappo_sampling_mean_std.csv`
- `../homogeneous_mappo/scenario_1_random_masked_mean_std.csv`
- `../homogeneous_mappo/scenario_1_homogeneous_mappo_sampling_mean_std.csv`
- `../heterogeneous_mappo/scenario_1_heterogeneous_mappo_sampling_mean_std.csv`
- `../rule_based_baselines/scenario_1_rule_based_v2_summary.csv`
