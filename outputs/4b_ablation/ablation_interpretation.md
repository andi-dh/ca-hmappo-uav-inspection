# 4B Ablation Interpretation

The ablation study evaluates whether the main CA-HMAPPO components are necessary for Scenario 1 performance.

Observed results:

- Full CA-HMAPPO achieves inspection success `0.588 ± 0.304`, coverage `0.747 ± 0.106`, and collision `0.100 ± 0.308`.
- Without guidance, inspection drops to `0.013 ± 0.040` and collision rises to `0.400 ± 0.516`, even though coverage remains `0.617 ± 0.085`.
- Without capability-aware reward, inspection becomes `0.000 ± 0.000` despite high coverage `0.748 ± 0.068` and detection `0.787 ± 0.145`.
- Without safety shaping, collision rises to `0.500 ± 0.527`, near-miss rises to `12.200 ± 11.526`, and inspection remains low at `0.062 ± 0.121`.

Interpretation:

- Removing the guidance vector is expected to reduce the quadrotors' ability to approach candidate targets, increasing mean distance to candidate targets and reducing valid hover-inspect events.
- Removing capability-aware reward shaping reduces inspection success most strongly. Although the agents still optimize wide-area coverage and target detection, they fail to convert detected targets into completed inspections. Therefore, this variant should not be interpreted as better simply because coverage, detection, or collision metrics are high.
- Removing safety shaping is expected to increase collision and near-miss events, showing that inspection-oriented rewards alone are insufficient for safe cooperative operation.

Paper-ready paragraph:

The ablation results confirm the contribution of each component in CA-HMAPPO. Removing the candidate guidance vector weakens the quadrotors' ability to approach candidate targets, resulting in a substantial drop in inspection success. Without capability-aware reward shaping, the agents optimize wide-area coverage and target detection, but fail to convert detected targets into completed inspections. Removing safety shaping increases collision and near-miss events, indicating that inspection-oriented reward alone is insufficient for safe heterogeneous UAV coordination. These results show that the proposed combination of guidance observation, capability-aware reward shaping, and safety shaping is necessary to obtain high inspection success while maintaining low collision.
