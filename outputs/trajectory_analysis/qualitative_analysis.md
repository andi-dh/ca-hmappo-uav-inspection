# 4C Qualitative Analysis

Seed: `16`

## Final Metrics

- Coverage ratio: `0.624`
- Target detection rate: `0.500`
- Inspection success rate: `0.500`
- Inspected targets: `4`
- Collision count: `0`
- Near-miss count: `4`
- Termination reason: `max_steps_reached`

## Action Diagnostics

- Quadrotor hover-inspect count: `873`
- Valid hover-inspect count: `195`
- Invalid hover-inspect count: `678`
- Mean distance to candidate: `305.23` m
- Time in candidate range: `274` agent-steps

## Event Summary

- Detected target events: `4`
- Inspected target events: `4`

## Paper-Ready Interpretation

The trajectory visualization shows that the fixed-wing UAV performs wide-area scouting, while quadrotor agents are directed toward detected candidate targets for close-range inspection. CA-HMAPPO produces role-consistent behavior: the fixed-wing UAV continues exploring uncovered regions, whereas quadrotors approach and inspect target candidates. This qualitative behavior is consistent with the quantitative results, where CA-HMAPPO achieves the highest inspection success rate and coverage ratio.

Versi Indonesia:

Visualisasi lintasan menunjukkan bahwa fixed-wing UAV berperan sebagai scout untuk coverage area luas, sedangkan quadrotor bergerak menuju candidate target untuk inspeksi jarak dekat. CA-HMAPPO menghasilkan perilaku yang konsisten dengan peran masing-masing UAV: fixed-wing tetap mengeksplorasi area yang belum tercakup, sementara quadrotor mendekati dan menginspeksi target kandidat. Perilaku ini konsisten dengan hasil kuantitatif, di mana CA-HMAPPO menghasilkan inspection success rate dan coverage ratio tertinggi.
