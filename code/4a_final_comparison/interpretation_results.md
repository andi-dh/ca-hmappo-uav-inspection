# 4A Final Comparison Interpretation

The proposed CA-HMAPPO achieves the highest coverage ratio and inspection success rate among all compared methods in Scenario 1. Homogeneous MAPPO and Heterogeneous MAPPO improve coverage and target detection, but both fail to generate effective inspection behavior. In contrast, CA-HMAPPO enables quadrotor agents to approach and inspect candidate targets through guidance observation, capability-aware reward shaping, and safety-aware action masking.

Compared with the strong Rule-Based V2 baseline, CA-HMAPPO improves coverage from 0.437 to 0.747 and inspection success from 0.300 to 0.588 while maintaining a low collision count of 0.1. The main trade-off is that CA-HMAPPO requires longer mission time and higher energy consumption, indicating that the proposed method prioritizes inspection completion and safety over energy efficiency.

These results support the claim that UAV-type-specific policy heads alone are insufficient for inspection-oriented heterogeneous UAV coordination. Capability-aware observation and reward design are needed to induce inspection behavior in the quadrotor agents.

Indonesian summary:

CA-HMAPPO menghasilkan coverage ratio dan inspection success rate tertinggi dibandingkan seluruh metode pembanding. Homogeneous MAPPO dan Heterogeneous MAPPO mampu meningkatkan coverage dan target detection, tetapi keduanya gagal menghasilkan perilaku inspeksi yang efektif. Sebaliknya, CA-HMAPPO membuat quadrotor mampu mendekati dan menginspeksi candidate target melalui guidance observation, capability-aware reward shaping, dan safety-aware action masking. Meskipun CA-HMAPPO membutuhkan energi dan waktu misi lebih besar, metode ini secara signifikan meningkatkan inspection success dengan collision rate yang tetap rendah.
