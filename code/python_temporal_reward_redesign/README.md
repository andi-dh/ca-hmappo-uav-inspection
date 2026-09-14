Temporal reward-redesign experiment for CA-HMAPPO.

This folder is intentionally separate from the existing reviewer-baseline code.
It trains a from-scratch CA-HMAPPO variant with:

- temporal hover masking: quadrotors can select hover-inspect only inside candidate inspection range
- retuned fixed-wing rewards to preserve coverage and detection
- retuned quadrotor rewards for approach, in-range hover, and inspection credit

Default outputs are isolated in:

- models/temporal_reward_redesign
- outputs/temporal_reward_redesign

Example:

```sh
python3 python_temporal_reward_redesign/train_temporal_reward_redesign.py --device cpu
```
