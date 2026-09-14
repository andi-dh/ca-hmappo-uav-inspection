Temporal reward-redesign experiment for CA-HMAPPO.

This folder contains the temporal-constrained CA-HMAPPO variant evaluated separately from the main implementation.
It trains a from-scratch CA-HMAPPO variant with:

- temporal hover masking: quadrotors can select hover-inspect only inside candidate inspection range
- retuned fixed-wing rewards to preserve coverage and detection
- retuned quadrotor rewards for approach, in-range hover, and inspection credit

Default outputs are isolated in:

- models/temporal_constrained_variant
- outputs/temporal_constrained_variant

## Example

```bash
python code/temporal_constrained_variant/train_temporal_reward_redesign.py --device cpu
```
