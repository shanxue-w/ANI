# Battery Q0 convention

This battery case study uses two reference-capacity conventions. This is already
stated in both the main text and the supplement:

- Main text: the paragraph beginning "Across the battery analyses..." states
  that SoC is computed by Coulomb counting with the reference-capacity
  convention specified for each evaluation protocol, and that the corresponding
  preprocessing/protocol details are summarized in Supplementary Table 2.
- Supplementary Table 2: the row "Reference capacity and aging proxy" gives the
  exact convention for each protocol. It states that the representative
  cycle-150 visualization uses the nominal-capacity convention
  `Q_ref = 2.0 Ah`, while the supplementary multicycle statistics use the
  dataset-initial-capacity convention based on the first readable B0005 capacity
  value, `Q_init` (`Q_init ~= 1.856 Ah`).

- Main-text representative cycle-150 visualization: use the nominal capacity
  convention `Q0 = Q_ref = 2.0 Ah` (`Q_total = 2.0 * 3600 A s`).
- Supplementary multicycle statistics over the held-out test cycles, including
  the supplementary table/figure results: use `meta_q0`, i.e. the value read
  from the processed data bundle as `meta["Q0_Ah"]`. For NASA B0005 this is the
  dataset-initial readable capacity value, about `1.856 Ah`.

These are protocol choices, not a mismatch: the paper states that the
state-of-charge input is computed by Coulomb counting with the
reference-capacity convention specified for each evaluation protocol, and that
numerical comparisons are made within each reported protocol.

Code mapping:

- Default rollout helpers use `2.0 Ah`, matching the main-text convention.
- Pass `--use_meta_q0` in multicycle/cross-battery evaluation scripts to use
  `meta["Q0_Ah"]`, matching the supplementary multicycle protocol.

## Rollout-window and test-cycle evaluation protocol

The processed bundle `dataset/processed_battery_data_rollout.pt` stores two
different kinds of tensors:

- `X_train`, `Y_train`, `X_val`, and `Y_val` are short rollout-window tensors
  used during model training and validation. In `2th/ANI2.py`, `4th/ANI4.py`,
  and `baseline/base.py`, each batch has shape `(batch, rollout_len, features)`.
  The training loop walks through the window, feeds the previous predicted
  voltage into the next step, and computes a short multi-step loss.
- `X_test` and `Y_test` are the held-out test-cycle step data. They are not
  stored as training rollout windows. The evaluation scripts group `X_test` by
  the normalized cycle feature and run autoregressive rollouts within each
  held-out discharge cycle.

Therefore, the rollout-window tensors are a training/validation batching format,
not the source of the reported test-cycle rollout metrics. The commands in the
top-level `run.sh` that call `plot.py`, `eval_cross_battery.py`,
`eval_multicycle_standalone.py`, and `plot_per_cycle_mse_compare.py` report
performance on held-out test cycles from `X_test` and `Y_test`.
