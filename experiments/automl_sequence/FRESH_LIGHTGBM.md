# Fresh static LightGBM control in notebook 05

Run the first two setup cells and the final four cells of the updated notebook 05. Fill its blank Snowflake options and use exactly the run identity/configuration from the completed notebooks. Do not rerun extraction or Transformer training. The final four cells are marked `FRESH_LGBM_EXTENSION` and are self-contained after the first two setup cells.

This additive control uses the frozen TRAIN-only AutoML transform and selected feature order from PREPARATION, current-snapshot summary values and the availability indicator. It does not use historical production MODEL_DATA or a production fitted model. This is a static-input control, not a same-temporal-input architecture comparison. Count features and historical tokens are not included. Existing upstream source-vintage limitations still apply.

All TRAIN snapshots are used with the existing patient weights and square-root class balance. Three seeds (42, 142, 242) use 1,000 maximum trees, 31 leaves, learning rate 0.05, minimum 20 rows per leaf, 80% feature and row fractions, L2 penalty 1 and 8 CPU threads. Validation Lift@10% controls early stopping with patience 100. There is no parameter search. Final scores average the three seeds. No TEST values enter fitting or stopping.

The baseline saves one model-text/prediction/identity artifact per seed, and a separate lift comparison artifact. Completed seeds resume only if source, settings, preparation, input, lock, transformed designs, feature order and runtime versions match. Corrupt payloads and changed identities stop rather than overwrite. Inputs are memory-mapped, transformed one split at a time, and released after comparison. Only the small static design matrices are materialized.

Validation is displayed first. TEST is disabled by default using `FRESH_LGBM_EVALUATE_TEST = False`. Enable only after fitting is locked and explicit retrospective evaluation is intended; previously saved, matching `_FINAL_TEST` Transformer predictions must exist. No Transformer retraining is triggered. An already-inspected TEST population cannot establish independent generalization.

Output is lift-only: lifts at 10/20/30%, Lift@10% ranks and differences, and 1,000 paired patient-cluster bootstrap draws using the shared metric implementation. Intervals condition on fitted models; validation is checkpoint-selection-conditioned. An interval crossing zero is inconclusive, not equivalence. The final cell shows only model, rank and Lift@10%.

Build the extension alone with `python -m experiments.automl_sequence.build_fresh_lgbm`. The existing reporting builder also appends it on a full rebuild. Existing core implementation hashes remain unchanged; the additive baseline has its own source hash and artifact prefix.

LightGBM must be installed on the execution compute. If absent, install `lightgbm==4.6.0`, then rerun the first two setup cells and extension cells after any Python restart. Do not reinstall an already working environment merely to match this version; the actual version is recorded and fixed for artifact resumption.

Runtime is not clinically benchmarked. Budget roughly 10–30 minutes as an unmeasured planning allowance, dependent on artifact loading, available CPU and bootstrap work. No completion-time guarantee is made.
