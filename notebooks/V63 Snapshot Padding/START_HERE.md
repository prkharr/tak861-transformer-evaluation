# V63 snapshot-padding notebooks

This folder contains the four self-contained notebooks recreated on September 27, 2026. Each code cell has preceding documentation explaining its purpose, expected output and interpretation. All notebook outputs are cleared.

## Run order

1. Run `01_tensor_initialization.ipynb` in the authorized Databricks/Snowflake environment. It validates the feature contract, exact snapshot alignment and frozen patient split, computes enrollment validity, and exports prepared tensors.
2. Copy the printed `EXPECTED_PREPARED_MANIFEST_SHA256` into the configuration cell of notebooks 02, 03 and 04.
3. Run `02_patient_level_split.ipynb`. It verifies the existing frozen patient split and fits shared preprocessing once per eligible TRAIN snapshot.
4. Run `03_transformer_training.ipynb`. It trains monthly and quarterly models with explicit L1/L2 penalties and selects checkpoints using VALIDATION average precision.
5. Run `04_transformer_evaluation.ipynb`. It restores models, evaluates TRAIN/VALIDATION, measures feature influence and correlation, draws architecture diagrams, investigates overfitting and compares six validation-selected configurations for each representation.

Provide the connection settings described in each notebook through the authorized environment. Do not put credentials into notebook source or Git. All four notebooks contain their own helpers; no companion notebook imports are required.

## Input meaning

All 49 feature values come from the matching V63 `MODEL_DATA` snapshot, joined on `(PATIENT_ID, END_DT)`. Values are already encoded and are not encoded a second time. The same vector is repeated across valid periods. This is snapshot reuse, not historical reconstruction or monthly-to-quarterly feature aggregation.

Monthly inputs have 12 periods and quarterly inputs have four snapshot-relative three-month periods. A period is valid if any enrollment interval overlaps it, inclusively. A quarterly period is valid if any of its three constituent months is valid. Coverage does not require closed medical or pharmacy flags or a full period of enrollment.

Unavailable timesteps remain zero and are masked. Valid zero-valued features remain real observations. Missing feature values on valid timesteps are handled using TRAIN-fitted preprocessing. All-padded snapshots are retained and reported separately. The same preprocessing state is used for both representations and is fitted once per eligible TRAIN snapshot, rather than weighting snapshots by their number of repeated valid periods.

The frozen source population is 23,151 snapshots, 12,447 patients and 1,345 positive snapshots. The notebooks require exact source identity, labels, feature order and split agreement. Source checks do not establish event-time leakage prevention in upstream V63 snapshot features.

## Training and evaluation controls

Notebook 03 includes explicit L1 and L2 loss terms. Notebook 04 compares a fresh baseline with higher L1, higher L2, increased dropout, reduced model capacity and a combined configuration. The same frozen TRAIN/VALIDATION populations are used throughout. `RUN_VALIDATION_SEARCH=True` launches six fits per representation, in addition to baseline training in notebook 03.

`EVALUATE_TEST=False` is the default in notebook 04. Keep TEST unscored while choosing settings. Enable retrospective TEST evaluation only after freezing the model choice. Validation-based selection and exploratory bootstrap comparisons are not independent confirmation of improvement.

Permutation importance and correlations are associations, not causal explanations. Snapshot repetition also means sequence-position and coverage effects can influence monthly-versus-quarterly results even when feature values are identical.

Fresh dataset and run identifiers are configured for this delivery. Artifact writes check identity and reject conflicting contents; use a new dataset/run/evaluation identifier for a genuinely new experiment. No earlier experiment outputs or checkpoints are required.

See [VALIDATION.md](VALIDATION.md) for the completed checks and their limits.
