# Revised client universe — R20261016A

Brian's October 14 message names `TAK861_TX_READY_PATIENT_UNIVERSE_DL` and says
the positive class is approximately 2.5 times larger. The screenshot does not
establish its database/schema, row count, or whether the increase is snapshots
or unique patients. The October 13 call confirms static/output fusion was the
completed experiment and historical AutoML sequences were the next experiment.
An improvement from temporal features was a hypothesis, not a reported result.

## Notebook 01 execution

1. Resolve the exact table name in `DSVC_TAKEDA_TA_PRIVATE.INFORMATION_SCHEMA.TABLES`.
   If zero or multiple matches are visible, set `CONFIG['client_universe']['table']`
   to the verified three-part name in all five notebooks. Never fall back to V63 MODEL_DATA.
2. Validate unique PATIENT_ID/END_DT, binary RESP and exact date grain. Compare
   new snapshots with the prior PATIENT_SPLIT; report added/removed rows,
   positive snapshots, positive patients and actual ratio. Shared-key label
   changes stop for producer-rule review. No labels are recomputed.
3. Inherit each known patient's partition across all their snapshots. New
   patients receive deterministic SHA256(seed:patient) 70/15/15 assignments.
   Both classes must exist in every partition. This does not create a fresh
   untouched holdout or guarantee exact split proportions.
4. Retain the existing 1,028 FEATURE_MAP names/order as the fixed count vocabulary.
   Reaggregate raw RX/PX/DX activity for **all** new universe keys. Old SNAPSHOTS
   and TENSOR_MONTHLY are not loaded. Current LATEST claim sources match the
   raw AutoML source family. RX counts include source transactions; AutoML
   ordinary RX counts are paid final/standalone transactions, as documented.
5. Resolve the CCSR dictionary by required columns and unique metadata match;
   provide `CONFIG['client_universe']['sources']['dx_reference']` if needed.
   An optional known common database/schema can narrow discovery via
   `CONFIG['client_universe']['common_scope']`. No guessed CCSR table is executed.
6. Extract unique patient/as-of AutoML rows using the real raw adapter, then
   assemble 12 positions. Current position uses exact END_DT; earlier ones
   use preceding month ends. Each rolling summary has its own 12-month cutoff.
7. Check medical count activity against current AutoML event coverage, reject
   null/non-finite counts, preserve unavailable AutoML windows with indicators,
   and save immutable run INPUTS with cohort/source/feature/split hashes.

Raw AutoML's 512-column ceiling now retains complete count/recent-percentage
pairs plus AGE (at most 511 columns for that ceiling). It is a resource limit,
not a model-selected optimal K. Features are not an exact reproduction of the
legacy target-fitted 261-feature table. The legacy non-null chronic-indicator
count is explicitly named DX_WITH_CHRONIC_INDICATOR_RECORDED, not chronic disease.
Window observation means a qualifying source event, including pharmacy rows
that may not contribute to paid-RX features; it is not enrollment coverage.

## Model and comparison scope

The five notebooks remain self-contained. Use R20261016A and identical CONFIG
throughout. COUNT, STATIC_FUSION and TEMPORAL_FUSION use matched inputs, splits,
selected subsets and seeds 42/142/242. Train from scratch; no old checkpoints.
Regularization and TRAIN-only feature selection remain in place. Transform
statistics now count the current sequence position once instead of twice.
Primary lift@10%, secondary lift@20%/30%; snapshot capacity, not unique outreach.
Validation bootstrap intervals condition on validation-selected models and are
descriptive, not unbiased independent confirmation. LightGBM remains deferred.
TEST stays off; its legacy patients were already evaluated, even when additional
snapshots are present. Completed TEST use is recorded in FINAL_TEST; the immutable
LOCK's false flag describes its creation time only.

No actual warehouse execution or new clinical results are claimed. Runtime
schema/permission checks, extraction coverage and compute sizing remain live
environment checks. Current references lack historical ingestion-time evidence;
12 rolling 12-month windows can use roughly 23 months of history. The new cohort
changes prevalence, so old absolute lift values are not like-for-like comparisons.

The unrelated rigorous_transformer source-builder and baseline edits were not
modified or included in this delivery. A task-owned copy of the source-builder
logic was adapted into `fresh_counts.py` for the revised universe.

## Final review guards

The new universe must retain every prior snapshot key; removed/reformatted keys
stop before assignment can silently move held-out patients. Shared labels must
also remain unchanged. If Brian intentionally removed keys, reconcile the source
contract rather than bypassing this check.

Before pivoting, the count builder prints inherited features without activity
and observed categories outside the dictionary. More than 10% inactive inherited
features stops extraction. This is a conservative diagnostic, not proof of a
mapping defect: after inspecting names/source drift, an explicit
`CONFIG['client_universe']['max_missing_feature_fraction']` below 1 can revise the
floor. Use identical configuration in all five notebooks; the floor and measured
fraction are recorded in the input manifest. Cohort comparisons ignore pandas
string-storage dtype while preserving exact key values.

Count RX/PX mappings retain all distinct mapping pairs; AutoML selects current
reference rows. Historical mapping-version equivalence remains unverified and
is not claimed. The Spark aggregation must still be exercised in Databricks;
local tests cover SQL construction and guards, not a live Spark connector.
