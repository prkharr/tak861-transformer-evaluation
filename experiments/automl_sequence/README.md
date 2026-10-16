# Two AutoML experiments — client universe revision

Current run: **R20261016A**. The five self-contained notebooks in
`notebooks/AutoML Sequential Features` now target Brian's
`TAK861_TX_READY_PATIENT_UNIVERSE_DL`. Read the
[client universe guide](CLIENT_UNIVERSE_20261016.md) for source resolution,
cohort audits and exact changes. Actual private-data execution is pending.

| Notebook | Purpose |
|---|---|
|01|Resolve client universe, preserve patient partitions, rebuild counts, extract real historical AutoML tensors|
|02|TRAIN-only feature quality, preprocessing, grouped proxy ranking and feature-count selection|
|03|COUNT, STATIC_FUSION, TEMPORAL_FUSION; three fixed seeds each, with regularization and resumable fits|
|04|Validation histories, feature reliance and immutable model/threshold lock|
|05|Paired lift comparison, capacity metrics and conditional patient-cluster uncertainty|

Use identical CONFIG/RUN_ID in all five notebooks. Existing private Spark and
Snowflake connection objects supply access; credentials are not embedded.
Table discovery requires a unique visible exact-name match. Supply the verified
three-part table identifier if ambiguous; never guess a schema. CCSR lookup also
requires a unique verified dictionary or explicit configured source.

Only old FEATURE_MAP and PATIENT_SPLIT are reused as the fixed 1,028 count
vocabulary and patient membership. No old tensors, scores, checkpoints or winner
are inputs. All count values are reaggregated for the new universe. New patients
use a seeded hash split; old patients never move between partitions. Added,
removed and label-changed snapshots are audited; shared-key label changes fail.

AutoML is computed at each unique patient/date, with 12 rolling monthly anchors
and a 12-month lookback per anchor. Step zero is the actual END_DT. No current
vector is broadcast backward. No target-aware legacy caps, filters, types or
importance are reused. Raw vocabulary uses TRAIN patient frequency only. The
512-column ceiling retains complete count/recent pairs plus AGE, so at most
511 columns are used before TRAIN feature selection. Missing windows retain
explicit masks; observed absence of a category is zero.

Lift@10% of snapshots is the primary declared default; 20%/30% are secondary.
AP, ROC AUC and calibration remain supporting metrics. Capacity is not a
confirmed unique-patient outreach requirement. LightGBM is deferred in this run.
Dropout, AdamW weight decay, patient weights, gradient clipping, validation-lift
checkpointing, early stopping and learning-rate scheduling remain enabled.

`RUN_FINAL_TEST=False`. Existing TEST patients were previously inspected; added
snapshots do not restore an untouched holdout. Validation intervals condition
on models selected with validation labels and are descriptive. New prevalence
means old absolute lift is not a like-for-like benchmark. Source LATEST tables
do not establish historical ingestion availability, and overlapping AutoML
windows can span roughly 23 months. Fixed count dictionary provenance remains
qualified; fresh counts are not claimed identical to historical tensor values.

See [validation receipt](local_validation.json) and [build manifest](build_manifest.json)
for the exact tested bytes. Tests run all five generated notebooks using a real
raw AutoML adapter with a simulated SQL response; Spark count aggregation and
warehouse I/O remain doubles. Snowflake syntax parsing is not live validation.
Review status is recorded in [review disposition](CLAUDE_REVIEW_20261016.md).

[Historical addendum findings](HISTORICAL_ADDENDUM_FINDINGS.md) remain descriptive
old-cohort evidence, not results of the new notebooks.

Current connection setup: each notebook's first code cell defines sf_options. Fill its blank sfPassword with your PAT locally in Databricks; do not commit the filled value. This replaces the earlier requirement for pre-existing connection globals. The adapter retains DS_ML for experiment artifacts. All five notebook filenames now end in _AutoML.ipynb.
