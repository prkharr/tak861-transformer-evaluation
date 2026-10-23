# AutoML sequence experiments — bounded memory

Current run: **R20261023_OLD_A**. The five self-contained notebooks in
`notebooks/AutoML Sequential Features` default to the original V63 cohort,
as requested for the first sequential AutoML run. Its population source is
`DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_MODEL_DATA`, projected to
`PATIENT_ID, END_DT, RESP` only. No previously fitted feature values are read.
Private-data execution is pending in the separate Databricks environment; this
local workspace supplies code, context and synthetic validation only. The [October 16 guide](CLIENT_UNIVERSE_20261016.md)
documents the earlier revised-cohort configuration, not the current default.

The delivered notebooks contain code cells only, with Python comments removed.
Required configuration identifiers are retained; runtime caveats remain in the
computed reports. Presentation cleanup is checked for unchanged executable AST.
Detailed allocation estimates, function changes and equivalence evidence are in
[the memory review](MEMORY_REVIEW_20261023.md).

| Notebook | Purpose |
|---|---|
|01|Resolve client universe, preserve patient partitions, rebuild counts, extract real historical AutoML tensors|
|02|TRAIN-only feature quality, preprocessing, grouped proxy ranking and feature-count selection|
|03|COUNT, STATIC_FUSION, TEMPORAL_FUSION; three fixed seeds each, with regularization and resumable fits|
|04|Validation histories, feature reliance and immutable model/threshold lock|
|05|Paired lift comparison, capacity metrics and conditional patient-cluster uncertainty|

Use identical CONFIG/RUN_ID in all five notebooks. Each first code cell creates
the Snowflake options with a blank PAT field for the separate runtime; credentials
are not embedded in the distributed notebooks.
Table discovery requires a unique visible exact-name match. Supply the verified
three-part table identifier if ambiguous; never guess a schema. CCSR lookup also
requires a unique verified dictionary or explicit configured source.

Only old FEATURE_MAP and PATIENT_SPLIT are reused as the fixed 1,028 count
vocabulary and patient membership. No old tensors, scores, checkpoints or winner
are inputs. All count values are reaggregated for the configured universe. New patients
use a seeded hash split; old patients never move between partitions. Added,
removed and label-changed snapshots are audited; removed keys or shared-key label changes fail.
Population and per-split totals must match `CONFIG['client_universe']['expected_population']`
after the audit and before raw-claim extraction. Counts are configuration, not module constants.

The default original population was recovered from the October 6 audit receipt:
23,151 snapshots / 12,447 patients / 1,345 positive snapshots. TRAIN is
16,256 / 8,712 / 941; validation 3,481 / 1,867 / 202; TEST 3,414 / 1,868 / 202.
The previous `R20261016A` or interactive fallback INPUTS cannot be reused by the
new implementation hash. The new run prefix prevents accidental replacement.

For a new cohort change exactly three configuration groups, identically in all five notebooks:

1. The population binding: `client_universe.table`, `table_name` and any verified raw-source overrides.
2. The independently audited `expected_population` totals and `per_split` counts.
3. `RUN_ID`, creating a new artifact prefix and refitting this run's raw vocabulary and models.

Keep the patient-level split rule (inherit known memberships; seeded SHA256 for new IDs)
and snapshot label definition unchanged unless separately approved. Do not auto-fill expected
counts from the same audit being checked. The later revised-cohort binding is
`DSVC_TAKEDA_TA_PRIVATE.TAK861.TAK861_TX_READY_PATIENT_UNIVERSE_DL`, with
`table_name='TAK861_TX_READY_PATIENT_UNIVERSE_DL'`; its exact independent population
receipt must be supplied before switching. It is not queried by the default notebooks.

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

`RUN_FINAL_TEST=False`. Original TEST patients were previously inspected; added
snapshots do not restore an untouched holdout. Validation intervals condition
on models selected with validation labels and are descriptive. New prevalence
means old absolute lift is not a like-for-like benchmark. Source LATEST tables
do not establish historical ingestion availability, and overlapping AutoML
windows can span roughly 23 months. Fixed count dictionary provenance remains
qualified; fresh counts are not claimed identical to historical tensor values.
Restoring the original cohort does not restore the old dated claim-source vintage.

## Memory execution

The scientific configuration is unchanged: three arms; seeds 42/142/242; the same
candidate sizes, fold-local one-SE selection, proxy, architecture, epochs, batch size,
thresholds and paired AP/AUC/lift bootstrap. No patients, features or training epochs
are sampled away. Correlation uses all eligible TRAIN rows, with disk-backed centering.

Count data stay float32 memmaps. Raw AutoML query rows are validated and canonically
hashed through an indexed SQLite spool; selected panels use float64 disk backing to
preserve their exact original byte/hash meaning. Sequence/summary tensors use float32.
TRAIN/CV/validation transforms and proxy descriptors are computed in bounded blocks;
float64 proxy arithmetic is retained on disk for numerical equivalence. Only the split
being modeled is transformed. Per-epoch full-TRAIN prediction remains, batched.

INPUTS keep the same NPZ/CSV/JSON files and raw-byte SHA256 meaning. Compression goes
to disk; the shared temporal-selection store streams 50,000-character chunks in at
most 1,024-row writes. Classic Spark verifies a staging table, then atomically publishes.
Without JVM access (such as Spark Connect), it uses an initial `errorifexists` write
followed by bounded appends to the final name. Chunk counts and hashes reject incomplete
tables; retries never overwrite them and require a new run ID. Readback
uses bounded ranges and disk extraction; raw arrays are read-only memory maps.
Each stage releases its mappings. `release_inputs(data)` must be the last use of
that input object and any views into its arrays; the generated cells follow this
rule. Array-level cleanup rejects views and accepts only the allocating owner.
Classic interruption can leave a run-prefixed
`__PENDING_...` staging table; direct mode can leave an invalid final table, which
cannot be loaded or reused. Complete artifacts are immutable in either mode.

Use sufficient fast local scratch space; memmaps and the SQL spool trade RAM for disk.
The first cell prefers `/local_disk0/tmp/tak861_automl` when available and exposes
`SCRATCH_DIRECTORY` for an explicit local-volume override. It checks write access;
aggregate receipts show the path, free disk and Python RSS/peak when available.
Do not point memmaps at DBFS or object storage. The long-row spool size is data-dependent;
the displayed free-space receipt must be checked on the actual cluster.
Python buffers are bounded, but Spark's largest result partition, JVM, Torch/BLAS,
OS file cache and actual raw-row volume still determine total driver usage. Local
synthetic validation is not a measured Databricks memory or runtime guarantee.
The generated notebooks embed `inputs`, `count_loader`, `fresh_counts`, `client_universe`,
`raw_automl`, `modeling`, `workflow`, `reliance`, `source_recovery`, the existing
`rigorous_transformer/comparison.py`, and the shared `temporal_selection/input_helpers.py`.
No auxiliary Python files are needed when importing the five notebooks into Databricks.

## Review boundaries

R1 sibling-folder isolation, single-counted step zero, finite counts, whole-pair cap,
VALUES requests, removed-key stop, dictionary floor and dtype-agnostic cohort checks
were already present at 2106cf1. No current-window observation-only hard stop exists:
missing history is masked; the medical-count/AutoML coverage consistency stop remains.
Unreachable missing LightGBM trainer/predictor calls are now explicit unsupported-arm
errors. Unrelated local `baseline49.py`/`test_baseline49.py` edits and untracked
`source_builder.py`/`source_stage.py` are excluded; `fresh_counts.py` is task-owned.

Still unverified: executing Spark/Snowflake integration, reference/schema permissions,
and count/raw-AutoML mapping-version parity (`CURR_IND`). The inherited dictionary's
historical fitting scope and source ingestion-as-of coverage remain qualified.
The source-bound `deterministic_raw` extractor is used; the old target-aware fallback
is not adopted. Never bypass the source, population or content-hash stops to make a run pass.

See [validation receipt](local_validation.json) and [build manifest](build_manifest.json)
for the exact tested bytes. Tests run all five generated notebooks using a real
raw AutoML adapter with a simulated SQL response; Spark count aggregation and
warehouse I/O remain doubles. Snowflake syntax parsing is not live validation.
Review status is recorded in [review disposition](CLAUDE_REVIEW_20261016.md).

[Historical addendum findings](HISTORICAL_ADDENDUM_FINDINGS.md) remain descriptive
old-cohort evidence, not results of the new notebooks.

Current connection setup: each notebook's first code cell defines sf_options. Fill its blank sfPassword with your PAT locally in Databricks; do not commit the filled value. This replaces the earlier requirement for pre-existing connection globals. The adapter retains DS_ML for experiment artifacts. All five notebook filenames now end in _AutoML.ipynb.
