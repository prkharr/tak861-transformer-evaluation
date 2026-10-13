# Two AutoML experiments

The delivery uses the existing five notebooks under `notebooks/V63 Snapshot Padding`. Run 01–04 in order; Notebook 05 compares completed predictors. Helpers are embedded, so no checkout, sibling notebook import or historical fitted model is required at execution.

## Current status

R20261013C includes the raw historical AutoML extractor and all five notebooks. All 58 local checks passed in 46.334 seconds, including synthetic execution of all five notebook cell sequences. Generated SQL passed a Snowflake syntax parse. These checks do not establish live source permissions, warehouse execution or clinical results. The user authorized the Git push and explicitly deferred Claude review; no fresh Claude approval is claimed.

LIFT at the top 10% of snapshots is the declared primary default for TRAIN grouped proxy ranking/K and validation checkpointing, scheduling, early stopping, reliance and paired comparison. Top 20%/30% are secondary. The capacity is an engineering default, not a recovered business decision. AP/AUC/calibration and threshold metrics remain supporting evidence. Freeze any change before fitting.

## Experiment and stage contracts

| Stage | Purpose | Output |
|---|---|---|
|01|Verify counts/cohort/splits; extract and align actual raw AutoML history|INPUTS|
|02|TRAIN-only quality, fold-local proxy selection and registered protocol|PREPARATION|
|03|Fit every registered arm/seed; save each completed fit|Per-arm/seed models|
|04|Validation histories, reliance and model-specific threshold lock|LOCK|
|05|Paired comparison, calibration and operational-capacity metrics|VALIDATION_COMPARISON; optional FINAL_TEST|

COUNT starts with all 1,028 count candidates. STATIC_FUSION adds the current AutoML vector and availability indicator to the count representation. TEMPORAL_FUSION uses real AutoML vectors at all 12 dates with availability indicators. The same count subset, common AutoML subset, cohort/RESP/patient split, seeds and training settings support matched contrasts. Optional AutoML-only controls remain off. LightGBM is deferred for this bounded run; these three arms do not establish superiority over LightGBM.

The default is nine encoder fits, not the earlier large search. K comes from grouped TRAIN proxy CV and one-SE selection rather than an arbitrary final K; this proxy is not proof of an encoder-optimal subset. Diagnostics/removal reasons, ranking and final feature order remain visible. Report actual parameter counts/runtime rather than a time promise.

## Newly declared raw source contract

Read [source evidence](INPUT_SOURCE_EVIDENCE.md). The received legacy code filters categories with RESP even in scoring mode and learns positive-class caps, types and importance. The new path must not reuse these fitted values or execute shared-output legacy writers. It uses raw event information and label-blind vocabulary discovery by TRAIN patient frequency, with a declared 512-feature candidate cap and 5,000-request extraction chunks. The cap is a resource bound, not a historically recovered final feature count. All1028 count candidates remain separate and intact.

Step 0 is exact END_DT, which may be the 21st; older dates are the preceding 11 calendar month ends. Extract each unique patient/date once. Do not move the current cutoff to month end or broadcast its vector backward. Twelve overlapping 12-month summaries can span roughly 23 months of unique history; this inferred support difference means improvement would not isolate architecture alone.

No-claim as-of requests retain NaN and maskFalse; they do not remove patients or become silently observed zeros. Indicators describe available extracted activity, not verified enrollment/clinical observation. Latest provider/plan/catalog references do not establish historical ingestion availability. New sanitization/counting grain and omitted positional ICD10PCS enrichment mean these features are not a reproduction of the old261 fields. Full source contracts/identities and these limitations must be saved.

Verified matching raw FEATURE_MAP/SNAPSHOTS/PATIENT_SPLIT/TENSOR_MONTHLY inputs may be reused after exact checks. No previous model/search/LOCK/FINAL/score/winner is an input. Same-run intermediates and exact matching patient/date caches are valid dependencies. Optional read-only GET_DDL supports deployed-version comparison; complete procedure text is no longer missing and recovery-and-stop is not the intended new default.

## Running and interpreting

Use the approved Spark/Snowflake connection and installed NumPy,pandas,scikit-learn,PyTorch,Matplotlib/IPython. Keep secrets in existing private connection objects. Use identical RUN_ID/CONFIG in all five notebooks. Changed source/data/config/code needs a new identity; unchanged completed fits resume only under matching contracts/hashes.

`RUN_FINAL_TEST=False` remains default. The cohort TEST has already been examined twice, so another evaluation is retrospective ordinal3, not an untouched OOT result. Lock models/thresholds before any permitted evaluation; immutable final artifacts are reused. Never choose K/seeds/epochs/thresholds/capacity using TEST.

Positive paired deltas mean first named model minus second. Patient-cluster intervals condition on fitted predictors and exclude search/training uncertainty. Crossingzero is inconclusive, not equivalence. Each model uses its own validation threshold. No new result is implied by generated code or synthetic scores.

[Historical addendum findings](HISTORICAL_ADDENDUM_FINDINGS.md) summarize recovered output-layer/lift results and upstream limitations. V4 used an additive static branch, not an AutoML sequence. Old results must not be labeled R20261013C outcomes.

## Validation and rebuild

`build_notebooks.py` embeds helpers and writes the exact content manifest. Tests exercise synthetic inputs, source/store doubles, fits, restoration, lift comparison, masks and gates. The latest receipt must match the current bytes; previous passing results do not certify later changes. Fresh raw integration results are pending at this documentation checkpoint.

Keep patient rows, source definitions with private details, prediction arrays and credentials out of Git and centralized memory. Private runtime artifacts remain in the approved store; context contains derived evidence and status only.
