# Fresh cohort AutoML and monthly NLP development

Status: development pipeline with runtime cohort audit binding. No clinical execution or new clinical performance is represented by the local tests. The previously delivered five-notebook bundle is unchanged.

## Five stages

The standalone, code-only notebooks are in `notebooks/Fresh Cohort AutoML NLP/`. All dependencies on repository source are embedded. The first cell of every notebook has seven empty Snowflake connection options. Use identical configuration in all five notebooks.

1. `01_tensor_initialization_AutoML`: audit the supplied expanded table, bound it to 48 snapshot months, reconstruct and audit forward outcomes, freeze POPULATION/COHORT_AUDIT, assign patient partitions using the existing rule, extract and persist monthly medical-description TF-IDF and its TRAIN salience ranking, build fresh count inputs, extract raw AutoML features, and persist INPUTS.
2. `02_patient_level_split_AutoML`: fit the existing fold-local TRAIN-only feature preparation and one-SE selection; persist PREPARATION and the registered protocol.
3. `03_transformer_training_AutoML`: train/resume every registered arm with seeds 42, 142 and 242; show per-seed train/validation lift, fitted model sizes, lift and log-loss learning curves.
4. `04_transformer_evaluation_AutoML`: lock the completed models, show validation lift, train-validation gaps, split-lift charts and paired patient-cluster bootstrap contrasts. TEST is disabled; explicit enablement adds locked retrospective TEST lift and a train/validation/test chart without refitting.
5. `05_model_comparison_AutoML`: fit/resume the existing TRAIN-only static LightGBM baseline, compare validation lift on exactly the same rows, and audit frozen production scores for those same keys. Show lift bars, paired patient-cluster lift intervals and a final Rank, Model, Lift@10% and Status table. Lift@20%/30% remain in the detailed output. TEST is disabled.

SUMMARY is the default development phase: AutoML has one current anchor and count history still has 12 steps. Its transformer arms are COUNT, STATIC_FUSION and NLP_FUSION. TEMPORAL additionally extracts all 12 AutoML anchors and registers TEMPORAL_FUSION. Each phase has its own run suffix, artifacts and preparation. SUMMARY's one-anchor proxy has degenerate historical descriptors by construction; it is not the earlier twelve-anchor experiment. Do not compare phases as if only one model component changed.

NLP_FUSION adds a 12-step monthly medical-description representation and medical-activity indicator to counts. This is a representation-plus-history contrast, not an isolated test of text. AutoML static LightGBM has different inputs; these results alone cannot establish an architecture advantage or superiority to production.

## Reused implementations

- `automl_sequence.inputs`: snapshot validation, request semantics, patient identity and feature panel utilities.
- `automl_sequence.client_universe`, `fresh_counts`, `count_loader`: inherited patient membership/dictionary and fresh claim aggregation, including missing-key/label-change/coverage stops.
- `automl_sequence.raw_automl`: label-independent raw extraction, resumable chunks and feature construction, with an explicit optional one-anchor development mode.
- `automl_sequence.modeling` and `workflow`: TRAIN preparation, grouped selection, encoder/training settings, seed artifacts, locking, comparison and release of memory maps. Legacy defaults remain twelve anchors.
- `automl_sequence.fresh_lgbm`: static TRAIN-only LightGBM representation control and lift comparison.
- `rigorous_transformer.comparison`: paired patient-cluster bootstrap.
- `temporal_selection.input_helpers`: streamed immutable artifact storage.

The new NLP adapter uses the medical NLP demonstration's DX/PX description and token semantics, with bounded extraction, reusable fitted state and TRAIN-only fitting for prediction. It uses sklearn's TF-IDF transformation. It does not ingest free-text notes or pharmacy descriptions.

## Cohort binding

For a new cohort, change three things consistently: (1) qualified eligibility population and label sources/evidence; (2) date-window configuration and its new measured population receipt; (3) run identity/output prefix. Keep the patient-level split rule unchanged. Keep the label definition unchanged unless an explicit new outcome version is approved and recorded.

In the generated notebooks `CONFIG['cohort_universe']` holds table, table_name, split_seed and expected_population. Expected population has this shape:

```python
{
    'snapshots': None, 'patients': None, 'positives': None,
    'per_split': {
        'train': {'snapshots': None, 'patients': None, 'positives': None},
        'validation': {'snapshots': None, 'patients': None, 'positives': None},
        'test': {'snapshots': None, 'patients': None, 'positives': None},
    },
}
```

The initial default is `None`. Stage 01 measures counts, reconciles an independent SQL aggregate query against the extracted metadata, persists the exact population, verifies its read-back hash, then freezes expected counts and per-split counts in COHORT_AUDIT. Later stages restore this configuration; they do not invent or silently update expected counts. These are `measured_at_binding` consistency checks, not independent clinical verification. The authorized mode is `bounded_reconstruction_development`; original `producer_sha256` remains `None`. The actual reconstruction query has its own hash. This does not establish original eligibility lineage.

The October 30 screenshots show 587,460 rows in `DSVC_TAKEDA_TA_PRIVATE.TAK861.TAK861_TX_READY_PATIENT_UNIVERSE_DL`. This is the eligibility base, not the original 23,151-snapshot population. Use its existing eligibility; do not rebuild raw eligibility or use the unexecuted V2 proposal. Snapshot cutoffs are restricted to `(upper_end - 48 months, upper_end]`. `upper_end` is the earlier of the candidate's maximum cutoff and 90 days before the earlier observed maximum basket/medical date. This only places outcomes inside the observed date range; it does not prove completeness or individual coverage. The wider filter cannot create snapshots or patients absent from the candidate.

The first `IS_GENERIC=FALSE` fill is found across all available history in the designated TAK861 `NARCOLEPSY_MARKET_BASKET_CLAIMS_NO_MDD` table. A snapshot is positive only when that event falls in days 1 through 90 after END_DT. Every qualifying positive snapshot is retained. Prior/same-day initiators are excluded; negatives require medical activity during that outcome window. Generic drug membership and paid/reversal semantics are unchanged. Unclassified basket flags and missing advanced-fill dates stop labeling for inspection. Existing future medical activity is an observability proxy, not enrollment proof.

The authorized missing-original-key tolerance is 16, checked against the unfiltered candidate. More missing keys stop. Further exclusions must be explained by the explicit date window, prior initiation or unobserved-negative rule. Stage 01 prints counts by reason and split plus 0-to-1 and 1-to-0 changes; exact removed/changed key hashes are frozen and enforced later. Old TEST patients remain inherited and any updated labels are retrospective; no untouched-holdout claim is made. All outputs use the new run prefix; original sources are never overwritten.

The supplied rule sheet specifies 135 covered days within a 270-day lookback, maximum gap 270 days, and INDEX_DATE minus 40 days as the effective cutoff. Whether the candidate producer implements every eligibility rule remains unverified. Stored END_DT is treated as the already adjusted cutoff; no second 40-day subtraction. The last positive date is never used as a completeness cutoff. No random patient subsampling is used.

## Memory, resume and timing

NLP queries use at most 5,000 requested documents and a 250,000 aggregate-row limit. An oversized result subdivides the request and re-queries; no patient or feature is silently truncated. Each completed request block is transactionally checkpointed in SQLite and hash-checked on resume. Mapping fingerprint, requests including split, configuration, SQL and implementation identity must match. Persisted arrays are float32 memory maps. Existing streamed artifact and blockwise modeling implementations are reused.

At the proposed 587,458 snapshots, a dense float32 count tensor alone is approximately 27.0 GiB on disk; a 256-term NLP tensor is approximately 6.72 GiB. One-anchor 512-feature AutoML is approximately 1.12 GiB; twelve-anchor AutoML is approximately 13.45 GiB. These are payload sizes, not measured peak RAM or scratch-space requirements. Temporary files, compression, transformed features and artifacts require additional space. Check actual driver memory and scratch capacity in the execution environment. No complete-run timing or OOM-free execution guarantee has been established at this population size.

NLP feature output is saved before the slower modeling work, so completed feature construction can be demonstrated independently. Salience ranking is TRAIN snapshot-position mean TF-IDF, not predictive importance or lift. Training stages resume completed seed artifacts; interrupted model fits restart that fit unless the existing trainer explicitly supplies a finer checkpoint. Never bypass an identity, source, coverage or cohort stop to resume.

## Interpretation and remaining limitations

- Month zero uses month start through the effective cutoff; older steps use complete preceding calendar months. Exact patient/as-of keys may be reused. A partial month is never merged with its later month end.
- Vocabulary, document frequency and IDF use unique TRAIN documents only. The cap is predeclared at 256 terms. Validation and TEST do not fit this state. Count/AutoML supervised selection retains the existing fold-local protocol.
- Activity masks describe observed medical claims, not insurance enrollment. An active document with only unknown terms can have a zero vector. No mapped TRAIN vocabulary is an error.
- Current reference descriptions and LATEST claim sources provide retrospective event-date-bounded data, not proven historical ingestion-time availability. The mapping fingerprint detects reference drift; it does not freeze raw claim tables.
- The count dictionary inherits the old 1,028 names/order; its original vocabulary fitting scope remains unresolved. Fresh count Spark aggregation still lacks a real warehouse execution test in this environment.
- Validation chooses model checkpoints, and multiple contrasts are reported without a multiplicity-adjusted superiority claim. Bootstrap intervals do not remove leakage, historical overlap or repeated evaluation bias.
- Frozen production comparison requires exact feature order, preprocessing, labels, common evaluation rows and training-overlap evidence. A newly trained static baseline is not the saved production model.

## Production comparison inputs

Notebook 05 defaults to auditing `DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_UNIVERSE_W_FEATURES_SCORED`. Its historical scores may not cover the expanded population's exact dates. If any evaluation key is missing, duplicated or malformed, or any required probability is invalid, the production row is BLOCKED with no lift or rank. There is no intersection, nearest-date or historical-metric fallback. The four controlled models remain reportable.

To supply newly generated frozen-production scores, change only the separate `PRODUCTION_CONFIG` in notebook 05. Scores must cover every evaluation `(PATIENT_ID, END_DT)` and use verified production feature order/preprocessing without refitting. Current frozen RESP values drive every metric; scored-table RESP values are audited, never substituted. Model identity, target compatibility, training-patient exclusion and historical input vintage must be documented. The notebook records this evidence as assertions, not independent verification. Validation remains development evaluation and the inherited TEST remains retrospective, even with complete provenance.

Completed production reports use a separate artifact identity containing the model lock, reporting implementation, production configuration and prediction hashes. Changes produce a separate report and do not invalidate notebooks 01–04 or overwrite the earlier comparison. Missing production inputs do not prevent the controlled lift comparison. The three Transformer arms and static LightGBM receive different feature representations; their comparison is a predictive-system comparison, not an isolated architecture experiment.

Run local tests with the designated Python runtime and the dependency paths recorded in `local_validation.json`. The local SQL test dependencies live in `work/sql_audit` (DuckDB and SQLGlot); validation records their versions and paths. Generated Snowflake queries are parsed, transpiled, and executed on synthetic DuckDB tables to test date and outcome boundaries; that is not Snowflake execution. Synthetic notebook integration substitutes remote I/O and raw Spark count aggregation and uses reduced settings only inside its fixture. It does not certify clinical results or warehouse execution.
