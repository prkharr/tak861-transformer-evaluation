# Brian's follow-up experiments

This package implements the proposed smaller feature sets and regularization experiments, plus a regularized logistic-regression comparator. These are candidate improvements, not evidence that overfitting is fixed. The notebooks have not been run on the private Snowflake data by Codex.

## What changed

| Experiment | Inputs | Change from reference |
|---|---|---|
| `all_baseline` | All 1,028 features | Reproduce the reference architecture with added diagnostics |
| `reduced150` | Up to 150 TRAIN-selected features | Feature reduction only |
| `reduced150_dropout` | Same selected features | Dropout 0.20 to 0.30 only |
| `reduced150_decay` | Same selected features | AdamW weight decay 0.0001 to 0.001 only |
| `reduced150_small` | Same selected features | Embedding 128 to 64; layers 2 to 1; feed-forward width 256 to 128 |
| `reduced150_logistic` | Same selected features | L2 logistic regression on 12-month count totals, with TRAIN-fitted scaling |

The regularization and smaller-model variants compare against `reduced150`. They are not cumulative changes. The logistic comparator uses the same snapshots and selected feature definitions but intentionally discards month order, so the comparison changes both representation and model family.

The client core-feature experiment is **pending the approved feature list**, as requested. No list was guessed from the screenshots. No demographic, payer or provider features have been invented or added.

**Clarification from the slide shown around 12:30:** the client's 150 custom features are engineered business features. Our `reduced150` experiment selects up to 150 of the existing monthly RX/DX/PX count features; it does **not** reproduce those 150 custom features. The slide also describes a much larger AutoML candidate pool, not proof that the final model uses 800,000 inputs.

The business objective is **advanced-therapy escalation in the next 90 days**. This is now recorded in the notebook configuration and saved training/evaluation metadata. Existing `RESP` labels are preserved. Their construction has not been independently verified against that horizon, so metadata explicitly records that verification as pending.

## The four notebooks

1. **01_tensor_initialization_DL_POC.ipynb** — validate and reuse the saved full tensor, feature map and snapshot tables; save an aggregate preparation audit.
2. **02_patient_level_split_DL_POC.ipynb** — validate and preserve the saved patient assignments; save an aggregate split audit.
3. **03_transformer_training_DL_POC.ipynb** — run the controlled experiments, save each completed run, compare TRAIN/VALIDATION, and freeze the run with highest validation average precision.
4. **04_transformer_evaluation_DL_POC.ipynb** — restore that one selected run and its selected features, then produce the TEST tables and six charts.

Each notebook contains eight numbered code cells and is self-contained. No repository installation or local configuration JSON is required on Databricks.

**Notebook 01 is a saved-tensor initialization/validation stage. It does not rebuild the raw-claims feature pipeline.** The authoritative original raw-claims initialization source was not available locally. This package deliberately reuses the tensors already used for RUN_001. Notebook 02 similarly preserves the existing split rather than creating new patient assignments.

## How to run

1. Import the four notebooks from this folder as new notebooks in your private Databricks workspace. Keep the RUN_001 notebooks/results for reference.
2. Attach the approved compute that successfully ran the original pipeline. Required packages: PyTorch, scikit-learn, NumPy, pandas and matplotlib, plus the working Spark Snowflake connector. No automatic package installation or compute changes occur.
3. In **cell 1 of each notebook**, paste your existing private `sf_options = {...}` setup above the check. Credentials are intentionally absent from this package.
4. Run notebooks **01 and 02**, cells 1–8 in order. Both write small audit tables in a new namespace after validating the original saved inputs.
5. In notebook **03 cell 1**, keep `SUITE_ID = "BRIAN_001"`, the default six `EXPERIMENT_NAMES`, and `CORE_FEATURES = []` for the initial experiment suite. Run cells 1–8.
6. Notebook **03 cell 5** performs the training. It prints progress and saves each completed run immediately. Cell 6 shows the comparison; cell 7 freezes the best validation-AP run. Exact AP ties use run ID ascending.
7. In notebook **04 cell 1**, use the same `SUITE_ID`. Run cells 1–8 only after the training suite is complete. The selected model is loaded from the saved selection record; no manual feature list or threshold needs to be re-entered.

Six runs take longer than RUN_001. The raw tensor uses approximately 1.06 GiB of temporary driver disk, and feature screening may need roughly 267 MB for its TRAIN matrix plus working memory. Use suitably sized approved compute; scoring/training is local PyTorch/scikit-learn, not distributed training.

## What remains fixed

- Original 23,151 snapshots, 12,447 patients and 1,345 positive snapshots.
- TRAIN: 16,256 snapshots / 8,712 patients / 941 positives.
- VALIDATION: 3,481 snapshots / 1,867 patients / 202 positives.
- TEST: 3,414 snapshots / 1,868 patients / 202 positives.
- Original patient assignments, target labels, full feature vocabulary and 12-month order: timestep 0 newest through 11 oldest.
- Transformer maximum 20 epochs, early-stopping patience 5, TRAIN-derived positive weight, learning rate 0.001, batch size 64 and gradient clipping 1.0, except settings explicitly varied by a named experiment.

The selected checkpoint still maximizes validation AP. The binary threshold still maximizes validation F1, with exact threshold ties resolved to the highest cutoff. Training diagnostic loss is now calculated with dropout disabled, just like validation loss. The original online optimization loss is retained separately as `optimization_loss`.

## How the 150 features are selected

Selection reads **TRAIN rows only**. It checks feature presence among distinct TRAIN patients, removes empty/constant window summaries and features present in fewer than 10 TRAIN patients, then fits a constrained ExtraTrees screening model. It summarizes each feature into four three-month windows (0–2, 3–5, 6–8, 9–11) and adds the four window importances to rank each original feature. At most 150 eligible features are selected; if fewer survive, the actual count is reported rather than padded.

The Transformer continues to receive the original **12 individual monthly values** for each selected feature. Three-month summaries are used only for screening. Screening collapses within-window timing and tree importance can favor some feature types; it is a starting experiment, not proof of the best feature subset or Transformer attribution. The full-feature reference is retained to measure whether screening helps.

The selected names, indices, source order, selection settings and checksum are saved with every model. Evaluation verifies and restores this manifest and never refits feature selection. Original source hashes and split metadata remain unchanged.

Preparation and split validation are also checked against the fingerprints in the saved `RUN_001` training summary (`REFERENCE_RUN_ID` in cell 1). A changed tensor or split is rejected even when its aggregate row counts remain the same.

## Adding the client's core features later

Enter the exact approved `FEATURE_NAME` strings into `CORE_FEATURES` in notebook 03 cell 1, choose a **new SUITE_ID**, and add `"core"` to `EXPERIMENT_NAMES`. Optional recipes include `core_dropout`, `core_decay`, `core_small` and `core_logistic`. Unknown, duplicate or empty core lists fail before training.

This option supports a subset of the existing monthly feature map. Legacy engineered/static variables such as discontinuation counts, demographic values or payer fields may not be in that map. Those need their exact definitions and historical availability rules implemented upstream first. Do not rename an unrelated monthly feature to force a match or duplicate a current static value into a fabricated monthly history.

The client slide identifies useful feature concepts: provider AT prescribing/patient history, MSLT/MWT counts, age at index, generic mix and unique generics tried, CNS stimulants, NT1/NT2/IH claims, generic discontinuations, and recent diagnosis proportions. These are human-readable descriptions, not executable column definitions. Reproducing them requires the actual source columns, lookback periods, counting rules and provider/payer joins. Those definitions and the original 90-day label construction are the next inputs needed for the true engineered-feature comparison.

## Saved outputs and reruns

Original input prefix: `DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_DL_POC`.

All new results use prefix `TAK861_TX_READY_V63_DL_POC_BRIAN_V2`:

- `_PREPARATION`: aggregate tensor audit.
- `_SPLIT_AUDIT`: aggregate patient-split audit.
- `_MODEL_<SUITE_ID>_<EXPERIMENT>`: checkpoint, summary, history, feature manifest/audit, and training/validation top-K tables.
- `_SELECTION_<SUITE_ID>`: validation comparison and frozen selected-run record.
- `_EVAL_<SELECTED_RUN_ID>`: aggregate TEST report and charts.

No original tensor/split table or RUN_001 result is overwritten. Writes use error-if-exists followed by read-back verification; matching completed results can be reused. No patient IDs or per-snapshot predictions are exported by the report cells. Checkpoints contain the learned model and are stored privately in Snowflake.

After an interrupted training suite, rerun notebook 03 with the **same settings and same SUITE_ID**. Matching completed runs are verified and reused; an unfinished run restarts rather than resuming its optimizer. Changed settings, feature-selection rules, implementation or training seed require a new SUITE_ID. An existing TEST report is not rescored automatically.

For training-seed checks, keep the patient split and `FEATURE_SELECTION_SEED` fixed, use a new SUITE_ID and change `TRAINING_SEED`. Compare validation results across those suites; do not pick the luckiest seed using TEST.

## How to judge the outcome

Compare validation AP primarily, supported by validation precision/recall and lift at the top 10%, 20% and 30%. A smaller train–validation loss gap alone does not establish a better model. The notebook freezes the highest validation-AP candidate for reproducibility; small differences still need stability checks because validation contains only 202 positives.

The original RUN_001 TEST results have already been inspected. Notebook 04 labels its evaluation as reuse of that cohort, not fresh confirmation. Develop further changes using TRAIN/VALIDATION and obtain a fresh, preferably later-period holdout before a final performance claim. Metrics remain snapshot-level; multiple snapshots from one patient are not independent patients. Upstream target timing and claims availability still require the source review discussed previously.

## Method references

- [Scikit-learn: preventing leakage during feature selection](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
- [Scikit-learn: ExtraTreesClassifier and importance limitations](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).
- [Scikit-learn: regularized logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [PyTorch: dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) and [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
