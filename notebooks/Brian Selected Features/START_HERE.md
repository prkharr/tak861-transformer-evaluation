# Brian's selected features: four self-contained notebooks

Run 01, 02, 03, then 04 on the approved Spark/Databricks compute previously used
to connect to Snowflake. Put your existing private `sf_options` connection setup
at the top of cell 1 in each notebook. No credentials are included. All application
code is embedded in the four notebooks; no repository checkout, helper upload,
CSV feature transcription or extra model file is needed.

The runtime needs the Spark Snowflake connector, NumPy, pandas, scikit-learn,
PyTorch, matplotlib and joblib. The notebooks do not download dependencies.
Use the same software runtime for training and reloading its saved models.

## Inputs and dates

- Read the exact ordered list in
  `DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_MODEL_TYPE.FEATURES`.
- Read only those feature values from `TAK861_TX_READY_V63_MODEL_DATA`.
- Audit the list against `TAK861_TX_READY_V63_FINAL_MODEL`. The configured list
  is authoritative: an omitted final-summary row does not silently remove a predictor.
- Preserve all 23,151 saved `PATIENT_ID + END_DT` snapshots, 12,447 patients and
  1,345 positive labels. Exact date/key/label coverage must pass again at extraction.
- Reuse `TAK861_TX_READY_V63_DL_POC_PATIENT_SPLIT`, checked against the original
  `TAK861_TX_READY_V63_DL_POC_MODEL_RUN_001` artifact fingerprints.
- Source manual-feature tables and the 1,028-feature monthly tensor are not input
  dependencies for these notebooks. No claim histories or monthly dates are reconstructed.

Keep DATASET_ID (`F001`) identical in all notebooks and SUITE_ID (`S001`) identical
in notebooks 03/04. Use a new dataset ID for changed source values/list and a new
suite ID for changed code, settings or seeds. Outputs live under
`TAK861_TX_READY_V63_DL_POC_BRIAN_SELECTED_V1` in DS_ML. Writes use error-if-exists
and checksum-verified read-back; matching saved runs can be resumed.

## What each notebook does

1. **Preparation:** freezes the complete configured feature list and exact snapshot
   matrix. Preserves fractional values and missingness; blocks duplicate/missing
   keys, label conflicts, prohibited metadata predictors, missing columns and infinities.
2. **Split/preprocessing:** verifies unchanged patient assignments and fits median
   imputation and standardization on TRAIN only. All-missing TRAIN columns remain
   in the vocabulary with zero imputation. Missingness is encoded per selected value.
3. **Training:** compares four feature-token Transformers (baseline, dropout,
   weight decay, smaller architecture), logistic regression and sklearn histogram
   gradient boosting. All use the same selected source features. The tree model
   is a comparator, not an exact reimplementation of Brian's LightGBM.
4. **Evaluation:** scores the frozen validation-selected winner on TRAIN, VALIDATION
   and TEST, using its saved preprocessing and validation-selected F1 threshold.

The Transformer has one token per selected feature plus a classification token.
It does not repeat static summaries across twelve pretend months. Each feature
token encodes its numeric value, identity and whether its source value was missing.
The tabular comparators receive the same values and missingness flags.

## Lift objective and training-set reporting

Primary checkpoint/model selection is **VALIDATION top-10% lift**; ties use
VALIDATION average precision, then earliest checkpoint / declared recipe order.
Training lift is reported every Transformer epoch and in each candidate's full
decile table. It is an in-sample diagnostic and never chooses the winner.

The final notebook provides:

- TRAIN/VALIDATION/TEST AP, ROC AUC, precision, recall, F1 and confusion counts;
- full decile lift, response rates, cumulative recall and cumulative lift;
- top-5%, 10%, 20% and 30% precision, recall and lift;
- training/validation loss and lift curves in notebook 03;
- lift/recall, precision-recall and TEST confusion charts in notebook 04;
- saved exact-key predictions, feature manifest, preprocessing, model, histories,
  metrics and plots within the private warehouse artifact tables.

Decile 10 means highest scores. Lift uses each split's own outcome prevalence.
Top-K uses ceiling rounding and deterministic patient/date tie-breaking. Metrics
count snapshots, including repeated snapshots of a patient within a split.
Prediction cutoffs/labels/splits stay fixed, but the feature representation and
Transformer architecture change compared with the original temporal model.

## Interpretation

The goal is increased lift; improvement is not guaranteed. Select from validation
and report TEST only for the frozen winner. Existing TEST results have already
been inspected. The client's feature-selection population is not established,
so this is a retrospective feature comparison, not untouched validation of feature
discovery. Confirm any improvement on a later, untouched cohort.

Using a matching END_DT does not itself prove that every feature was available
historically. Existing RESP is retained; its detailed 90-day construction and
the feature availability rules have not been independently verified.

Saved model files use joblib serialization. Load only artifacts generated by these
notebooks in the controlled experiment tables, with matching provenance/checksums;
never substitute an external model file.
