# Manual features → tensor → split → Transformer → lift

Run the four notebooks in order on the approved Databricks compute. Paste your private `sf_options` at the top of cell 1 in each. All application code is embedded. Required runtime packages: PyTorch, NumPy, pandas, scikit-learn, matplotlib, joblib and the Spark Snowflake connector.

1. **01 — Prepare input.** Read Brian's exact ordered feature list from MODEL_TYPE.FEATURES and values from MODEL_DATA. Your screenshots confirm 49 features. Preserve patient/date keys, labels, fractional values and missingness.
2. **02 — Split and preprocess.** Reuse original patient assignments and dates, verified against RUN_001. Fit imputation/scaling on TRAIN only. Keep 16,256 training, 3,481 validation and 3,414 test snapshots.
3. **03 — Train one Transformer.** Numeric input is a [snapshots, features] tensor with a matching missingness mask. Internally each feature becomes a token, producing [batch, features + 1, 64] including the classification token. Show training/validation loss and lift every epoch.
4. **04 — Evaluate.** Reload the best validation checkpoint; report training/validation/test lift, full deciles, top-5/10/20/30% results, AP, ROC AUC, precision, recall and confusion counts.

Exactly one configuration is trained: 2 layers, 4 heads, width 64, feedforward 128, dropout 0.35, AdamW learning rate 0.0005, weight decay 0.0001, batch size 128, seed 42, maximum 30 epochs, patience 6. These are the dropout Transformer settings already selected on validation in the supplied results. No other models or Transformer variants run.

Snapshot summaries are not monthly sequences. This feature-token Transformer does not repeat summaries over artificial months. The original model used monthly claim-category tokens, so this is not an architecture-controlled comparison with the original 1,028-feature model.

## Preserve the completed experiment

Run all four notebooks. New outputs use TAK861_TX_READY_V63_DL_POC_MANUAL_TRANSFORMER_V1, dataset F001, run S001. The completed SELECTED_V1 experiment remains available. Keep dataset ID identical across all notebooks and run ID identical in 03/04. Change dataset ID for changed source values; change run ID for changed settings/code. Conflicting saved outputs are rejected rather than overwritten.

## Overfitting discussion

- Training lift is in-sample. Select checkpoints by validation top-10% lift, then validation AP, then earliest epoch. Threshold uses validation F1. TEST never selects either.
- Examine training/validation loss and lift together. Growing training lift with stalled validation lift indicates a generalization gap. Lower training lift than a reference may also reflect representation or fit.
- Earlier screenshots show top-10% lift **3.782 TRAIN, 2.963 VALIDATION, 3.608 TEST**. These are the earlier observed results; Codex has not run the revised notebooks on private data.
- Before comparing against Brian's reported lift of 6, confirm its split, population, dates, outcome and targeting fraction. Lift here is selected response rate divided by that split's response rate. Decile 10 contains highest scores; ties use patient/date order.

Existing TEST results have already been inspected. Brian's feature-selection population and historical feature availability have not been independently verified. Later untouched data is needed to confirm generalization. Fewer features do not guarantee lift 6 or eliminate overfitting.

Predictions remain in the private warehouse. Load only this pipeline's own verified model artifacts; joblib files must be trusted.
