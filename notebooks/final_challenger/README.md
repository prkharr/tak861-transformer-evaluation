# Development challenger — not yet run on project data

This is the intended final development workflow, not a proven winning model. **Project-data results: pending.** The three notebooks were previously numbered 11–13; the files now use 01–03. Their executable contents and saved artifact identities are unchanged.

## Run these three stages

| Order | Notebook | Action and expected output |
| --- | --- | --- |
| 1 | [01_input_audit.ipynb](01_input_audit.ipynb) | Verify existing frozen inputs, patient splits, monthly orientation, provenance and timing channels; resolve blockers before training. |
| 2 | [02_tuning_and_blend.ipynb](02_tuning_and_blend.ipynb) | Tune static LightGBM, timing LightGBM and NLP fusion within TRAIN; learn the blend from TRAIN out-of-fold predictions; save resumable stages and print the completed report table. |
| 3 | [03_validation_report.ipynb](03_validation_report.ipynb) | Load saved predictions, compare captures/lift with incumbents, calculate patient-cluster intervals and save the aggregate report without retraining. |

The notebooks reuse the completed `R20261030_NEW_SUMMARY` inputs and preparation. **Do not rerun tensor creation or splitting if those verified artifacts already exist.** If they are missing, use the [established route](../established_experiments/README.md) to investigate/recover them rather than inventing replacement inputs.

## Fill in before running

| Setting | What to do |
| --- | --- |
| `sf_options` | Fill all seven blank connection fields privately in each notebook, using the same approved connection. Never commit the filled copy. |
| Parent `RUN_ID`, `RUN_MODE`, `CONFIG` | Keep `R20261030_NEW`, `SUMMARY` and the existing parent configuration unchanged. These identify the existing tensors. |
| `EXACT_CHANNELS` | Leave blank if the audit resolves one unambiguous sleep-study and sleep-wake channel. Otherwise provide the exact verified dictionary names consistently in all three notebooks. |
| `CHALLENGER_SETTINGS` | Defaults: `seconds=28800`, `device='auto'`, `threads=8`, `bootstrap=2000`. A GPU is strongly recommended. |
| `CHALLENGER_REPORT_TABLE` in stage 03 | Copy the completed table printed by stage 02; do not guess a table name or report an incomplete search as final. |

Use approved Databricks classic Spark compute with JVM access, the Snowflake Spark connector, PyTorch, LightGBM, NumPy, pandas, scikit-learn and matplotlib. Shared/serverless Spark Connect without JVM access is rejected by the artifact publisher.

## Runtime and resuming

The default eight-hour budget starts after input loading and checks between iterations/batches. Input loading, feature selection and warehouse calls can extend elapsed time. CPU execution may require multiple overnight resumptions.

Rerun **stage 02** unchanged to reuse completed work; an interrupted fit restarts. Changing only the time budget preserves reusable stages. Changing model settings or runtime versions creates a separate experiment. Keep the same environment when resuming. Saved artifacts are verified and never silently overwritten.

## What the result will mean

Report Lift@10/20/30%, positive snapshot captures, distinct captured positive patients, AP, ROC-AUC, probability losses, uncertainty, overlap and runtime. Compare against the saved NLP incumbent and static control on identical validation records. A blend is not guaranteed to outperform either parent.

Supervised selection and tuning use patient-separated TRAIN folds. Existing NLP vocabulary/IDF uses all TRAIN documents without labels; it is not refitted within every fold, so OOF estimates are conditional on that representation. The existing validation set has already informed development. TEST and production verification remain disabled.

For detailed search grids and artifact behaviour, see the [technical protocol](../../experiments/evidence_challenger/README.md). Notebook text that refers to stages 11–13 uses the historical numbering: 11 = current 01, 12 = current 02, 13 = current 03.
