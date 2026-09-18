# TAK861 / NT1 Transformer: split, train and evaluate

Continue from the existing completed `tensor_complete` monthly table on the work laptop. Run the three notebooks under **`notebooks/Transformer Encoder/`** to preserve snapshot identity, freeze the patient split, train a compact Transformer and evaluate it on held-out TEST snapshots. LightGBM remains deferred.

The [shared project reference](https://chatgpt.com/share/6aacb1e6-dad0-83ee-9fce-3a370814f508) describes the completed Spark monthly table and a proposed model. It does not confirm an aligned NumPy tensor, saved checkpoint or completed model training. The upstream `tensor_initialization.ipynb` was not available for inspection here and is neither recreated nor included. No real project model is trained or evaluated on this machine; actual data and results remain on the work laptop.

## Run order

| Notebook | Action | Local outputs |
| --- | --- | --- |
| `notebooks/Transformer Encoder/01_patient_level_split.ipynb` | Load the completed monthly data, validate and save an identity-preserving bundle, then freeze or reuse patient-level splits | Bundle; frozen manifest; aggregate split summary; integrity record |
| `notebooks/Transformer Encoder/02_transformer_training.ipynb` | Train on TRAIN, select the checkpoint and one operating threshold on VALIDATION | Best checkpoint; epoch history; metadata; training chart |
| `notebooks/Transformer Encoder/03_transformer_evaluation.ipynb` | Verify the frozen artifacts, score TEST and evaluate the selected run | Aggregate global/threshold metrics; decile and top-K tables; gains/lift charts; report |

The root `06_transformer_evaluation.ipynb` remains a separate **predictions-only** route for users who already have the original frozen manifest and existing Transformer scores. It does not train or load this pipeline's model and requires no PyTorch installation. Do not run both routes unless you intentionally want to cross-check the same predictions.

## Prepare the work environment

Use an approved Python environment and notebook frontend (Databricks, Jupyter or VS Code). For the full pipeline, use Python 3.10 or newer and an approved CPU/GPU PyTorch build compatible with your environment. Choose the wheel from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) or your organization's package mirror; this project does not assume a CUDA version.

After PyTorch is installed, run from the repository directory:

```sh
python -m pip install -r requirements-training.txt
```

For the predictions-only root notebook, install `requirements.txt` instead; compatible dependency versions support Python 3.9 or newer. The requirements provide a Python kernel, not a notebook frontend. On Databricks, use the supported package installation mechanism and restart the kernel if required by that environment.

Copy `config.example.json` to the ignored `config.local.json` at the repository root and fill it in on the work laptop. Alternatively, set `TAK861_CONFIG` to the actual local JSON path. On PowerShell, use `$env:TAK861_CONFIG = 'your local path'`; on a Unix shell, use `export TAK861_CONFIG='your local path'`. Set the variable in the environment used by the notebook kernel. All configured relative paths resolve against the configuration file.

Keep private input files, bundle exports, manifests, checkpoints and reports in approved local artifact storage. The default `artifacts/` directory is ignored by Git. Custom destinations need the same protection. No credentials, table connections or patient data belong in the shared configuration example.

## Connect the completed monthly table

The source contract is the already-completed `tensor_complete` table, with:

- `PATIENT_ID`: nonmissing string identifier preserving any leading zeros;
- `END_DT`: valid calendar snapshot date;
- `RESP`: 0 or 1, constant across the snapshot's months;
- `TIME_STEP`: each integer from 0 through 11 exactly once per snapshot, newest to oldest;
- the established `RX__`, `DX__` and `PX__` feature columns containing finite, nonnegative **raw counts**.

`RESP`, dates and timestep columns are metadata rather than predictors. `TIME_STEP_MONTH` is not a feature even if an earlier monthly export includes it. Missing activity months must already be present with zero feature values. Missing month rows or malformed snapshots cause an error rather than being dropped or silently filled.

The default population contract is historical V63 with claims vintage `20260825`: 23,151 snapshots, 12,447 patients, 1,345 positive snapshots, and `X.shape = (23151, 12, 1028)` with 42 RX, 490 DX and 496 PX features. The bundle saves the ordered `(PATIENT_ID, END_DT)` rows separately from `X`, along with `y`, the exact feature vocabulary and timestep order. It never guesses how an unrelated tensor's rows align with patient keys.

Choose one input mode:

| `input.mode` | Configuration or existing object |
| --- | --- |
| `in_memory` | Existing `tensor_complete` Spark/pandas DataFrame in the kernel running notebook 01 |
| `monthly_parquet` | `input.path` points to an approved completed monthly Parquet file or Spark Parquet directory |
| `bundle` | `bundle_dir` points to a complete bundle previously exported by notebook 01 |

A complete existing bundle takes precedence over collecting the monthly input again. The loader verifies its fingerprint and metadata. An incomplete bundle or population mismatch fails; choose the correct source or a new artifact directory instead of overwriting it.

When using the current Spark session, exporting the existing completed table can make future runs independent of that session:

```python
# approved_monthly_export_path must be an actual approved location supplied locally.
# This exports the completed table; it does not rerun claims extraction or mappings.
tensor_complete.write.mode("errorifexists").parquet(approved_monthly_export_path)
```

Spark's [`errorifexists` write mode](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameWriter.mode.html) prevents replacing an existing export. Configure a Python-accessible path when later loading the Parquet directory through pandas; do not assume a Spark filesystem URI is directly readable by local Python.

If an existing `feature_list` is available in in-memory mode, notebook 01 uses it. You can instead write the original list to an approved local JSON file and set `input.feature_order_json`. Feature names may be proprietary, so keep that file with local artifacts. Without an explicit list, the adapter sorts all established prefixed feature names, matching the shared reference's convention; verify that convention against the actual upstream code.

The raw float32 tensor alone occupies about **1.06 GiB**; the monthly DataFrame and conversion copies require substantially more memory. Spark [`toPandas()` collects data into driver memory](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toPandas.html). Use an appropriately sized driver for notebook 01; later runs can memory-map the saved `X.npy` rather than collecting Spark data again.

## Complete the upstream review locally

`config.example.json` leaves the source-review flags false because the upstream implementation has not been inspected here. Before running, verify the actual feature order, V63 snapshot semantics, predictor cutoff and exclusion of outcome-window events. Record the review in an approved location, set `review_reference`, and mark only completed checks true.

The required declared versions are `population_version="V63"`, `claims_vintage="20260825"` and `value_representation="raw_counts"`. The pipeline applies `log1p` once during training/inference by default; do not supply already-transformed values as raw counts. Tensor shape and numeric checks cannot prove event-time leakage prevention. Review flags are explicit local attestations, not evidence inferred from the tensor.

## Split and training behavior

Notebook 01 reuses an existing manifest exactly. If no manifest exists, it splits unique patients into TRAIN/VALIDATION/TEST using 70%/15%/15% defaults and seed 42, stratified by each patient's maximum RESP. All snapshots from one patient stay together, including patients with both positive and negative snapshots. Fractions refer to patients; snapshot fractions can differ. The natural outcome imbalance is retained. If you are evaluating an already-trained model, supply its original manifest rather than making a new split after training.

Reruns verify and preserve the original split audit, including its recorded creation seed and fractions. If an existing manifest has no audit, the new audit marks its original creation settings as unknown instead of assigning the current configuration to it.

The default Transformer uses a 128-dimensional learned projection, learned temporal positions, two encoder layers, four heads, a 256-dimensional feedforward block, dropout 0.2, mean pooling and a binary-logit head. Zero-activity months remain actual timesteps. The pipeline learns no preprocessing statistics from held-out data; `log1p` is applied per batch. Positive loss weight is computed from TRAIN only. AdamW and gradient clipping train the model, with early stopping on VALIDATION average precision.

The highest validation-AP checkpoint is saved; exact ties retain the earliest epoch. A fixed classification threshold is then selected by maximum VALIDATION F1, with exact ties choosing the highest threshold. TEST predictions and metrics are not computed during training. Integrity checks bind the checkpoint to the full bundle, frozen manifest, feature vocabulary and time order.

Choose a new or empty `run_dir` for training. Existing nonempty runs are not overwritten. Select any new experiment using TRAIN/VALIDATION only and freeze the final choice before notebook 03. Random seeds and deterministic settings improve reproducibility within a fixed software/device environment; they do not promise identical results across all hardware. Class-weighted sigmoid scores support ranking but are not established calibrated clinical probabilities.

## Held-out evaluation

Notebook 03 scores the frozen TEST snapshots, checks exact identity/label coverage and reuses the threshold saved by notebook 02. It reports average precision, ROC-AUC, threshold precision/recall/F1 and confusion counts, along with the detailed targeting analysis:

- Decile 1 is highest propensity and decile 10 lowest, using descending predicted probability plus a fixed label-independent identity-hash tie break.
- For `N` TEST snapshots, decile `d` ends at rank `ceil(d*N/10)`. TEST labels never choose boundaries.
- Each decile reports snapshots, actual positives, response rate, positive share, cumulative positive count/capture, individual and cumulative lift, and within-decile/cumulative precision.
- Top-10%/20%/30% capture/recall, precision and lift use the first one/two/three deciles and state the actual selected sizes. Top-decile lift equals `Precision@10% / overall TEST response rate`.
- Cumulative gains and lift charts show random-ranking references; top-K charts summarize capacity-based targeting performance.

Average precision uses [scikit-learn's non-interpolated definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), not trapezoidal PR-AUC. Missing classes produce explicitly undefined metrics where appropriate. See `EVALUATION_SPEC.md` for formulas, tie handling and edge cases.

The unit remains a **snapshot**, not a unique patient reached. Repeated snapshots are not deduplicated. These are point estimates without confidence intervals. The Transformer can be assessed for how strongly it concentrates positives at each capacity, but incremental value over LightGBM remains unmeasured until that model is evaluated on the identical frozen population, split and snapshot set. Global ROC-AUC alone would not establish improved prioritization.

## Existing predictions-only route

To use the root `06_transformer_evaluation.ipynb`, supply these environment variables instead of the three-notebook pipeline configuration:

| Variable | Meaning |
| --- | --- |
| `TAK861_SPLIT_MANIFEST` | Original full TRAIN/VALIDATION/TEST manifest |
| `TAK861_TRANSFORMER_PREDICTIONS` | CSV/Parquet with exactly the TEST snapshot keys and `P_RESP1` |
| `TAK861_PROVENANCE` | Optional run-declaration JSON |
| `TAK861_EVALUATION_OUTPUT` | Optional output path, default `artifacts/transformer_evaluation` |

The manifest requires `PATIENT_ID` (string), `END_DT` (`YYYY-MM-DD`), `RESP` (0/1) and `SPLIT` (`TRAIN`, `VALIDATION`, `TEST`). Predictions require the same two keys and finite `P_RESP1` values in `[0,1]`. Optional prediction labels/split values are cross-checked. Duplicate, missing and extra TEST rows fail. Labels come from the canonical manifest; scores align by keys, not incoming row order.

That notebook also supports existing pandas DataFrames named `frozen_snapshot_manifest` and `transformer_test_predictions` by setting `USE_IN_MEMORY=True`. Preserve keys and scores together through the original inference pipeline. A shuffled loader's scores cannot safely be paired with an unrelated list of keys merely because their lengths match. Apply sigmoid exactly once for a binary-logit model; do not guess a two-class probability array's positive-class column.

Optional `Transformer` provenance declarations require `snapshot_manifest_sha256`, `population_version="V63"`, `claims_vintage="20260825"`, `model_frozen_before_test=true`, `training_split="TRAIN"`, `tuning_split="VALIDATION"`, `scoring_split="TEST"`. Compute the manifest digest with `manifest_fingerprint`. Missing declarations remain explicitly unverified; matching declarations do not independently audit training history. This compatibility notebook focuses on ranking and global discrimination and does not invent a classification threshold.

## Repository layout and verification

```text
notebooks/
  Transformer Encoder/
    01_patient_level_split.ipynb
    02_transformer_training.ipynb
    03_transformer_evaluation.ipynb
src/
  configuration.py
  data_utils.py
  evaluation.py
  model.py
  training.py
  metrics.py
  reproducibility.py
tests/
06_transformer_evaluation.ipynb
targeting_evaluation.py
config.example.json
requirements.txt
requirements-training.txt
EVALUATION_SPEC.md
VALIDATION.md
README.md
.gitignore
```

Run the synthetic tests from the repository root using `python -m pytest -q`. Read `VALIDATION.md` for what has actually been checked and any remaining environment limitations. Synthetic tests validate code behavior; they do not measure performance on the V63 cohort.

To verify notebook execution in fresh kernels, run:

```sh
python scripts/smoke_notebooks.py --output-dir artifacts/notebook_smoke_001
```

The output directory must not already exist. This check creates synthetic monthly inputs, executes notebook 01 twice to verify frozen-split reuse, then executes 02 and 03 in separate kernels. It includes categorical patient identifiers and literal identifiers such as `NA` to check identity preservation through saved artifacts. All synthetic inputs, executed copies and reports remain in the chosen local directory; source notebooks stay cleared. See `REVIEW.md` for the implementation review and corrections.

Keep local configuration, patient/monthly data, tensors, manifests, checkpoints, prediction exports and reports outside Git. Clear notebook outputs before sharing a notebook that has been run, and inspect every staged file. `.gitignore` does not remove previously tracked files. This repository does not include the upstream initialization notebook, private context attachments, credentials or proprietary source exports.
