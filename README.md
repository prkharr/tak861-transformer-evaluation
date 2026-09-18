# TAK861 / NT1: evaluate the existing Transformer

This package evaluates **held-out predictions from your Transformer once trained**, using its frozen patient-level TEST split. It produces detailed decile, cumulative gains, lift and top-10%/20%/30% results. LightGBM scoring and comparison are deferred until its artifacts are available.

The [shared project reference](https://chatgpt.com/share/6aacb1e6-dad0-83ee-9fce-3a370814f508) describes the tensor and proposed Transformer but does not confirm a saved checkpoint or training results. Model training status, artifact paths and the original split must be established on the work laptop. These artifacts are not included here, and this package contains no measured model-performance results. It does not recreate the patient split, train a model, invent an architecture, rebuild V63 or query claims.

## Files

| File | Purpose |
| --- | --- |
| `06_transformer_evaluation.ipynb` | Cleared-output notebook for running and reviewing the evaluation |
| `targeting_evaluation.py` | Shared input checks, deterministic ranking, metrics, charts and report |
| `EVALUATION_SPEC.md` | Definitions, denominators, edge cases and interpretation |
| `tests/` | Synthetic checks of evaluation behavior; no real patient records |
| `requirements.txt` | Evaluation dependencies; a training framework is not required |
| `.gitignore` | Keeps local data, credentials, artifacts and checkpoints out of Git |

## Run on the work laptop

1. Download or clone this package. Keep the notebook and `targeting_evaluation.py` in the same directory.
2. Use Python 3.9 or newer and install dependencies into the approved environment:

   ```sh
   python -m pip install -r requirements.txt
   ```

3. Point these environment variables to the existing local CSV or Parquet inputs before launching Jupyter/VS Code/Databricks. Use your real paths locally; no machine-specific paths are embedded in the notebook.

   | Variable | Meaning |
   | --- | --- |
   | `TAK861_SPLIT_MANIFEST` | Full frozen cohort snapshot manifest, including TRAIN, VALIDATION and TEST |
   | `TAK861_TRANSFORMER_PREDICTIONS` | Probabilities from the frozen Transformer for every TEST snapshot |
   | `TAK861_PROVENANCE` | Optional JSON declaration for the existing model run |
   | `TAK861_EVALUATION_OUTPUT` | Optional output directory; defaults to `artifacts/transformer_evaluation` |

   On PowerShell, use `$env:VARIABLE_NAME = 'your local value'`; on a Unix shell, use `export VARIABLE_NAME='your local value'`. Set variables in the environment that starts the notebook kernel. The notebook also supports existing in-memory pandas DataFrames; see its configuration cell.

4. Open `06_transformer_evaluation.ipynb`, select the prepared Python kernel, and run all cells. A Jupyter frontend or VS Code notebook extension must already be installed; `ipykernel` supplies the kernel.
5. Read the decile table, gains/lift charts, top-K results, global discrimination summary and generated Markdown report. Input errors fail before evaluation; do not remove rows to make a failed join pass.

The ordinary run writes aggregate reports and charts under the configured output directory. It does not write per-snapshot ranks or patient predictions. Aggregate reports may still be confidential; keep run outputs within the approved environment. Clear all notebook outputs before committing a notebook that has been run.

## Required input contracts

**Frozen manifest:** one row per original cohort snapshot, with `PATIENT_ID`, `END_DT`, `RESP`, `SPLIT`.

- `PATIENT_ID` must be a string, preserving leading zeros. Recover identifiers from the authoritative source if they have already been converted to numbers.
- `END_DT` must be a calendar date in `YYYY-MM-DD` form. Do not silently truncate times or shift dates.
- `RESP` must be 0 or 1 and retain the established first-advanced-therapy initiation within 90 days definition.
- `SPLIT` must be exactly `TRAIN`, `VALIDATION` or `TEST`. The manifest must contain all three, with every patient's snapshots entirely within one split.
- `(PATIENT_ID, END_DT)` must be unique and complete. Supply the manifest that was frozen for the trained model; this evaluator cannot reconstruct or prove its history.

**Transformer predictions:** one row per TEST snapshot, with `PATIENT_ID`, `END_DT`, `P_RESP1`. `P_RESP1` must be a finite probability in `[0, 1]` for `RESP=1`. Optional `RESP` and `SPLIT` columns are checked against the manifest. Duplicate, missing, extra or non-TEST snapshots fail validation. Labels come from the manifest, and score rows are aligned by both snapshot keys rather than their incoming row order.

The project context specifies historical V63, claims vintage `20260825`, and a sample grain of patient plus snapshot date. These describe the required upstream artifacts; this evaluator does not derive claims vintage or model leakage from scores. It reports the actual input sizes rather than treating the full cohort's previously documented size as the TEST size. It retains repeated snapshots and the TEST response balance.

### Connect the existing inference code

Prefer the predictions already exported by the trained model. If predictions are currently in memory, preserve the snapshot keys **through the existing inference pipeline**. The following adapter can run after inference; it does not infer a model architecture or load an unknown checkpoint:

```python
import numpy as np
import pandas as pd

# Existing objects, obtained together from the same ordered inference records:
# test_snapshot_keys: pandas DataFrame with PATIENT_ID and END_DT
# test_probabilities: one probability for RESP=1 per inference record
# frozen_snapshot_manifest: original full manifest from model training

probabilities = np.asarray(test_probabilities)
if probabilities.ndim == 2 and probabilities.shape[1] == 1:
    probabilities = probabilities[:, 0]
if probabilities.ndim != 1 or len(test_snapshot_keys) != len(probabilities):
    raise ValueError("Keys and probabilities must have one matching row per snapshot.")
if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
    raise ValueError("Expected finite RESP=1 probabilities in [0, 1].")
transformer_test_predictions = test_snapshot_keys[["PATIENT_ID", "END_DT"]].copy()
transformer_test_predictions["P_RESP1"] = probabilities
```

Length equality cannot prove correct score-to-key correspondence. Obtain keys and scores from the same inference record/batch ordering and audit that correspondence in the existing model code. Do not sort one object independently, join by row number, or pair a shuffled loader's scores with an unrelated list of keys. If generating scores from a binary-logit model, use the existing inference code in evaluation mode without gradient tracking, apply sigmoid exactly once, and preserve the two snapshot keys. A two-class probability array needs its known `RESP=1` column; do not flatten it or guess class order. The evaluator rejects logits outside `[0, 1]` but cannot detect every mislabeled probability array.

For in-memory use, set `USE_IN_MEMORY = True` in the notebook after the two DataFrames exist in that kernel. If importing this notebook into the training environment, keep the same `targeting_evaluation.py` accessible on the Python path.

## Optional provenance declaration

Use a JSON file with a `Transformer` object containing:

```json
{
  "Transformer": {
    "snapshot_manifest_sha256": "the digest of the actual frozen manifest",
    "population_version": "V63",
    "claims_vintage": "20260825",
    "model_frozen_before_test": true,
    "training_split": "TRAIN",
    "tuning_split": "VALIDATION",
    "scoring_split": "TEST"
  }
}
```

Compute the digest with `manifest_fingerprint(frozen_snapshot_manifest)` and retain it with the actual training run. A declaration created retrospectively is not proof that the split or model was frozen before TEST inspection. A missing declaration is allowed and explicitly recorded as unverified. A supplied declaration must match the manifest and expected project versions or the run fails. For in-memory mode, an optional `transformer_run_metadata` dictionary uses the same structure.

## How to read the results

Decile 1 contains the highest predicted propensities; decile 10 contains the lowest. Ranking uses probabilities plus a fixed, label-independent identity hash to resolve score ties. No TEST labels define the buckets or their boundaries. Top-K metrics use the first one, two or three complete deciles, with the actual selected size and fraction reported when the TEST size is not divisible by ten.

The report explicitly distinguishes within-decile response rate/precision from cumulative precision in the selected population. `Recall@10%`, `Recall@20%` and `Recall@30%` are the fractions of all TEST positive snapshots captured; top-decile lift equals `Precision@10% / overall TEST response rate`. The gains chart uses the actual cumulative selected fraction on the horizontal axis. The lift chart shows both individual-decile and cumulative lift against a random-ranking baseline of 1.

Average precision and ROC-AUC provide supporting global discrimination context. Average precision uses scikit-learn's non-interpolated definition, not trapezoidal integration of the precision-recall curve. See the [official metric documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). No classification threshold is selected using TEST labels.

This run quantifies how well the Transformer concentrates positive **snapshots** at a fixed review capacity. It does not deduplicate patients or estimate unique patients reached. It does not establish improvement over LightGBM; that requires the later benchmark on the identical frozen manifest and exact TEST snapshots. Keep the manifest fingerprint and evaluation definitions for that comparison. See [EVALUATION_SPEC.md](EVALUATION_SPEC.md) for all formulas and limitations.

## Verification and sharing

Run the synthetic unit tests from this package directory:

```sh
python -m pytest -q
```

Before sharing code, clear notebook outputs, inspect every staged file, and exclude private inputs, local run reports, model artifacts and secrets. `.gitignore` reduces accidental additions; it does not remove files that Git already tracks. This package includes neither original context attachments nor proprietary source exports. It does not publish a repository by itself.
