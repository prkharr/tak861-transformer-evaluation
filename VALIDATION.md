# Validation record

## Downstream pipeline update — 18 September 2026

Added the executable path from the completed monthly table through patient splitting, Transformer training and held-out evaluation. The source reference was the shared conversation and master context; the authoritative `tensor_initialization.ipynb` was not available in the accessible repository branches or this machine. It was not modified or included.

Verified with synthetic fixtures only:

- Three downstream notebooks parse and execute in **three fresh kernels**, using only saved artifacts to communicate: 01 split, 02 training, 03 TEST evaluation.
- A shuffled monthly Parquet input with 80 synthetic patients, 160 snapshots, 12 months and 6 synthetic feature channels was converted with metadata/feature order preserved. Production defaults continue to require the stated V63 population and 1,028 channels.
- The pipeline generated a frozen patient manifest, complete memory-mappable tensor bundle, best checkpoint, validation-selected operating threshold, epoch history, five aggregate evaluation CSV tables and six evaluation charts. No synthetic patient identifiers appeared in displayed notebook outputs or aggregate reports. Synthetic execution outputs are outside the deliverable.
- Data/split tests cover patient-level stratification, mixed-label patients, reuse despite changed requested seed, complete cohort matching, preserved row order, raw-count validity, zero months, source-review declarations, feature order and fingerprint corruption.
- Training tests verify class weights from TRAIN only, AP-based checkpoint selection, deterministic checkpoint reload, unchanged model weights/validation threshold after flipping every TEST label, and refusal of incomplete or overwritten runs.
- Evaluation integration tests verify the validation-selected threshold independently, exact TEST key coverage, checkpoint/input/split fingerprints, all six charts, aggregate-only exports, and refusal to overwrite an existing evaluation report.
- Notebook 01 preserves the original split-creation audit instead of replacing its seed/configuration during a reuse. Missing provenance for an existing manifest remains explicitly unknown.

Full test suite: **107 tests passed** (35 original evaluation + 56 data/split + 8 training + 8 pipeline integration). The complete pipeline smoke used PyTorch **2.8.0+cpu** with Python 3.9.7 on Windows. A current approved Python/PyTorch combination is recommended for the work laptop; no CUDA or Databricks runtime was available for validation here.

The initialized default 1,028-feature, 12-month Transformer has **406,785 trainable parameters**. Its architectural dimensions were checked locally; no V63 training epoch or measured project-model result was produced. Actual training, cutoff review and performance assessment require the private work-laptop inputs.

All delivered notebooks remain cleared. Security review covers the complete staged source/docs/config/test set; only synthetic fixtures appear in tests, and no patient export, generated tensor, weight file, credential, source SQL or original context attachment is included.

## Original predictions-only package validation

Validated on 18 September 2026 using synthetic records only. No work-laptop data, trained checkpoint, or real model predictions were available on this system. This is a validation of the evaluator and notebook, not a measured assessment of Transformer performance.

## Completed checks

- **35 synthetic unit tests passed.** Coverage includes hand-calculated counts/rates/capture/lift, top-K consistency, unequal decile sizes, deterministic ties, invariance of ranking to labels and input row order, TEST-only lift denominators, exact snapshot matching, patient disjointness, missing/duplicate/invalid inputs, no/all-positive outcomes, provenance checks, and aggregate-only report exports.
- **Notebook executed end to end** with synthetic CSV inputs, shuffled prediction rows, repeated patient snapshots and 23 TEST snapshots. The notebook generated four aggregate CSV tables, three PNG charts, an audit JSON and a Markdown report.
- Notebook schema and all code-cell syntax validated; the delivered notebook has empty outputs and null execution counts.
- Synthetic chart layouts were visually inspected for legible labels, plotted baselines and distinct decile/cumulative lift panels.
- The evaluator refuses fewer than ten TEST snapshots; zero positives yield undefined recall/lift rather than fabricated zeros.
- The package contains source code, a notebook, documentation and synthetic tests. It contains no trained weights, tensor, patient export, credential file, proprietary SQL, source context attachment or actual performance output.

## Tested environment

Windows, Python 3.9.7; NumPy 2.0.2; pandas 2.3.3; scikit-learn 1.6.1; matplotlib 3.9.4; nbformat 5.10.4; nbclient 0.10.2; ipykernel 6.31.0; pytest 8.4.2; PyArrow 21.0.0. Requirements allow compatible versions; this is the environment used for the recorded checks.

Matplotlib emitted upstream pyparsing deprecation warnings during tests. They did not affect test outcomes or report generation.

## Still to verify on the work laptop

- Provide the existing frozen full snapshot split manifest and correctly keyed Transformer TEST probabilities.
- Verify score-to-snapshot correspondence in the original inference code. Matching row counts alone cannot establish that correspondence.
- Verify historical V63 population, fixed `20260825` claims vintage, first-initiation target, time cutoffs and observability in the upstream pipeline.
- Confirm training, preprocessing, early stopping, calibration and operating-threshold selection excluded TEST data.
- Run the notebook in the work environment and review the actual metrics and tie diagnostics. No model-superiority conclusion is available; LightGBM comparison is deferred.
