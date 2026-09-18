# Validation record

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
