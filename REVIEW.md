# Transformer implementation review

Reviewed on 18 September 2026, starting from commit `5763b03` and the previously delivered ZIP. The ZIP's 29 files matched the local source byte for byte. This review covers notebooks 01-03, their supporting modules, and the existing predictions-only evaluator. LightGBM and ablations remain deferred.

The Transformer implementation follows the intended training and evaluation design. The review found four categories of correctness defects and fixed them locally. The source notebooks did not require changes.

## Corrections

| Finding | Previous behavior | Corrected behavior |
| --- | --- | --- |
| Validation F1 ties | Floating-point rounding could choose a lower threshold when two cutoffs had mathematically identical F1. | Compare F1 using integer counts and exact cross-products. Select the highest threshold on a true tie; keep equal-score groups together. |
| CSV patient identities | Default missing-value parsing converted valid literal keys such as `NA`, `NULL`, or `N/A` into missing values, breaking saved split reuse. | Preserve literal strings when loading manifests and predictions. Empty keys and invalid required fields still fail validation. |
| Categorical patient identities | Unused categories could create nonexistent groups. Category ordering could change split construction or a manifest fingerprint after CSV reload. | Group only observed values and order validated patient keys by their literal strings. Saved and reloaded identities retain their meaning. |
| Complex numerical inputs | Casting complex counts or scores to real numbers could silently discard imaginary components. | Reject complex counts and scores before conversion, including object arrays containing complex scalars. |

For the threshold defect, descending scores `8/9, 7/9, ..., 1/9` with labels `[0, 0, 1, 1, 1, 0, 0, 1]` give F1 = 2/3 at both `4/9` and `1/9`. The old implementation chose `1/9`; the corrected implementation selects `4/9`, as documented. No TEST labels enter this choice.

## Verified behavior

- Snapshot keys, labels, feature order and timestep order remain bound to tensor rows. Missing, duplicate or inconsistent snapshot-month records fail validation.
- Splits use patients, including patients with both response labels. Existing frozen assignments are reused and checked against the complete input population.
- Positive class weighting uses TRAIN only. Per-batch `log1p` fits no statistics on held-out data.
- AdamW receives gradients from backpropagation, with clipping and nonfinite-gradient checks. Regression tests verify that projection, temporal embedding, attention, feedforward, normalization and classifier parameters change during training.
- Validation and TEST scoring preserve model weights. Altering TEST labels does not alter learned weights or the validation-selected threshold.
- Early stopping uses validation average precision. Checkpointing preserves the best observed AP even when its improvement is below the patience `min_delta`; equal AP retains the earliest epoch.
- Evaluation verifies the bundle, feature/time order and frozen manifest against checkpoint metadata. It applies the saved validation threshold unchanged.
- Deciles rank scores with a label-independent tie break. Cumulative boundaries are `ceil(d*N/10)`; recall uses TEST positives, and lift uses TEST prevalence. Top-10/20/30% results agree with the corresponding cumulative deciles. Zero-positive cases retain undefined recall/lift.
- Reports contain aggregate tables and charts. The source notebooks retain empty outputs.

## Validation and execution

The unmodified baseline passed all 107 existing tests. Updated test and notebook-execution results are recorded in `VALIDATION.md`.

Run the updated tests with `python -m pytest -q`. The reusable `scripts/smoke_notebooks.py` executes the notebooks in fresh kernels with synthetic data, including a second execution of notebook 01 to check saved bundle, split and audit reuse. It produces local evidence without populating the delivered notebooks with results.

## Remaining project checks

The original `tensor_initialization.ipynb` and private work-laptop inputs were unavailable. Consequently, this review does not establish the actual upstream feature order, storage schema, snapshot-date semantics, predictor cutoff, outcome exclusion or observability rules. The existing source-review flags correctly remain false in the example configuration until those checks are completed against the authoritative source.

The default contract remains `(23151, 12, 1028)`, with 12,447 patients and 1,345 positive snapshots. Tests use small synthetic fixtures; they do not validate those private population totals or provide clinical performance estimates. Real training, GPU/Databricks execution and held-out performance remain unverified. Class-weighted scores are not established calibrated probabilities, and no superiority claim over LightGBM is supported.

The fixes do not rewrite an existing checkpoint or frozen split. Already completed runs retain their saved threshold; any corrected threshold must be recomputed from that checkpoint's VALIDATION scores, recorded as a separate reviewed artifact, and never selected from TEST performance.
