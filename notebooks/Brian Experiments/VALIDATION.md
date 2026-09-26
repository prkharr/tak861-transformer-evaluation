# Verification of this notebook delivery

Date: 22 September 2026.

**105 local tests passed.** Tests used synthetic inputs and an in-memory substitute for Snowflake transport. These are software checks, not new client-model results.

Coverage:

- All 32 notebook code cells compile; notebooks contain no saved outputs or credentials.
- Original tensor and split audits are checked against RUN_001 fingerprints, including changes that preserve aggregate counts.
- Reduced-feature selection reads TRAIN only, counts presence by unique TRAIN patient, retains broad timing-only signals and preserves exact selected-column order.
- Transformer fitting and logistic scaler fitting do not read TEST values or labels.
- Saved numeric checkpoints reproduce predictions and reject changed inputs, feature manifests, model settings and inconsistent training summaries.
- Actual training/evaluation notebook cells execute all six candidate recipes on small synthetic data, save each completed candidate, resume without refitting and select the validation-AP winner.
- Only the selected run is evaluated; evaluation uses its saved features and threshold, and exports six charts plus five aggregate data/metadata artifacts.
- Duplicate TEST evaluation is refused. Successful completion and tested provenance failures release temporary tensor storage.
- Synthetic examples of the new comparison/history charts were visually inspected.

Test groups: 36 feature-selection tests, 27 training/runtime tests, 20 early-stage audit tests, and 22 notebook integration tests.

The local runtime was Python 3.9.7, PyTorch 2.8.0 CPU, scikit-learn 1.6.1. The only test warnings were third-party matplotlib/pyparsing deprecation warnings.

Not performed locally: live Spark/Snowflake connector execution, training on the private client data, rebuilding raw-claims features, verifying the original RESP label construction against the stated 90-day horizon, or producing a fresh holdout evaluation. The client's engineered core-feature definitions remain pending.
