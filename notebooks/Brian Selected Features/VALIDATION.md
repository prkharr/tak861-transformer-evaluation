# Validation — 22 September 2026

## Completed locally

- 11 focused tests passed using Python 3.9, PyTorch 2.8 and scikit-learn 1.6.
- Exact list parsing/order and leading underscores/quoted names checked.
- Fractional Decimal values retained; missing feature values remain missing until
  TRAIN-only preprocessing. Duplicate/missing source keys and conflicting labels fail.
- Frozen patient split fingerprints checked, including rejection of changed assignments.
- TRAIN-only medians/scaling, all-missing columns and input fingerprints checked.
- Lift calculations checked for unequal decile sizes and tied scores; tie order is
  deterministic and independent of the outcome.
- Transformer, logistic and histogram-gradient-boosting fit/predict/serialization
  round trips passed on synthetic data.
- All four notebook code paths ran in separate Python namespaces against an
  in-memory synthetic warehouse. The synthetic population has the expected
  23,151 snapshots / 12,447 patients / 1,345 positives and original split sizes,
  but contains invented identifiers and generated feature values, not private data.
- That test exercised all six candidates with shortened settings, authoritative
  list/summary disagreement, saved preprocessing, model reload, validation-only
  winner selection, training lift, test reporting, plots and artifact saving.
- Warehouse artifact chunk round-trip and missing-chunk rejection checked.

The generated notebooks contain eight code cells each and no execution outputs.
Compilation and notebook-schema checks are performed during delivery. No local
helper files are required by the delivered notebooks.

## Not established by these checks

- No live Snowflake query, write, model training or evaluation was performed by Codex.
- Spark/Snowflake connector behavior, role permissions and runtime packages must
  be available on the user's approved compute. The synthetic warehouse exercises
  orchestration, not a live integration.
- No improvement in real-data lift, exact reproduction of Brian's LightGBM, source
  feature historical availability or label-construction correctness is claimed.
- Matplotlib's bundled pyparsing dependencies emitted deprecation warnings during
  plotting tests; these did not prevent chart generation or test completion.

The original 1,028-feature notebooks and prior Brian V2 artifacts were not replaced.
This delivery uses its own selected-feature directory and warehouse namespace.
