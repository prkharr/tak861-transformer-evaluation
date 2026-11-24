# Local validation — 22 September 2026

27 tests passed using Python 3.9 and PyTorch 2.8 on CPU.

Checks cover:

- Original saved feature-map ordering, population counts, snapshot labels,
  patient separation and input/split fingerprints against RUN_001.
- Zero-activity sequence retention and rejection of changed source inputs.
- Reproducible internal fitting/stopping/ranking groups containing TRAIN patients
  only, with no patient overlap between roles.
- One latest snapshot per ranking patient; exact-date donor groups and rejection
  of groups with no alternative donor.
- Whole 12-month feature sequences move together. Other features and the source
  array stay unchanged; no original validation/test values are read by ranking.
- A known synthetic signal ranks above irrelevant features.
- Selected indices preserve the original feature order and all 12 timesteps.
- Temporal training, train/validation diagnostics, checkpoint serialization,
  reload and contract mismatch rejection.
- Training and evaluation notebook cells execute in separate Python namespaces
  against an in-memory synthetic artifact store with shortened training settings.
  This exercises both fitting stages, permutation chunks, ranking persistence,
  selected model inference, metrics, plots and saved predictions.
- All four notebooks compile and contain embedded helpers. Packaging verification
  checks notebook schemas, exact filenames, absence of outputs/credentials, and
  byte-identical repository and ZIP copies.

No private data was queried or trained locally. Spark/Snowflake integration and
the real-data ranking remain to run on approved compute. Synthetic tests cannot
establish real-data lift improvement or historical feature/label correctness.
Matplotlib dependency deprecation warnings did not prevent test completion.
