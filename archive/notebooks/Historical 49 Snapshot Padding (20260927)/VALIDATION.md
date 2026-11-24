# Validation of the September 27, 2026 snapshot-padding rebuild

The four notebooks passed static syntax checks, documentation checks and an offline synthetic integration run before publication. These checks do not report performance on the client population.

## Completed checks

- All 60 code cells parse. Every code cell is immediately preceded by Markdown documentation. Outputs and execution counts are cleared.
- Eight independent semantic checks passed: equal snapshot weighting in preprocessing; strict metadata and duplicate rejection; exact split alignment and patient separation; snapshot-summary equality; monthly/quarterly calendar semantics; strict Boolean parsing; source alignment and malformed-value rejection; and coverage-key alignment.
- Notebook 01 passed ten preparation checks, including mocked warehouse reads, source validation, padding and artifact round trips.
- The four notebooks ran in separate fresh execution namespaces against 120 synthetic patients with 49 features, including three all-padded snapshots.
- The integration run executed 59 code cells and trained/restored 14 CPU checkpoints: two baselines plus six candidate configurations for each representation. The synthetic budget was two epochs with reduced model width; this is not a production training run.
- Thirteen cross-notebook checks passed, covering feature/encoding contracts, prepared artifact hashes, shared preprocessing, identical inputs, checkpoint save/restore, validation AP reproduction, feature influence/correlations, all candidate configurations, selected checkpoint restoration, evaluation diagnostics, aggregate exports and default TEST isolation.
- Four architecture SVGs were generated and exported. JSON/CSV/SVG exports and artifact pack/unpack logic were exercised using an in-memory warehouse stand-in.
- The actual evaluation control flow was separately checked with TEST disabled and enabled. The full integration run did not score TEST.

## Limits

No private warehouse/client data was used, no client model was trained locally, and no improvement on the client cohort is claimed. The integration run used a warehouse stand-in and reduced training settings. Matplotlib chart rendering was stubbed, and one chart-only cell was excluded from numeric integration. Snowflake SQL execution, private source schemas/data, real artifact permissions, full-budget training and Matplotlib rendering still require verification in the authorized runtime.

Source alignment, masking and hash checks validate the stated snapshot-reuse contract. They do not demonstrate historical feature reconstruction, upstream absence of future information, causal feature effects or an independently confirmed overfitting fix.
