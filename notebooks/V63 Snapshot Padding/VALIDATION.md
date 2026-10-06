# Validation of the October 6, 2026 five-notebook revision

**107 tests and 70 subtests passed in 16.49 seconds, exit 0**, with:

```text
python -B -m pytest experiments/rigorous_transformer -q
```

These are synthetic software checks. No private project model performance was measured.

## Scope

The tests cover TRAIN-only preprocessing and patient-grouped feature screening, exact quality/removal rules, temporal summary ranking and stable ordering; sequence models, regularization gradients, deterministic seeds, validation stopping and complete-grid finalist selection; lineage corroboration limitations; patient-cluster uncertainty and metric correctness; encoded-49 input alignment; immutable artifact serialization, identity checks and changed-input rejection. Two-phase tests verify family-specific count anchors across all seeds, persistence before phase B, interruption/resume and changed-anchor rejection.

The integration test builds the five notebooks in a temporary directory and executes every generated code cell in fresh stage namespaces. It uses real feature selection, small two-epoch Transformer fits, reduced LightGBM fits across all three seeds, the encoded-49 baseline, checkpoint restoration, the real warehouse artifact chunk codec, locked thresholds, post-lock validation reliance, TEST predictions, final comparison and cached-result reuse. It checks altered models, policies and inputs are rejected. Preprocessing tamper cases update valid inner/outer hashes and are still rejected by the frozen per-seed predictor identities. A cached final comparison with a different reliance report is rejected. Matplotlib uses a noninteractive backend; plots execute, while rich frontend display is suppressed.

Reliance tests exercise non-contiguous original feature identities, representation-specific LightGBM gain aggregation, per-seed ranks and scale-aware norms, fixed-seed within-month permutation of whole trajectories, joint correlated groups, unchanged labels/other columns/decision locks, latest-patient selection, TEST refusal, probe coverage and explicit unknowns for unmeasured effects. Screening rank, parameter magnitude and predictive reliance are separate diagnostics; equality is not required. The LightGBM reference cross-check matches feature IDs and records any temporal-information difference.

Only private warehouse access, population dimensions and training budgets are replaced. Synthetic inputs include patient repetition, exact duplicate/constant features, missing values and a deliberately reversed historical feature-order report. These cases exercise failure paths without claiming they describe the project data.

The integration test is in `experiments/rigorous_transformer/test_notebook_flow.py`; the builder and independently testable embedded sources are in the same folder. The standalone notebooks require none of these source files at runtime.

## Reproducibility

All five generated notebooks carry the same implementation SHA256:

```text
48f7a1e0b10113320b4be9cd3a52039aec5888ba52c2c6d54a479373b56641a3
```

Observed local numerical packages for this verification: Python 3.12.14, PyTorch 2.8.0+cpu, NumPy 2.5.3, pandas 3.0.1, scikit-learn 1.9.1 and LightGBM 4.6.0. Actual authorized-runtime versions are saved with the experiment. Local package versions are a test record, not an instruction to upgrade the work environment.

All 32 code cells passed AST parsing and notebook-format validation, have preceding Markdown, and have cleared execution counts/outputs. Embedded helpers match the current source files. The earlier September 27 four notebooks and documentation are preserved in `notebooks/Historical 49 Snapshot Padding (20260927)/`; their older verification record applies only to that source. All six archived files match Git revision `38b02295a6b9cc7bd5a5293166b49d8776905bed` after normalizing Windows/Unix line endings; the original documentation is also byte-identical. The delivery includes `static_delivery_validation.json` with these checks and observed versions.

## Limits

The actual Snowflake reads/writes, private source contracts, production dimensions, complete search budget and client outcome metrics have not executed in this session. Synthetic tests cannot certify upstream cutoff correctness, historical model fitting, event availability, label maturity, clinical utility or an untouched OOT cohort. Successful artifact checks cannot retroactively prove those facts.

Actual Claude provided a substantive design review and subsequently inspected the predecessor implementation, confirming feature mapping and identifying the two-phase search and predictive-reliance corrections. Both were implemented and tested. The actual review artifacts and accepted/qualified recommendations are preserved in the repository. A final verification request could not be submitted after Windows activation and fresh-window recovery failed with `GetCursorPos / Access denied (0x80070005)`; no final Claude approval of this revision is claimed. Independent Codex review added the lock/cache protections above and found no remaining material integration issue within its reviewed scope. This is not equivalent to executing on the client data.
