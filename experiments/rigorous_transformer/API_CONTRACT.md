# Five-notebook experiment API

Implemented modules are embedded in the delivered notebooks. This contract
describes the 1,028-count experiment, the separately identified encoded-49
reference, and their artifact/evaluation interfaces. It does not certify
historical source lineage or report measured clinical performance.

## Count-feature selection

- `validate_count_inputs(X, metadata, feature_map, valid=None, expected_features=1028, expected_steps=12)` checks the frozen identity/split contract and raw nonnegative integer counts. NaN means missing; zero is activity absence, never an inferred padding mask. `valid=None` retains all twelve positions. It cannot verify upstream provenance.
- `fit_count_preprocessor(X, train_rows, patient_ids, valid=None, config=None)` fits quality checks and count medians exclusively on supplied fitting rows. Default `scaling="log1p"` does not center or divide by variance. `scaling="floored_standard"` is an ablation with a TRAIN-fitted scale floored at 1, so rare features cannot be amplified by tiny standard deviations. Medians replace only genuine missing values. `transform_counts(X, state, valid=None, rows=None, feature_indices=None, out=None)` applies the frozen state in row batches; an optional caller-owned memory map avoids another full tensor in RAM.
- `select_ranked_features(X, metadata, feature_map, valid=None, config=None)` reads only TRAIN values/labels for selection. Three patient-disjoint inner folds refit quality, preprocessing and ranking. Patient inverse-snapshot weights give each patient equal total weight inside screening; this does not dictate final Transformer weighting. Four quarter sums and cumulative newest 1/3/6/12-position sums preserve coarse sequence shape. All summaries receive log1p. A fixed L2 logistic model supplies a transparent screening floor. Its linear inductive bias can miss interactions; neither the rank nor its shortlist establishes an optimal nonlinear feature set.
- `FeatureSelectionResult` returns `selected_indices` (proxy one-SE), `eligible_indices`, `candidate_indices` (all in original FEATURE_MAP order), `candidate_tags`, separate `ranking`, `quality`, `cv_scores`, `cv_summary`, `fold_rankings`, `fold_audit`, and fitted `preprocessor`. `chosen_candidate` is the inner-CV resource-grid tag for the smallest candidate inside one standard error of the best mean AP. The shortlist contains proxy-best, proxy-one-SE, the next smaller resource-grid set where available, and all-eligible. Exact equal index arrays are deduplicated; `candidate_tags` maps all conceptual roles to the retained candidate key. The literal `all_eligible` candidate always remains. This is a proxy shortlist, not the optimal Transformer feature count.
- Final training compares every distinct shortlisted count for both families under their declared base settings in phase A. Phase B applies the registered ablations to each family's own winning count. The final feature set is selected from the complete registered search using outer VALIDATION, not by the proxy alone. Feature ranking never forces learned importance to match marginal association.
- `feature_diagnostics` provides TRAIN-only redundancy/effective-rank summaries. `shift_summary` compares TRAIN and VALIDATION only; it refuses TEST and does not affect selection. TEST diagnostics belong after configuration lock.
- Preprocessing state is plain JSON-compatible metadata/vectors (`to_dict` / `from_dict`). Result tables are ordinary DataFrames. Tensor arrays can be saved using NPZ with `allow_pickle=False`; no executable model pickle is required by this module.
- Row batching limits temporary tensor copies. Fold screening matrices contain eight log1p count summaries per retained category, not another 12-step tensor. The newest-three sum intentionally duplicates the newest quarter for a stable explicit summary contract. No missing upstream provenance is inferred from successful validation.

Ranking uses the maximum across these summaries and a recent-minus-oldest block
contrast of weighted absolute Spearman association and corrected binned mutual
information. Quantile bins prevent large count values becoming unique categories.
The patient-count MI bias adjustment is a descriptive safeguard, not a p-value or
proof of statistical significance. Fold eligibility and rank dispersion measure
stability; low support is reported without automatically deleting rare features.
Exact constants, all-missing columns and exact duplicate sequences are the only
automatic quality removals. Constant observed values with varying missingness are
retained and flagged, although median imputation can eliminate that signal; a
missing-indicator/group ablation must be separately declared if required.

`validate_count_inputs(..., value_rows=...)` can limit value inspection. Selection
passes TRAIN rows automatically. Unspecified value_rows audits the entire supplied
tensor, so call that only at preparation or after the permitted evaluation lock.
Metadata identity checks cover all splits without fitting a statistic.

## Extraction and safe serialization example

```python
import json
import numpy as np
from experiments.rigorous_transformer.features import (
    select_ranked_features, CountPreprocessor, transform_counts,
)

# X follows FEATURE_MAP exactly; metadata follows X row order and frozen SPLIT.
result = select_ranked_features(X, metadata, feature_map, valid=None)
state_json = json.dumps(result.preprocessor.to_dict(), allow_nan=False)
restored = CountPreprocessor.from_dict(json.loads(state_json))
candidate_json = json.dumps({k: v.tolist() for k, v in result.candidate_indices.items()})
roles_json = json.dumps(result.candidate_tags)

# Select only the rows needed for this stage; TEST stays deferred until locked.
train_rows = np.flatnonzero(metadata.SPLIT.to_numpy() == "train")
key = result.candidate_tags["proxy_one_se"]
original_indices = result.candidate_indices[key]
train_X = transform_counts(X, restored, rows=train_rows,
                           feature_indices=original_indices, valid=None)
np.savez_compressed("candidate_train.npz", X=train_X,
                    original_indices=original_indices)
# Reload with np.load("candidate_train.npz", allow_pickle=False).
# Save result.ranking / quality / cv_summary / cv_scores / fold_rankings /
# fold_audit as CSV or JSON. Do not rename the original feature vocabulary.
```

Status: implemented and tested synthetically; no project model performance was
measured by these tests. Production defaults are 1,028 features and 12 steps;
explicit smaller dimensions support synthetic verification only.

## Registered two-phase search: protocol version 3

`workflow.build_protocol(candidate_indices, candidate_tags, input_hashes,
implementation_hash)` returns an immutable **DESIGN**, with `search_stage="design"`.
It records all count subsets, candidate templates, three seeds (42, 142, 242),
budgets, the count-anchor rule, and bounded reliance settings before fitting.
The caller saves `protocol.json` and `runtime.json` in `<prefix>_PROTOCOL`.
The runtime must bind the warehouse helpers and training helpers as the
delivered notebook bootstrap does.

- `phase_a` contains every shortlisted count with a fixed base configuration
  within each family. Both families receive the same count/seed budget.
- `phase_b` contains the architecture/regularization templates. Their `columns`
  are `None` and `subset` is `__PHASE_A_FAMILY_ANCHOR__` until resolution. These
  templates must never be passed directly to fitting or a throughput pilot.
  Select an explicit phase-A candidate for the TRAIN-only pilot.
- Each family chooses its own count anchor using the highest mean VALIDATION
  AP across **every** prespecified seed. Exact ties prefer fewer features, then
  lower mean model complexity, then candidate name. No confidence-band or
  non-significance equivalence rule selects the anchor.
- The `without_step0` sensitivity retains all eligible features, independently
  of either family's winning count. Finalists are later selected across all
  completed phase-A and phase-B candidates, not necessarily the anchor itself.

`execute_two_phase_search(data, design, run_prefix)` returns
`(resolved_protocol, results)`. It first verifies the saved DESIGN, completes or
resumes phase A, computes its anchors and persists `<prefix>_ANCHORS/resolution.json`
**before any phase-B fit**. The resolution records all count-arm means, exact
seed metrics, chosen feature identities, and hashes of phase-A summaries,
models and validation predictions. `resolved_protocol` contains concrete
phase-B columns, `design_protocol`, `design_sha256`, `resolution_sha256` and
`count_anchor_resolution`. The input DESIGN is never mutated.

Later notebooks load the original DESIGN and call
`read_resolved_protocol(design, run_prefix)`. This reloads complete phase-A
artifacts, recomputes the anchors and rejects any disagreement with the saved
resolution. `read_search(resolved_protocol, run_prefix)` repeats the resolution
verification and loads all candidate/seed runs. Version-3 calls to the legacy
`execute_search` or attempts to read an unresolved search fail explicitly.

```python
design = wf.build_protocol(candidate_indices, candidate_tags, input_hashes,
                           implementation_hash)
wh.save_artifacts(run_prefix + "_PROTOCOL", {
    "protocol.json": wf.json_bytes(design),
    "runtime.json": wf.json_bytes(wf.runtime_versions()),
})
protocol, results = wf.execute_two_phase_search(data, design, run_prefix)

# In subsequent notebooks: read the saved DESIGN, then resolve and verify it.
protocol = wf.read_resolved_protocol(design, run_prefix)
results = wf.read_search(protocol, run_prefix)
```

Artifact candidate indices use the same sorted full-candidate order in both
phases. Phase-A contracts reference the DESIGN hash; phase-B contracts also
reference the resolution hash and resolved candidate configuration. Interrupted
phase B reuses verified phase-A fits. Changed inputs, code, settings, seeds,
anchors or budgets require a new run identifier; failed or missing runs cannot
silently shrink the search. This adaptive reuse of VALIDATION is declared
development work, not independent confirmation of performance.

## Run artifacts, restored predictors and final selection

Training returns a dictionary containing `summary`, `history`, `model_blob`,
`train_scores` and `validation_scores`. Summary hashes identify the model,
resolved configuration and fitted transform; `raw_input_shape` records the
private model input contract. `workflow.pack_run(result, contract)` produces
the six-artifact bundle enumerated by `RUN_ARTIFACT_NAMES`. Its version-2
envelope hashes every payload. `unpack_run(..., expected_lengths=None,
expected_config=None)` verifies the envelope, exact contract, completeness,
seed, internal hashes, input shape, nonempty history and finite aligned
probabilities. Production contracts carry TRAIN/VALIDATION prediction counts.
The optional arguments add explicit caller checks; omitting them does not
remove the counts already declared in the contract.

`training.make_saved_predictor(result, device="cpu")` verifies and restores one
saved model **once**, freezes copied configuration/transform metadata and
returns `predict(raw_selected_counts)`. Reuse the callable for repeated
validation perturbations; it does not refit preprocessing or reload the
checkpoint for every probe. Raw input shape and the configured representation,
including step-0 removal, remain enforced. Transformer weights load with
`weights_only=True`; count LightGBM uses its text model. `predict_saved` is the
one-shot convenience wrapper. The calling notebook keeps TEST behind its
persisted decision lock; these numerical predictors cannot infer split identity
from an array alone.

`training.choose_finalists(results, expected_candidates, seeds)` requires the
complete registered candidate and seed set. Mean seed VALIDATION AP chooses
each family finalist; exact-score ties prefer fewer features, lower complexity
and candidate name. `workflow.ensemble_scores(runs, split)` uniformly averages
complete aligned development predictions and rejects duplicate seeds. No seed
is selected by its outcome. The final TEST predictor averages the same frozen
three seed models, after the lock.

The encoded reference has a separate interface in `baseline49.py`:
`load_encoded_baseline(read_table, frozen_metadata)` preserves MODEL_TYPE order,
validates exact frozen keys/RESP, records the FINAL_MODEL.SEQ discrepancy, and
returns `X`, `features`, `feature_sha256`, `input_sha256`, `order_audit` and
`provenance`. Its input digest includes ordered values and metadata, so changing
values without changing the feature list is detected across notebooks.
`train_encoded_baseline`/`predict_encoded` preserve encoded fractions and AGE
and use a fixed LightGBM context-refit configuration. Their run bundles conform
to the same artifact envelope but have a one-dimensional raw input shape.
Original V63 score recovery remains separate, VALIDATION-only by default and
always explicit about unknown historical fitting scope.

## Ranking versus internal diagnostics and bounded reliance

`reliance.join_rank_and_internal_diagnostics(feature_map, train_ranking, runs,
expected_columns, expected_seeds, top_k=10)` verifies original index/name/order,
complete seed identities, model/config hashes and temporal-block attribution.
It returns `full_table` (one row per original feature), `seed_table`,
`agreement_table` and `summary`. Selected model positions stay keyed to original
FEATURE_MAP indices. Each seed's Transformer projection-column norm or
LightGBM temporal-block gain is joined to TRAIN rank/stability. Projection norm
times transformed TRAIN standard deviation is included when available; none
of these internal quantities is itself measured predictive reliance.

`prespecify_probes(joined, max_features=30, random_seed=20261006,
top_train=10, top_internal=10, lowest_train=5, random_remaining=5,
correlation_pairs=None, correlation_threshold=.9, max_correlation_groups=10)`
returns a hashed plan. Its individual probes are the union of those declared
quotas; duplicates reduce the count rather than filling it adaptively. Joint
probes cover selected DX/PX/RX families, selected-feature TRAIN-rank terciles,
and up to ten largest connected TRAIN correlation components. Every included
and omitted component is recorded. Correlation-pair provenance remains a caller
contract; the function cannot prove a matrix came from TRAIN by its values.

`validation_permutation_diagnostic(X, metadata, feature_names, predictor, plan,
lock_hash, *, split="validation", repeat_seeds=(271,811,1297),
all_validation_metadata=None)` requires
raw 12-position counts and one latest VALIDATION snapshot per patient. The
delivered caller supplies the full VALIDATION metadata so latest-row selection
is verified. TEST rows are rejected. Whole feature trajectories are permuted
within END_DT calendar month; members of a joint group move together. The
frozen predictor receives no changed labels, preprocessing state or model
parameters. One baseline prediction is reused, and the same seeded patient
permutations are shared across probes.

The result contains:

- `probe_summary`: mean AP/lift@10% drops, repeat SD, prediction changes and
  fractions of patients whose values actually changed;
- `repeats`: all declared repeat results, never the favorable subset;
- `coverage`: each selected feature's individual-test status and joint-group
  membership; omitted individual effects are `NOT_TESTED_UNKNOWN`;
- `protocol`: lock/plan hashes, exact seeds, population/strata, predictor-call
  count, no-fitting/no-TEST declarations and interpretation limits.

Notebook 4 saves the plan and result separately from the model lock and reuses
unchanged cached diagnostics. At the declared maximum there are 46 probes and
three repeats: 139 ensemble calls per family, plus the notebook's checkpoint
consistency check. No runtime duration is guaranteed. The full repeated-snapshot
population and all 1,028 individual effects are **not** measured by this bounded
latest-patient diagnostic. Repeat SD quantifies permutation randomness, not
patient-sampling uncertainty. Marginal permutations can break correlations;
joint groups preserve only their within-group dependence. These are conditional
reliance findings, not causal effects or proof that a feature group adds signal
after refitting. They must not alter the locked model or feature selection.

## Final comparison and claim boundaries

`comparison.binary_metrics` reports AP separately from trapezoidal PR area,
ROC-AUC, Brier/log loss, fixed-bin calibration, confusion-derived metrics and
deterministic top-K quantities. `select_validation_threshold` returns a hashed
F1 policy selected only on VALIDATION. Operational top-K locks a population
fraction and tie rule rather than optimizing a TEST cutoff.
`paired_patient_bootstrap` resamples patients with both models paired and all
their snapshots retained. Its intervals condition on the fitted predictions;
training, feature-selection and adaptive-search uncertainty are not included.
`comparison_conclusion` restricts claims for unknown historical fitting scope,
unmatched populations, prior holdout inspection or unresolved source timing.
An interval containing zero is not evidence of equivalence. Different winning
feature sets or representations compare pipelines, not architecture alone.

The existing TEST remains retrospective and no complete certified OOT cohort
is available. Successful software tests, source hash checks and saved-input
contracts do not resolve that limitation or create historical performance.
