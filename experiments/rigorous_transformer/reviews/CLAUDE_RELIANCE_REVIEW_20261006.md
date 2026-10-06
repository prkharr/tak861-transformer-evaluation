# SME review — reliance-vs-ranking consistency and final pre-push checks (2026-10-06)

Reviewer: Claude (independent SME). Read-only inspection of the actual files: `experiments/rigorous_transformer/{features,training,workflow,comparison,lineage,build_notebooks}.py` (plus `baseline49.py` signatures), `API_CONTRACT.md`, `REVIEW_DECISIONS.md`, and the five generated notebooks under `notebooks/V63 Snapshot Padding/`. The implementation hash `a6a5caff3fb78f41cd4a2eef865143ee89be45e6a62cffbe99604e0573517fe6` reproduces from the on-disk module sources plus `build_notebooks.py`, and every notebook's metadata carries it. No code was edited; no model reliance values exist yet and none are asserted here.

## 1. Feature mapping — verified correct

- `select_ranked_features` returns every candidate as `np.sort(original FEATURE_INDEX)`; `build_protocol` stores `columns = sorted(indices)`; `candidate_views` slices `X[rows][:, :, columns]`, so model input position `j` ↔ original index `columns[j]` ↔ `FEATURE_MAP.FEATURE_NAME[columns[j]]` (features list is built from `FEATURE_MAP` ordered by `FEATURE_INDEX` in `load_count_inputs`).
- `train_candidate` returns `importance = projection.weight.norm(dim=0)` (length = input width, position order). For LightGBM, `flattened` is `x.reshape(N, 12·F)` with feature fastest, so `importance.reshape(12, F).sum(0)` attributes gain to the right original position; `windows` (4 blocks) and `annual` (1) reshape the same way.
- Notebook 04 builds `selected_orders[family] = [features[i] for i in protocol candidates[name]["columns"]]` and pairs it with the seed-mean `importance`; a length mismatch would raise in the DataFrame constructor. `read_search`/`unpack_run` verify the saved config `columns` against the protocol, and `predict_saved` verifies model/config/transform hashes and raw input shape before inference.

Verdict: the position ↔ original-index ↔ name mapping is correct and hash-guarded. Independent Codex finding confirmed.

## 2. Leak review of the search, lock and TEST path — no leak found

TRAIN-only: proxy folds, preprocessing medians/scales, class weight, ranking, redundancy diagnostics, lineage pairs and step-0 screen. VALIDATION use is limited to early stopping, plateau scheduler, finalist choice and threshold policy, all disclosed. TEST rows are never passed to `train_candidate` (no argument exists); Notebook 05 reads the lock, verifies protocol/model/baseline hashes, writes an intent artifact, then scores TEST once and persists; reruns reuse the saved final artifact. `shift_summary` refuses TEST. `feature_diagnostics` is TRAIN latest-snapshot only. `drop_step0` is applied identically in `candidate_views` and `_configured_representation` (idempotent).

## 3. The proposed reliance diagnostic — assessment

The current Notebook 04 table (`mean_projection_norm`) is a parameter-magnitude diagnostic, and under the default `scale=False` (log1p only, no standardization) it is confounded by input scale: a frequent high-count feature and a rare 0/1 feature have different input ranges, so projection norms are not comparable across features. It must not be read as reliance. The proposed addition is the right correction, with these refinements:

- **Scope all selected features, not a prespecified top-N.** Cost is small: for the all-eligible finalist, 1,028 features × 3 permutation seeds × 3 ensemble members × 3,481 VALIDATION rows is a few minutes on CPU. Bound only the displayed table. If a cap is kept, it must be registered before computing and be the union of top-N by screening rank and top-N by internal magnitude, so neither view is privileged.
- **Permutation unit and manifold.** Permute the whole 12-step raw-count trajectory across VALIDATION rows (before `numeric_transform`, so per-feature marginals stay on-manifold), with fixed permutation seeds and the locked uniform ensemble as the predictor (`predict_saved` path, never a re-instantiated model). Prefer permuting within END_DT calendar-month strata: it is the same code path with a group loop, preserves the calendar distribution of each feature, and reduces off-manifold combinations. Report AP drop (primary), lift@10 % drop (secondary), per-permutation-seed SD, and the fraction of rows actually changed (sparse features move few rows).
- **Groups, TRAIN-defined only.** (a) RX/DX/PX families; (b) redundancy clusters = connected components of `feature_diagnostics` pairs at |Spearman| ≥ 0.9 (TRAIN latest snapshot); (c) screening terciles by TRAIN composite rank, permuted jointly — the bottom tercile jointly permuted answers the original "do weak features matter as a group" requirement without any refit. Joint permutation uses one row permutation for all members so within-group structure is preserved.
- **Patient-valid secondary.** One latest VALIDATION snapshot per patient (`cmp.latest_patient_indices`): 1,867 rows, all 202 positives retained (positives are last snapshots), donors from other patients. Report as secondary; the full snapshot-level run is primary because it matches the scoring unit.
- **Comparison with ranking, descriptive only.** Keyed audit per selected feature: original index, name, model position, TRAIN composite rank and fold stability, per-seed projection norm rank, LightGBM gain rank from the twin `gb_` arm on the same subset (no new fits), permutation AP-drop rank. Report Spearman rank correlations and top-25/top-50 overlaps, and quadrant flags (high reliance/low screening; high screening/low reliance). Flags are observations about proxies, interactions, redundancy or instability — not errors, never an input to selection.
- **Immutability.** Write the audit to a separate artifact table keyed to the lock hash (`_RELIANCE`), after `_LOCK` is saved and before any TEST inference; the diagnostic must not touch `lock.json` or the finalist list. Declare its settings (seeds, strata, groups, thresholds) in a `RELIANCE_PROTOCOL` dict whose hash is stored with the output.

Stronger feasible checks, in priority order: (1) the LightGBM-gain cross-check above (free); (2) scale-aware internal magnitude = projection norm × TRAIN standard deviation of the transformed input (free, comparable across features); (3) bounded leave-group-out refits of the locked Transformer configuration for the three terciles and three families (6 groups × 3 seeds, VALIDATION early stopping only, post-lock, pre-TEST) — this measures information content rather than fixed-model reliance and is the only check that can say a weak group "adds" signal; optional if budget is short. Gradient-based attributions add little beyond these and are not needed.

## 4. Material blockers before push

B1. **Ablation arms anchor on the proxy one-SE subset only** (`build_protocol`: `default_subset = candidate_tags["proxy_one_se"]`). Capacity, dropout, decay, class weight, L1/L2, pooling, scaling, quarterly and patient-weight arms are all run on that subset; the other count arms get only the base configuration. If `all_eligible` or `proxy_best` wins the count comparison, the final Transformer is untuned on its own feature set, and the project's "strongest defensible model" claim is weakened. Fix without post-hoc freedom: a two-phase registered protocol — phase A = count arms (base config, 3 seeds); phase B = ablation arms anchored on the count arm with the highest mean VALIDATION AP across seeds, with the anchor rule declared now and the resolved anchor, its hash and phase-A results recorded in the protocol artifact before phase B fits. Both families follow the same rule. If the user prefers to keep the single-phase protocol, record the limitation explicitly in Notebook 04/05 ("ablations tuned on the proxy subset only").

B2. **Reliance diagnostic as currently delivered is not a reliance diagnostic** (projection norm, scale-confounded) and no screening-rank join exists. Implement §3 (keyed audit + post-lock VALIDATION permutation reliance + groups + LightGBM cross-check) so the single authorized run yields the "reliance consistent with ranking" answer; relabel the projection-norm table as "parameter magnitude (scale-confounded)" wherever it remains.

## 5. Required correctness tests for the addition (offline, synthetic)

1. Mapping: FEATURE_MAP with distinctive names, a subset with non-contiguous original indices, and a synthetic signal only at original index `s`; after a short fit, the largest permutation AP drop is at the model position whose audit row names index `s`, and `selected_orders[j] == FEATURE_MAP.FEATURE_NAME[columns[j]]` for all `j`.
2. Permutation integrity: sorted per-feature values unchanged before/after; the 12 values of a row move together; other features and labels untouched; stratified permutation only moves rows within the same END_DT month; a function called with TEST rows raises.
3. Determinism: identical drops for identical seeds; different seeds give the reported SD.
4. Group permutation: two perfectly correlated synthetic features remain equal row-wise after joint permutation; family and tercile membership derived from TRAIN tables only.
5. Rank diagnostics: Spearman and top-K overlap match hand-computed values; flags fire only beyond the declared thresholds; no selection object is modified (lock bytes hash-equal before and after).
6. LightGBM cross-check: in the flattened representation, gain attributed to the signal feature lands on the correct original index after `reshape(12, F).sum(0)`.
7. Patient-valid secondary: exactly one row per patient, all positives retained, no TEST rows.
8. Protocol: `RELIANCE_PROTOCOL` hash stored with the output; rerun with an unchanged protocol reuses the saved artifact; a changed protocol requires a new identifier.

## 6. Non-blocking observations

- `cpu_threads=8` while the runtime reports 16 CPUs; raise to 16 for both torch and LightGBM.
- `overlap_month_consistency` deliberately excludes the boundary cells, so it tests orientation only; the pair data already loaded could also report, descriptively, the later-snapshot step-`delta` versus earlier-snapshot step-0 agreement/≥ frequency on active cells (the capping signature), labeled corroboration only.
- `same_feature_information` in Notebook 05 should also require equal `drop_step0` on both winners rather than forbidding it on LightGBM; otherwise a without-step-0 pair is mislabeled "different information".
- Add an explicit assertion with a message that `len(importance) == len(columns)` in Notebook 04 (currently an implicit pandas error).
- Fit budget: ≈ 96 fits (16 arms × 2 families × 3 seeds); LightGBM all-eligible flattened arms use 12,336 columns — the TRAIN-only throughput cell in Notebook 03 should print projected totals before the grid starts so the session budget is visible.

## 7. Verdict

Mapping correct, no leak found, lock and TEST discipline sound, implementation hash reproducible. Do not push with the projection-norm table as the reliance evidence. Implement B2 (and B1, or record its limitation explicitly), add the §5 tests, then push and run.
