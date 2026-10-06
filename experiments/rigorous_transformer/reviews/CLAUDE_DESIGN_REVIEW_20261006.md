# SME design review — final five-notebook experiment (2026-10-06)

Reviewer: Claude (independent SME/overseer). Scope: the eight-area design proposal for the 1,028-primary experiment plus the secondary 49-feature experiment and the comparison notebook. Evidence basis: `project_context/evidence/recovered_outputs/collector_run_a8a58e97fe6f_20261006.md` (collector run `TAK861_EVIDENCE_a8a58e97fe6f`), `features/feature_ranking.md`, `session_memory/final_experiment_contract.md`, `CURRENT_STATE.md` (2026-10-06 checkpoint), and the previously inspected notebook code. No metric is produced here; nothing below is a measured result. Claim restrictions from the SME addendum stand: upstream selection/encoding provenance UNKNOWN; original TEST previously inspected; 1,028 cutoff/counting lineage unresolved; existing V63 fit scope UNKNOWN.

## Collector facts that change the design (verified in the collector receipt)

- Runtime: 16 CPUs, 113 GiB RAM, CPU-only torch, lightgbm 4.6.0 available. The 1,028 tensor (23,151×12×1,028 float32 ≈ 1.1 GB) fits in RAM; no memmap needed; set `torch.set_num_threads(16)`.
- Every END_DT falls on day 19–22 of its month (E04B). Whole-month aggregation, if that is what ran, would admit 8–12 post-cutoff days — the first days of the 90-day outcome window — for every snapshot. The cutoff question is therefore material, not academic.
- A positive snapshot is always the patient's last snapshot (positives before latest = 0). Median spacing between a patient's snapshots is 122 days (min 28); 10,704 adjacent pairs. Consecutive snapshots share most of their 12-month windows, so within-patient dependence is strong and the effective sample size is closer to patients (8,712 TRAIN, 941 positives) than to snapshots.
- `TAK861_TX_READY_V63_MAPPING` is readable but empty; `ENROLLMENT_20260825` and `HCP_LOOKUP` are UNAVAILABLE under the current binding; the DX map is unbound. An event-level recomputation of the 1,028 counts is not possible right now; the enrollment mask cannot be recomputed either, but saved INPUTS manifests (TEMPORAL_CLEAN_V1 D20260925A: 276,044 available / 1,768 unavailable / 37 all-padded monthly cells) show the mask is nearly all-valid.
- `UNIVERSE_W_FEATURES_SCORED` holds a SCORE in [0,1] for all 23,151 frozen snapshots across TRAIN/VALIDATION/TEST — the existing V63 model scored the whole cohort; fit provenance UNKNOWN. `BACK_TESTING_SCORED` is a later-period population (E11: 2025-06-21 to 2025-10-21, 12,972 patients, 4,704 outside the frozen cohort; E09 disagrees — under audit) with only 19 of the 49 feature columns and no 1,028 tensor: not usable as an OOT set for these models regardless of prior-use status.
- E06: MODEL_TYPE order ≠ FINAL_MODEL.SEQ for the same 49 names. Irrelevant to the 1,028 route; for the 49 route keep MODEL_TYPE order (what every existing notebook uses) and record the discrepancy.
- The 1,028 tensor contains no demographics: AGE, the top-reliance feature of the 49 models, is absent from Experiment B by construction.

## Area-by-area challenge

**1 — Input contract.** Agree: FEATURE_MAP index order is the contract (hash recorded), orientation declared `0 = newest`, provenance UNKNOWN. Challenge: declared orientation is harmless for learning (a mirrored encoding spans the same function class) but not for interpretation (recency ablations, step-0 reliance). Add the two tensor-only corroborations below (B3); they need only `TENSOR_MONTHLY` and `SNAPSHOTS`, both READ_OK.

**2 — Preprocessing.** Counts have no missingness; "imputation" is undefined for the 1,028 and must not be applied. `log1p` is right. Challenge: TRAIN-fitted per-feature standardization of sparse log1p counts divides rare features by tiny standard deviations and amplifies noise; make scaling an ablation (log1p only vs log1p + TRAIN standardization with a floored scale), default log1p only. Drop exact constants and exact duplicate columns (zero information); report near-constant/rare but keep them. The original contract has no enrollment mask; with ~0.6 % padded cells, run no-mask as default and mask-reuse (from a hash-verified saved INPUTS artifact) as a low-priority ablation — do not wait on `ENROLLMENT_20260825`.

**3 — Ranking and feature-count selection.** The annual-sum proxy with a one-SE rule is too indirect to *choose* the Transformer's feature count: (a) sums discard temporal shape; (b) inner-CV SE at ~190 positives per fold is ≈0.01–0.02 AP, so the one-SE band will often reach very small counts and over-prune; (c) the proxy evaluator's inductive bias (linear or tree) is transferred to the sequence model. Correction: the proxy produces the *ranking and a shortlist only*; the Transformer's own VALIDATION AP (3 seeds, paired patient-cluster bootstrap) selects among a predeclared count grid {all-eligible, proxy-best, proxy-one-SE, one step smaller}, with a parsimony rule (smallest count whose mean AP lies inside the bootstrap CI of the best). Make the proxy less indirect: evaluator = LightGBM (nonlinear, sparse-aware; available) on 3/6/12-month sums plus the step-0 value, patient-grouped 5-fold inside TRAIN with within-fold refit; report per-feature "recency share" (gain carried by the 3-month and step-0 columns) and rank stability across folds. Inverse-patient-snapshot weighting is acceptable inside the proxy; for training it is an ablation, not the default (see dependence below).

**4 — Architecture and ablations.** Agree with a small pre-norm encoder, month tokens, linear projection of the (selected) counts, fixed sinusoidal time encodings, mean pooling. Challenges: (i) the projection (k×d) dominates the parameter count for k = 1,028 — this is where L1 belongs: L1 on the input projection only (feature-sparsity), with per-feature projection norms reported as a model-internal relevance measure to compare against the proxy ranking; global L1 on attention/FF weights is a weaker, less interpretable lever. (ii) Bound the grid and order it by information value: capacity first (d 64/1 layer/FF 128 vs d 128/2/256), then feature count, then regularization/dropout/class weight, then pooling and scheduler; print per-fit wall-clock after a one-epoch dry run before launching. (iii) A single-token/flattened MLP control is unnecessary for the 1,028 route because LightGBM on windowed sums already serves as the "same information, no sequence model" comparator; keep the single-token control only for the 49 replay.

**5 — Loss and optimization.** Agree. Corrections: the class-weight ablation must include the historical full-ratio arm (`pos_weight` ≈ 16.28) alongside unweighted and √ratio (≈4), or the new runs cannot be placed in the chronology. Keep one L2 convention: either explicit penalty or decoupled AdamW decay, never both (the existing guard is right); run one matched-strength ablation explicit-L2 vs decoupled-decay so the user's "true L2" requirement is answered with evidence rather than convention. Validation-AP early stopping (patience 5) plus validation-based config selection double-uses VALIDATION; disclose it and report TEST once for the frozen finalists only.

**6 — Protocol integrity.** Agree. Additions: pre-register the protocol (grids, seeds, counts, selection rule, budgets) in a `PROTOCOL` dict whose SHA-256 is printed before the first fit and stored with every artifact; define comparison budgets in *validation evaluations × seeds*, not wall-clock, equal across model families; record the number of validation evaluations consumed per family in the final table. No OOT: `BACK_TESTING_SCORED` lacks 30 of the 49 columns and any 1,028 tensor, and its prior use is UNKNOWN.

**7 — Baselines.** Agree on same-population matched refits. Required set: LightGBM on windowed sums of all-eligible and of each selected count (same count grid as the Transformer), LightGBM on 12-month sums only (tests whether recency windows carry information), logistic floor on the same columns, and LightGBM-49 refit on the encoded V63 vector (cheap; the only fair reading of the existing feature set). Existing V63 SCORE from `UNIVERSE_W_FEATURES_SCORED`: descriptive AP/AUC/lift on VALIDATION and TEST partitions, labeled "scored the full cohort; fit provenance UNKNOWN; possibly in-sample".

**8 — Evaluation and claims.** Agree. Additions: thresholds locked on VALIDATION (F1 and top-10 % rule), recall/precision/lift@10 % with the deterministic tie-break, patient-level (latest snapshot) metrics as secondary, generalization gap per seed, calibration (Brier) optional. Superiority rule: paired cluster-bootstrap 95 % CI of ΔAP on VALIDATION excluding zero **and** same-sign retrospective TEST Δ **and** the provenance restriction stated; otherwise "not distinguishable at this sample size".

## Duplicated-snapshot dependence — how to handle it

Treat the patient as the independent unit everywhere that inference happens: patient-grouped inner CV, patient-cluster bootstrap for every CI, patient-disjoint frozen split (already). Train on all snapshots (more signal) but report snapshot-level metrics as primary (matches V63 decile scoring) and patient-level as secondary. Because a positive is always a last snapshot, inverse-count weighting shifts effective prevalence toward positives; run it as an ablation together with "one random snapshot per patient per epoch" sampling (cheap augmentation that removes redundancy), not as the default. Note in the report that the 10,704 adjacent pairs share overlapping windows, so epoch-level train AP is optimistic about memorization risk.

## Comparison resource fairness

Same frozen split, same inputs per comparison (Transformer-k vs LightGBM-k on sums of the same k features), same early-stopping metric (VALIDATION AP), same number of validation evaluations and seeds per family, pre-registered grids. Wall-clock is reported, never used to equalize. A family that received more validation evaluations is flagged in the comparison table.

## Blocking recommendations (fix in the design before coding runs)

- **B1 Pre-registration.** `PROTOCOL` dict + printed hash before any fit; a changed protocol gets a new ID; no post-hoc grid edits.
- **B2 Selection rule.** Proxy ranks and shortlists only; Transformer VALIDATION with 3 seeds and paired cluster bootstrap selects the count from the predeclared grid including the all-eligible control; parsimony within CI. LightGBM uses the identical count grid.
- **B3 Lineage corroboration in Notebook 01 (1,028 branch), recorded before interpretation.** (a) Overlapping-window consistency: for patients with two snapshots k months apart, the later snapshot's step t and the earlier snapshot's step t−k denote the same calendar month; exact agreement on full months establishes orientation; disagreement confined to the earlier snapshot's step 0 (later ≥ earlier) corroborates END_DT capping, equality there corroborates whole-month aggregation. (b) Step-0 label screen on TRAIN: per feature, univariate AP and positive-class nonzero rate at step 0 versus step 1; list the top 20 step-0 anomalies by name. Outcomes CORROBORATED / CONTRADICTED / INCONCLUSIVE, scoped to the audited pairs. If CONTRADICTED (whole-month indicated), Experiment B results carry a leakage flag and a step-0-masked refit becomes a required sensitivity arm.
- **B4 Equal-budget tabular baselines** on the same inputs and count grid, with early stopping on VALIDATION AP; existing V63 SCORE descriptive only.
- **B5 Class-weight ablation includes the historical full-ratio arm** so the chronology connects to the 09-25 runs.
- **B6 Dependence handling** as above: patient-grouped CV and bootstrap; snapshot-level primary and patient-level secondary metrics; weighting only as an ablation.

## Non-blocking recommendations (budget permitting, in priority order)

- N1 L1 on the input projection only, with per-feature projection norms reported against the proxy ranking.
- N2 Matched-strength explicit-L2 versus decoupled-decay ablation (2 fits).
- N3 "Counts + static AGE" arm for Experiment B (AGE concatenated to the pooled vector): tests whether the 49-model's AGE reliance carried information absent from the counts; clearly labeled as a hybrid, not the primary experiment.
- N4 One-random-snapshot-per-patient-per-epoch sampling and inverse-count weighting ablations.
- N5 Scaling ablation (log1p only vs log1p + floored TRAIN standardization); mask-reuse ablation from a hash-verified saved INPUTS artifact.
- N6 Optional patient-grouped 5-fold confirmation on TRAIN+VALIDATION for the two finalists (never TEST).
- N7 Scheduler and attention-pool ablations last; batch 64 (historical) vs 256 (CPU efficiency) once.
- N8 49-feature replay kept minimal: hash-verified saved inputs, baseline + ≤6 configs × 3 seeds, single-token control, LightGBM-49 refit, AGE diagnostics; MODEL_TYPE order with the SEQ discrepancy recorded.
- N9 Chronology: reprint the clipped E10 reports (RUN_001, TEMPORAL_CLEAN_V1 D20260925A, BUSINESS49_TEMPORAL_V1 H002/H003, SELECTED_V1 F001 baselines) with intrinsic `completed_at_utc` before Notebook 05 is finalized; the SELECTED_V1_F001 family already contains histogram-gradient-boosting and logistic artifacts on the 49 features — recover them rather than assuming no tabular baseline was ever run.

## Tests Codex should add (bounded, offline where possible)

- Protocol hash reproduces from the dict; artifact summaries carry it.
- Count-grid selection function: given synthetic per-seed validation APs and bootstrap CIs, picks the smallest count within the best's CI; the all-eligible control is never dropped from the table.
- Overlapping-window check: synthetic tensors built under capped and whole-month rules with both orientations yield the four expected outcome labels.
- Step-0 screen: a synthetic feature nonzero only at step 0 among positives is listed first.
- Patient-cluster bootstrap keeps all snapshots of a resampled patient together; patient-level metrics use exactly one (latest) snapshot per patient.
- L1-on-projection: gradient of the penalty is zero for non-projection weights; penalty share logged per epoch.
- Class-weight arms produce the historical `pos_weight` value from TRAIN counts (≈16.28) when the full-ratio arm is selected.

Nothing in this review authorizes a superiority claim; it bounds the search so that whatever the results are, they are interpretable under the recorded restrictions.
