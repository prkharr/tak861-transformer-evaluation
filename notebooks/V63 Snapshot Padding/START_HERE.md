# Final 1,028-feature experiment: five-notebook run guide

Updated 2026-10-06. The four existing notebook paths are revised, and Notebook 5 completes the comparison. The earlier four notebooks and their original instructions are preserved unchanged in `../Historical 49 Snapshot Padding (20260927)/`.

**Status: implemented and synthetically verified; no new project model has been trained or evaluated by this delivery.** Final feature count, model settings, performance and winner are computed by the authorized runtime. Cleared notebook outputs are intentional.

## Run in order

| Notebook | Purpose | Saved stage |
|---|---|---|
| `01_tensor_initialization.ipynb` | Read all 1,028 saved count features; check current vocabulary/order, frozen keys/labels/splits and count validity; record timing and leakage corroboration limits | PREPARATION |
| `02_patient_level_split.ipynb` | TRAIN-only quality checks, removal ledger, temporal ranking, patient-grouped fold stability, count shortlist and distribution/redundancy diagnostics | SELECTION |
| `03_transformer_training.ipynb` | Register the full two-phase protocol, compare counts, freeze each family's winning count anchor before ablations, checkpoint/resume, and fit the encoded-49 LightGBM reference | PROTOCOL, ANCHORS, per-fit artifacts, SEARCH, BASE49 |
| `04_transformer_evaluation.ipynb` | Inspect all seeds and overfitting; lock finalists and thresholds; audit ranking against internal diagnostics and frozen-model validation permutation effects | LOCK, RELIANCE_PLAN, RELIANCE |
| `05_model_comparison.ipynb` | Score the locked finalists on retrospective TEST; show uncertainty, metrics, calibration, confusion matrices, top-K, complexity and a qualified conclusion | FINAL_INTENT, FINAL |

Import the five notebooks into the existing approved Databricks/Spark environment. Use its existing `spark` and `sf_options_dl_poc` (or `sf_options`) objects. Each notebook embeds all project helpers; no repository checkout or companion-module import is needed. Do not paste credentials into shared notebook source.

Use the same `RUN_ID` in all five notebooks (default `R20261006A`). Artifacts use the new `TAK861_TX_READY_V63_DL_POC_RIGOROUS1028_V1_<RUN_ID>` prefix. Existing conflicting artifacts are rejected. Changed inputs, code, protocol, seeds or budgets require a new ID. Resume the same unchanged run to reuse completed fits. Notebook 5 reuses a completed final artifact instead of recomputing predictions.

The runtime needs its existing Snowflake Spark connector plus NumPy, pandas, SciPy, scikit-learn, PyTorch, LightGBM, Matplotlib and IPython. Use the organization's approved packages; the collector reported CPU PyTorch and LightGBM 4.6.0 already available. Notebook 3 saves actual runtime versions. The local verification environment and its limits are documented in `VALIDATION.md`.

## What the experiment tests

The count experiment begins with 1,028 candidates (490 DX, 496 PX, 42 RX) over 12 saved positions and the frozen 23,151-snapshot population. It does not turn the 49 previously encoded snapshot fields into a historical sequence. All twelve saved count positions remain real inputs; zero is not inferred to mean missing enrollment.

All-missing, exact constant and exactly duplicate feature sequences are removed with a recorded reason. Rare, low-variance and correlated features remain visible and eligible. Default log1p is suitable for nonnegative counts; standardization is a declared sensitivity arm. Actual missing values are audited and, only where necessary, imputed using TRAIN statistics. Quantiles, extreme counts, missingness, sparsity, variance, fold-rank stability, correlation and a bounded spectral diagnostic are recorded. No global VIF or clinical code semantics are invented.

A temporal-summary logistic proxy evaluated within three patient-grouped TRAIN folds produces a shortlist, not a final Transformer feature count. The shortlist includes all eligible features. Each model family selects its own finalist using mean VALIDATION average precision across seeds 42, 142 and 242; exact ties prefer fewer features and lower complexity. The final predictor is the uniform three-seed ensemble. Ranking order and original FEATURE_MAP input order are saved separately.

The two phases are registered before training: phase A compares all shortlisted counts using the reference settings; each family's count winner is then recorded in a hashed anchor artifact before phase B tests architecture and regularization on that family's winning feature set. The step-0 sensitivity retains all eligible features. Resumption recomputes anchors from complete saved phase-A evidence and rejects changes. Validation is adaptively reused for this declared search; it is development evidence, not independent confirmation.

The registered grid tests feature count, capacity, dropout, weight decay, explicit L2, projection L1, class weighting, pooling, scaling, quarterly aggregation, patient weighting and step-0 exclusion. It contains 12 paired ablation configurations plus each distinct shortlisted feature set per family: `6 * (12 + number_of_shortlisted_sets)` count-model fits (78-96), plus three fixed encoded-49 LightGBM fits. Maximum training is 30 Transformer epochs per fit and 600 boosting iterations for count LightGBM; validation-AP early stopping can shorten runs. Eight CPU threads are configured. Notebook 3 prints a TRAIN-only throughput estimate before fitting; it is not a completion-time promise. Keep the registered grid complete rather than dropping inconvenient seeds or slow candidates.

LightGBM receives the same saved count information through flattened monthly inputs by default; windows and annual sums are separate ablations. Family-level finalists may select different features or representations, so a best-pipeline comparison is not automatically an isolated architecture comparison. Actual trial counts, validation evaluations, time and complexity are reported. The encoded-49 LightGBM refit is an additional comparison with a different input history. It is not a reproduction of the original fitted V63 model.

## Ranking and model reliance

Notebook 4 preserves an audit row for every original feature. It joins the TRAIN rank/stability to the selected model position and every seed's projection norm or aggregated LightGBM gain. Projection norm multiplied by transformed TRAIN standard deviation is an additional scale-aware internal diagnostic. These quantities are not predictive reliance. A matching count LightGBM arm supplies gain on the Transformer's exact selected feature identities.

After the model/threshold lock, a frozen ensemble measures validation AP and lift@10% changes under whole-trajectory permutation. It uses one latest snapshot per validation patient, permuted within END_DT calendar month, as a secondary diagnostic. The registered individual-feature union includes top 10 screening ranks, top 10 internal ranks, bottom 5 screening ranks and 5 seeded remaining features, at most 30. Joint probes cover DX/PX/RX, selected-feature TRAIN-rank terciles and up to ten largest TRAIN correlation components. All individual omissions remain UNKNOWN; group results do not substitute for individual effects.

The maximum is 139 ensemble prediction calls per family, plus one checkpoint-output consistency check. No runtime is promised. Identical plans reuse cached diagnostics. Repeat SD measures permutation randomness, not a confidence interval. Actual changed-value fractions reveal weak perturbations of sparse features; correlated/off-manifold effects remain limitations. Rank disagreement is a finding to investigate, not a reason to force learned importance to match screening or change the locked model. Notebook 5 requires the saved audit before TEST inference.

## Evidence and interpretation

- Successful checks reproduce current structural and fitting contracts only. Original event cutoff, count grain, feature-builder execution, selection/encoding fit scope and some historical feature ordering remain unresolved.
- The prior TEST was inspected historically. It is called retrospective TEST, never untouched OOT. Model/threshold selection in the revised run uses TRAIN/VALIDATION only.
- No complete, certified untouched OOT cohort with all required model inputs is available. The notebooks do not fabricate one from calendar eligibility or table timestamps.
- Notebook 5 reports AP and trapezoidal PR-AUC separately, ROC-AUC, precision/recall/F1/specificity, confusion matrices, Brier/log loss and calibration, top-10% precision/recall/lift, training gaps and seed variation. Paired patient bootstrap intervals condition on fixed fitted predictions; they do not capture all training/search uncertainty. An interval crossing zero is not equivalence.
- Top-10% is a prespecified capacity illustration, not a confirmed business operating requirement. Better retrospective discrimination alone does not establish clinical or business benefit.
- Original V63 scores are recovered for descriptive VALIDATION context only; historical fit exclusion and score semantics remain unknown. Rounded historical screenshots are not new model results.

See `REVIEW_DECISIONS.md` for Claude's actual design review and dispositions, `VALIDATION.md` for completed software checks, and `RUN_HANDOFF.md` for the remaining execution/recovery request. The canonical project memory is `C:/Users/prakhar/Desktop/Response/DLPOCfinal/project_context/TAK861_MASTER_CONTEXT.md`.
