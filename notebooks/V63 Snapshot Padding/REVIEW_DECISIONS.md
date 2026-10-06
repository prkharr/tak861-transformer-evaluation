# Reviewer decisions and evidence boundaries

Date: 2026-10-06. Reviewer: actual Claude in the existing TAK861 deep-learning project review conversation, followed by independent Codex implementation review.

Source: `C:/Users/prakhar/Documents/Codex/DLPOC/work/final_experiment_20261006/CLAUDE_DESIGN_REVIEW.md`, preserved verbatim in the centralized project context under `evidence/sme_review/CLAUDE_DESIGN_REVIEW_20261006.md`. SHA256: `fab9b8da6de480648f9a2b7c976cc4b6d662da7d42d1b8d6abd1f3daa81f411e`. The design review is a recommendation artifact, not an independent warehouse measurement. See the canonical context's SME disposition for source-linked detail.

## Accepted and implemented

| Recommendation | Implementation and reason |
|---|---|
| Do not let a proxy prescribe Transformer feature count | Proxy-best, one-SE, one-smaller and all-eligible sets form a shortlist. Actual Transformer validation performance over all three seeds chooses its count. |
| Register the protocol and retain every seed | Immutable protocol/input/code identities, fixed seeds, complete-grid requirement, fit-level checkpointing and full trial tables. |
| Preserve rare sparse signals | Only all-missing, exact constants and exact duplicate sequences are automatically removed. Default log1p; floored standardization is a sensitivity arm. |
| Challenge temporal lineage | TRAIN-only overlapping-window corroboration and step-0/step-1 label screens; an all-eligible step-0-excluded fit is always included. None certifies original cutoff enforcement. |
| Stronger regularization and compact architecture | Small pre-norm month-token encoder plus controlled capacity, dropout, decay, explicit L2, projection L1, class-weight, pooling, sequence and patient-weight arms. Validation-AP checkpoint restoration, scheduler and clipping. |
| Include full historical class-weight ratio | Unweighted, square-root and full TRAIN-derived class-weight arms; no TEST-derived weights. |
| Fairer baselines | Count LightGBM gets the same count shortlist and seed count, plus representation/capacity controls. A separate encoded-49 LightGBM refit uses the frozen cohort and documents its unresolved upstream history. |
| Account for repeated snapshots | Patient-disjoint inner folds and frozen outer split; patient-cluster bootstrap; latest-snapshot-per-patient secondary metrics; inverse-patient training weight is an ablation. |
| Lock before TEST | All finalists, exact ordered features, model hashes, ensemble membership and validation thresholds, including encoded-49, are locked before TEST inference. |
| Show uncertainty and operational views | All seeds, learning curves, train/validation/test gaps, conditional paired patient intervals, calibration and prespecified top-10% capacity metrics. |

## Qualified or rejected after checking evidence

| Proposal or assertion | Decision and reason |
|---|---|
| Counts have no missingness | Not established: the collector skipped a full value scan. The implementation audits actual values and fits imputation only for real missing values. |
| Saved 49-feature padding statistics establish count coverage | Rejected. The 49-feature replay contract does not establish availability for the 1,028 count tensor. Zero counts remain observations; no historical mask is inferred. |
| Exact overlapping-window equality proves orientation/capping | Rejected as proof. Static, absent or replayed activity can agree under more than one construction. Results are corroboration with explicit limitations. |
| Select smallest count inside the best validation bootstrap interval | Not used. Adaptive overlapping intervals do not establish equivalence or a principled optimum. The proxy one-SE rule only shortlists; final selection uses preregistered mean validation AP with exact-tie parsimony. |
| Validation significance plus same-sign TEST establishes superiority | Rejected. Validation is reused for stopping and search. The final paired retrospective TEST interval is conditional on locked fits; crossing zero does not establish equivalence. No prospective claim is made. |
| Window sums preserve the same information as monthly sequences | Rejected. Default count LightGBM uses flattened monthly positions. Window/annual summaries remain separate ablations. Finalists with different features/representations compare pipelines, not architecture alone. |
| Explicit L2 and AdamW decay can be matched by nominal coefficient | Qualified. These are different optimization operations. Declared coefficients are tested separately, never presented as mathematically equivalent strengths. |
| Original V63 score can be treated as a verified baseline probability | Rejected. Fit membership and score meaning remain unknown. Its VALIDATION score ranking is descriptive only; no original-model superiority/calibration claim is authorized. |

## Deferred within the registered scope

A five-fold nonlinear LightGBM screening proxy was not added on top of the three-fold transparent temporal logistic proxy: it would introduce another selection model/search budget. Its nonlinear bias is a limitation, and all-eligible features remain an actual Transformer control.

A counts-plus-static-AGE hybrid, new tuning of the archived 49-feature Transformer, random-one-snapshot augmentation, full global VIF, mask transfer and additional five-fold finalist confirmation are not executed by this version. They must not appear as completed experiments. The bounded ablations do not exhaust every architecture or prove a global optimum. Saved 49-feature experiments remain historical evidence, not fresh controlled results.

## Pre-push reliance and actual-code review

After the user's additional pre-push request, native access to Claude was restored. Claude independently read the actual helper sources and five notebooks and saved [the actual-code review](../../experiments/rigorous_transformer/reviews/CLAUDE_RELIANCE_REVIEW_20261006.md). Its SHA256 is `b3693eeab6ffbd2debd0d0bf98ca269cd2f547796f958533d4383e0863cb185f`. It confirmed the feature-index/name/input/projection mapping and found no new leakage in the reviewed model-selection/TEST control paths; unresolved upstream lineage remains outside that finding. This review inspected the predecessor implementation `a6a5caff...`, before the corrections below. The [original design review](../../experiments/rigorous_transformer/reviews/CLAUDE_DESIGN_REVIEW_20261006.md) is also preserved verbatim.

Two substantive recommendations were accepted before publication:

1. Replace proxy-subset-only ablations with a registered two-phase search. Each model family selects its phase-A count anchor using all prescribed seeds, records its identity and evidence, then runs phase-B ablations on that feature set. Count anchors are recomputed and verified on resume; no discretionary post-hoc anchor change is allowed.
2. Add the keyed ranking-to-model audit and post-lock validation permutation diagnostics. The audit covers full original vocabulary, each seed, ordered identities, scale-aware internal measures, matching count-LightGBM gain, individual probes, source families, TRAIN-rank terciles and TRAIN-correlation groups. It preserves locked decisions and exposes unmeasured individual reliance.

Claude's proposal to run every individual feature in “a few minutes” was not accepted as a runtime fact: no production benchmark supports that estimate. Its permitted bounded union is registered instead, with explicit omission coverage. Latest-patient, within-month permutation is used as a secondary diagnostic; naive repeated-snapshot shuffling is not presented as patient-independent inference. Leave-group-out refits after the final lock were deferred because they add experiments and could reopen selection. All-source-family group effects measure fixed-model reliance, not proof that a group adds information to retraining. Eight CPU threads remain a declared resource choice; sixteen is not assumed faster without a benchmark.

## Review completion

Claude's design review actively challenged all eight requested areas; its later actual-file review identified the corrections above. A subsequent request for verification could not be submitted: Windows activation failed, and the fresh-window recovery attempt returned `GetCursorPos failed: Access is denied (0x80070005)`. Therefore no final Claude approval of the revised implementation is claimed. Independent Codex reviewers and synthetic execution checked the corrections, mapping, artifact integrity and release contents; the delivery's validation record states their scope. The final review also tightened preprocessing/configuration identities in the decision lock and cached reliance-report consistency. No reviewer result establishes clinical performance or original upstream provenance; authorized-runtime execution remains outstanding.
