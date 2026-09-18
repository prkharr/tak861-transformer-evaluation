# Transformer evaluation specification

## Scope and unit

Evaluate existing Transformer probabilities for every snapshot in the trained model's original, held-out TEST assignment. The intended upstream population is historical V63, using claims vintage `20260825` and the established `RESP=1` definition: first advanced-therapy initiation within the forward 90-day outcome window. This evaluation does not rebuild that population or verify upstream outcome/observability logic.

The evaluation unit is `(PATIENT_ID, END_DT)`. A patient may contribute multiple snapshots, each counted once; the results measure snapshot targeting. All snapshots for a patient must be in one split. The original full TRAIN/VALIDATION/TEST manifest is required to check that patient separation. An internally consistent manifest cannot prove that no leakage occurred during training; model freezing, feature cutoff logic and training-only transformations require upstream verification.

LightGBM execution, prediction loading and comparative claims are deferred. There is no fabricated baseline and no side-by-side comparison in the current run.

## Checks before ranking

1. Require valid, unique snapshot keys; string patient identifiers; ISO calendar dates; binary, nonmissing labels; and all three named splits.
2. Require every patient to belong to exactly one split across the full manifest.
3. Require at least ten TEST snapshots and exact Transformer coverage of those snapshots. Reject duplicate, missing or extra predictions rather than silently taking an intersection.
4. Require finite `P_RESP1` values between 0 and 1. If prediction rows include labels or split names, check them against the manifest.
5. Align by both keys. Only the manifest supplies canonical outcomes. Record a SHA-256 fingerprint of the canonical full manifest, independent of its input row order.
6. If run provenance is supplied, verify its declared fingerprint, V63 population, `20260825` vintage and split roles. Declarations are not an independent training-code audit.

## Rank and assign deciles

Let `N` be the number of TEST snapshots and `P` their total number of `RESP=1` snapshots. Let `pi = P/N` be the overall TEST response rate.

Sort descending by the Transformer probability. For tied scores, sort ascending by SHA-256 of the JSON representation of `["tak861-targeting-v1", PATIENT_ID, END_DT]`; the canonical keys provide a deterministic fallback for hash collisions. Only keys and probabilities enter the ordering function. Labels are joined to the resulting order for evaluation. A hash used for ties is not anonymization and is not exported.

For decile `d` from 1 to 10, define `b_d = ceil(d*N/10)`, with `b_0 = 0`. Assign ranks `b_(d-1)+1` through `b_d` to decile `d`. Thus decile 1 has the highest propensity, decile 10 the lowest, all buckets are nonempty for `N >= 10`, and bucket sizes differ by at most one. No outcome-dependent quantiles, cutpoints, downsampling or label stratification enter this step. This is rank-based bucketing; a probability value may legitimately span multiple deciles when tied.

The sort uses explicit score and tie keys, following [pandas' multiple-column sorting interface](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html). The tie policy is fixed before evaluating outcomes. Tie diagnostics at 10%, 20% and 30% report the cutoff score, total tied snapshots, selected tied snapshots and whether the tie spans the cutoff. Large ties mean the model cannot distinguish all candidates at that boundary; the hash supplies reproducibility rather than additional predictive signal.

## Decile table definitions

For decile `d`, let `n_d` be its number of snapshots and `p_d` its actual positive count. Let `N_d = sum(n_i, i<=d)` and `P_d = sum(p_i, i<=d)`.

| Report field | Formula | Meaning |
| --- | --- | --- |
| `n_snapshots` | `n_d` | Snapshots in this decile |
| `n_resp1` | `p_d` | Actual RESP=1 snapshots in this decile |
| `population_fraction` | `n_d/N` | Fraction of TEST snapshots in this decile |
| `response_rate` | `p_d/n_d` | Within-decile response rate |
| `decile_precision` | `p_d/n_d` | Within-decile precision; deliberately equal to response rate |
| `share_of_all_resp1` | `p_d/P` | Fraction of all TEST positives captured in this decile |
| `cumulative_n_snapshots` | `N_d` | Snapshots selected through this decile |
| `cumulative_population_fraction` | `N_d/N` | Actual fraction of TEST selected |
| `cumulative_resp1` | `P_d` | Cumulative actual RESP=1 captured |
| `cumulative_recall` | `P_d/P` | Cumulative capture rate / recall |
| `cumulative_precision` | `P_d/N_d` | Precision within the entire selected population |
| `decile_lift` | `(p_d/n_d)/pi` | This decile's response rate relative to TEST overall |
| `cumulative_lift` | `(P_d/N_d)/pi` | Selected population's response rate relative to TEST overall |
| `overall_test_response_rate` | `pi` | Shared denominator for all lift metrics |

CSV rates and shares are fractions in `[0, 1]`; multiply by 100 to express percentages. Notebook and report presentation label percentages explicitly. Lift is a multiple of baseline, such as `2.0x`, not a percentage. Counts are snapshot counts, not unique-patient counts.

## Explicit top-K metrics

For `K` in 10%, 20% and 30%, let `d=K/10` and select the first `d` deciles:

- Number selected: `N_d = ceil(K*N/100)`.
- Actual selected fraction: `N_d/N`, which can slightly exceed the nominal fraction.
- `Recall@K = P_d/P`: top-K capture of all TEST positives.
- `Precision@K = P_d/N_d`: positive share of the selected population.
- `Lift@K = Precision@K/pi`.
- Top-decile lift is `Lift@10%`, equal to decile 1 lift.

Top-K results always reuse the decile boundaries rather than independently rounding or finding score thresholds. This keeps the table, gains curve and headline metrics consistent.

## Charts and global metrics

The cumulative gains chart plots `(N_d/N, P_d/P)` and includes the origin. The random-ranking reference is `y=x`. The lift chart shows individual-decile lift and cumulative lift, with a reference at 1. The top-K chart summarizes Transformer recall, precision and lift at 10%, 20% and 30%.

The supporting global metrics are average precision and ROC-AUC. Average precision follows [scikit-learn's definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): the sum of precision at each recall increment, weighted by that recall increment. It is non-interpolated and is not trapezoidal PR-AUC. ROC-AUC requires both outcome classes.

No probability classification threshold is learned from TEST. Capacity cutoffs are predeclared at 10%, 20% and 30% using ranks only. Threshold-based F1 and a confusion matrix require a separately frozen operating threshold selected on VALIDATION; they are outside this ranking-focused run.

## Edge cases and interpretation

- **No positive TEST snapshots:** counts, response rates and precision remain defined; positive-capture fractions, recall, lift and average precision are N/A. ROC-AUC is also N/A. Undefined values are not replaced with zero.
- **All TEST snapshots positive:** precision, recall, gains and lift remain defined; ROC-AUC is N/A. Every selected snapshot is positive, so lift is 1.
- **Fewer than ten TEST snapshots:** fail explicitly; ten nonempty deciles are impossible.
- **Repeated patients:** retain each distinct snapshot. This does not answer a one-contact-per-patient operational question; that needs a separately specified patient-level aggregation policy.
- **Uncertainty:** point estimates only. The current run has no confidence intervals or significance claims; correlated snapshots must not be treated as independent in a later uncertainty analysis.
- **Probability calibration:** scores support ranking, but these summaries do not establish probability calibration. Training class weights can affect calibration, and recalibration must not be fitted on TEST.
- **Later model comparison:** use the identical full frozen manifest, snapshot labels, TEST keys, cutoff definitions and tie policy. Compare captured positives, Recall@K, Precision@K and Lift@K at equal selection sizes, with paired patient-aware uncertainty if required. Global ROC-AUC alone does not establish incremental prioritization value.

The generated results summarize only the supplied artifacts. The required upstream provenance remains to be verified when it is absent, and LightGBM incremental value remains unmeasured until that separate benchmark is run.
