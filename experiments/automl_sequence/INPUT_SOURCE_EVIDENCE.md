# AutoML input recovery and execution contract

## Latest delivery and review instruction

Active new raw construction is R20261013C. Complete source receipt permits integration without the old custom275-column pivot; it does not validate legacy equivalence. New relevant tests/build identity are pending; previous48-test R20261013B remains predecessor evidence. The latest user explicitly requested Git push and deferred Claude review. Fresh review is no longer a push condition, and no new Claude approval is claimed. See [historical addendum/output-layer findings](HISTORICAL_ADDENDUM_FINDINGS.md) for recovered results, not new experiment metrics.

Recorded 2026-10-13. This is code/source inspection, not clinical-data execution.

## Complete procedure intake now supersedes the earlier missing-body status

**Recovered from user-supplied code, not live warehouse verification:** Procedure 2, `FEATURE_CREATION_PROC`, and Procedure 6, `FEATURE_ENGINEERING_PROD`, are now complete across the supplied continuations. The DX/PX/RX extraction, AGE/RND, final QUALIFY, cap/type/model-data/importance/final-summary logic, cleanup and return are received. Do not ask for these bodies or the unavailable `auto_ml_previous.txt` again. The exact custom pivot that produced the old 275-column AutoML table remains unreceived, but it is not a prerequisite for a new explicitly documented raw extractor.

The user now prioritizes the five notebooks immediately, with a real AutoML tensor as the minimum concrete stage. New raw-extractor integration is in progress; it must build genuine patient/as-of values, not reuse legacy target-aware encodings or broadcast a current row. The new extractor deliberately removes learned/label-dependent legacy behavior and therefore must not claim numerical equivalence to the old feature table. The 48-test `57780af...` source-recovery build is a predecessor checkpoint; it does not validate subsequent raw-extractor changes. No new clinical tensor, metric, runtime, fresh Claude approval or push has been produced by this intake.

Earlier OCR references and the old static loader were incomplete recovery leads. The supplied static-fusion code reads `TAK861_TX_READY_V63_DL_POC_AUTOML_FEATURES_R20261009A`; its exact custom wide pivot and historical executed revision still require separate provenance if reproducing that artifact. Current source bodies are sufficient to continue the new declared construction without blocking on exact historical reproduction.

The previous static table alone cannot establish twelve historical values. This implementation permits exact patient/date reuse only with matching source-version, extraction, feature-definition and content fingerprints. It constructs unique historical requests; each shared patient/date is extracted once. It never repeats the current snapshot vector into history. Raw numeric declarations are explicit; unknown categorical/encoded semantics must be resolved before training. The fusion notebook default is `month_end_history`: current END_DT followed by prior calendar month-ends. A same-day shift exists as a separate input API policy and is not the count-fusion default.

## Optional current-deployment verification in the authorized connection

```sql
SELECT PROCEDURE_CATALOG, PROCEDURE_SCHEMA, PROCEDURE_NAME,
       ARGUMENT_SIGNATURE, CREATED, LAST_ALTERED, PROCEDURE_DEFINITION
FROM DSVC_TAKEDA_TA_PRIVATE.INFORMATION_SCHEMA.PROCEDURES
WHERE PROCEDURE_SCHEMA IN ('DS_ML_DEV', 'DS_ML', 'DS_ML_PRO', 'DS_ML_PRD')
  AND PROCEDURE_NAME IN ('FEATURE_CREATION_PROD', 'FEATURE_CREATION_PROC', 'FEATURE_ENGINEERING_PROD')
ORDER BY PROCEDURE_NAME, ARGUMENT_SIGNATURE;
```

Use the returned signature (never a guessed overload) with `DESCRIBE PROCEDURE <fully-qualified-name>(<returned argument types>)` / `SELECT GET_DDL('PROCEDURE', '<fully-qualified-name>(<returned argument types>)')` if the definition is not exposed. These are recovery instructions, not a claim that access is available. Do not execute a procedure that writes shared old-run tables during inspection.

## Source contract for the new raw construction

Adapt the now-received source logic into a read-only query accepting `(PATIENT_ID, AS_OF_DATE)` requests. Document feature definitions/types, source versions, fixed lookbacks, actual event bounds and unresolved availability. Use the same unfitted schema at all dates, with no RESP-dependent filtering or upstream target encoding; fit any learned selection/transform within TRAIN folds. Missing history requires an honest coverage contract, not a fabricated all-observed flag. A deployed-DDL read can verify current definitions if available but is no longer required merely to obtain the supplied procedure bodies. The old custom pivot is necessary only for exact historical-table reproduction, not for this new extractor.

The local implementation, synthetic input/sequence checks and notebook wiring can continue while this production source binding is recovered. No warehouse result or fitted model is implied by passing local tests.

## Count-plus-AutoML fusion calendar

The original count initialization explanation (`outputs/dl_poc_manual_work/manual.md`, Block 2) and recovered count construction code use calendar-month buckets: step zero is truncated at the original END_DT; older steps are complete preceding calendar months. Original cohort cutoffs include the 21st day, so a same-day-month shift is not equivalent to the count bucket end. Count-plus-sequence fusion must therefore select `month_end_history` for the AutoML panel. It must not silently align a historical AutoML cutoff of the 21st with count events from the 22nd through month-end.

`count_loader.py` reuses only the four saved raw count input tables, retaining exact keys/labels/feature ordering and recorded split metadata. Every saved sequence position is marked structurally present, including zero-activity buckets. That position mask does not establish enrollment, clinical observation coverage or source availability; this limitation remains explicit in its contract. No previous model/search/prediction artifacts are required.

Local validation: 15 synthetic input tests passed. A separate synthetic workflow check passed uppercase split normalization, identifier-preserving CSV roundtrip, source/array save/load integrity, explicit count mask/calendar rejection, and streamed `.npy` fingerprint equivalence. Spark/warehouse execution is not available in this validation; no clinical extraction or model result was produced.

## Later user-supplied procedure supplement

Recovered from supplied source, not independently executed: the wrapper `FEATURE_CREATION_PROD`, full lower-level `FEATURE_CREATION_PROC`, and full `FEATURE_ENGINEERING_PROD` are now received, together with registration/training/comparison/scoring wrappers. Earlier statements that the event and engineering bodies were omitted describe the prior checkpoint only.

The supplied wrapper uses positive RESP counts from its model universe to determine annual mapping thresholds; it selects current/latest mapping versions, overwrites the project mapping and universe-with-features tables, then calls `Feature_Creation_Proc`. That is not a target-free deterministic extraction over the complete cohort. A TRAIN-only mapping-selection artifact must be frozen and reused identically across historical requests and splits. Current/latest mapping selection also requires a recorded source version; the wrapper does not itself establish historical point-in-time availability.

The complete lower-level procedure expects PATIENT_ID, RESP, START_DT and END_DT; additional fields are unpivoted with positive-value filtering and RESP-dependent counts. Medical service dates and pharmacy fill dates use inclusive START_DT/END_DT bounds. The recent window runs from END_DT minus four months through minus one month inclusively. Raw category values are counts; `value_b` is binary except the CLAIMS/HCPS families. Recent percentages are emitted only when total count exceeds 6 and recent count exceeds 2. AGE is calculated as of the supplied date. RND is added only with scoring flag N.

The final SQL unconditionally applies `QUALIFY resp_cnt > (r.resp_all_cnt * 0.1)` after the DX/PX/RX temporary rows join the input universe. Scoring flag Y narrows the earlier category list but does not bypass this label-dependent filter. Frozen mapping reuse alone is not safe raw extraction. Repeated wrapper calls would also overwrite shared outputs; the new raw path must not execute those writer procedures.

The now-complete engineering body computes `percentile_disc(0.9)` on RESP=1, caps values, forms response-rate summaries and classifies feature type using a regression of capped value against response rate. Numeric classification uses r-value at least 0.8 or AGE/RND; otherwise a response-rate-ratio condition determines binary classification. MODEL_DATA uses all positives plus a negative sample capped at 250,000, without a TRAIN filter. Transformations are AGE decade bins, positive binary indicators, and capped ratios using VALUE_P for other fields. LightGBM then fits that MODEL_DATA and supplies gain importance. The final FEATURES_SUMMARY joins features and importance, compares each importance with RND, assigns model/final/dropped status, orders SEQ by descending importance, writes the project/version summary, cleans temporary objects and returns. These operations are label-aware, not an unfitted feature extractor.

The complete comparison procedure uses up to `max_f` (default 50) non-dropped features and PyCaret five-fold comparison sorted by Recall, with no explicit patient grouping shown. This legacy policy is not the new LIFT protocol. The orchestration wrapper also passes the session identifier into the training procedure's fourth argument, where its received signature expects model parameters; preserve this supplied-code mismatch and do not execute the legacy wrapper to obtain new inputs. Neither finding alone proves what historical deployed version executed.

## October 13 screenshot supplement and recovery route

The user confirmed the pasted text is all the AutoML source available to them;
the `auto_ml_previous.txt` file mentioned by Genie is not available locally.
Searches of project/context, Desktop, Downloads, attachment stores and relevant
archives did not recover it. Do not continue requesting the same unavailable file.
`source_recovery.py` can verify the three supplied procedure signatures with
read-only GET_DDL through the existing Snowflake connection. The preceding
48-test notebook build saved definitions and a receipt privately before stopping.
Now that full bodies are received, the next build is wiring direct raw extraction;
deployment verification remains separate from received-source completeness.

The 13 unique screenshots are **Recovered from existing artifact**, not a live
warehouse verification. They report that the 23,151-row universe included all
splits, START_DT was ADD_MONTHS(END_DT,-12), and both FEATURE_CREATION_PROD and
FEATURE_ENGINEERING_PROD ran on that universe. The wrapper's RESP-dependent
mapping and the positive-class cap are corroborated by the pasted SQL. Genie
also reports full-universe feature typing, LightGBM importance and RND-threshold
selection. Consequently the old AutoML feature table must not be treated as
leakage-free input; a later TRAIN-only scaler cannot undo upstream selection.

The full source now corroborates inclusive per-row service_date/fill_date cutoffs and a
recent-period window from END_DT minus four months through END_DT minus one
month. There is no reported claim-availability cutoff. Even scoring mode is
confirmed in the supplied code to retain RESP-based rare-feature pruning in a final QUALIFY condition.
Frozen mapping reuse alone is therefore not sufficient. No claims, unavailable
history and missing features reportedly become indistinguishable after fillna(0).
Do not infer complete observation from zero values or fabricate availability flags.

Discrepancies remain explicit: the screenshots mix 261 selected and 272 wide
feature counts; the supplied fusion code instead visibly removes 11 fields from
272. A claim that notebook preprocessing was not persisted conflicts with supplied
Cells 18/22. A description of V4 concatenation conflicts with its additive code.
A screenshot's array-index orientation does not override the recovered count
loader's TIME_STEP contract. Live definitions describe current source, not proof
of the historically executed revision.

For a new 12-anchor panel, retaining a 12-month lookback at every anchor yields
roughly 23 months of combined event history. This remains an **Inferred** design
implication until a real panel executes, even with the inclusive source bounds
now supplied. The comparison must disclose that information, history length and
model capacity can all change.
