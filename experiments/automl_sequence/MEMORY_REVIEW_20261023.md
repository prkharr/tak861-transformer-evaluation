# Memory revision and local validation

The five notebooks are code-only by request; scientific explanations, review evidence and operational instructions live outside them. The default run is R20261023_OLD_A on the original cohort. Local validation passed 102 tests plus 27 shared-helper regressions. The full notebook execution test uses synthetic data and source/store doubles. Private runtime execution is pending in a separate environment.

Implementation: `69adef9395638375cd5497f598acce84c20c6dff886279601c4953df0e95303b`.

Independent review completed in the shared review mailbox (Claude #5) and its
chat. The reviewer independently ran 102 suite tests and 27 shared-helper tests,
reproduced the controlled before/after equivalence, verified the five notebook
and ZIP hashes against the manifest, and closed M1–M4. O1–O4 were accepted;
O5 (manifest-only replay optimization) remains optional and deferred. The review
approved this scope for commit, retaining the live-runtime limitations below.

The controlled baseline/current fixture has identical selected lists, preparation hash `6987f361a113bdfb590feb10f62e11d209abd4c6c8d74ccdf8f8280f03547ce7`, lock hash `bd6c2dbfc8368cf3b94683f6aca3b4d97af8c469f488523920a3f018df5c0aef`, and all nine validation score vectors (maximum absolute difference 0). Implementation identity and elapsed telemetry are fixed only in the comparison harness. The fixture uses 96 synthetic patients, 1,028 count candidates (four active), five AutoML measurements and the production model/selection configuration.

# A1 memory diagnosis (source audit, original cohort)

This is an analytical allocation report, not a measured Databricks RSS profile. No clinical data were queried or executed. The latest direction is to run the original cohort first; therefore this report uses its verified population, rather than substituting it for the unknown revised-universe population. The future revised universe must supply its own audit counts before sizing.

Population: **23,151 snapshots**, **12,447 patients**, **1,345 positives**; TRAIN **16,256**, VALIDATION **3,481**, TEST **3,414** snapshots. Source: `work/final_experiment_20261006/CLAUDE_FINAL_PROJECT_REVIEW_20261006.md:13â€“14`, which summarizes the user-run collector E04/E05. The supplied execution screenshot reported **213,202 distinct historical extraction keys** and **277,812 tensor positions**. These are different quantities: repeated patient/date keys are reconstructed once and reused at matching positions. This report uses 1,028 count columns and the conservative 512 AutoML-column cap; pair preservation plus AGE allows at most 511 retained columns in the current implementation. No selected-K count or historical-observation rate is invented.

## Allocation units and interpretation

All sizes below use binary MiB/GiB. Let `C` be compressed INPUTS NPZ bytes, `R_long` the returned long-row count, and `L` the observed TRAIN time-position count (at most `12*16256=195072`). Array values and array-backed disk storage are sized exactly by shape/dtype. Pandas object strings, SQL results, runtime/native buffers, framework state and page cache cannot be inferred from shape alone. **Do not add every row of the stage table: stages run sequentially.** Mapped-file capacity is disk demand; mapped pages can still appear in RSS and require OS reclaimable-memory headroom.

| Array | Size at the declared bound |
|---|---:|
| Entire fresh count tensor, float32 NÃ—12Ã—1,028 | 1.064 GiB |
| TRAIN count tensor, float32 | 0.747 GiB |
| Historical AutoML panel, float64 QÃ—512 | 0.813 GiB |
| Expanded panel at 12N rows, float64 | 1.060 GiB |
| AutoML sequence, float32 NÃ—12Ã—512 | 0.530 GiB |
| AutoML summary, float32 NÃ—512 | 45.22 MiB |
| Count+AutoML sequence+summary raw arrays | 1.638 GiB |
| TRAIN count proxy, float64 TRAINÃ—1,028Ã—4 | 0.498 GiB |
| TRAIN AutoML proxy, float64 TRAINÃ—512Ã—4 | 254.00 MiB |
| Max selected TEMPORAL_FUSION TRAIN+VALIDATION (1,541 columns including indicator) | 1.360 GiB |

## Where memory peaks and what changed

| Stage/function | Frozen baseline | Current bounded implementation |
|---|---|---|
| Fresh counts / `attach_count_reference` | Fresh tensor was already a 1.064 GiB float32 memmap. `values[order].astype(float32)` made two full heap arrays, **2.128 GiB** while both temporaries overlapped, even for identity order. | Identity order retains the same read-only float32 mapping. Reorder/cast writes another mapping in 512-row chunks; float32 gather â‰¤24.09 MiB. Existing exact-count checks remain, including finite/integer/float32-roundtrip audits. Their cast/check scratch can approach **128 MiB**, so it is included in notebook01 allowance. |
| Raw long extraction | `_read_long` retained every per-query frame and concatenated them; then feature, TRAIN and merged-check frames plus sort copies accumulated. For `D=long.memory_usage(deep=True).sum()`, long collection alone is roughly `2D` during concat; later extra terms depend on coverage/category density. **No defensible numeric D is available.** The512 cap is applied after extraction and does not bound R_long. | Spark aggregate rows stream in4096-row frames into indexed SQLite with a16MiB cache target and disk sorting. Python long payload is O(4096Ã—deep-row-bytes), not O(R_long). Coverage/key metadata remains O(Q), and Spark's connector/partition/JVM memory remains an unmeasured term. A pandas-returning reader already owns its full query result; the production Spark iterator avoids `.toPandas()`. |
| AutoML panel / assembly | `np.zeros` allocated 0.813 GiB float64 values; panel construction/validation copied wide frames. The grid join expanded to 1.060 GiB numeric bytes before float32 sequence conversion (0.530 GiB). | Same float64 values remain on disk to preserve panel hashes. Wide merging is replaced by an integer key index and2048-row gathers; one float64 block is8.00 MiB, its float32 conversion4.00 MiB. Sequence and summary are float32 mappings. Full key/coverage frames remain resident and must be counted separately. |
| `inputs.frame_hash` | Repeated panel validation/sorts and full Python row passes; long sort copies could be large. Hashing itself was incremental, not one giant JSON blob. Repetition inflated CPU and the lifetime of copied frames. | Canonical long-row bytes are sorted/read from disk once; panel hashing iterates2048-row blocks once and caches the digest keyed to validated immutable panel metadata/specification. Byte meaning is unchanged. Hash buffers are small; key metadata and2048-row numeric frames remain. |
| `modeling._fit_transform` | `sequence[valid].copy()` produced **two float32 observed arrays transiently**, up to1.494 GiB for count and762.00 MiB for AutoML. A per-feature float64 conversion followed. | No full observed tensor. One feature at a time uses at most8L bytes=1.49 MiB per vector; median/unique/sort/imputation/hash operations retain several such vectors (roughly12â€“24MiB numeric scratch at this population, distribution dependent). This preserves exact quantiles and source-order reductions. Large inputs first receive a feature-major float32 scratch mapping (F,N,T), filled in 512-row/64-feature blocks. It costs one additional TRAIN tensor on disk (0.747 GiB for count, 0.372 GiB for AutoML), then makes per-feature statistic reads contiguous instead of performing 1,028 strided passes through a row-major mapping. |
| `_transform_all` and split selection | Baseline already used one float64 feature at a time; the problem was the full float32 output plus an advanced-index selected output and pre-sliced raw split/fold. | Selected output writes directly into a C-contiguous float32 mapping,512rows at a time. One float64 column chunk is0.05 MiB. Source checks still inspect unselected columns. TRAIN/fold gathers go to mappings and release explicitly. Repeated transforms validate only requested rows, while continuing to check every source feature in those rows; the separate full-source input audit is unchanged. |
| `_proxy` / per-fold proxy models | Baseline already blocked64features, but across all TRAIN rows: **95.25 MiB per float64 temporal temporary**, several concurrent. Proxy output0.498 GiB stayed resident, including across CV. Fancy candidate matrices could add more copies. | Blocks are512rowsÃ—12stepsÃ—64features: **3.00 MiB each**. Full proxy and C-contiguous candidate designs are float64 mappings, preserving logistic solver values; no downcasting or reduced candidates. Full proxy is freed before CV and fold artifacts/designs are explicitly released. Native sklearn/BLAS working memory remains additional. |
| INPUTS compression, base64 and Spark transfer | Compressed NPZ `C` lived in BytesIO/bytes. `pack_artifacts` built full base64 string and all substrings: `C+2B`, with `B=4*ceil(C/3)`, **about3.67C payload bytes** before Spark serialization/native overhead. Readback collected every chunk and repeated decode. All raw arrays remained resident too. | NPZ is written to disk; hash reads1MiB blocks. Transfer batches at most 1,024Ã—50,000 base64 characters = **48.83 MiB payload** plus small encoding/string/row overhead. Numeric serialization may use NumPy's roughly16MiB block buffer. Readback uses bounded ordered connector requests and reconstructs archive/arrays on disk. Publication performs one full staging byte-hash readback, then atomic rename and metadata verification; it does not repeat a second full publication readback. No full compressed/base64/row-list triple. |
| Every-epoch TRAIN prediction | Already predicted in512-row batches; it was not a single full-TRAIN device forward. `_array` validation still gathered all observed float32 values, and `_predict` retained score chunks then concatenated them. | Same complete per-epoch TRAIN pass and512-row inference batches. `_array` checks512rowchunks; maximum fusion gather36.12 MiB plus boolean checks. Preallocated TRAIN scorevector=0.06 MiB instead of about0.12 MiB. Training batch256 remains unchanged; max fusion input batch=18.06 MiB, independent of N. Model/optimizer/activation/native workspace unchanged and unmeasured. |
| Eligible count correlation | Diagnostic matrix alone8Ã—TRAINÃ—eligible_count â‰¤127.50 MiB; fancy slicing, `np.cov` centering and previous proxies added concurrent copies. | All eligible rows retained. Centered float64 matrix is disk-backed, one feature at a time; covariance matrix â‰¤**8.06 MiB** (1,028Â²Ã—8), plus temporary covariance/boolean/eigensolver work. No diagnostic row sampling. |

## Before/after by notebook

These are **array-ledger scenarios and explicit numeric-workspace planning allowances**, not upper bounds on total driver RSS. Baseline INPUTS loading adds compressed/base64 decode overhead depending on C to each notebook. Larger key strings, source density or Spark/native allocations require extra memory in both versions.

| Notebook | Baseline allocation scenario | After: large storage and explicit numeric workspace |
|---|---|---|
| 01 tensor initialization | During panel assembly, two QÃ—F float64 panels + expanded12N panel + float32 sequence+summary total **3.260 GiB**, excluding raw longframes, key metadata and the count mapping. Count alignment could instead overlap **3.515 GiB** of heap arrays including retained raw panel/AutoML arrays. Artifact packing adds about3.67C. | Raw count+sequence+summary **1.638 GiB on disk**, panel **0.813 GiB on disk**, plus SQLite long/index spool and compressed archive C. Allow **256MiB** for explicit numeric/block buffers, then add resident key metadata/4096longrows, SQLite cache, Spark/runtime and mapped pages. |
| 02 split/preparation | Raw arrays + TRAIN count gather + two observed count copies already **3.879 GiB**. A representative later balanced CV-fold fit with retained full/fold proxies reaches **4.979 GiB** before per-feature/native/allocator costs. This is an illustrative ledger, not a bound: actual grouped fold sizes and eligible counts differ. | Raw inputs mapped; TRAIN/fold tensors, proxies and candidate matrices disk-backed. Largest count proxy0.498 GiB; candidate design storageâ‰¤that size across a partition. Allow **128MiB** explicit numeric scratch (512-row checks, per-feature exact statistics,3MiB proxy blocks,8.06MiB covariance and matrix temporaries), plus metadata, sklearn/BLAS/runtime and mapped pages. |
| 03 training | Raw arrays + largest TRAIN/VALIDATION fusion outputs total **2.998 GiB**. A noncontiguous TRAIN array's TensorDataset conversion can add another TRAIN tensor, giving **4.117 GiB**; arm assembly also has transient raw/transformed copies. These are cap scenarios; selected K may be smaller. | Raw inputs and current arm's TRAIN/VALIDATION outputs mapped (1.360 GiB max arm output on disk). Outputs are C-contiguous, so TensorDataset shares backing. Allow **128MiB** explicit host numeric scratch for validation/gathers,256-row training/512-row inference inputs and outputs; add unchanged PyTorch/optimizer/activations/native work. Arm mappings released before next arm. |
| 04 evaluation/reliance | Same **1.638 GiB** raw heap floor. Atâ‰¤1,867latest validation patients, four max-width subset/permutation buffers give a representative **2.152 GiB** total, plus full validation transform temporaries, models and artifact decode. | Only required validation subset transformed; base and perturbed inputs mapped. Distribution stats one feature at a time. Allow **128MiB** explicit numeric buffers, plus models, summaries/metadata, native inference work and mapped pages. |
| 05 comparison | Default validation bootstrap still loaded all raw tensors: **1.638 GiB** heap floor plus artifact decode, although predictions/metrics need little numeric memory. | Raw tensors mapped and validated in chunks; default validation scores/cluster bootstrap are small O(N_validation) vectors. Allow **128MiB** for explicit numeric/block payload buffers (including the 48.83 MiB artifact-read batch), plus metadata/model artifacts/metric overhead and mapped pages. An explicitly enabled TEST pass needs the same bounded transform/inference allowances as04; TEST remains disabled by default. |

The implementation loads the complete archive to **disk** and maps its members, then transforms only the requested split. It does not yet avoid downloading/decompressing unused splits because that would require altering the unchanged INPUTS artifact layout. Peak local-disk usage includes the compressed archive C, its uncompressed arrays, source panel/spool while01 runs, and current preparation/training intermediates. A deployment must check scratch-disk capacity; the report cannot supply a fixed GB requirement without C and the long-spool size.

## What this establishes and what remains unknown

1. The source has deterministic paths to multi-GiB overlapping heap arrays. These plausibly explain the observed driver OOM, but no driver log/process profile was supplied to identify the exact failing allocation. Fresh SQL scans/shuffles or Spark-to-Python transfer may also peak independently.
2. The revised code removes full resident NÃ—TÃ—F float64 intermediates and full artifact-encoding triples. The original baseline already used some feature blocking; this review does not claim those existing protections were newly added.
3. Memory usage now depends on bounded numeric chunks plus required metadata/native execution and reclaimable mapped pages. **Successful local synthetic tests do not prove a private Databricks run cannot OOM.** Real driver/executor specifications, free local disk and aggregate-only runtime peak telemetry remain needed for operational confirmation.
4. Long-row count/deep byte size, Q-key frame deep size, compressed C, Spark largest partition/JVM/shuffle memory, and PyTorch/native workspace are intentionally recorded as unknown, not guessed. An aggregate-only runtime receipt should record stage start/end/peak RSS, free disk, R_long and archive byte length without exporting patient data.
5. No scientific reduction was used: no patient subsampling for fitting, no reduced candidate grid/cap, no mixed precision, no changed seeds/epochs/batches/labels/splits, no change to proxy reduction order. Float64 panel/proxy storage remains where needed for exact hashes/solver parity. Full-row correlation is unchanged.


# Changed functions and classes

Each line describes a changed function/class relative to 2106cf1. Scientific configuration is unchanged; cohort selection and presentation follow the latest explicit instructions.

## build_notebooks.py
- `notebook_source`: removes Python comment tokens and rejects any executable-AST change.
- `build`: embeds existing modules, original-cohort expectations, scratch/RSS receipts and cleanup; emits only code cells with blank PATs and synchronized identities.
## client_universe.py
- `resolve_universe`: resolves the explicitly configured population name and fully qualified binding.
- `align_universe`: retains audits/stops; uses neutral cohort wording in errors.
- `assert_expected_population`: checks independently declared total/per-split snapshots, patients and positives before extraction.
- `load_client_counts`: applies configured population assertions and neutral provenance text before fresh extraction.
- `check_activity_alignment`: performs the same count/AutoML consistency check in bounded row blocks.
## fresh_counts.py
- `build_source_frames`: only changes an error's wording from client to cohort; query logic is unchanged.
- `POLICY.population_labels`: only changes descriptive wording; its source-contract identity consequently changes in the new run.
## inputs.py
- `disk_array`: allocates owned float32/float64 local mappings with cleanup.
- `_require_array_owner`: rejects attempts to close mapped views.
- `readonly_array`: safely transfers ownership to a read-only mapping.
- `release_array`: closes/removes only owned mappings, idempotently.
- `DiskPanel`: holds disk-backed numeric columns with validated immutable metadata and cached canonical digest.
- `validate_panel`: validates mapped panels in blocks while preserving the same checks.
- `canonical_row_bytes`: centralizes the unchanged canonical row encoding for both streamed and frame hashing.
- `frame_hash`: streams canonical bytes and reuses a valid immutable panel digest.
- `make_panel_manifest`: supports the mapped panel without another full frame materialization.
- `obtain_panel`: validates/reuses mapped inputs with the original source-contract requirements.
- `combine_panels`: handles mapped-panel inputs without weakening duplicate/key checks.
- `assemble_inputs`: maps historical request keys into preallocated float32 sequence/summary outputs in bounded blocks.
## raw_automl.py
- `_read_long`: yields bounded Spark iterator frames instead of retaining all query frames.
- `_spool_long`: validates and indexes canonical long rows in a disk SQLite spool.
- `_build_raw_automl_inputs`: selects the same TRAIN vocabulary and writes the same float64 panel values on disk.
- `build_raw_automl_inputs`: owns the spool lifetime and cleanup around extraction.
## modeling.py
- `_close_backing`: closes a mapping before removing its temporary file.
- `allocate_array`: uses bounded-memory owned mappings for large arrays.
- `release_array`: rejects views so cleanup cannot invalidate a sibling mapping.
- `take_rows`: gathers requested rows into preallocated outputs in blocks.
- `_array`: validates bounded chunks and requested split rows without changing full-source audits.
- `_fit_transform`: creates feature-major scratch for large TRAIN arrays, preserving observation order.
- `_fit_transform_featurewise`: computes identical statistics one feature at a time without full observed-tensor copies.
- `_transform_all`: writes selected transformed rows/features directly into float32 mappings.
- `transform_inputs`: passes row selection into the same transform implementation.
- `_proxy`: retains float64 descriptor arithmetic with both row and feature blocking.
- `_transformed_proxy`: bounds and releases intermediate transformed arrays.
- `_correlation`: retains all eligible TRAIN rows using disk-backed centering and bounded covariance storage.
- `_proxy_design`: writes candidate designs contiguously without full heap gathers.
- `prepare_features`: reuses the same selection/proxy/one-SE logic with mapped folds and explicit releases.
- `_predict`: retains all prediction passes and inference batches but preallocates the score vector.
- `distribution_diagnostics`: computes feature-wise diagnostics without wide float64 copies.
## workflow.py
- `save_inputs`: compresses NPZ to disk and streams it with cleanup on failures.
- `file_sha256`: hashes file bytes in bounded blocks without changing hash meaning.
- `_InputDirectory`: owns loaded mappings/files and releases them in Windows-safe order.
- `release_inputs`: closes stage inputs at their final use.
- `load_inputs`: downloads/validates the archive to disk and maps extracted arrays read-only.
- `_validate_input_arrays`: retains shape/value checks with row blocking.
- `attach_count_reference`: retains an aligned float32 count mapping or gathers into another mapping.
- `prepare_run`: passes row indices instead of copied full splits, retaining original protocol decisions.
- `arm_arrays`: creates only requested split/arm arrays with bounded assembly and exception cleanup.
- `train_registered`: releases each arm after fitting and explicitly rejects unsupported LightGBM paths.
- `compare_run`: preserves metrics/bootstrap/TEST gating and releases optional TEST transforms.
## reliance.py
- `run`: transforms only the required validation subset and safely releases base/perturbed mappings.
## temporal_selection/input_helpers.py
- `_blob_reader`: provides a context-managed reader for bytes or disk files.
- `artifact_signature`: computes the same content SHA256/length incrementally.
- `iter_artifact_rows`: streams the unchanged base64 chunk representation.
- `pack_artifacts`: retains the existing convenience API over the streaming encoder.
- `unpack_artifacts`: verifies ordered chunks and optionally reconstructs large payloads on disk.
- `_checked_table`: validates artifact identifiers used in connector queries.
- `_query_artifacts`: reuses the existing Snowflake reader for bounded metadata/chunk queries.
- `_read_artifact_rows`: queries bounded ranges and locally sorts each collected batch.
- `read_artifacts`: exposes bounded reads and optional disk reconstruction through the existing API.
- `_verify_saved_artifacts`: validates published bytes once without retaining the whole archive.
- `_artifact_ddl`: performs classic-Spark staging publication through its existing JVM connector.
- `_has_artifact_jvm`: detects direct-write fallback before any upload.
- `save_artifacts`: uploads 1,024-row batches, verifies content, and preserves immutable fail-closed publication in both modes.
