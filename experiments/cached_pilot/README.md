# Cached-feature comparison and experiment tracker

The two notebooks work from verified completed extraction chunks. They do not require the full 90-chunk extraction or notebooks 01–06 to finish. This is a separate development pilot: it does not change the full experiment or establish superiority over production.

## Files

- `notebooks/expanded cohort/07_cached_feature_comparison_AutoML.ipynb`: AutoML versus AutoML plus nine timing features first; two NLP arms follow only when matching saved NLP artifacts pass verification.
- `notebooks/expanded cohort/08_experiment_tracker_AutoML.ipynb`: aggregate evidence tables and charts; no training or raw-claim extraction.
- `experiments/cached_pilot/HANDOFF_CELL.py`: standalone code to paste into a new cell in notebook 01 after gracefully stopping its extraction cell.
- `experiments/cached_pilot/GENIE_PROMPT.txt`: execution instructions.

The notebook files embed their implementation. They do not require a repository checkout at runtime. `HANDOFF_CELL.py` must be available to perform the initial handoff.

## Freeze completed work and resume extraction

1. Gracefully interrupt the active AutoML extraction cell. Wait for it to stop. Keep the kernel, `counts`, `CONFIG`, `RUN_PREFIX`, `IMPLEMENTATION_SHA256` and `SCRATCH_DIRECTORY` intact. Do not restart the kernel, delete checkpoint files or rerun notebooks 01–06.
2. Inspect the actual running extraction call and its `checkpoint_directory`. The standard builder uses this exact formula:

   ```python
   from pathlib import Path
   candidate = Path(SCRATCH_DIRECTORY) / (RUN_PREFIX + "_RAW_CHECKPOINT") / "raw.sqlite"
   if not candidate.is_file():
       raise FileNotFoundError("The configured extraction checkpoint does not exist")
   print(str(candidate))
   ```

   If the running cell has been changed, follow its actual expression. Do not guess a timestamped directory or select an arbitrary `raw.sqlite`. Verify read-only checkpoint `state.identity.run_identity == RUN_PREFIX`; the handoff also verifies source identity, requested keys, completed-chunk hashes and content. Do not print credentials or patient rows.
3. Paste `HANDOFF_CELL.py` into a new cell. Set `RAW_SQLITE_PATH` to the verified path. Keep `HANDOFF_ROOT` separate from the original checkpoint directory. `CHUNK_LIMIT=None` freezes all fully committed chunks available at handoff; unfinished work is not exported. Keep `PREVIOUS_CACHE_DIRECTORY=""` for the first version.
4. The count subset uses exactly one sleep-study PX channel and one sleep-wake DX channel. Automatic resolution must be unambiguous. If it stops, inspect `counts["features"]` and set both `EXACT_COUNT_CHANNELS` entries to the exact dictionary names; do not substitute similar categories.
5. Run the handoff cell. It copies only two monthly count channels and exports verified raw-feature chunks through bounded read-only SQLite queries. It never copies a live SQLite file or changes source checkpoints. Retain the printed `handoff.json` path.
6. Resume the original extraction cell immediately with its existing configuration and checkpoints. The initial export takes time proportional to committed content; it is not an instantaneous filesystem copy. Do not repeatedly interrupt extraction to inspect progress.

The default paths are under `/local_disk0/tmp/tak861_cached_pilot/`. They are driver-local and may disappear if the cluster is replaced. Notebook 07 must use a separate notebook session on the same driver, or storage actually accessible from both sessions. A different cluster cannot read another driver's local path. Concurrent sessions still share driver memory and CPU.

## Run notebook 07

Set `HANDOFF_MANIFEST` to the printed path and choose a separate `PILOT_OUTPUT` directory. Leave `PREVIOUS_PREPARED=""` for the first pilot. Connection values ship empty. The base AutoML/timing pair requires no Snowflake connection; populate `sf_options` privately only to request the optional saved NLP package. Never print or save those values.

Check that `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `psutil` and `matplotlib` are available in the analysis session before fitting. Do not change packages in the actively extracting session. The pilot runs one model fit at a time, two CPU threads, seed 42 and 300 fixed trees. It uses the existing patient-weighting and class-balancing conventions. No validation early stopping or tuning is performed.

Only complete patients in the frozen chunk prefix enter the pilot; their original TRAIN/VALIDATION membership is preserved and TEST is unused. The prefix is not a random sample. All arms use identical training and evaluation rows. Timing features use the saved two-channel twelve-position histories. NLP uses monthly TF-IDF means and active-document fraction only after matching metadata, cutoff and TRAIN-fitting provenance are verified; its encoder may have been fitted on the full parent TRAIN population. If NLP is unavailable or mismatched, retain the completed two-arm result and report the NLP arms as blocked.

The memory guard stops analysis when the configured free-memory reserve is insufficient. Leave extraction running and resume analysis later; do not disable the guard to force both through. Each completed fit and prediction artifact is saved with its receipt. After interruption, rerun with the same handoff and output path to reuse valid completed work. Memory checks are not a guarantee against another process suddenly consuming memory.

Outputs include `prepared.json`, per-arm models/predictions/receipts, `report.json`, `report_receipt.json` and `lift_diagnostics.png`. Compare validation Lift@10%, positive captures, gained/lost/shared positives and paired patient-cluster intervals against AutoML. TRAIN lift is an overfitting diagnostic. Historical production scores are not included in the pilot ranking. Measure runtime from the first export and fit; no completion time is guaranteed.

## Refresh and track

For a later handoff, use a new `HANDOFF_ROOT` and set `PREVIOUS_CACHE_DIRECTORY` to the prior handoff's `cache` directory. The exporter reuses verified completed chunks and adds newly verified ones. Use a new `PILOT_OUTPUT`; set `PREVIOUS_PREPARED` to the earlier result's `prepared.json`. The initial validation panel, feature definitions and pilot settings stay fixed while newly complete TRAIN patients can be added. Preserve all earlier versions.

In notebook 08, set `LOCAL_REPORT_PATHS` to explicit pilot `report.json` paths. It verifies the adjacent `report_receipt.json`. `REMOTE_REPORT_TABLES` is optional and must name explicit saved aggregate-report tables; the bounded reader loads `report.json` only, not predictions or clinical records. Leave connection values empty when using local reports. Set `OUTPUT_DIRECTORY` to save aggregate CSV/JSON exports.

The tracker separates screenshot-reported evidence, recorded historical findings, checked current report artifacts, pending experiments and engineering improvements. A checksum verifies file integrity, not freedom from leakage. It makes charts only within recorded matched comparisons and fixed-validation refresh groups, with no overall ranking across incompatible populations or metrics.

## Local verification

Synthetic tests cover committed-chunk integrity, key alignment, complete-patient filtering, patient separation, TRAIN-only selection, feature parity, fixed-panel refresh, failed-fit resume, report checksums and metric arithmetic. Notebook 07's six embedded cells were exercised with synthetic sources and real fixed-tree LightGBM fits, including restart without refitting. Notebook 08's six cells and matched charts were also executed. These checks do not replace execution and source-lineage verification in the data environment.
