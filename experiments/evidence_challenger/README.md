# Evidence-based development challenger

Run these independent Databricks notebooks in order:

1. `notebooks/expanded cohort/11_challenger_evidence_and_input_audit.ipynb`
2. `notebooks/expanded cohort/12_challenger_tuning_and_blend.ipynb`
3. `notebooks/expanded cohort/13_challenger_validation_report.ipynb`

Fill the seven blank `sf_options` entries with the existing private connection in each notebook. Keep the existing parent `RUN_ID='R20261030_NEW'`, `RUN_MODE='SUMMARY'` and `CONFIG` unchanged. These values identify the completed immutable tensor artifacts; they do not request rebuilding the cohort. A fresh copy of each notebook is standalone and contains its dependencies as checked embedded source.

Use an approved Databricks runtime containing PyTorch, LightGBM, NumPy, pandas, scikit-learn and matplotlib, with the existing Snowflake Spark connector. Classic Spark compute with JVM access is required for verified atomic artifact publication; shared/serverless Spark Connect without JVM access is rejected. Local synthetic validation used LightGBM 4.6.0. A GPU is strongly recommended for the neural search; `device='auto'` resolves and records the actual device before creating the run identity. CPU may need several overnight resumptions. The full experiment is not a one-hour demonstration. Its default eight-hour budget starts after input loading and checks between training iterations and batches; blocking feature-selection and warehouse operations may exceed the target.

Notebook 11 checks frozen input identities, patient/snapshot separation, class support, raw AutoML provenance, monthly orientation and the two timing channels. Blank channel overrides auto-resolve only an unambiguous match. If a channel is ambiguous, supply its exact dictionary name in `EXACT_CHANNELS` consistently across notebooks.

Notebook 12 searches 12 LightGBM configurations for each of the static and timing arms, and 8 NLP fusion configurations. Three patient-grouped outer TRAIN folds each use an inner patient-separated 80/20 split. Supervised feature selection is refitted inside each inner training set and again on each outer training set. It uses the existing grouped logistic proxy and feature-size grid, not the original full-TRAIN selected feature list. Fixed-round refits never use the external validation labels for stopping. Final configurations maximize mean inner Lift@10%; median inner rounds determine the final fit length. Final fits use seeds 42, 142 and 242.

The NLP input remains the existing, label-free, all-TRAIN TF-IDF representation. Thus OOF predictions are conditional on that fixed representation; vocabulary and IDF are **not** refitted inside each outer fold. The report makes this limitation explicit. This is a reuse-of-tensors development experiment, not a fully inductive representation-validation study.

Blend weights run from 0 to 1 in increments of 0.05 and are selected only from TRAIN OOF predictions. Endpoints retain the individual parents. The blend combines timing LightGBM and NLP fusion. Static LightGBM is retained as a matched control. OOF uses each fold's locally selected configuration at seed 42; final predictions average three seeds using the configuration chosen by mean inner lift. Transfer of the blend weight between those predictors is explicitly a development hypothesis. The neural challenger uses the same architecture with a simplified fixed-learning-rate AdamW schedule. There is no guarantee the blend or tuned model improves over the saved NLP incumbent.

Completed preparation, candidate trials, outer predictions and final seed fits are persisted in private warehouse artifact tables. Each table has one `bundle.zip` artifact containing its full payload and checksum receipt, compatible with the warehouse reader's exact-name-set contract. Every stage is keyed by source, parent input/protocol, modeling settings, device and library versions. Row identities are checked on resume. An interrupted fit is restarted; completed stages are reused. Rerun notebook 12 unchanged to resume; changing only `seconds` preserves completed work. Changing modeling settings or runtime versions creates a separate experiment to avoid mixing protocols. Publication uses a verified temporary table followed by atomic rename. Corrupt published artifacts fail closed with the affected table name; they are never silently overwritten. A report marked COMPLETE is written only after all prescribed folds, candidates and seeds finish. Copy its printed `CHALLENGER_REPORT_TABLE` into notebook 13.

Notebook 13 reads saved predictions without loading tensors or refitting. It reports Lift@10/20/30, positive snapshot captures, distinct positive patients among those captures, AP, ROC-AUC, log loss, Brier score, overlap with the NLP incumbent, seed gaps and runtime. It computes 2,000 paired patient-cluster bootstrap draws for lift differences versus the frozen NLP incumbent, versus the saved static LightGBM when available, and for tuned timing versus tuned static LightGBM. It saves the report separately. Unique-patient counts are descriptive; they are not a new unique-patient ranking policy.

The existing validation set has already informed development. Bootstrap intervals condition on fitted predictions and exclude tuning/retraining uncertainty. OOF scores used to optimize blend weights are also not an independent performance estimate. Production verification and TEST scoring remain disabled. The frozen input loader may read the existing archive containing all partitions; only TRAIN and validation indices are used by the challenger.

The current raw-input provenance and reconstructed labels retain the limitations documented by the parent experiment. This notebook does not alter therapy definitions, eligibility, follow-up rules or labels. The proposed model improvements do not resolve those upstream questions.

Local validation covers real LightGBM/PyTorch training on synthetic arrays, nested patient partitions, full artifact save/resume, reporting, corruption rejection, duplicate snapshot rejection, blend endpoints and deadline behavior. It does not establish runtime or clinical performance on the private Databricks dataset.

Rebuild: `python -m experiments.evidence_challenger.build_notebooks`.
Tests: `python -m unittest experiments.evidence_challenger.test_pipeline -v`.
