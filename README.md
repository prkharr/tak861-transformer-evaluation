# TAK861 deep learning proof of concept

**Start with the [final challenger run guide](notebooks/final_challenger/README.md).**
The development challenger has **not yet been run on project data**. Its results are pending; it is not a validated winner or a production release.

## Choose the right route

| Folder | Purpose | Status / when to use |
| --- | --- | --- |
| [Final challenger](notebooks/final_challenger/README.md) | Audit existing inputs, tune models and blend, then report validation results | Next experiment; run 01 → 02 → 03. Reuse existing tensors. |
| [Established experiments](notebooks/established_experiments/README.md) | Input preparation and the count, static fusion, NLP fusion and timing comparisons | Existing development evidence; do not rebuild inputs merely to run the challenger. |
| [Utilities](notebooks/utilities/README.md) | Optional feature selection and experiment tracking | Not prerequisites for the challenger. |
| [Experiment summary](reports/experiment_summary.md) | Results, conclusions and pending work | Aggregate findings only; no patient data. |
| [Latest presentation](reports/presentations/TAK861_Progress_20261124.pptx) | Stakeholder report dated 24 November 2026 | Business implications, implementation options and comparison limitations. |
| [Archive](archive/README.md) | Earlier notebook routes and superseded presentations | Historical reference, not the default run path. |

There are **11 active notebooks**: six established experiments, three challenger stages and two optional utilities. Older notebooks remain available in the archive and Git history.

## Before running

- Use the approved private Databricks/Snowflake environment. Keep credentials, tensors, row-level predictions and model artifacts out of Git.
- Fill the seven blank `sf_options` values privately. For the challenger, preserve parent `RUN_ID='R20261030_NEW'`, `RUN_MODE='SUMMARY'` and the input `CONFIG`.
- The challenger is an overnight, resumable experiment, not a one-hour demo. Its default training budget is eight hours per invocation; input loading and blocking operations can add time.
- Current validation has informed development. TEST scoring and production-lift verification remain separate activities; no production superiority has been established.

## Repository maintenance

`experiments/` contains reproducible builders, supporting modules and tests. `src/`, `tests/` and `scripts/` retain shared and historical support. Some active notebooks reuse helpers from older experiment packages; do not delete those packages solely because their notebooks are archived.

Use the [migration index](archive/path_index.md) to find a notebook by its old name. Published notebook contents and their embedded source identities were preserved during the reorganisation. Historical release registries retain their original Git paths because they address pinned commits.

Validate the layout and notebook contracts with:

```powershell
python scripts/validate_repository_layout.py
python -m unittest experiments.evidence_challenger.test_pipeline -v
```

The synthetic pipeline tests require the approved scientific Python dependencies (including PyTorch and LightGBM). They do not establish performance on project data. See the [cleanup validation record](reports/repository_cleanup_validation.json) for checks performed during this migration.
