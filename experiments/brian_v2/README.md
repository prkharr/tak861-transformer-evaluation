# Controlled feature and regularization experiments

The four generated notebooks are in `notebooks/Brian Experiments`. Start with that folder's `START_HERE.md`.

This version preserves the completed V63 input tables and patient split. It tests a TRAIN-selected claims subset, separate dropout/weight-decay changes, a smaller Transformer, and regularized logistic regression. The true client engineered-feature comparison remains pending exact definitions; the slide's 150 custom features are not equivalent to the `reduced150` claims subset. The stated 90-day business target is recorded without claiming independent verification of existing labels.

## Rebuild

From the repository root, using an environment with the documented dependencies:

```text
python experiments/brian_v2/build_experiment_notebooks.py
python -m pytest experiments/brian_v2 -q
```

The builder embeds these reviewed helper modules and the existing `src/model.py`, `src/metrics.py`, and `src/reproducibility.py` in self-contained notebooks. Databricks users do not need to install this repository. Editing generated notebooks alone is not the maintained source of truth.

## Components

- `early_stages.py`: validates saved tensors and patient assignments against the original RUN_001 fingerprints; never reconstructs unknown raw-claims SQL.
- `input_helpers.py`: frozen input identity, Decimal-safe labels, memory-mapped tensor loading and private Snowflake artifact storage.
- `feature_selection.py`: TRAIN-only temporal-window screening and immutable feature manifest.
- `experiment_training.py`: Transformer experiments and a numeric-only logistic checkpoint format, with comparable diagnostic loss and TRAIN/VALIDATION targeting metrics.
- `experiment_reporting.py`: controlled recipes, saved-run consistency checks and validation comparison.
- `evaluation_helpers.py`: snapshot-level classification, deciles, lift, gains, tie handling and charts.
- `build_experiment_notebooks.py`: builds the four eight-cell notebooks.

New artifact writes use `TAK861_TX_READY_V63_DL_POC_BRIAN_V2`; the original tables and RUN_001 artifacts are read-only dependencies. Every candidate is saved before the next is trained. The selected-run record is frozen on validation AP before TEST inference.

Local verification used Python 3.9.7, PyTorch 2.8.0 CPU and scikit-learn 1.6.1. No private credentials, source records or live Snowflake session were used. See the delivered `VALIDATION.md` for scope and limitations.
