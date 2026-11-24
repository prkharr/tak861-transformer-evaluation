# Established expanded-cohort experiments

These are the canonical notebooks behind the reported development comparisons. Their file contents are unchanged by the repository cleanup.

| Order | Notebook | Purpose |
| --- | --- | --- |
| 01 | [Tensor initialisation](01_tensor_initialization_AutoML.ipynb) | Build or reuse aligned count, AutoML and NLP inputs. |
| 02 | [Patient splits and preparation](02_patient_level_split_AutoML.ipynb) | Preserve patient partitions and save TRAIN feature preparation. |
| 03 | [Transformer training](03_transformer_training_AutoML.ipynb) | COUNT, STATIC_FUSION and NLP_FUSION fits. |
| 04 | [Transformer evaluation](04_transformer_evaluation_AutoML.ipynb) | Development validation and training-gap diagnostics. |
| 05 | [Model comparison](05_model_comparison_AutoML.ipynb) | Static LightGBM comparison and production-score comparability audit. |
| 06 | [Time-aware LightGBM](06_time_aware_lightgbm_AutoML.ipynb) | Matched fixed-budget static versus timing experiment. |

All connection fields remain blank. These are reproducibility notebooks, not an instruction to rebuild completed tensors. The [next challenger](../final_challenger/README.md) reuses their saved inputs and preparation. It does not require repeating the six stages.

Results and limitations are in the [experiment summary](../../reports/experiment_summary.md). Production lift is not verified by these development comparisons.
