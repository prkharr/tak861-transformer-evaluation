# Temporal feature selection: four notebooks

Run these in order on the approved Databricks compute. Paste your private
`sf_options` connection dictionary into cell 1 of each notebook. Required
packages are PyTorch, NumPy, pandas, scikit-learn and matplotlib, plus the Spark
Snowflake connector. All application helpers are embedded in each notebook;
no additional code upload is needed.

1. **01_tensor_initialization:** load and verify the saved original tensor,
   **23,151 snapshots × 12 monthly timesteps × 1,028 features**. It reuses
   FEATURE_MAP, SNAPSHOTS and TENSOR_MONTHLY. It does not rebuild raw claims.
2. **02_patient_level_split:** preserve the original patient split and index
   dates. Within TRAIN only, create mutually exclusive fitting/stopping/ranking
   patient groups in 60%/20%/20% proportions. These internal groups do not
   replace the original train/validation/test assignments.
3. **03_transformer_training:** fit an auxiliary copy of the original temporal
   encoder on internal fitting patients; select its checkpoint on separate
   internal stopping patients. Rank all 1,028 features on the separate ranking
   patients using whole-sequence permutation. Freeze the top 50, then train a
   fresh temporal encoder on all original TRAIN snapshots using only those
   features. Select its checkpoint and threshold on original VALIDATION.
4. **04_transformer_evaluation:** restore exactly that feature order, checkpoint
   and threshold. Report TRAIN/VALIDATION/TEST lift, full deciles, top-5/10/20/30%
   targeting metrics, AP, AUC, precision, recall, F1 and confusion counts.

The final input is **23,151 × 12 × 50**, including:

| Split | Shape |
|---|---|
| TRAIN | 16,256 × 12 × 50 |
| VALIDATION | 3,481 × 12 × 50 |
| TEST | 3,414 × 12 × 50 |

These 50 inputs are selected original monthly claim-category features. They
are not the 49 snapshot-summary features from the prior experiment. Time step
0 remains newest and step 11 oldest. Empty months remain valid all-zero input.

## Ranking definition

For ranking only, take the latest snapshot for each internal ranking patient.
This gives one independent patient unit per permutation. A whole 12-month
feature sequence is exchanged between different patients with the **same
exact END_DT**. Months inside that sequence are never shuffled. No claims
history is invented and the source tensor is never changed.

Cutoff groups with only one ranking patient cannot be permuted and are excluded
from ranking. The notebook reports retained patients/positives and stops if
fewer than 80% of ranking patients remain, fewer than 20 snapshots remain,
or either outcome class is absent. Full original populations remain in final
training and evaluation. The ranking reflects this latest-snapshot cohort,
not necessarily every earlier snapshot equally.

Importance = unshuffled top-10% lift minus shuffled top-10% lift, averaged over
five repeats. Save the mean, standard deviation and each repeat. Average AP
drop breaks ties, followed by original feature index. Model inputs retain the
original vocabulary order. The top 50 count is declared before ranking; the
notebook does not search different counts against TEST. It reports selected
features whose measured lift drop is zero or negative; top 50 does not mean
all 50 are proven useful. Correlated features can mask each other's importance.
The repeat standard deviation is not a population confidence interval.

## Preserve the original training setup

Both fits use the original temporal architecture: learned monthly positions,
128-wide projection, 4 attention heads, 2 encoder layers, feedforward 256,
dropout 0.2 and mean pooling. Raw counts receive log1p once per batch.
AdamW uses learning rate 0.001, weight decay 0.0001, batch size 64 and gradient
clipping 1.0. Seed is 42; maximum epochs 20, patience 5 and minimum AP improvement
0.0001. Class weights use only each fitting population. Checkpoints retain
original validation-AP selection; the classification threshold maximizes
validation F1. Lift is reported every epoch in addition to AP and loss.

The ranking model sees no original VALIDATION or TEST during fitting,
checkpoint choice or ranking. Original VALIDATION is used only for the final
reduced model. TEST is scored only in notebook 04. Existing TEST was previously
inspected, so this remains retrospective evaluation. A new untouched cohort
is needed for independent confirmation.

## Runtime and resuming

Ranking requires **5,140 holdout inference passes** plus one unshuffled pass,
so it can take substantially longer than one training run. GPU is recommended;
no duration or lift improvement is guaranteed. Completed feature batches are
saved every 32 features. An interrupted batch is recalculated; completed
matching batches and fitted models are reused when rerunning notebook 03.

The driver needs approximately 1.06 GiB of temporary disk for the original
float32 tensor plus working memory. Feature slicing is per snapshot; no full
reduced tensor copy is required.

Outputs use `TAK861_TX_READY_V63_DL_POC_TEMPORAL_SELECTION_V1`, run `P001`.
Keep RUN_ID and PLAN_SEED identical across all four notebooks. Change RUN_ID
before changing ranking/training settings or notebook code. The original
RUN_001 artifacts and all original source tables are read only. Exact input
fingerprints, model contracts and saved feature indices are verified on reload.

Use the same split and targeting fraction when comparing with the original
model. Feature selection may improve generalization, but may also remove
useful interactions; it does not promise lift 6 or eliminate overfitting.
