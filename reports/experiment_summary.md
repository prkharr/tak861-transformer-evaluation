# Experiment summary — 24 November 2026

These are established development observations, not new runs performed during repository cleanup. Current comparison population: 67,969 validation snapshots, 3,919 patients and 1,141 positive snapshots. A snapshot is one patient at one date.

| Experiment | Validation Lift@10% | Useful conclusion |
| --- | ---: | --- |
| Static LightGBM, validation early stopping | 2.532829 | Tabular development control. |
| COUNT Transformer | 2.734403 | Count history provides a useful sequence reference. |
| STATIC_FUSION | 2.786988 | Modest gain over the count reference. |
| NLP_FUSION | 2.953506 | Strongest current top-10% result; 337 captures versus 289 for static LightGBM at the same 6,797 selected rows. |
| Fixed-budget static LightGBM | 2.059567 | Matched control for the timing ablation; not the early-stopped control above. |
| Fixed-budget static + timing | 2.445188 | Timing added 44 captures (235 → 279), but did not exceed the separate stronger static control. |
| Final tuned/blended challenger | **Pending** | Not yet run on project data; no improvement claimed. |

NLP versus static LightGBM: +0.420677 lift, approximately +16.6%; reported paired interval +0.009027 to +0.801666. These are additional positive **snapshots**, not necessarily additional unique patients. Validation selected the incumbent checkpoints, so the result is development evidence.

Static fusion leads slightly at 20% and 30% selection capacity; the top-10% ordering is not universal. Timing needs comparison under a common tuning protocol. Training-to-validation lift gaps and early validation peaks warrant monitoring; they do not establish performance on a new population.

The historical production lift of about 4.59 is not directly comparable yet. An original-cohort TRAIN-only 49-feature refit was about 2.52, but that does not establish the cause of the gap. Exact-key production score coverage and training/evaluation provenance remain unresolved. Production verification and independent evaluation are separate future activities.

The latest [stakeholder presentation](presentations/TAK861_Progress_20261124.pptx) documents these results and the implementation proposals. Its [evidence index](../docs/presentations/progress_20261124/evidence.json) links the supporting implementation and earlier evidence records. Historical records remain in the [archive](../archive/README.md).
