# TAK861 progress review, 17 November 2026

The 16-slide PowerPoint summarizes the current development comparison, seed variation, training-versus-validation gaps, the separate timing-feature ablation, production score coverage, the next experiments, and historical results. It contains five editable charts and eight editable tables.

`evidence.json` preserves the aggregate values transcribed from the supplied outputs. Current clinical results are screenshot-reported and were not rerun locally. All four current main-comparison arms used validation for model selection. The notebook 06 timing comparison uses a separate fixed-budget static control. Historical cohorts are shown separately. The production row has no score on the current validation population.

The Med-BERT slide describes a bounded monthly code-token adaptation, not use of the original pretrained weights. The official repository states that the pretrained weights are not shareable: https://github.com/ZhiGroup/Med-BERT#sharing-pre-trained-model. No Med-BERT clinical result is claimed.

## Rebuild

Use the bundled Node.js runtime and `@oai/artifact-tool`. Copy `build_progress.mjs` and `evidence.json` into a private build directory, and link that directory's `node_modules` to the bundled packages. Set `RUNTIME_NODE_MODULES`, `RUNTIME_PYTHON`, and `PRESENTATIONS_SKILL_DIR` to the installed runtime and presentation skill paths, then run the builder with the bundled Node.js executable. Set `DECK_FILENAME` to a new output filename for each revision. The builder writes previews, exports the draft, materializes workbook snapshots for the native charts, and runs the presentation finalizer.

The reviewed output is copied unchanged into `archive/presentations/TAK861_Progress_20261117.pptx`. Private render images and validation receipts remain in the workspace build directory.

## Evidence scopes

- Current main comparison: `R20261030_NEW_SUMMARY`, 67,969 validation snapshots and 1,141 positive snapshots. Notebook 05 control Lift@10% is 2.532829.
- Timing ablation: same validation keys, separate fixed 1,000-tree TRAIN-only protocol. Static control Lift@10% is 2.059567. Static plus time is 2.445188.
- Original validation: 3,481 snapshots and 202 positives. Separate from the expanded cohort.
- Original retrospective TEST: 3,414 snapshots and 202 positives. Previously inspected, with historical upstream feature-selection limitations.
- Production: all 67,969 current exact keys lack frozen scores. No numeric production comparison is available.

Speaker notes contain additional interpretation and source context. Metrics are snapshot-based; unique-patient deployment decisions require corresponding patient-level reporting.
