# Missing-input evidence collection

Run **[00_collect_missing_inputs.ipynb](00_collect_missing_inputs.ipynb)** in the approved Databricks environment after its existing Snowflake connection setup. It accepts `sf_options_dl_poc` or `sf_options` and embeds its runtime helpers. No repository imports or new credentials are needed.

This is an auxiliary discovery notebook requested for the evidence handoff. It does not replace the existing four model notebooks or the planned fifth comparison notebook.

## How to return results

1. Run the cells in order, starting at E00. Keep the optional full tensor value scan disabled on the first pass.
2. Send every printed `EVIDENCE_BEGIN` / `EVIDENCE_END` block, retaining its E-number, run ID and page number. Text is preferred; screenshots are acceptable. The 1,028-feature dictionary is paginated metadata.
3. Return failed/skipped sections too. A permission failure should not stop the remaining independent sections. Private connector errors are retained in `private_errors` for Genie to inspect locally; never print or export that dictionary.
4. Return E13's checklist and any source-backed answers for E12. Restart at E00 after changing source configuration so the results share one coherent run identity.

The notebook runs read-only SELECTs. It does not train, change source tables, create splits, calculate new TEST metrics, load model binaries, or export files. Bounded returned results do not limit warehouse scan cost. Cancel an unexpectedly slow section in the work interface and report its E-number.

## What still requires historical evidence or an owner

Only supply an answer where an existing audit/log cannot establish it; “unknown” or “unavailable” is valid.

- The practical compute/session time budget for final tuning.
- Which original 1,028-feature initializer/version actually ran, including its counting, mapping, cutoff and export order; whether that source is still recoverable.
- The historical V63 cutoff/claims-lag and negative-label follow-up rules, and the population used to fit feature selection and encoding parameters.
- Original LightGBM training, tuning and refit membership relative to the frozen Transformer split.
- Whether historical TEST results influenced feature selection, architecture, hyperparameters or thresholds. TEST has already been viewed; the question is whether it influenced decisions.
- If another holdout candidate is found, its previous use in fitting, selection, threshold setting, back-testing or other development. A date/key difference cannot prove it was unused.

Genie prompts are included before the relevant code cells. Ask for exact object/revision/run references, not a reconstruction from memory. Current source definitions, sample matches and code-generated flags do not certify historical execution or leakage safety.

The evidence packet contains metadata and aggregate reports only. Keep executed notebooks and private outputs in the work environment; do not commit them. Missing history must remain unavailable rather than being converted into a fabricated metric or a passed audit.
