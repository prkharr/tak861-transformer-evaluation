"""Build a separate five-stage development pipeline without replacing prior notebooks."""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

from experiments.automl_sequence import build_notebooks as base

HERE = Path(__file__).resolve().parent
DEST = base.REPO / "notebooks" / "Fresh Cohort AutoML NLP"


def cleaned(text):
    return base.notebook_source(text).replace("client", "cohort").replace("Client", "Cohort")


def assigned(source, name):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body
                if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets))


def replace_assignment(source, name, value):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            node.value = ast.parse(repr(value), mode="eval").body
    return ast.unparse(tree)+"\n"


def build():
    recipes = base.build(write=False)
    recipes = {name: [copy.deepcopy(c) for c in cells if c["cell_type"] == "code"]
               for name, cells in recipes.items()}
    first = next(iter(recipes.values()))
    config_source, helper = ["".join(c["source"]) for c in first[:2]]
    config = assigned(config_source, "CONFIG")
    config["client_universe"].update(table="DSVC_TAKEDA_TA_PRIVATE.TAK861.TAK861_TX_READY_PATIENT_UNIVERSE_DL",
                                     table_name="TAK861_TX_READY_PATIENT_UNIVERSE_DL", expected_population=None)
    config["cohort_evidence"] = dict(producer_sha256=None,
        lineage_status="bounded_reconstruction_development", candidate_labels_accepted=False, automatic_relabel=True,
        eligibility_reference="Supplied 2026-10-30 rule sheet: qualifying NT1 claim dates at least 90 days apart; generic exposure 135 days within 270 days; max gap 270 days. Existing table implementation unverified.",
        first_initiation_reference="First IS_GENERIC=FALSE fill in all available designated basket history; recompute every eligible snapshot in days1..90 without manual drug classification.",
        negative_followup_reference="Supplied rule sheet requires observable medical activity during the 90-day outcome; existing table implementation and completeness date unverified.",
        effective_cutoff_reference="Rule sheet INDEX_DATE minus 40 days; aggregate audit says stored END_DT already includes this adjustment. Do not subtract again.",
        source_snapshot_reference="Candidate table as delivered; aggregate audit dated 2026-10-30. Source values and population fingerprint recorded at extraction; historical availability unverified.",
        retention_rule="all_eligible_preinitiation_snapshots", outcome_days=90,
        outcome_boundary="cutoff < first_event <= cutoff + 90 days", historical_availability="retrospective_latest_sources",
        previously_reported_label_discrepancies=223,
        unresolved_rules=["Reconstructed labels audited against the candidate; original producer unavailable",
            "Intended advanced-therapy basket versus IS_GENERIC=FALSE; no manual drug exclusions applied",
            "Paid/final versus rejected/reversed source parity and treatment initiation",
            "Original producer, diagnosis flags, prior advanced treatment exclusion and first-event history",
            "Coverage, second-claim and MONTHS_SINCE_FIRST_NT1 implementation versus supplied rule specification",
            "Negative follow-up observability and source completeness date",
            "Up to16 missing original keys explicitly accepted; exact set and bounded-window exclusions frozen in audit",
            "NO_MDD versus main basket equality verified only by reported group totals"])
    config['cohort_plan'] = dict(months=48, max_missing_original_snapshots=16,
        candidate='DSVC_TAKEDA_TA_PRIVATE.TAK861.TAK861_TX_READY_PATIENT_UNIVERSE_DL',
        basket='DSVC_TAKEDA_TA_PRIVATE.TAK861.NARCOLEPSY_MARKET_BASKET_CLAIMS_NO_MDD',
        medical='DSVC_TAKEDA_FULLMAP_PLAID_PROD.COHORT_1009719.MEDICAL_EVENTS_LATEST',
        end_policy='latest_candidate_cutoff_with_90_days_inside_observed_source_date_range')
    config["include_nlp"] = True
    config["nlp"] = dict(medical="DSVC_TAKEDA_FULLMAP_PLAID_PROD.COHORT_1009719.MEDICAL_EVENTS_LATEST",
        reference="DSVC_TAKEDA_TA_PRIVATE.DS_ML_PRD.DX_PX_RX_PLAID", event_id_column="MEDICAL_EVENT_ID",
        max_terms=256, chunk_size=5000, max_source_rows_per_chunk=250000)
    config_source = replace_assignment(config_source, "CONFIG", config)
    config_source = replace_assignment(config_source, "RUN_ID", "R20261030_NEW")
    config_source += '''
RUN_MODE = "SUMMARY"
if RUN_MODE not in ("SUMMARY", "TEMPORAL"):
    raise ValueError("RUN_MODE must be SUMMARY or TEMPORAL")
RUN_ID = RUN_ID + "_" + RUN_MODE
CONFIG["experiment_phase"] = RUN_MODE
CONFIG["raw_automl"]["request_steps"] = 1 if RUN_MODE == "SUMMARY" else 12
print("Development phase:", RUN_MODE, "AutoML anchors:", CONFIG["raw_automl"]["request_steps"])
print("Compare within this phase only. Changing phase changes extraction and selection scope.")
print("48-month candidate snapshot window; first-initiation labels reconstructed for every eligible days1..90 window.")
print("Development results: original eligibility producer and source completeness remain unverified.")
'''
    sources = assigned(helper, "EMBEDDED_SOURCES")
    for name in ("pipeline", "nlp_features", "cohort_binding", "plots"):
        path = HERE/(name+".py")
        if not path.exists():
            raise RuntimeError("Implementation pending: " + name)
        sources[name] = path.read_text(encoding="utf-8-sig")
    sources["fresh_lgbm"] = (base.HERE/"fresh_lgbm.py").read_text(encoding="utf-8-sig")
    sources = {name.replace("client", "cohort"): cleaned(source) for name, source in sources.items()}
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    implementation = hashlib.sha256(
        (source_hash+Path(__file__).read_text(encoding="utf-8")+Path(base.__file__).read_text(encoding="utf-8")).encode()).hexdigest()
    helper = cleaned(helper)
    helper = replace_assignment(helper, "EMBEDDED_SOURCES", sources)
    helper = replace_assignment(helper, "EMBEDDED_SOURCE_SHA256", source_hash)
    helper = replace_assignment(helper, "IMPLEMENTATION_SHA256", implementation)
    helper += '''
fp, nlp, fresh_lgbm, binding = [modules[n] for n in ("pipeline", "nlp_features", "fresh_lgbm", "cohort_binding")]
plots = modules["plots"]
def write_population(table, metadata):
    small = metadata[["PATIENT_ID", "END_DT", "RESP"]].copy()
    small["PATIENT_ID"] = small["PATIENT_ID"].astype(object)
    small["END_DT"] = small["END_DT"].dt.date
    small["RESP"] = small["RESP"].astype("int32")
    spark.createDataFrame(small, "PATIENT_ID string, END_DT date, RESP integer").write.format("snowflake").options(**sf_options_dl_poc).option("dbtable", table).mode("errorifexists").save()
COHORT_BOUND = False
def bind_population(create=False):
    global COHORT_BOUND
    if not COHORT_BOUND:
        receipt = binding.bind(read_query, wh.read_sf, write_population, CONFIG, IMPLEMENTATION_SHA256,
            RUN_PREFIX, inp, population, wf, wh, create=create)
        fp.validate_evidence(CONFIG)
        COHORT_BOUND = True
        return receipt
original_load_stage = load_stage
def load_stage():
    bind_population()
    return original_load_stage()
def preparation():
    saved = wh.read_artifacts(RUN_PREFIX+"_PREPARATION", wf.PREP_NAMES)
    return json.loads(saved["preparation.json"]), json.loads(saved["protocol.json"])
'''
    config_cell, helper_cell = base.code(cleaned(config_source)), base.code(helper)
    def cell(s):
        return base.code(cleaned(s))
    new = {}
    names = list(recipes)
    new[names[0]] = [config_cell, helper_cell, cell('''
cohort_receipt = bind_population(create=True)
if cohort_receipt is not None:
    aggregate(pd.DataFrame([cohort_receipt["source_audit"]]))
    print("Accepted missing original snapshots:", cohort_receipt["missing_original_snapshots"])
    aggregate(pd.DataFrame(cohort_receipt["prior_changes_by_split"]), 30)
    print(cohort_receipt["test_scope"])
    print("Any changed old TEST labels or reused TEST patients make evaluation retrospective; this is not an untouched holdout.")
    print("Source date horizon is a development guard; completeness is unverified.")
metadata, population_audit = fp.load_metadata(read_query, wh.read_sf, CONFIG, population)
aggregate(pd.DataFrame(population_audit["population_by_split"]))
population_counts = []
for split, group in [("ALL", metadata)] + list(metadata.groupby("SPLIT")):
    population_counts.append({"Split": split, "Snapshots": len(group), "Unique patients": group.PATIENT_ID.nunique(),
        "RESP=0 snapshots": int(group.RESP.eq(0).sum()), "RESP=1 snapshots": int(group.RESP.eq(1).sum()),
        "Positive patients": group.loc[group.RESP.eq(1), "PATIENT_ID"].nunique(), "Positive rate": group.RESP.mean(),
        "First cutoff": str(group.END_DT.min().date()), "Last cutoff": str(group.END_DT.max().date())})
print("Frozen population counts before feature extraction")
aggregate(pd.DataFrame(population_counts), 10)
print("Expansion audit:", json.dumps({k:v for k,v in population_audit.items() if np.isscalar(v)}, default=str))
if CONFIG["include_nlp"]:
    if not wh.table_exists(RUN_PREFIX+"_NLP_INPUTS"):
        nlp_config = dict(CONFIG["nlp"], run_identity=RUN_PREFIX+":"+IMPLEMENTATION_SHA256)
        nlp_result = nlp.extract_and_fit(read_query, metadata, nlp_config,
            os.path.join(SCRATCH_DIRECTORY, RUN_PREFIX+"_NLP_CHECKPOINT"))
        try:
            fp.save_nlp(nlp_result, metadata, CONFIG, IMPLEMENTATION_SHA256, RUN_PREFIX, inp, wf, wh)
        finally:
            nlp.release_result(nlp_result)
    nlp_package = fp.load_nlp(metadata, CONFIG, IMPLEMENTATION_SHA256, RUN_PREFIX, inp, wf, wh)
    try:
        print("Monthly NLP sequence:", nlp_package["X_sequence"].shape, "step 0 is newest")
        print("TRAIN mean TF-IDF ranking; this is descriptive feature salience, not model lift or importance.")
        aggregate(fp.nlp_ranking(nlp_package), 256)
        print("Saved NLP feature package:", RUN_PREFIX+"_NLP_INPUTS")
    finally:
        fp.release_nlp(nlp_package)
''')]
    for original in recipes[names[0]][2:]:
        source = "".join(original["source"])
        source = source.replace('inp.make_requests(metadata, date_policy=',
                                'inp.make_requests(metadata, steps=CONFIG["raw_automl"]["request_steps"], date_policy=')
        source = source.replace('inp.assemble_inputs(metadata, panel, feature_specs, date_policy=',
                                'inp.assemble_inputs(metadata, panel, feature_specs, steps=CONFIG["raw_automl"]["request_steps"], date_policy=')
        before = 'data["manifest"] = wf.save_inputs'
        if before in source:
            addition = '''if CONFIG["include_nlp"]:
        nlp_package = fp.load_nlp(data["metadata"], CONFIG, IMPLEMENTATION_SHA256, RUN_PREFIX, inp, wf, wh)
        data["X_nlp"], data["nlp_mask"] = nlp_package["X_sequence"], nlp_package["mask"]
        data["nlp_manifest"] = nlp_package["manifest"]
    '''
            source = source.replace(before, addition+before)
            source = source.replace('wf.release_inputs(data)', '''wf.release_inputs(data)
if "nlp_package" in globals():
    fp.release_nlp(nlp_package)''')
        new[names[0]].append(cell(source))
    new[names[1]] = [config_cell, helper_cell, cell('''
data = load_stage()
try:
    prep, protocol = wf.prepare_run(data, RUN_PREFIX, CONFIG, IMPLEMENTATION_SHA256)
    for family in ("automl", "count"):
        print(family, "TRAIN-only selection and fold-local preprocessing")
        aggregate(pd.DataFrame(prep[family]["cv_summary"]))
        aggregate(pd.DataFrame(prep[family]["ranking"]), 1100)
    print("Registered arms:", list(protocol["arms"]), "Seeds:", protocol["seeds"])
    print("Common count feature order:", prep["count"]["selected_names"])
    print("Selected AutoML feature order:", prep["automl"]["selected_names"])
    if "nlp" in prep:
        print("Fixed TRAIN-fitted NLP feature order:", prep["nlp"]["feature_names"])
    print("Protocol hash:", wf.digest(protocol))
finally:
    wf.release_inputs(data)
''')]
    new[names[2]] = [config_cell, helper_cell, cell('''
data = load_stage()
try:
    prep, protocol = preparation()
    results = wf.train_registered(data, prep, protocol, RUN_PREFIX)
    rows = []
    for arm, runs in results.items():
        for run in runs:
            s = run["summary"]
            rows.append(dict(Model=arm, Seed=run["identity"]["seed"], Best_epoch=s["best_epoch"],
                Train_lift=s["train_metrics"]["primary_lift"], Validation_lift=s["validation_metrics"]["primary_lift"],
                Seconds=s["wall_seconds"], Parameters=s["parameter_count"]))
    aggregate(pd.DataFrame(rows), 100)
    display(plots.training_curves(results))
finally:
    wf.release_inputs(data)
'''), cell('''
print("Per-seed validation lift; all registered seeds must complete before comparison.")
aggregate(pd.DataFrame(rows).sort_values(["Model", "Seed"]), 100)
''')]
    new[names[3]] = [config_cell, helper_cell, cell('''
data = load_stage()
try:
    prep, protocol = preparation()
    results = wf.read_runs(protocol, RUN_PREFIX)
    lock = wf.make_lock(data, prep, protocol, results, RUN_PREFIX)
    report = wf.compare_run(data, prep, protocol, results, lock, RUN_PREFIX, evaluate_test=False, n_bootstrap=1000)
    gaps = pd.DataFrame([dict(Model=arm, Train_lift=metrics["train"]["primary_lift"],
        Validation_lift=metrics["validation"]["primary_lift"],
        Train_validation_gap=metrics["train"]["primary_lift"]-metrics["validation"]["primary_lift"])
        for arm, metrics in lock["metrics"].items()])
    aggregate(gaps)
    history = pd.DataFrame([dict(Model=arm, Seed=r["identity"]["seed"], **epoch)
        for arm,runs in results.items() for r in runs for epoch in r["history"]])
    aggregate(history, 500)
    print("Validation selected checkpoints; these are development results. TEST remains disabled.")
    aggregate(wf.report_table(report), 100)
    display(plots.split_lifts(lock))
finally:
    wf.release_inputs(data)
'''), cell('''
EVALUATE_TEST = False
if EVALUATE_TEST:
    data = load_stage()
    try:
        test_report = wf.compare_run(data, prep, protocol, results, lock, RUN_PREFIX, evaluate_test=True, n_bootstrap=1000)
        display(plots.split_lifts(lock, test_report))
        print("Retrospective TEST diagnostics only; do not tune or select models from this output.")
        aggregate(wf.report_table(test_report), 100)
    finally:
        wf.release_inputs(data)
else:
    print("TEST disabled. Showing actual validation results only.")
    aggregate(wf.report_table(report), 100)
''')]
    new[names[4]] = [config_cell, helper_cell, cell('''
import importlib.util
if importlib.util.find_spec("lightgbm") is None:
    raise RuntimeError("Install the reviewed lightgbm==4.6.0 package on this compute and rerun this cell.")
data = load_stage()
try:
    prep, protocol = preparation()
    lock = json.loads(wh.read_artifacts(RUN_PREFIX+"_LOCK", {"lock.json"})["lock.json"])
    baseline = fresh_lgbm.fit_or_resume(data, prep, protocol, lock, wf, mdl, wh,
                                      RUN_PREFIX, IMPLEMENTATION_SHA256)
finally:
    wf.release_inputs(data)
'''), cell('''
EVALUATE_TEST = False
data = load_stage()
try:
    prep, protocol = preparation()
    results = wf.read_runs(protocol, RUN_PREFIX)
    lock = json.loads(wh.read_artifacts(RUN_PREFIX+"_LOCK", {"lock.json"})["lock.json"])
    lift_reports = {}
    for split in (["validation", "test"] if EVALUATE_TEST else ["validation"]):
        report = wf.compare_run(data, prep, protocol, results, lock, RUN_PREFIX,
            evaluate_test=(split == "test"), n_bootstrap=1000)
        table = RUN_PREFIX+("_FINAL_TEST" if split == "test" else "_VALIDATION_COMPARISON")
        payload = wh.read_artifacts(table, {"report.json"} | {arm+".npy" for arm in protocol["arms"]})
        stored = json.loads(payload["report.json"])
        if stored["lock_hash"] != wf.digest(lock) or stored["split"] != split:
            raise ValueError("Stored predictions belong to a different lock or split")
        scores = {arm: wf.bytes_array(payload[arm+".npy"]) for arm in protocol["arms"]}
        lift_reports[split] = fresh_lgbm.compare(data, prep, baseline, scores, split, wf, mdl, cmp, wh,
            allow_test=EVALUATE_TEST, n_bootstrap=1000, expected_arms=protocol["arms"])
        print(lift_reports[split]["scope"])
        print("Static LightGBM is a representation control. Augmented models receive additional information.")
        print("Frozen production comparison is pending verified model inputs and independent evaluation provenance.")
        aggregate(fresh_lgbm.ranking(lift_reports[split]), 100)
        display(plots.lift_ranking(fresh_lgbm.ranking(lift_reports[split]), split.upper()+" — controlled development comparison"))
        aggregate(pd.DataFrame(lift_reports[split]["intervals"]), 100)
finally:
    wf.release_inputs(data)
'''), cell('''
for split, result in lift_reports.items():
    print(RUN_MODE, split.upper(), "Lift@10% ranking")
    display(fresh_lgbm.ranking(result)[["Rank", "Model", "Lift@10%"]])
''')]
    DEST.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, cells in new.items():
        for i, c in enumerate(cells):
            c["id"] = name[:2]+"-"+str(i)
            ast.parse("".join(c["source"]))
        book = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "implementation_sha256": implementation})
        raw = json.dumps(book, ensure_ascii=False, indent=1)+"\n"
        if re.search(r"\b(codex|claude|github|git|client|brian)\b", raw, re.I):
            raise ValueError("Unexpected presentation text")
        path = DEST/name
        path.write_text(raw, encoding="utf-8")
        entries.append(dict(path=str(path.relative_to(base.REPO)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cells=len(cells)))
    manifest = dict(implementation_sha256=implementation, source_sha256=source_hash, notebooks=entries,
                    clinical_execution=False, status="LOCAL_VALIDATION_PENDING", default_phase="SUMMARY",
                    cohort_policy="bounded_reconstruction_development", snapshot_months=48)
    (HERE/"build_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
