"""Build the five self-contained notebooks for the two AutoML add-on experiments."""
import ast
import hashlib
import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEST = REPO / "notebooks" / "V63 Snapshot Padding"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)}


def code(text):
    text = dedent(text).strip() + "\n"
    ast.parse(text)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}


def build():
    sources = {n: (HERE / (n + ".py")).read_text(encoding="utf-8-sig")
               for n in ("inputs", "count_loader", "modeling", "workflow", "reliance", "source_recovery", "raw_automl")}
    sources["comparison"] = (REPO / "experiments/rigorous_transformer/comparison.py").read_text(encoding="utf-8-sig")
    sources["warehouse"] = (REPO / "experiments/temporal_selection/input_helpers.py").read_text(encoding="utf-8-sig")
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    implementation = hashlib.sha256((source_hash + Path(__file__).read_text(encoding="utf-8")).encode()).hexdigest()
    config = code('''
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    RUN_ID = "R20261013C"  # Raw AutoML extraction bound. Same in all five; new ID after any change.
    SOURCE_PREFIX = "TAK861_TX_READY_V63_DL_POC"
    SOURCE_VERSION = "frozen_saved_count_inputs_content_verified_at_load"
    CONFIG = {
        "date_policy": "month_end_history", "seeds": [42, 142, 242],
        "require_count_reference": True, "include_automl_only_controls": False,
        "include_lightgbm": False, "primary_lift_fraction": .10,
        "raw_automl": {"date_policy": "month_end_history", "chunk_size": 5000, "max_features": 512},
        "selection": {"n_splits": 3, "seed": 42,
                      "candidate_sizes": [16, 32, 64, 128, 256, 512, 1028],
                      "proxy_C": .1, "proxy_max_iter": 1000, "weighting": "patient"},
        "model": {"d_model": 64, "heads": 4, "depth": 1, "ff": 128,
                  "dropout": .2, "weight_decay": .001, "learning_rate": .0003,
                  "epochs": 30, "patience": 5, "batch_size": 256,
                  "pooling": "mean", "gradient_clip": 1., "scheduler": "plateau",
                  "weighting": "patient", "balance": "sqrt", "threads": 8, "device": "cpu"}}
    if "spark" not in globals():
        raise RuntimeError("An active Spark session and the existing private source connection are required.")
    connection = globals().get("sf_options_dl_poc", globals().get("sf_options"))
    if not isinstance(connection, dict) or not connection:
        raise RuntimeError("Initialize the private source connection through the existing secret configuration.")
    sf_options_dl_poc = dict(connection)
    sf_options_dl_poc.update(sfDatabase="DSVC_TAKEDA_TA_PRIVATE", sfSchema="DS_ML")
    ''')
    helper = code('''
    import types, sys, json, hashlib, io, time
    import numpy as np, pandas as pd
    from IPython.display import display
    EMBEDDED_SOURCES = ''' + repr(sources) + '''
    EMBEDDED_SOURCE_SHA256 = ''' + repr(source_hash) + '''
    IMPLEMENTATION_SHA256 = ''' + repr(implementation) + '''
    if hashlib.sha256(json.dumps(EMBEDDED_SOURCES, sort_keys=True).encode()).hexdigest() != EMBEDDED_SOURCE_SHA256:
        raise ValueError("Embedded implementation changed. Rebuild all five notebooks together.")
    modules = {}
    for name, source in EMBEDDED_SOURCES.items():
        module = types.ModuleType("_tak861_automl_" + name)
        sys.modules[module.__name__] = module
        exec(compile(source, module.__name__, "exec"), module.__dict__)
        modules[name] = module
    inp, cl, mdl, wf, cmp, wh = [modules[n] for n in
        ("inputs", "count_loader", "modeling", "workflow", "comparison", "warehouse")]
    rel = modules["reliance"]
    recovery = modules["source_recovery"]
    raw = modules["raw_automl"]
    raw.inputs = inp
    wf.inputs, wf.modeling, wf.comparison, cl.inputs = inp, mdl, cmp, inp
    wh.spark, wh.sf_options_dl_poc, wh.PREFIX = spark, sf_options_dl_poc, SOURCE_PREFIX
    wf.save_artifacts, wf.read_artifacts, wf.table_exists = wh.save_artifacts, wh.read_artifacts, wh.table_exists
    RUN_PREFIX = wf.prefix(RUN_ID)
    def read_query(query):
        return spark.read.format("snowflake").options(**sf_options_dl_poc).option("query", query).load()
    def load_stage():
        data = wf.load_inputs(RUN_PREFIX, IMPLEMENTATION_SHA256)
        if data["manifest"]["configuration_hash"] != wf.digest(CONFIG):
            raise ValueError("Notebook configuration differs from Notebook 01; synchronize configuration or use a new RUN_ID.")
        return data
    def aggregate(frame, n=40):
        # Only feature descriptions, aggregate counts and metrics are allowed here.
        display(frame.head(n))
    print("Run:", RUN_ID, "Implementation:", IMPLEMENTATION_SHA256)
    ''')
    header = """
    The four-stage modeling pipeline is followed by a fifth comparison notebook.
    COUNT is the matched count-only control; STATIC_FUSION adds per-snapshot AutoML;
    TEMPORAL_FUSION adds genuine historical AutoML values at all 12 aligned steps.
    All three use the same cohort, labels, patient splits, count selection and seed list.
    This bounded experiment does not repeat the former 84-fit search or reuse its fitted models.
    Saved raw count tables may be reused after content/order/cohort/calendar checks.
    Historical construction/availability limitations remain visible; reuse is not a leakage certificate.
    Lift is the primary objective. Top 10% is the declared default operating capacity;
    top 20% and 30% are secondary. Fix this choice before training, not after viewing results.
    The prediction unit is a patient snapshot; capacity is not a unique-patient outreach count.
    """
    notebooks = {}
    notebooks["01_tensor_initialization.ipynb"] = [md("# 01 · Source inputs and aligned AutoML tensors\n" + header), config, helper,
        md('''
        ## Historical AutoML extraction
        The raw extractor is included, derived from the supplied DX/PX/RX and AGE procedure.
        Each distinct patient/date request gets its own 12-month lookback, with the original
        recent window from four months to one month before that date. No current vector is
        copied into earlier steps. Labels are never sent to this SQL. It reads current raw
        source tables and the current code dictionary; it does not call legacy procedures or
        use their RESP-dependent mapping, pruning, caps, feature typing or model selection.
        This is a revised raw feature definition, not a numerical reproduction of the old
        261-column table. Names use an explicit deterministic sanitizer; the original
        unspecified regexp_comb UDF and ICD-10-PCS positional enrichment are not used.
        At most 512 categories enter the bounded run, selected by distinct TRAIN-patient
        frequency without labels; the complete candidate/exclusion ledger is retained.
        Learned feature selection and preprocessing remain inside TRAIN as described in 02.
        `feature_specs` is an ordered list of `{name, kind, definition}`; kinds are count, binary,
        or continuous. Include legitimate psychotherapy/respiratory features: substring matching is
        not a leakage test. The contract must document source/version, extraction code hash,
        cutoff rule, feature semantics and target-aware fit scope. A target-aware procedure fitted
        on the complete cohort is not an acceptable raw feature generator. Upstream
        target-aware fitting on all TRAIN is also rejected at feature selection:
        it would contaminate the inner TRAIN folds. Recover deterministic raw values
        and a target-independent vocabulary, or implement upstream fitting within
        each fold under a separately reviewed protocol.
        The extractor receives only PATIENT_ID and AS_OF_DATE and returns every requested key,
        IS_OBSERVED, features and, where recoverable, MAX_EVENT_DATE/MAX_AVAILABLE_DATE.
        IS_OBSERVED means at least one eligible source claim exists in the lookback. It does
        not prove continuous coverage. No-claim windows remain missing with an explicit
        indicator, including the current window; no patient is dropped. Within observed
        windows, absence of a selected count/category is zero. AGE can remain missing.
        Current latest source/reference tables cannot establish historical ingestion-time
        availability. Report results as retrospective event-date-bounded comparisons.
        '''), code('''
        # Optional explicit source override for a separately documented extraction/cache.
        # The default is the included raw historical extractor; no source code is missing.
        AUTOML_SETUP = globals().get("AUTOML_SETUP")
        REUSE_SAVED_INPUTS = wh.table_exists(RUN_PREFIX+"_INPUTS")
        '''), md('''
        ## Reuse and alignment
        Read only the raw count FEATURE_MAP, SNAPSHOTS, PATIENT_SPLIT and TENSOR_MONTHLY tables.
        Preserve all 1,028 candidates and all saved splits. No earlier checkpoint, metric or
        training-history table is a runtime dependency. Count time step zero is the partial
        current month through END_DT; older steps are complete calendar months. AutoML as-of
        anchors therefore use END_DT followed by prior month-ends. AutoML rolling lookbacks
        can overlap; they are not described as disjoint monthly event counts.
        The recovered AutoML description uses a 12-month lookback at each anchor. Across
        12 anchors this can extend the event history to roughly 23 months before END_DT.
        Such an experiment changes both feature detail and effective history length;
        it does not isolate a pure temporal-architecture advantage.
        '''), code('''
        started = time.perf_counter()
        if "counts" in globals():
            cl.release_count_inputs(counts)
        if REUSE_SAVED_INPUTS:
            data = load_stage()
            print("Verified and reused this run's frozen tensors; no source extraction repeated.")
        else:
            counts = cl.load_count_inputs(wh.read_sf, SOURCE_PREFIX, SOURCE_VERSION)
            metadata = wf.assign_patient_split(counts["metadata"])
            requests, date_grid = inp.make_requests(metadata, date_policy=CONFIG["date_policy"])
            aggregate(metadata.groupby("SPLIT").agg(snapshots=("RESP", "size"),
                patients=("PATIENT_ID", "nunique"), positives=("RESP", "sum")).reset_index())
            print("Count universe:", len(counts["features"]), "Historical extraction keys:", len(requests),
                  "Tensor positions:", len(date_grid), "Positions served by reusable keys:", len(date_grid)-len(requests))
            print("Count provenance:", counts["contract"]["provenance_status"])
            if AUTOML_SETUP is None:
                AUTOML_SETUP = raw.build_raw_automl_inputs(read_query, metadata, source_config=CONFIG["raw_automl"])
                aggregate(pd.DataFrame([{k: v for k, v in AUTOML_SETUP["receipt"].items() if np.isscalar(v) or v is None}]))
            feature_specs = inp.validate_feature_specs(AUTOML_SETUP["feature_specs"])
            source_contract = inp.verify_fitting_scope(AUTOML_SETUP["contract"], metadata)
            extractor = AUTOML_SETUP.get("extractor")
            if AUTOML_SETUP.get("sql_template"):
                extractor = inp.make_query_extractor(read_query, AUTOML_SETUP["sql_template"], chunk_size=5000)
                if extractor.extraction_sha256 != source_contract["extraction_sha256"]:
                    raise ValueError("Extraction SQL hash differs from its declared source contract")
        '''), code('''
        # obtain_panel reuses exact patient/date rows only when definition and content hashes agree.
        if not REUSE_SAVED_INPUTS:
            panel, panel_manifest = inp.obtain_panel(requests, feature_specs, source_contract,
                cached_panel=AUTOML_SETUP.get("cached_panel"), cached_manifest=AUTOML_SETUP.get("cached_manifest"),
                extractor=extractor)
            if AUTOML_SETUP.get("receipt"):
                panel_manifest["raw_extraction_receipt"] = AUTOML_SETUP["receipt"]
            data = inp.assemble_inputs(metadata, panel, feature_specs, date_policy=CONFIG["date_policy"])
            data = wf.attach_count_reference(data, counts["X"], counts["metadata"], counts["contract"], valid=counts["valid"])
            data["manifest"] = wf.save_inputs(RUN_PREFIX, data, panel_manifest, CONFIG, IMPLEMENTATION_SHA256)
            aggregate(pd.DataFrame([data["report"], panel_manifest["extraction_report"]]))
            cl.release_count_inputs(counts)
            del counts, panel
        print("Summary tensor:", data["X_summary"].shape, "Historical tensor:", data["X_sequence"].shape,
              "Count tensor:", data["X_count"].shape)
        # Full row-level values remain in the private artifact store; never display them.
        print("Notebook 01 complete in seconds:", round(time.perf_counter()-started, 2))
        del data
        '''), md("**Output:** immutable run INPUTS artifact containing tensors, exact feature order, cohort/splits, source contracts and content hashes. No model or performance result has been produced in this notebook.")]
    notebooks["02_patient_level_split.ipynb"] = [md("# 02 · Frozen splits, feature diagnostics and selection\n" + header), config, helper,
        md('''
        Patient membership is retained from the validated count reference; this stage verifies it.
        Fit diagnostics, imputation, count log transforms, scaling and selection on TRAIN only.
        Remove all-missing, truly constant and exact duplicate features, with recorded reasons.
        Near-constant, low variance and highly correlated features are flagged rather than
        automatically discarded; rare clinical signals and meaningful count outliers can matter.
        No blanket percentile clipping is applied. Binary values are preserved.
        Rank original features using TRAIN-only summary/mean/SD/slope descriptors. Choose the
        smallest prefix within one standard error of the best three-fold patient-grouped
        proxy lift at the predeclared capacity; repeat quality checks, transforms and
        ranking inside each training fold. AP remains a secondary diagnostic.
        Count and AutoML subsets are selected separately and then shared across all three arms.
        Proxy selection is a bounded screening decision, not proof of an optimal Transformer K.
        '''), code('''
        data = load_stage()
        # Source semantic kinds come from the stored source contract, not feature-name guesses.
        preparation_config = dict(CONFIG, feature_kinds=[s["kind"] for s in data["feature_specs"]])
        prep, protocol = wf.prepare_run(data, RUN_PREFIX, preparation_config, IMPLEMENTATION_SHA256)
        rows = []
        for family, p in prep.items():
            rows.append({"universe": family, "raw_candidates": len(p["transform"]["feature_names"]),
                         "quality_eligible": len(p["transform"]["eligible_indices"]),
                         "selected_features": len(p["selected_names"]), "selection_rule": p["selection_rule"]})
        aggregate(pd.DataFrame(rows))
        for family, p in prep.items():
            print(family, "quality/removal ledger (full table retained in PREPARATION)")
            aggregate(pd.DataFrame(p["quality"]), 1100)
            print(family, "TRAIN proxy feature-count comparison")
            aggregate(pd.DataFrame(p["cv_summary"]))
            print(family, "ranked feature order, fold-rank stability and final selection")
            aggregate(pd.DataFrame(p["ranking"]), 1100)
            print(family, "high-correlation pairs:", len(p["correlation_pairs_ge_095"]),
                  "singular correlation:", p["singular_correlation"],
                  "condition number:", p["correlation_condition_number"])
            aggregate(pd.DataFrame(p["correlation_pairs_ge_095"]))
        '''), md('''
        ## Final input order and registered experiment
        COUNT: selected count columns in saved rank order.
        STATIC_FUSION: the same count tokens, plus the selected AutoML summary vector
        through a separate additive branch at the pooled representation.
        TEMPORAL_FUSION: selected count columns, then selected historical AutoML columns,
        then an explicit AutoML-history-available indicator at each of the 12 steps.
        Missing AutoML history never masks an available count token.
        '''), code('''
        orders = {"COUNT": ["COUNT::"+s for s in prep["count"]["selected_names"]],
                  "STATIC_FUSION_static": ["AUTOML::"+s for s in prep["automl"]["selected_names"]]+["AUTOML_CURRENT_AVAILABLE"]}
        orders["STATIC_FUSION_sequence"] = orders["COUNT"]
        orders["TEMPORAL_FUSION"] = orders["COUNT"] + ["AUTOML::"+s for s in prep["automl"]["selected_names"]] + ["AUTOML_HISTORY_AVAILABLE"]
        aggregate(pd.DataFrame([{"arm_input": arm, "position": i, "feature": name}
                               for arm, names in orders.items() for i, name in enumerate(names)]), 5000)
        print("Registered arms:", list(protocol["arms"]), "Seeds:", protocol["seeds"])
        print("Protocol hash:", wf.digest(protocol))
        print("No TEST distribution or prediction was used for selection.")
        ''')]
    notebooks["03_transformer_training.ipynb"] = [md("# 03 · Matched Transformer fits\n" + header), config, helper,
        md('''
        Train the three registered arms with identical seeds and the same bounded architecture:
        one encoder layer, 64-dimensional representation, four heads and feed-forward width 128.
        Use dropout, AdamW, validation-lift early stopping, validation-lift learning-rate scheduling,
        gradient clipping, explicit normalization, patient-balanced weights and square-root class
        weighting. These are declared settings, not a claim that each setting won an ablation.
        Each arm starts fresh. The static branch starts at zero contribution; no previous
        fitted checkpoint is loaded. Parameter counts and elapsed times reveal added capacity.
        Every completed seed is saved immediately; verified identical artifacts resume without
        refitting. An incomplete seed set may be reported as incomplete but cannot be locked.
        '''), code('''
        data = load_stage()
        saved = wh.read_artifacts(RUN_PREFIX+"_PREPARATION", wf.PREP_NAMES)
        prep, protocol = json.loads(saved["preparation.json"]), json.loads(saved["protocol.json"])
        results = wf.train_registered(data, prep, protocol, RUN_PREFIX)
        rows = []
        for arm, runs in results.items():
            for r in runs:
                s = r["summary"]
                rows.append(dict(arm=arm, seed=r["identity"]["seed"],
                    best_epoch=s["best_epoch"], parameters=s["parameter_count"],
                    train_lift=s["train_metrics"]["primary_lift"],
                    validation_lift=s["validation_metrics"]["primary_lift"],
                    train_ap=s["train_metrics"]["average_precision"],
                    validation_ap=s["validation_metrics"]["average_precision"],
                    seconds=s["wall_seconds"]))
        aggregate(pd.DataFrame(rows))
        print("All registered fits saved. No TEST predictions computed.")
        ''')]
    notebooks["04_transformer_evaluation.ipynb"] = [md("# 04 · Validation robustness and predictor lock\n" + header), config, helper,
        code('''
        data = load_stage()
        saved = wh.read_artifacts(RUN_PREFIX+"_PREPARATION", wf.PREP_NAMES)
        prep, protocol = json.loads(saved["preparation.json"]), json.loads(saved["protocol.json"])
        results = wf.read_runs(protocol, RUN_PREFIX)
        lock = wf.make_lock(data, prep, protocol, results, RUN_PREFIX)
        rows = []
        for arm, metrics in lock["metrics"].items():
            for split, values in metrics.items():
                rows.append(dict(arm=arm, split=split, **{k: values.get(k) for k in
                    ("primary_lift", "average_precision", "roc_auc", "log_loss", "brier_score", "precision", "recall", "f1")}))
        aggregate(pd.DataFrame(rows))
        history = pd.DataFrame([dict(arm=arm, seed=r["identity"]["seed"], **epoch)
            for arm, runs in results.items() for r in runs for epoch in r["history"]])
        aggregate(history, 500)
        import matplotlib.pyplot as plt
        for arm in results:
            fig, ax = plt.subplots(figsize=(7, 3))
            for seed in protocol["seeds"]:
                h = history[(history.arm == arm) & (history.seed == seed)]
                ax.plot(h.epoch, h.validation_lift, label=str(seed))
            ax.set(title=arm+" validation lift by epoch", xlabel="Epoch", ylabel="Lift at frozen capacity")
            ax.legend(title="Seed"); plt.show(); plt.close(fig)
        print("Frozen predictor/threshold lock:", wf.digest(lock))
        '''), md('''
        Thresholds are selected separately for each ensemble using the same validation-F1
        rule; a shared raw numeric cutoff would be misleading across score scales.
        Lift at the frozen population capacity drives the primary comparison, independently
        of these secondary F1 thresholds. Lift is precision in the selected population
        divided by prevalence in the same evaluation population. A lift of 3 means three
        times the positive rate of random selection. Weighted-loss scores are not assumed
        to be calibrated probabilities.
        Validation performance is development evidence. The source cutoff and historical
        availability limitations remain even when these code-level checks pass.
        '''), code('''
        # Descriptive stability diagnostics; no selection or TEST access.
        ti, vi = data["indices"]["train"], data["indices"]["validation"]
        stability = mdl.distribution_diagnostics(
            mdl.transform_inputs(data["X_summary"][ti], prep["automl"], valid=data["summary_mask"][ti]),
            mdl.transform_inputs(data["X_summary"][vi], prep["automl"], valid=data["summary_mask"][vi]), prep["automl"]["selected_names"],
            valid_train=data["summary_mask"][ti], valid_validation=data["summary_mask"][vi])
        aggregate(pd.DataFrame(stability), 1100)
        '''), md('''
        ## Does model reliance follow feature ranking?
        After freezing the predictors, permute top/middle/bottom TRAIN-rank groups on
        one latest VALIDATION snapshot per patient, within the same snapshot date.
        Use a common complete-history subset so unavailable history is not confused
        with a feature intervention. Report excluded patients and shuffleable strata.
        Positive lift drop means predictive reliance on the perturbed group; AP drop is
        also retained. A negative
        drop or disagreement with the screening rank is retained, not hidden or used
        for retuning. Correlation and jointly permuted groups limit individual-feature
        interpretation. This is a diagnostic, not causal importance or a TEST analysis.
        '''), code('''
        RUN_RELIANCE_DIAGNOSTICS = True
        if RUN_RELIANCE_DIAGNOSTICS:
            reliance_table = RUN_PREFIX + "_RELIANCE"
            if wh.table_exists(reliance_table):
                reliance_report = json.loads(wh.read_artifacts(reliance_table, {"reliance.json"})["reliance.json"])
                if reliance_report["lock_hash"] != wf.digest(lock):
                    raise ValueError("Saved reliance diagnostic belongs to a different predictor lock")
            else:
                reliance_report = rel.run(data, prep, protocol, results, wf, mdl, cmp)
                reliance_report["lock_hash"] = wf.digest(lock)
                wh.save_artifacts(reliance_table, {"reliance.json": wf.json_bytes(reliance_report)})
            print("Reliance status:", reliance_report["status"])
            print("Diagnostic patients:", reliance_report["patients"],
                  "excluded incomplete histories:", reliance_report["excluded_incomplete_latest"])
            aggregate(pd.DataFrame(reliance_report["rows"]), 200)
        else:
            print("Reliance diagnostic skipped by explicit setting; no rank-versus-reliance conclusion.")
        ''')]
    notebooks["05_model_comparison.ipynb"] = [md("# 05 · Incremental value and final comparison\n" + header), config, helper,
        md('''
        Primary contrasts: STATIC_FUSION minus COUNT, TEMPORAL_FUSION minus COUNT,
        and TEMPORAL_FUSION minus STATIC_FUSION. Positive lift deltas favor the first arm.
        Top 10% is the declared default primary capacity, with 20% and 30% secondary;
        do not choose the most favorable capacity after examining outcomes.
        Use paired patient-cluster bootstrap intervals, accounting for repeated snapshots.
        They condition on the fitted predictors and do not include model-selection uncertainty.
        The count cohort has already been evaluated twice on TEST. Any further TEST evaluation
        here is a third retrospective evaluation, not an independent holdout or OOT result.
        Default to validation for today's development comparison. Freeze the complete protocol,
        models and thresholds before an explicitly requested final TEST pass; reruns read the
        identical persisted report. Do not change models after inspecting TEST outcomes.
        '''), code('''
        RUN_FINAL_TEST = False
        BOOTSTRAP_RESAMPLES = 1000
        data = load_stage()
        saved = wh.read_artifacts(RUN_PREFIX+"_PREPARATION", wf.PREP_NAMES)
        prep, protocol = json.loads(saved["preparation.json"]), json.loads(saved["protocol.json"])
        results = wf.read_runs(protocol, RUN_PREFIX)
        lock = json.loads(wh.read_artifacts(RUN_PREFIX+"_LOCK", {"lock.json"})["lock.json"])
        report = wf.compare_run(data, prep, protocol, results, lock, RUN_PREFIX,
                                evaluate_test=RUN_FINAL_TEST, n_bootstrap=BOOTSTRAP_RESAMPLES)
        aggregate(wf.report_table(report))
        uncertainty = []
        for name, pair in report["comparisons"].items():
            for metric in ("primary_lift", "average_precision", "roc_auc"):
                uncertainty.append(dict(comparison=name, metric=metric, **pair[metric]))
        aggregate(pd.DataFrame(uncertainty), 100)
        aggregate(pd.DataFrame([dict(comparison=name, **interval)
            for name, pair in report["comparisons"].items()
            for fraction, interval in pair["lift_by_fraction"].items()]), 100)
        capacity = []
        for arm, values in report["metrics"].items():
            for metrics in values["topk"]:
                capacity.append(dict(arm=arm, **metrics))
        aggregate(pd.DataFrame(capacity))
        aggregate(pd.DataFrame([dict(arm=arm, **row) for arm, values in report["metrics"].items()
                                for row in values["calibration"]]), 100)
        aggregate(pd.DataFrame([dict(arm=arm, **{key: values.get(key) for key in
            ("threshold", "tn", "fp", "fn", "tp", "specificity", "expected_calibration_error")})
            for arm, values in report["metrics"].items()]))
        aggregate(pd.DataFrame([dict(arm=arm,
            train_lift=lock["metrics"][arm]["train"]["primary_lift"],
            validation_lift=lock["metrics"][arm]["validation"]["primary_lift"],
            train_validation_lift_gap=lock["metrics"][arm]["train"]["primary_lift"]-lock["metrics"][arm]["validation"]["primary_lift"],
            train_ap=lock["metrics"][arm]["train"]["average_precision"],
            validation_ap=lock["metrics"][arm]["validation"]["average_precision"],
            train_validation_ap_gap=lock["metrics"][arm]["train"]["average_precision"]-lock["metrics"][arm]["validation"]["average_precision"],
            parameters_per_seed=runs[0]["summary"]["parameter_count"],
            total_training_seconds=sum(r["summary"]["wall_seconds"] for r in runs))
            for arm, runs in results.items()]))
        print("Split:", report["split"], "TEST evaluation ordinal:", report["test_evaluation_ordinal"])
        print("Count comparison:", report["count_reference_status"], "LightGBM:", report["lightgbm_status"])
        for limitation in report["limitations"]:
            print("Limitation:", limitation)
        '''), md('''
        ## Evidence-based readout
        For each primary contrast, report lift difference and its patient-bootstrap interval
        at the frozen capacity. Show top-10/20/30% recall/precision/lift, lift generalization gap,
        calibration, secondary AP/AUC, parameters and runtime. A higher lift means greater
        concentration of positives at the same capacity and in the same population.
        A confidence interval crossing zero is inconclusive, not equivalence or proof of no value.
        An additive input gain does not establish a causal mechanism or a general interaction.
        Do not choose a new winner from TEST. LightGBM comparison is explicitly deferred in this
        bounded run; no claim that the Transformer beats LightGBM is supported by these arms.
        The previously supplied static-fusion results remain recovered historical evidence:
        V0 TEST AP 0.1397; AutoML-only LightGBM 0.1451; logistic fusion 0.1469. All supplied
        AP comparison intervals crossed zero, and the original printed subtraction direction
        was reversed. Those numbers are not results from this implementation.
        '''), code('''
        for contrast, pair in report["comparisons"].items():
            lift = pair["primary_lift"]
            if not pair["lift_bootstrap"]["interval_reliable"]:
                statement = "insufficient valid patient-bootstrap draws; interval unavailable"
            elif lift["lower"] > 0:
                statement = "positive lift difference at frozen capacity within this evaluation; inspect size and limitations"
            elif lift["upper"] < 0:
                statement = "negative lift difference at frozen capacity within this evaluation"
            else:
                statement = "lift difference remains inconclusive; interval includes zero"
            print(contrast, statement)
        print("No independent OOT conclusion. No completed LightGBM comparison in this run.")
        ''')]
    files = []
    for name, cells in notebooks.items():
        notebook = dict(nbformat=4, nbformat_minor=5, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "tak861": {"implementation_sha256": implementation, "source_sha256": source_hash,
                       "clinical_execution": "not_executed_by_build", "protocol": "automl_two_experiments_v1"}}, cells=cells)
        for idx, cell in enumerate(cells):
            cell["id"] = f"{name[:2]}-{idx:03d}"
        path = DEST / name
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
        files.append({"path": str(path.relative_to(REPO)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cells": len(cells)})
    manifest = dict(implementation_sha256=implementation, source_sha256=source_hash, notebooks=files,
                    generated_from="experiments/automl_sequence/build_notebooks.py", clinical_execution=False)
    (HERE / "build_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build()
