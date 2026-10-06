"""Build the revised five-stage delivery, with isolated embedded helper modules."""
import ast
import hashlib
import json
import shutil
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
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def build():
    sources = {n: (HERE / (n + ".py")).read_text(encoding="utf-8-sig")
               for n in ("features", "training", "workflow", "lineage", "comparison", "baseline49", "reliance")}
    sources["warehouse"] = (REPO / "experiments/temporal_selection/input_helpers.py").read_text(encoding="utf-8-sig")
    implementation = hashlib.sha256((json.dumps(sources, sort_keys=True) + Path(__file__).read_text(encoding="utf-8")).encode()).hexdigest()
    config = '''
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    RUN_ID = "R20261006A"  # Same across all five. New ID required for any changed input/code/protocol.
    SOURCE_PREFIX = "TAK861_TX_READY_V63_DL_POC"
    EXPERIMENT_PREFIX = SOURCE_PREFIX + "_RIGOROUS1028_V1"
    if "spark" not in globals():
        raise RuntimeError("Run in the existing authorized Databricks/Spark environment.")
    connection = globals().get("sf_options_dl_poc", globals().get("sf_options"))
    if not isinstance(connection, dict) or not connection:
        raise RuntimeError("Use the existing private Snowflake connection; never paste credentials into shared source.")
    sf_options_dl_poc = dict(connection)
    sf_options_dl_poc.update(sfDatabase="DSVC_TAKEDA_TA_PRIVATE", sfSchema="DS_ML")
    import re
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,23}", RUN_ID):
        raise ValueError("Invalid run ID")
    RUN_PREFIX = EXPERIMENT_PREFIX + "_" + RUN_ID
    '''
    helper = '''import types, sys, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
EMBEDDED_SOURCES = ''' + repr(sources) + '''
modules = {}
for name, source in EMBEDDED_SOURCES.items():
    module_name = "_tak861_final_" + name
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    exec(compile(source, module_name, "exec"), module.__dict__)
    modules[name] = module
fs, tr, wf, lin, cmp, b49, rel, wh = [modules[n] for n in
    ("features", "training", "workflow", "lineage", "comparison", "baseline49", "reliance", "warehouse")]
wh.spark, wh.sf_options_dl_poc, wh.PREFIX = spark, sf_options_dl_poc, SOURCE_PREFIX
for name in ("read_sf", "checked_metadata", "input_fingerprints", "table_exists", "read_artifacts", "save_artifacts"):
    setattr(wf, name, getattr(wh, name))
wf.stable_hash, wf.train_candidate = tr.stable_hash, tr.train_candidate
IMPLEMENTATION_SHA256 = ''' + repr(implementation) + '''
def read_table(table):
    return spark.read.format("snowflake").options(**sf_options_dl_poc).option("dbtable", table).load()
def safe_table(frame, limit=30):
    # Feature/aggregate tables only; never pass patient-level metadata here.
    display(frame.head(limit))
def frozen_predictor_identities(runs):
    # A predictor includes preprocessing and configuration, not just model bytes.
    return [{key:run["summary"][key] for key in
             ("seed","model_sha256","config_sha256","transform_sha256")} for run in runs]
def verify_frozen_predictors(runs, expected, label):
    if frozen_predictor_identities(runs) != expected:
        raise ValueError("Locked predictor model/config/preprocessing identity changed: " + label)
def count_input_contract(config):
    return {"columns":config["columns"],"representation":config["representation"],
            "step0_excluded":bool(config.get("drop_step0",False) or config["representation"]=="without_step0")}
def same_count_feature_information(transformer_config, lightgbm_config):
    tf,gb=count_input_contract(transformer_config),count_input_contract(lightgbm_config)
    return (tf["columns"]==gb["columns"] and tf["representation"] in {"monthly","without_step0"}
            and gb["representation"]=="flattened" and tf["step0_excluded"]==gb["step0_excluded"])
RELIANCE_NAMES = {"reliance.json", "transformer_matched_count_lgb_internal.csv"} | {
    family+"_"+kind+".csv" for family in ("transformer", "lightgbm")
    for kind in ("feature_audit", "internal_seeds", "rank_agreement", "probes", "repeats", "coverage")}
print("Implementation:", IMPLEMENTATION_SHA256)
'''
    load = '''
    if "data" in globals():
        wf.release_count_inputs(data)
    data = wf.load_count_inputs()
    prep = json.loads(wh.read_artifacts(RUN_PREFIX+"_PREPARATION", {"preparation.json", "lineage.json", "step0_screen.csv"})["preparation.json"])
    wf.verify_stage(data, prep, IMPLEMENTATION_SHA256)
    print("Verified all stage input hashes; no TEST predictions computed.")
    '''
    common = [md("""### Authorized runtime and immutable run identity
        Use the existing private Spark/Snowflake session. Each notebook embeds its helpers, uses no repository imports,
        and writes only new run-specific artifacts. Changing code, data, seeds, or protocol requires a new run ID.
        Patient-level arrays and predictions stay in the authorized environment. Do not export notebook outputs containing private rows.
        """), code(config), md("### Versioned implementation\nThe isolated modules below prevent helper-name collisions and make this notebook independently executable."), code(helper)]
    n1 = [md("""# 01 · Data, the full 1,028-feature universe, and provenance
        **Current delivery: implementation ready for execution; no new clinical model performance is prefilled.**
        This revises the first stage of the existing four-notebook workflow. The prior 49-feature snapshot-replay notebooks
        are preserved under `Historical 49 Snapshot Padding (20260927)`; their repeated vectors are not temporal count histories.

        The primary input is the saved 12-position count tensor with all **1,028** mapped categories: 490 DX, 496 PX, 42 RX.
        We preserve the frozen 23,151 snapshots / 12,447 patients and original patient split. No new split is invented.

        **Evidence boundary:** E13 completed 206 queries with six failures and remained inconclusive. Structural alignment is verified
        within the collector's scope; original event cutoff, count grain, feature-building execution and fitting provenance remain unresolved.
        E06 found the same 49 names in a different order, not a proven scoring defect. The old TEST has been inspected before.
        No candidate later cohort is certified untouched or provides a complete 1,028 input tensor.
        """)] + common + [
        md("""### Read and audit saved counts without reconstructing source history
        Full-map index order is the **current tensor contract**. A hash mismatch with historical split metadata is reported separately;
        it is not silently repaired. Values must be nonnegative integer counts or explicitly missing. NaNs stay missing until TRAIN-only fitting.
        All 12 positions remain real saved inputs. A zero is not an enrollment mask. Physical-schema and count-integrality checks may cover all rows;
        held-out distributions, rankings and predictions remain unused during development.
    """), code('''
        data = wf.load_count_inputs()
        train_rows = data["indices"]["train"]
        structural_audit = fs.validate_count_inputs(data["X"], data["metadata"], data["feature_map"], value_rows=train_rows)
        prep = wf.preparation_summary(data, IMPLEMENTATION_SHA256)
        safe_table(pd.DataFrame(prep["cohort"]))
        safe_table(data["feature_map"], 25)
        print("Original candidates:", len(data["features"]), "tensor shape:", data["X"].shape)
        print("Current map hash:", data["hashes"]["feature_names_sha256"])
        print("Matches historical split feature hash:", prep["historical_split_feature_hash_matches_current"])
        '''),
        md("""### TRAIN-only timing checks and leakage review
        Compare overlapping patient windows under both declared orientations; report activity-supported agreement and limitations.
        Equal values can reflect persistent/zero/replayed activity and cannot establish historical cutoff enforcement.
        The step-0/step-1 association screen flags hypotheses, not proven leakage. Counts in a partial current month are especially material
        for a mid-month prediction cutoff. The search always includes an all-eligible **step-0-excluded sensitivity arm**.
        That arm does not certify point-in-time data availability or repair all possible upstream leakage.
        """), code('''
        lineage, step0_screen = lin.audit_count_lineage(data["X"], data["metadata"], data["feature_map"], max_pairs=200)
        display(lineage)
        safe_table(step0_screen, 20)
        '''),
        md("""### Persist the exact input and evidence contract
        The full ordered vocabulary, input hashes, cohort aggregates and source restrictions travel to every later stage.
        Source strings are preserved verbatim. No patient rows are printed. A conflicting existing run is rejected.
        """), code('''
        wh.save_artifacts(RUN_PREFIX+"_PREPARATION", {"preparation.json": wf.json_bytes(prep),
            "lineage.json": wf.json_bytes(lineage), "step0_screen.csv": wf.frame_bytes(step0_screen)})
        print("Stage 1 complete: current saved-input structure reproduced; historical source provenance still unresolved.")
        ''')]
    n2 = [md("""# 02 · Frozen split, feature diagnostics, and TRAIN-only shortlist
        **1,028 raw candidates → quality ledger → justified removals → temporal ranking → count shortlist → ordered inputs.**
        All fitting uses TRAIN; three inner folds keep whole patients together and refit quality checks, imputation and ranking within each fold.
        The frozen outer split remains unchanged. Inverse snapshot-count weights are used for screening; model-training weighting is tested separately.

        Only all-missing, exact constant and exact duplicate sequences are removed automatically. Rare, near-constant, correlated and low-variance
        features remain visible and eligible: sparse healthcare signals can be useful. Continuous counts use log1p; no blanket winsorization,
        correlation pruning or VIF filtering is justified for a regularized nonlinear model. Extreme values and scaling sensitivity are reported.
        """)] + common + [md("### Verify stage 1 and frozen population"), code(load),
        md("""### Fold-local temporal ranking and proxy evaluation
        A transparent fixed L2 logistic proxy evaluates cumulative 1/3/6/12-position and quarter summaries. Association ranking combines weighted
        Spearman and binned mutual-information diagnostics, with rank stability across patient folds. This can miss nonlinear interactions.
        The resource grid (16,32,64,128,256,512,all eligible) tests counts rather than prescribing the final number.
        Proxy-best, one-SE, one-smaller and all-eligible become a deduplicated shortlist. **The Transformer selects its final count using its own
        validation results across all three prespecified seeds.** Proxy AP is neither Transformer AP nor independent final evidence.
        """), code('''
        selection = fs.select_ranked_features(data["X"], data["metadata"], data["feature_map"],
                       config={"candidate_sizes": (16,32,64,128,256,512), "n_splits": 3, "seed": 42})
        safe_table(selection.quality, 30)
        display(selection.quality.groupby(["ELIGIBLE", "QUALITY_REASON"], dropna=False).size().rename("features").reset_index())
        safe_table(selection.ranking, 30)
        display(selection.cv_summary)
        display(pd.DataFrame([{"candidate": k, "feature_count": len(v)} for k,v in selection.candidate_indices.items()]))
        print("Candidate roles:", selection.candidate_tags)
        print("Ranking order is separate from model order, which always follows FEATURE_MAP indices.")
        '''),
        md("""### Redundancy, multicollinearity, stability and distributions
        Annual-count Spearman pairs are TRAIN latest-patient diagnostics, not automatic removals. The reported spectral condition/effective rank
        is a bounded 128-column diagnostic submatrix; it is not a global VIF result. Distribution differences in VALIDATION are descriptive
        and cannot retroactively change this shortlist. TEST distribution checks are deferred. The quality ledger includes missingness,
        nonzero support, variance, quantiles, extreme values, duplicate representatives and removal reasons for every original feature.
        """), code('''
        diagnostics = fs.feature_diagnostics(data["X"], data["metadata"], data["feature_map"], selection.preprocessor)
        shift = fs.shift_summary(data["X"], data["metadata"], data["feature_map"], selection.preprocessor)
        display(diagnostics["summary"])
        safe_table(diagnostics["correlation_pairs"], 25)
        safe_table(shift.sort_values("STANDARDIZED_MEAN_DIFFERENCE", key=lambda x:x.abs(), ascending=False), 25)
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        selection.cv_summary.plot.bar(x="CANDIDATE", y="MEAN_AP", yerr="SE_AP", ax=axes[0], legend=False)
        axes[0].set_title("TRAIN proxy CV — not Transformer performance")
        shift["STANDARDIZED_MEAN_DIFFERENCE"].hist(bins=40,ax=axes[1])
        axes[1].set_title("TRAIN–VALIDATION log annual count shift")
        plt.tight_layout(); plt.show()
        '''),
        md("### Save the full audit and every candidate's exact order\nNo fixed final count is declared here. All removals and candidate memberships are recoverable."), code('''
        selection_report = {"input_hashes":data["hashes"], "implementation_hash":IMPLEMENTATION_SHA256,
            "candidates":{k:v.tolist() for k,v in selection.candidate_indices.items()}, "candidate_tags":selection.candidate_tags,
            "preprocessor":selection.preprocessor.to_dict(), "config":selection.config,
            "input_audit":selection.input_audit, "diagnostics":diagnostics["summary"],
            "final_transformer_feature_count":"Not selected until stage 4"}
        selection_artifacts = {"selection.json":wf.json_bytes(selection_report)}
        for name in ("quality","ranking","cv_scores","cv_summary","fold_rankings","fold_audit"):
            selection_artifacts[name+".csv"] = wf.frame_bytes(getattr(selection,name))
        selection_artifacts["correlations.csv"] = wf.frame_bytes(diagnostics["correlation_pairs"])
        selection_artifacts["validation_shift.csv"] = wf.frame_bytes(shift)
        wh.save_artifacts(RUN_PREFIX+"_SELECTION", selection_artifacts)
        print("Full feature audit saved; proceed to protocol registration and training.")
        ''')]
    read_selection = '''
    selection_names = {"selection.json","quality.csv","ranking.csv","cv_scores.csv","cv_summary.csv",
                       "fold_rankings.csv","fold_audit.csv","correlations.csv","validation_shift.csv"}
    selection_artifacts = wh.read_artifacts(RUN_PREFIX+"_SELECTION", selection_names)
    selection_report = json.loads(selection_artifacts["selection.json"])
    wf.verify_stage(data, selection_report, IMPLEMENTATION_SHA256)
    '''
    n3 = [md("""# 03 · Registered model preparation and controlled training
        This is the fitting stage. No TEST/OOT predictions are made here. A small pre-norm Transformer is the reference:
        linear feature projection → fixed position encoding → encoder → layer norm → mean pooling → binary logit.
        Source orientation is a recovered declaration, so recency interpretations remain qualified.

        Phase A compares feature counts at the reference settings. Each model family's highest mean validation-AP count arm
        becomes its immutable phase-B anchor, with exact ties resolved by parsimony. The rule is registered before any fit;
        anchor results and hashes are saved before architecture/regularization fits. Step-0 exclusion always uses all eligible features.
        Architecture, count, dropout, capacity, class weights, projection L1, explicit L2 vs AdamW, pooling, count scaling,
        sequence aggregation and patient weighting are tested in a fixed bounded grid. Explicit L2 and decoupled decay are not mathematically
        interchangeable; nominal settings are labeled separately and never combined. No regularizer is presumed beneficial.
        Both families receive the same number of candidate fits and three seeds. Actual iterations, validation evaluations, time and complexity
        are reported because nominal trial equality does not equal identical optimization opportunity.
        """)] + common + [md("### Verify upstream artifacts"), code(load + read_selection),
        md("""### Register the exact experiment before any validation-driven fitting
        Seeds 42/142/242 are repetitions; no seed can be discarded for a poor result. The all-eligible candidate is mandatory.
        LightGBM receives all 12 raw monthly values flattened by default, preserving the same input information as the temporal model;
        cumulative-window and annual variants are separate representation ablations. Projection/temporal variants are not claimed to be identical
        feature transformations. The fixed 49-feature LightGBM refit is an additional reference, separately identified.
        """), code('''
        protocol = wf.build_protocol(selection_report["candidates"], selection_report["candidate_tags"], data["hashes"], IMPLEMENTATION_SHA256)
        print("PROTOCOL SHA256:",tr.stable_hash(protocol))
        print("Candidate fits:",len(protocol["candidates"])*len(protocol["seeds"]))
        print("Maximum Transformer epochs/fit:",protocol["epochs"],"CPU threads:",protocol["cpu_threads"])
        safe_table(pd.DataFrame([dict(candidate=k, family=v["family"],features=len(v["columns"]) if v["columns"] is not None else "Phase-A anchor pending",
                         representation=v["representation"], balance=v["balance"]) for k,v in protocol["candidates"].items()]),100)
        wh.save_artifacts(RUN_PREFIX+"_PROTOCOL", {"protocol.json":wf.json_bytes(protocol), "runtime.json":wf.json_bytes(wf.runtime_versions())})
        '''),
        md("""### TRAIN-only throughput check
        Time a few disposable forward/backward batches before launching the grid. This uses no validation outcomes and cannot choose settings.
        The reference estimate excludes validation inference, I/O and larger models; it is a planning measurement, not a completion promise.
        Pause/resume the same immutable run if needed. To revise a budget, do so before inspecting validation outcomes and use a new registered run ID.
        """), code('''
        import time,torch,math
        torch.set_num_threads(protocol["cpu_threads"])
        pilot_cfg=next(c for c in protocol["candidates"].values() if c["family"]=="transformer" and c["subset"]=="all_eligible" and c["columns"] is not None)
        pilot_rows=data["indices"]["train"][:min(256,len(data["indices"]["train"]))]
        pilot_raw=np.asarray(data["X"][pilot_rows])[:,:,pilot_cfg["columns"]]
        pilot_state=tr.fit_numeric_transform(pilot_raw)
        pilot_x=torch.from_numpy(tr.numeric_transform(pilot_raw,pilot_state))
        pilot_y=torch.as_tensor(data["y"][pilot_rows],dtype=torch.float32)
        pilot_model=tr.make_encoder(pilot_x.shape[-1],pilot_cfg)
        timings=[]
        for repeat in range(3):
            started=time.perf_counter();pilot_model.zero_grad(set_to_none=True)
            torch.nn.functional.binary_cross_entropy_with_logits(pilot_model(pilot_x),pilot_y).backward()
            timings.append(time.perf_counter()-started)
        seconds_per_batch=float(np.median(timings[1:]))
        estimate=seconds_per_batch*math.ceil(len(data["indices"]["train"])/len(pilot_rows))*protocol["epochs"]
        print("Measured TRAIN-only reference batch seconds:",round(seconds_per_batch,3))
        print("Reference maximum-epoch backprop estimate per fit (minutes), excluding evaluation/I/O:",round(estimate/60,1))
        print("Transformer fits:",sum(c["family"]=="transformer" for c in protocol["candidates"].values())*len(protocol["seeds"]))
        print("Whole Transformer grid reference-only backprop estimate (hours), not a runtime guarantee:",
              round(estimate*sum(c["family"]=="transformer" for c in protocol["candidates"].values())*len(protocol["seeds"])/3600,2))
        del pilot_model,pilot_x,pilot_y,pilot_raw
        '''),
        md("""### Fit and checkpoint every declared repetition
        Transformer: BCE logits; TRAIN-derived class weighting only in declared arms; AdamW or explicit L2; gradient clip 1;
        validation-AP early stopping; plateau scheduler; deterministic seeds; restoration of the highest validation-AP epoch.
        Fitted numeric state uses TRAIN only, log1p by default, missing-value medians only where needed. Floored standardization is a separate arm.
        LightGBM also stops on validation AP. Validation is reused for stopping and configuration selection: all such estimates are development evidence.
        Each completed fit is saved immediately and verified when resumed. A failure does not silently reduce the declared grid.
        """), code('''
        protocol, results = wf.execute_two_phase_search(data, protocol, RUN_PREFIX)
        winners, search_rows = tr.choose_finalists(results, list(protocol["candidates"]), protocol["seeds"])
        display(pd.DataFrame(search_rows).sort_values(["family","mean_validation_ap"],ascending=[True,False]))
        wh.save_artifacts(RUN_PREFIX+"_SEARCH", {"search.json":wf.json_bytes({"protocol_hash":tr.stable_hash(protocol),
                          "winners":winners,"rows":search_rows,"all_trials_complete":True})})
        '''),
        md("""### Refit the existing 49-feature LightGBM input contract
        This reference fits on the same frozen TRAIN and stops on VALIDATION. It is **a new refit**, not the original V63 estimator.
        The encoded inputs can still carry unresolved historical upstream selection/encoding contamination. Current order follows MODEL_TYPE;
        FINAL_MODEL.SEQ differences are recorded. No count-only transformation is applied to AGE or other encoded continuous values.
        """), code('''
        baseline_data = b49.load_encoded_baseline(read_table, data["metadata"])
        display(baseline_data["order_audit"])
        baseline_runs = []
        for seed in protocol["seeds"]:
            table = RUN_PREFIX+"_BASE49_S"+str(seed)
            contract = {"protocol_hash":tr.stable_hash(protocol),"candidate":"lightgbm49_refit","seed":seed,
                        "feature_sha256":baseline_data["feature_sha256"],"input_sha256":baseline_data["input_sha256"]}
            if wh.table_exists(table):
                fitted = wf.unpack_run(wh.read_artifacts(table,wf.RUN_ARTIFACT_NAMES),contract,
                          expected_lengths={s:len(data["indices"][s]) for s in ("train","validation")},
                          expected_config=b49.DEFAULT_BASELINE_CONFIG)
            else:
                ti,vi = data["indices"]["train"],data["indices"]["validation"]
                fitted = b49.train_encoded_baseline(baseline_data["X"][ti],data["y"][ti],data["metadata"].PATIENT_ID.to_numpy()[ti],
                          baseline_data["X"][vi],data["y"][vi],seed)
                wh.save_artifacts(table,wf.pack_run(fitted,contract))
            baseline_runs.append(fitted)
        wh.save_artifacts(RUN_PREFIX+"_BASE49", {"baseline.json":wf.json_bytes({"feature_sha256":baseline_data["feature_sha256"],
                "input_sha256":baseline_data["input_sha256"],"features":baseline_data["features"],
                "order_audit":baseline_data["order_audit"],"provenance":baseline_data["provenance"]})})
        print("Matched 1,028-family search and separately identified 49-feature reference saved.")
        ''')]
    read_protocol = '''
    design_protocol = json.loads(wh.read_artifacts(RUN_PREFIX+"_PROTOCOL", {"protocol.json","runtime.json"})["protocol.json"])
    if design_protocol["input_hashes"] != data["hashes"] or design_protocol["implementation_hash"] != IMPLEMENTATION_SHA256:
        raise ValueError("Protocol and input/code identity mismatch")
    protocol = wf.read_resolved_protocol(design_protocol, RUN_PREFIX)
    results = wf.read_search(protocol,RUN_PREFIX)
    winners,search_rows = tr.choose_finalists(results,list(protocol["candidates"]),protocol["seeds"])
    '''
    n4 = [md("""# 04 · Generalization, robustness, and the final decision lock
        Review every candidate and seed, selected epochs, training/validation gaps, preprocessing sensitivity and model complexity.
        No TEST score is available in this stage. Choose the maximum mean validation AP across the three seeds; exact ties prefer fewer features
        and lower complexity. A nonsignificant difference is not proof of equivalence and is not used as a feature-count rule.
        The final predictor is the uniform three-seed ensemble; individual results remain visible.
        """)] + common + [md("### Load all complete trials"), code(load + read_protocol + read_selection),
        md("### All-seed results and controlled experiments\nThese validation comparisons are exploratory after tuning; do not label their intervals independent confirmation."), code('''
        seed_report = cmp.summarize_seed_runs(results)
        safe_table(pd.DataFrame(seed_report["all_seeds"]),200)
        safe_table(pd.DataFrame(search_rows).sort_values(["family","mean_validation_ap"],ascending=[True,False]),100)
        fig,ax = plt.subplots(figsize=(12,5))
        rows = pd.DataFrame(search_rows)
        ax.errorbar(np.arange(len(rows)),rows.mean_validation_ap,yerr=rows.sd_validation_ap,fmt="o")
        ax.set_xticks(np.arange(len(rows)),rows.candidate,rotation=90)
        ax.set_ylabel("Mean validation AP ± seed SD"); ax.set_title("All declared candidates; no seed picking")
        plt.tight_layout(); plt.show()
        '''),
        md("### Learning curves and overfitting evidence\nCheckpoint epochs are chosen by validation AP. Penalized training loss is not directly comparable with unpenalized validation log loss."), code('''
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for family,name in winners.items():
            for run in results[name]:
                history=pd.DataFrame(run["history"])
                axes[0].plot(history.epoch,history.validation_ap,label=family+" seed "+str(run["summary"]["seed"]))
                if "train_ap" in history:
                    axes[1].plot(history.epoch,history.train_ap-history.validation_ap,label="seed "+str(run["summary"]["seed"]))
        axes[0].set_title("Finalist validation learning curves"); axes[0].legend(fontsize=8)
        axes[1].set_title("Transformer TRAIN minus VALIDATION AP"); axes[1].legend()
        for ax in axes: ax.set_xlabel("Epoch / boosting iteration")
        plt.tight_layout(); plt.show()
        '''),
        md("""### Freeze models, thresholds, feature ordering and the comparison contract
        Threshold F1 is optimized on ensemble VALIDATION, not TEST. Top-10% is a capacity rule with deterministic frozen-key tie ordering,
        not a test-optimized probability threshold. Feature projection norms are scale-confounded internal parameter magnitude,
        not predictive reliance or causal importance. The separate post-lock audit below measures perturbation effects.
        The selected ordered list, full universe, removal ledger and rank list all remain in the saved artifacts.
        """), code('''
        vi=data["indices"]["validation"]
        policies={family:cmp.select_validation_threshold(data["y"][vi],wf.ensemble_scores(results[name],"validation"))
                  for family,name in winners.items()}
        selected_orders={family:[data["features"][i] for i in protocol["candidates"][name]["columns"]] for family,name in winners.items()}
        feature_counts={family:len(v) for family,v in selected_orders.items()}
        display({"finalists":winners,"feature_counts":feature_counts,"thresholds":policies})
        for family,name in winners.items():
            safe_table(pd.DataFrame({"model_position":range(len(selected_orders[family])),"feature":selected_orders[family]}),40)
        tfname=winners["transformer"]
        projection=np.mean([r["summary"]["importance"] for r in results[tfname]],axis=0)
        if len(projection)!=len(selected_orders["transformer"]): raise ValueError("Projection columns do not match selected feature order")
        reliance=pd.DataFrame({"feature":selected_orders["transformer"],"mean_projection_norm":projection})
        safe_table(reliance.sort_values("mean_projection_norm",ascending=False),25)
        base_manifest=json.loads(wh.read_artifacts(RUN_PREFIX+"_BASE49",{"baseline.json"})["baseline.json"])
        baseline_runs=[]
        for seed in protocol["seeds"]:
            contract={"protocol_hash":tr.stable_hash(protocol),"candidate":"lightgbm49_refit","seed":seed,
                      "feature_sha256":base_manifest["feature_sha256"],"input_sha256":base_manifest["input_sha256"]}
            baseline_runs.append(wf.unpack_run(wh.read_artifacts(RUN_PREFIX+"_BASE49_S"+str(seed),wf.RUN_ARTIFACT_NAMES),contract,
                  expected_lengths={s:len(data["indices"][s]) for s in ("train","validation")},expected_config=b49.DEFAULT_BASELINE_CONFIG))
        baseline_lock={"input_sha256":base_manifest["input_sha256"],"feature_sha256":base_manifest["feature_sha256"],
             "threshold_policy":cmp.select_validation_threshold(data["y"][vi],wf.ensemble_scores(baseline_runs,"validation")),
             "model_hashes":[r["summary"]["model_sha256"] for r in baseline_runs],
             "predictor_identities":frozen_predictor_identities(baseline_runs)}
        lock={"input_hashes":data["hashes"],"implementation_hash":IMPLEMENTATION_SHA256,
              "protocol_hash":tr.stable_hash(protocol),"winners":winners,"threshold_policies":policies,
              "seeds":protocol["seeds"],"ordered_features":selected_orders,"feature_counts":feature_counts,
              "model_hashes":{f:[r["summary"]["model_sha256"] for r in results[n]] for f,n in winners.items()},"baseline49":baseline_lock,
              "predictor_identities":{f:frozen_predictor_identities(results[n]) for f,n in winners.items()},
              "test_used_for_new_selection":False,"historical_test_previously_inspected":True,
              "final_predictor":"uniform_three_seed_ensemble","primary_metric":"average_precision",
              "test_label":"retrospective frozen TEST; not an untouched prospective/OOT claim"}
        wh.save_artifacts(RUN_PREFIX+"_LOCK", {"lock.json":wf.json_bytes(lock),"projection_parameter_magnitude.csv":wf.frame_bytes(reliance)})
        print("Immutable final decision saved. Complete the post-lock validation reliance audit before Notebook 5.")
        '''),
        md("""### Does measured model reliance agree with the TRAIN ranking?
        A screening rank is not a requirement for the fitted model's importance order. Join every original feature by index/name to
        its selected input position, TRAIN ranking/stability and each seed's internal magnitude. Projection norms are scale-confounded;
        norm × TRAIN transformed-input SD is also an internal diagnostic. Gain from the reference count LightGBM arm with the same feature IDs
        provides another view. Its temporal representation or step-0 policy may differ; both are recorded. This is not an architecture-controlled
        reliance comparison, even when the effective input information matches.

        The registered bounded probe union uses top 10 TRAIN ranks, top 10 internal ranks, bottom 5 TRAIN ranks and 5 seeded remaining
        features (at most 30); untested individual reliance stays UNKNOWN. Joint groups cover source families, selected-feature TRAIN-rank
        terciles and up to 10 largest TRAIN |Spearman|≥0.9 connected components. Omitted groups are reported.
        Only one **latest VALIDATION snapshot per patient** is used, with whole 12-position trajectories permuted within END_DT month.
        This is a secondary diagnostic population. It avoids treating repeated patient snapshots as independent permutation units.

        The locked uniform ensemble and its preprocessing never refit. Three fixed permutations report AP drop, lift@10% drop,
        prediction change and the fraction whose actual raw values changed. Repeat SD is not a confidence interval. Sparse constant values
        or singleton month strata may yield no perturbation. Correlated inputs can hide individual reliance or create unrealistic combinations;
        joint groups preserve within-group relationships only. VALIDATION was already used for selection, so these are exploratory diagnostics,
        not independent confirmation. They cannot change the locked features, models, thresholds or TEST policy.
        """), code('''
        import io
        train_ranking=pd.read_csv(io.BytesIO(selection_artifacts["ranking.csv"]),keep_default_na=False)
        train_pairs=pd.read_csv(io.BytesIO(selection_artifacts["correlations.csv"]),keep_default_na=False)
        audits={};plans={};settings=protocol["reliance_diagnostic"]
        for family,name in winners.items():
            audits[family]=rel.join_rank_and_internal_diagnostics(data["feature_map"],train_ranking,results[name],
                                protocol["candidates"][name]["columns"],protocol["seeds"])
            plans[family]=rel.prespecify_probes(audits[family],correlation_pairs=train_pairs,
                max_features=settings["max_individual_features"],random_seed=settings["probe_seed"],
                top_train=settings["top_train"],top_internal=settings["top_internal"],
                lowest_train=settings["lowest_train"],random_remaining=settings["random_remaining"],
                correlation_threshold=settings["correlation_threshold"],max_correlation_groups=settings["max_correlation_groups"])
        plan_record={"lock_hash":tr.stable_hash(lock),"settings":settings,"plans":plans}
        wh.save_artifacts(RUN_PREFIX+"_RELIANCE_PLAN",{"plan.json":wf.json_bytes(plan_record)})
        if wh.table_exists(RUN_PREFIX+"_RELIANCE"):
            audit_artifacts=wh.read_artifacts(RUN_PREFIX+"_RELIANCE",RELIANCE_NAMES)
            audit_report=json.loads(audit_artifacts["reliance.json"])
            if audit_report["lock_hash"]!=tr.stable_hash(lock) or audit_report["plan_record_sha256"]!=tr.stable_hash(plan_record):
                raise ValueError("Saved reliance audit changed its lock or declared plan")
        else:
            audit_artifacts={};family_reports={}
            tf_columns=protocol["candidates"][winners["transformer"]]["columns"]
            twin_names=[name for name in protocol["phase_a"] if protocol["candidates"][name]["family"]=="lightgbm"
                        and protocol["candidates"][name]["columns"]==tf_columns]
            if len(twin_names)!=1: raise ValueError("No unique count LightGBM arm matches Transformer feature identities")
            twin=rel.join_rank_and_internal_diagnostics(data["feature_map"],train_ranking,results[twin_names[0]],tf_columns,protocol["seeds"])
            tf_config=protocol["candidates"][winners["transformer"]];twin_config=protocol["candidates"][twin_names[0]]
            twin_contract={"label":"Same feature IDs; reference count-arm gain, not an architecture-controlled reliance comparison",
                "transformer_input":count_input_contract(tf_config),"reference_count_lgb_input":count_input_contract(twin_config),
                "same_effective_input_information":same_count_feature_information(tf_config,twin_config),
                "architecture_controlled_reliance_comparison":False}
            audit_artifacts["transformer_matched_count_lgb_internal.csv"]=wf.frame_bytes(twin["full_table"])
            for family,name in winners.items():
                x_val,y_val,groups_val=wf.candidate_views(data,protocol["candidates"][name],"validation")
                val_meta=data["metadata"].iloc[data["indices"]["validation"]].reset_index(drop=True)
                latest=cmp.latest_patient_indices(val_meta.PATIENT_ID,val_meta.END_DT)
                x_latest=x_val[latest];latest_meta=val_meta.iloc[latest].reset_index(drop=True)
                predictors=[tr.make_saved_predictor(run) for run in results[name]]
                def frozen_ensemble(values): return np.mean([predict(values) for predict in predictors],axis=0)
                np.testing.assert_allclose(frozen_ensemble(x_latest),wf.ensemble_scores(results[name],"validation")[latest],rtol=1e-5,atol=1e-6)
                diagnostic=rel.validation_permutation_diagnostic(x_latest,latest_meta,selected_orders[family],frozen_ensemble,
                    plans[family],tr.stable_hash(lock),repeat_seeds=tuple(settings["repeat_seeds"]),all_validation_metadata=val_meta)
                joined=audits[family];full=joined["full_table"].copy();covered=diagnostic["coverage"].set_index("FEATURE_INDEX")
                for column in covered.columns:
                    if column not in {"FEATURE_NAME","MODEL_POSITION","PERMUTATION_STATUS"}:
                        full[column]=full.FEATURE_INDEX.map(covered[column])
                full["PERMUTATION_STATUS"]=full.FEATURE_INDEX.map(covered.PERMUTATION_STATUS).fillna("NOT_IN_MODEL")
                measured=full.loc[full.PERMUTATION_STATUS.eq("MEASURED_VALIDATION_MARGINAL")].copy()
                measured["AP_DROP_RANK_WITHIN_PROBES"]=measured.MEAN_AP_DROP.rank(ascending=False,method="average")
                measured["TRAIN_RANK_WITHIN_PROBES"]=measured.TRAIN_RANK.rank(method="average")
                measured["RANK_DIFFERENCE_WITHIN_PROBES"]=measured.TRAIN_RANK_WITHIN_PROBES-measured.AP_DROP_RANK_WITHIN_PROBES
                for column in ("AP_DROP_RANK_WITHIN_PROBES","TRAIN_RANK_WITHIN_PROBES","RANK_DIFFERENCE_WITHIN_PROBES"):
                    full[column]=full.FEATURE_INDEX.map(measured.set_index("FEATURE_INDEX")[column])
                rho=None
                if len(measured)>1 and measured.MEAN_AP_DROP.nunique()>1 and measured.TRAIN_RANK.nunique()>1:
                    rho=float(measured.TRAIN_RANK.corr(measured.AP_DROP_RANK_WITHIN_PROBES,method="spearman"))
                if family=="transformer":
                    twin_table=twin["full_table"].set_index("FEATURE_INDEX")
                    for column in ("MEAN_INTERNAL_MAGNITUDE","MEAN_INTERNAL_RANK"):
                        full["MATCHED_COUNT_LGB_"+column]=full.FEATURE_INDEX.map(twin_table[column])
                family_reports[family]={"mapping":joined["summary"],"permutation":diagnostic["protocol"],
                    "screen_vs_permutation_spearman_among_probed":rho,"individually_measured_features":len(measured),
                    "agreement_rule":"No required rank agreement; inspect signed rank differences and groups; do not retune the locked model"}
                for kind,frame in {"feature_audit":full,"internal_seeds":joined["seed_table"],"rank_agreement":joined["agreement_table"],
                                   "probes":diagnostic["probe_summary"],"repeats":diagnostic["repeats"],"coverage":diagnostic["coverage"]}.items():
                    audit_artifacts[family+"_"+kind+".csv"]=wf.frame_bytes(frame)
                del predictors,x_val,x_latest
            audit_report={"lock_hash":tr.stable_hash(lock),"plan_record_sha256":tr.stable_hash(plan_record),"families":family_reports,
                "evidence_class":"Reproduced","matched_count_lgb_candidate":twin_names[0],
                "matched_count_lgb_reference_contract":twin_contract,
                "interpretation":"Secondary exploratory VALIDATION marginal reliance; no causal claim, no TEST use, no forced rank agreement"}
            audit_artifacts["reliance.json"]=wf.json_bytes(audit_report)
            wh.save_artifacts(RUN_PREFIX+"_RELIANCE",audit_artifacts)
        display(audit_report)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,family in zip(axes,("transformer","lightgbm")):
            frame=pd.read_csv(io.BytesIO(audit_artifacts[family+"_feature_audit.csv"]))
            safe_table(frame.sort_values("MEAN_AP_DROP",ascending=False),30)
            measured=frame[frame.PERMUTATION_STATUS.eq("MEASURED_VALIDATION_MARGINAL")]
            ax.errorbar(measured.TRAIN_RANK,measured.MEAN_AP_DROP,yerr=measured.REPEAT_SD_AP_DROP,fmt="o")
            ax.set(title=family+": secondary VALIDATION reliance",xlabel="TRAIN screening rank (1=highest)",ylabel="Permutation AP drop ± repeat SD")
            safe_table(pd.read_csv(io.BytesIO(audit_artifacts[family+"_probes.csv"])),50)
        plt.tight_layout();plt.show()
        print("Reliance audit saved; immutable decisions unchanged. Notebook 5 may now evaluate retrospective TEST.")
        ''')]
    n5 = [md("""# 05 · Final LightGBM vs Transformer comparison
        **Question:** does the selected Transformer add reproducible predictive value over LightGBM?
        This notebook computes the answer from locked artifacts. It does not contain invented or copied historical performance.

        Primary: locked three-seed Transformer ensemble versus selected same-cohort 1,028-input LightGBM refit.
        Secondary: fixed three-seed LightGBM refit on the 49 encoded V63 fields.
        Original V63 SCORE: descriptive context only, because fit exclusion, score semantics and upstream fitting provenance remain unresolved.
        The previously inspected frozen TEST supports a **retrospective** comparison. No result here certifies an untouched OOT evaluation
        or resolves the original count-building cutoff. A larger AP alone cannot establish clinical or business value.
        """)] + common + [md("### Verify the immutable decision before any held-out inference"), code(dedent(load) + dedent(read_protocol) + dedent('''
        lock=json.loads(wh.read_artifacts(RUN_PREFIX+"_LOCK",{"lock.json","projection_parameter_magnitude.csv"})["lock.json"])
        wf.verify_stage(data,lock,IMPLEMENTATION_SHA256)
        if lock["protocol_hash"]!=tr.stable_hash(protocol) or lock["winners"]!=winners:
            raise ValueError("Finalist/protocol changed after lock")
        for family,name in winners.items():
            if lock["model_hashes"][family] != [r["summary"]["model_sha256"] for r in results[name]]:
                raise ValueError("Locked model bytes changed")
            verify_frozen_predictors(results[name],lock["predictor_identities"][family],family)
        baseline_data=b49.load_encoded_baseline(read_table,data["metadata"])
        base_manifest=json.loads(wh.read_artifacts(RUN_PREFIX+"_BASE49",{"baseline.json"})["baseline.json"])
        for key in ("feature_sha256","input_sha256"):
            if baseline_data[key]!=base_manifest[key] or baseline_data[key]!=lock["baseline49"][key]:
                raise ValueError("Baseline input/order changed after lock")
        baseline_runs=[]
        for seed in protocol["seeds"]:
            contract={"protocol_hash":tr.stable_hash(protocol),"candidate":"lightgbm49_refit","seed":seed,
                      "feature_sha256":baseline_data["feature_sha256"],"input_sha256":baseline_data["input_sha256"]}
            baseline_runs.append(wf.unpack_run(wh.read_artifacts(RUN_PREFIX+"_BASE49_S"+str(seed),wf.RUN_ARTIFACT_NAMES),contract,
                expected_lengths={s:len(data["indices"][s]) for s in ("train","validation")},expected_config=b49.DEFAULT_BASELINE_CONFIG))
        if [r["summary"]["model_sha256"] for r in baseline_runs]!=lock["baseline49"]["model_hashes"]:
            raise ValueError("Baseline model changed after lock")
        verify_frozen_predictors(baseline_runs,lock["baseline49"]["predictor_identities"],"baseline49")
        base_policy=lock["baseline49"]["threshold_policy"]
        audit_artifacts=wh.read_artifacts(RUN_PREFIX+"_RELIANCE",RELIANCE_NAMES)
        audit_report=json.loads(audit_artifacts["reliance.json"])
        plan_record=json.loads(wh.read_artifacts(RUN_PREFIX+"_RELIANCE_PLAN",{"plan.json"})["plan.json"])
        if audit_report["lock_hash"]!=tr.stable_hash(lock) or plan_record["lock_hash"]!=tr.stable_hash(lock) or audit_report["plan_record_sha256"]!=tr.stable_hash(plan_record):
            raise ValueError("Post-lock reliance audit identity mismatch")
        print("Decision hash:",tr.stable_hash(lock))
        ''')),
        md("""### Held-out inference, once per frozen run
        A saved final artifact is reused on rerun. The run's evaluation intent is written first; changed choices cannot overwrite it.
        No threshold or model is refitted from TEST. Paired patient bootstrap conditions on the fixed fitted predictions and does not include
        model-search uncertainty. Calibration bins, confusion matrices, AP, trapezoidal PR area, ROC-AUC, Brier, precision/recall/F1,
        specificity and fixed top-K results are computed from exactly aligned rows.
        """), code('''
        final_names={"comparison.json","metrics.csv","test_transformer.npy","test_lightgbm.npy","test_lightgbm49.npy"}
        if wh.table_exists(RUN_PREFIX+"_FINAL"):
            final_artifacts=wh.read_artifacts(RUN_PREFIX+"_FINAL",final_names)
            final_report=json.loads(final_artifacts["comparison.json"])
            if final_report["lock_hash"]!=tr.stable_hash(lock): raise ValueError("Final report lock mismatch")
            if final_report.get("validation_reliance")!=audit_report: raise ValueError("Final report reliance mismatch")
        else:
            wh.save_artifacts(RUN_PREFIX+"_FINAL_INTENT",{"intent.json":wf.json_bytes({"lock_hash":tr.stable_hash(lock),"evaluation":"retrospective TEST"})})
            scores={}; locked_seed_test_metrics={}
            for family,name in winners.items():
                xx,yy,gg=wf.candidate_views(data,protocol["candidates"][name],"test")
                predictions=[tr.predict_saved(r,xx) for r in results[name]]
                scores[family]=np.mean(predictions,axis=0)
                locked_seed_test_metrics[family]=[{"seed":r["summary"]["seed"],
                    "test_metrics":cmp.binary_metrics(yy,p,lock["threshold_policies"][family]),
                    "threshold_note":"Same ensemble-validation policy applied to all seeds; no per-seed threshold tuning"}
                     for r,p in zip(results[name],predictions)]
                del xx
            test_rows=data["indices"]["test"]; vi=data["indices"]["validation"]
            provenance={"baseline_kind":"matched_refit","fit_membership_verified":True,"same_cohort":True,"same_labels":True,
                "patient_disjoint_split":True,"selection_locked":True,"test_used_for_selection":False,"comparison_split":"test",
                "upstream_cutoff_verified":False,"holdout_never_previously_inspected":False,
                "same_feature_information":same_count_feature_information(protocol["candidates"][winners["transformer"]],
                                                                            protocol["candidates"][winners["lightgbm"]])}
            paired=cmp.compare_predictions(data["y"][test_rows],scores["transformer"],scores["lightgbm"],
                   data["metadata"].PATIENT_ID.to_numpy()[test_rows],threshold_policies=lock["threshold_policies"],
                   provenance=provenance,n_bootstrap=1000,seed=20261006)
            base_predictions=[b49.predict_encoded(r,baseline_data["X"][test_rows]) for r in baseline_runs]
            scores["lightgbm49"]=np.mean(base_predictions,axis=0)
            locked_seed_test_metrics["lightgbm49"]=[{"seed":r["summary"]["seed"],
                "test_metrics":cmp.binary_metrics(data["y"][test_rows],p,base_policy)} for r,p in zip(baseline_runs,base_predictions)]
            baseline49_comparison=cmp.compare_predictions(data["y"][test_rows],scores["transformer"],scores["lightgbm49"],
                data["metadata"].PATIENT_ID.to_numpy()[test_rows],
                threshold_policies={"transformer":lock["threshold_policies"]["transformer"],"lightgbm":base_policy},
                provenance={**provenance,"same_feature_information":False},n_bootstrap=1000,seed=20261006)
            metric_rows=[]
            full_metrics={}
            for family,name in winners.items():
                full_metrics[family]={}
                for split in ("train","validation","test"):
                    rr=data["indices"][split]
                    ss=scores[family] if split=="test" else wf.ensemble_scores(results[name],split)
                    m=cmp.binary_metrics(data["y"][rr],ss,lock["threshold_policies"][family])
                    full_metrics[family][split]=m
                    metric_rows.append(dict(model=family,split=split,**{k:v for k,v in m.items() if np.isscalar(v) or v is None}))
            baseline_metrics={}
            for split in ("train","validation","test"):
                rr=data["indices"][split]
                ss=scores["lightgbm49"] if split=="test" else wf.ensemble_scores(baseline_runs,split)
                m=cmp.binary_metrics(data["y"][rr],ss,base_policy)
                baseline_metrics[split]=m
                metric_rows.append(dict(model="lightgbm49_refit",split=split,**{k:v for k,v in m.items() if np.isscalar(v) or v is None}))
            latest=cmp.latest_patient_indices(data["metadata"].iloc[test_rows].PATIENT_ID,data["metadata"].iloc[test_rows].END_DT)
            patient_secondary={f:cmp.binary_metrics(data["y"][test_rows][latest],ss[latest],
                                  base_policy if f=="lightgbm49" else lock["threshold_policies"][f]) for f,ss in scores.items()}
            final_report={"lock_hash":tr.stable_hash(lock),"paired_comparison":paired,"metrics":full_metrics,
                "baseline49_metrics":baseline_metrics,"baseline49_provenance":base_manifest["provenance"],"validation_reliance":audit_report,
                "baseline49_comparison":baseline49_comparison,"locked_seed_test_metrics":locked_seed_test_metrics,
                "baseline49_seed_report":cmp.summarize_seed_runs({"lightgbm49_refit":baseline_runs}),
                "patient_latest_secondary":patient_secondary,"search":search_rows,"all_seeds":cmp.summarize_seed_runs(results),
                "heldout_status":"retrospective TEST; OOT unavailable","feature_counts":lock["feature_counts"],
                "business_limit":"Top10% is a declared illustrative capacity; no cost/benefit or clinical utility claim without operational evidence"}
            final_artifacts={"comparison.json":wf.json_bytes(final_report),"metrics.csv":wf.frame_bytes(pd.DataFrame(metric_rows)),
                **{"test_"+f+".npy":wf.array_bytes(s) for f,s in scores.items()}}
            wh.save_artifacts(RUN_PREFIX+"_FINAL",final_artifacts)
        display(pd.read_csv(__import__("io").BytesIO(final_artifacts["metrics.csv"])))
        display(final_report["paired_comparison"])
        '''),
        md("""### Discrimination, calibration, operating points and complexity
        Curves reuse the saved final predictions. Calibration is descriptive; no calibration model is fitted on TEST.
        The confusion matrices use locked validation thresholds. Top-10% uses a fixed population-capacity rule.
        Large training–validation gaps, seed variability, inference complexity and lower operational yield count against a claimed improvement.
        """), code('''
        from sklearn.metrics import precision_recall_curve,roc_curve
        import io
        test_rows=data["indices"]["test"]; y=data["y"][test_rows]
        test_scores={f:np.load(io.BytesIO(final_artifacts["test_"+f+".npy"]),allow_pickle=False)
                     for f in ("transformer","lightgbm","lightgbm49")}
        test_metrics={f:final_report["metrics"][f]["test"] for f in ("transformer","lightgbm")}
        test_metrics["lightgbm49"]=final_report["baseline49_metrics"]["test"]
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for family,s in test_scores.items():
            precision,recall,_=precision_recall_curve(y,s); fpr,tpr,_=roc_curve(y,s)
            axes[0].plot(recall,precision,label=family); axes[1].plot(fpr,tpr,label=family)
            calibration=pd.DataFrame(test_metrics[family]["calibration"])
            calibration=calibration[calibration["count"]>0]
            axes[2].plot(calibration.mean_probability,calibration.observed_rate,"o-",label=family)
        axes[0].axhline(y.mean(),color="grey",linestyle="--");axes[0].set(xlabel="Recall",ylabel="Precision",title="Retrospective TEST precision–recall")
        axes[1].plot([0,1],[0,1],"--",color="grey");axes[1].set(xlabel="False-positive rate",ylabel="Recall",title="ROC")
        axes[2].plot([0,1],[0,1],"--",color="grey");axes[2].set(xlabel="Mean probability",ylabel="Observed event rate",title="Calibration (fixed bins)")
        for ax in axes:ax.legend(fontsize=8)
        plt.tight_layout();plt.show()
        fig,axes=plt.subplots(1,3,figsize=(12,3))
        for ax,(family,m) in zip(axes,test_metrics.items()):
            matrix=np.array(m["confusion_matrix"]);ax.imshow(matrix,cmap="Blues")
            for (row,col),value in np.ndenumerate(matrix):ax.text(col,row,str(value),ha="center",va="center")
            ax.set(title=family,xlabel="Predicted",ylabel="Actual",xticks=[0,1],yticks=[0,1])
        plt.tight_layout();plt.show()
        display(pd.DataFrame([{**m["topk"][0],"model":f} for f,m in test_metrics.items()]))
        display(pd.DataFrame(final_report["search"]))
        display(final_report["baseline49_comparison"])
        display(final_report["patient_latest_secondary"])
        '''),
        md("""### Original V63 scores and historical reports: separate evidence tier
        E10's saved histories and metrics were read successfully but clipped in supplied output; use the bounded recovery helper to recover their
        exact values without re-running training. Table timestamps are not run chronology. The original V63 score vector, if readable and exactly
        aligned, is reported below for VALIDATION only as potentially in-sample ranking context. Probability semantics are not established:
        no original-model calibration, Brier score, log loss or probability threshold is reported. These recomputed ranking metrics are not
        recovered historical evaluation numbers. Failure to retrieve it cannot silently become a matched comparison.
        """), code('''
        try:
            original=b49.load_original_v63_scores(read_table,data["metadata"],split="validation")
            from sklearn.metrics import average_precision_score,roc_auc_score
            original_y=original["metadata"].RESP.to_numpy()
            original_metrics={"snapshots":len(original_y),"positives":int(original_y.sum()),
                "average_precision":float(average_precision_score(original_y,original["scores"])),
                "roc_auc":float(roc_auc_score(original_y,original["scores"]))}
            original_context={"evidence_class":"Reproduced","source_evidence_class":"Recovered from existing artifact",
                     "comparison_use":"descriptive ranking only; fit/score provenance unresolved",
                     "original_v63_validation":original_metrics,"provenance":original["provenance"],"lock_hash":tr.stable_hash(lock)}
            wh.save_artifacts(RUN_PREFIX+"_ORIGINAL_CONTEXT",{"original_context.json":wf.json_bytes(original_context)})
            display(original_context)
        except Exception as error:
            print("Original V63 descriptive score recovery unavailable:",type(error).__name__,"— no value invented; inspect private runtime error.")
        '''),
        md("""### Evidence-based conclusion
        The code below states the measured AP difference and its paired patient interval, then applies provenance restrictions.
        An interval containing zero does not prove equivalence. Secondary metrics, calibration and operational yields must be reviewed together;
        no retrospective AP difference establishes prospective benefit. If LightGBM wins, retain that conclusion. If execution has not occurred,
        the correct conclusion remains **unresolved**, not a placeholder performance claim.
        """), code('''
        conclusion=final_report["paired_comparison"]["conclusion"]
        print(conclusion["conclusion"])
        print("Claim restrictions:",conclusion["restrictions"])
        print("Final counts:",final_report["feature_counts"])
        print("Business interpretation:",final_report["business_limit"])
        print("No independent OOT comparison is established. Historical source/fit provenance remains in the evidence registry.")
        ''')]
    names=["01_tensor_initialization.ipynb","02_patient_level_split.ipynb","03_transformer_training.ipynb",
           "04_transformer_evaluation.ipynb","05_model_comparison.ipynb"]
    archive=REPO/"notebooks"/"Historical 49 Snapshot Padding (20260927)"
    if not archive.exists():
        shutil.copytree(DEST,archive)
    for name,cells in zip(names,(n1,n2,n3,n4,n5)):
        notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python"},"tak861":{"implementation_sha256":implementation,"execution_status":"not_production_executed"}},"cells":cells}
        for i,cell in enumerate(cells): cell["id"]=f"tak861-{i:03d}"
        (DEST/name).write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"notebooks":names,"implementation_sha256":implementation,"archive":str(archive)}))


if __name__=="__main__":
    build()
