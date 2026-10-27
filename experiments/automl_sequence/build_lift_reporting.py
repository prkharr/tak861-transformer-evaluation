"""Append reporting-only cells while preserving the reviewed training identity."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.automl_sequence import build_notebooks as core

BASELINE_REVISION = "2106cf1"
LOADER_FUNCTIONS = {"_frame", "_read_frozen_source", "_normalized_columns", "_keys", "_align", "load_original_v63_scores"}


def loader_source():
    source = subprocess.check_output([
        "git", "-c", "safe.directory=" + core.REPO.as_posix(), "show",
        BASELINE_REVISION + ":experiments/rigorous_transformer/baseline49.py"], cwd=core.REPO).decode("utf-8")
    tree = ast.parse(source)
    functions = [ast.get_source_segment(source, node) for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in LOADER_FUNCTIONS]
    if len(functions) != len(LOADER_FUNCTIONS):
        raise ValueError("Pinned shared score loader is incomplete")
    return core.notebook_source('import numpy as np\nimport pandas as pd\n'
        'DEFAULT_SOURCE_PREFIX = "DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_"\n\n' + "\n\n".join(functions))


def reporting_setup(loader, reporting, identity):
    return core.code(f'''
import io, types
existing_scores = types.ModuleType("existing_scores")
exec({loader!r}, existing_scores.__dict__)
lift_reporting = types.ModuleType("lift_reporting")
exec({reporting!r}, lift_reporting.__dict__)
REPORTING_SHA256 = {identity!r}
class ExistingScoresUnavailable(Exception):
    pass
def read_existing_score_table(table):
    try:
        return spark.read.format("snowflake").options(**sf_options_dl_poc).option("dbtable", table).load()
    except Exception as error:
        raise ExistingScoresUnavailable("Source access failed (" + type(error).__name__ + "); inspect permissions and source coverage privately") from None
def existing_lift_report(split, scores_by_model, allow_test=False, selection_locked=False):
    if split == "test" and not (allow_test and selection_locked):
        raise ValueError("TEST reporting requires explicit evaluation and a persisted lock")
    metadata = lift_metadata.loc[lift_metadata.SPLIT.eq(split)].reset_index(drop=True)
    canonical = metadata[["PATIENT_ID", "END_DT", "RESP", "SPLIT"]].copy()
    canonical["END_DT"] = pd.to_datetime(canonical.END_DT).dt.strftime("%Y-%m-%d")
    metadata_hash = wf.digest(canonical.to_dict("records"))
    baseline_table = RUN_PREFIX + "_EXISTING_V63_" + split.upper()
    names = {{"scores.npy", "identity.json"}}
    if wh.table_exists(baseline_table):
        saved_scores = wh.read_artifacts(baseline_table, names)
        receipt = json.loads(saved_scores["identity.json"])
        baseline_scores = np.load(io.BytesIO(saved_scores["scores.npy"]), allow_pickle=False)
        if receipt["metadata_hash"] != metadata_hash or receipt["scores_hash"] != wf.array_sha256(baseline_scores):
            raise ValueError("Existing-score snapshot identity mismatch")
        baseline = dict(metadata=metadata, scores=baseline_scores, provenance=receipt["provenance"])
    else:
        try:
            baseline = existing_scores.load_original_v63_scores(read_existing_score_table, metadata, split,
                allow_test=allow_test, selection_locked=selection_locked)
        except ValueError as error:
            raise ExistingScoresUnavailable(str(error)) from None
        except ExistingScoresUnavailable:
            raise
        except Exception as error:
            raise ExistingScoresUnavailable("Source collection failed (" + type(error).__name__ + "); inspect the source privately") from None
        baseline["provenance"]["source_table"] = existing_scores.DEFAULT_SOURCE_PREFIX + "UNIVERSE_W_FEATURES_SCORED"
        receipt = dict(metadata_hash=metadata_hash, scores_hash=wf.array_sha256(baseline["scores"]), provenance=baseline["provenance"])
        wh.save_artifacts(baseline_table, {{"scores.npy": wf.array_bytes(baseline["scores"]), "identity.json": wf.json_bytes(receipt)}})
    identity = dict(metadata_hash=metadata_hash, baseline_hash=receipt["scores_hash"],
        predictions={{name: wf.array_sha256(values) for name, values in scores_by_model.items()}}, reporting_hash=REPORTING_SHA256)
    report_table = baseline_table + "_LIFT"
    if wh.table_exists(report_table):
        saved_report = json.loads(wh.read_artifacts(report_table, {{"report.json"}})["report.json"])
        if saved_report["identity"] != identity:
            raise ValueError("Saved lift comparison belongs to different predictions or reporting implementation")
        result = saved_report["report"]
    else:
        result = lift_reporting.build_report(metadata, scores_by_model, baseline, split,
            comparison=cmp, workflow=wf, allow_test=allow_test, selection_locked=selection_locked, n_bootstrap=1000)
        wh.save_artifacts(report_table, {{"report.json": wf.json_bytes(dict(identity=identity, report=result))}})
    print("Evaluation split:", split.upper(), "TRAIN is resubstitution; validation is development evidence; TEST is retrospective.")
    print("Existing LightGBM fitting scope is unknown. This comparison is descriptive and does not establish generalization superiority.")
    aggregate(lift_reporting.report_table(result))
    aggregate(lift_reporting.interval_table(result), 100)
    return result
def try_existing_lift_report(split, scores_by_model, allow_test=False, selection_locked=False):
    try:
        return existing_lift_report(split, scores_by_model, allow_test, selection_locked)
    except ExistingScoresUnavailable as error:
        print("Existing scores unavailable for", split.upper(), ":", str(error))
        print("Transformer results remain available; no LightGBM lift, difference or rank is inferred.")
        metadata = lift_metadata.loc[lift_metadata.SPLIT.eq(split)].reset_index(drop=True)
        result = lift_reporting.build_report(metadata, scores_by_model, None, split,
            comparison=cmp, workflow=wf, allow_test=allow_test, selection_locked=selection_locked)
        aggregate(lift_reporting.report_table(result))
        return result
''')


def build():
    core.build()
    loader = loader_source()
    reporting = core.notebook_source((core.HERE / "lift_reporting.py").read_text(encoding="utf-8"))
    identity = hashlib.sha256((loader + reporting + Path(__file__).read_text(encoding="utf-8")).encode()).hexdigest()
    manifest_path = core.HERE / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["notebooks"]:
        path = core.REPO / entry["path"]
        book = json.loads(path.read_text(encoding="utf-8"))
        stage = path.name[:2]
        if stage in ("03", "04", "05"):
            for cell in book["cells"]:
                source = "".join(cell["source"])
                if "wf.release_inputs(data)" in source:
                    source = source.replace("wf.release_inputs(data)", 'lift_metadata = data["metadata"][["PATIENT_ID", "END_DT", "RESP", "SPLIT"]].copy()\nwf.release_inputs(data)')
                source = source.replace("No independent OOT conclusion. No completed LightGBM comparison in this run.",
                                        "No independent OOT conclusion. Existing-score comparison follows.")
                cell["source"] = source.splitlines(True)
            setup = reporting_setup(loader, reporting, identity)
            if stage in ("03", "04"):
                splits = '("train", "validation")' if stage == "03" else '("validation",)'
                body = core.code(f'''
for evaluation_split in {splits}:
    score_key = "train_scores" if evaluation_split == "train" else "val_scores"
    ensemble_scores = {{name: np.mean([run[score_key] for run in runs], axis=0) for name, runs in results.items()}}
    lift_comparison = try_existing_lift_report(evaluation_split, ensemble_scores)
''')
            else:
                body = core.code('''
evaluation_split = report["split"]
prediction_table = RUN_PREFIX + ("_FINAL_TEST" if RUN_FINAL_TEST else "_VALIDATION_COMPARISON")
prediction_artifacts = wh.read_artifacts(prediction_table, {"report.json"} | {name + ".npy" for name in protocol["arms"]})
ensemble_scores = {name: np.load(io.BytesIO(prediction_artifacts[name + ".npy"]), allow_pickle=False) for name in protocol["arms"]}
lift_comparison = try_existing_lift_report(evaluation_split, ensemble_scores,
    allow_test=RUN_FINAL_TEST, selection_locked=True)
''')
            setup["source"] += ["\n"] + body["source"]
            book["cells"].append(setup)
            if stage == "05":
                book["cells"].append(core.code('''
print("Lift@10% ranking — " + evaluation_split.upper() + " — descriptive existing-score comparison")
if lift_comparison["interpretation"] == "TRANSFORMER_ONLY_EXISTING_SCORES_UNAVAILABLE":
    print("Existing LightGBM unavailable; ranking contains only the three Transformer ensembles.")
display(lift_reporting.ranking_table(lift_comparison))
'''))
        book["metadata"]["tak861"]["reporting_sha256"] = identity
        for index, cell in enumerate(book["cells"]):
            cell["id"] = f"{stage}-{index:03d}"
            ast.parse("".join(cell["source"]))
        path.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cells=len(book["cells"]))
    manifest.update(reporting_sha256=identity, reporting_generator="experiments/automl_sequence/build_lift_reporting.py",
                    existing_score_loader_revision=BASELINE_REVISION,
                    existing_score_loader_sha256=hashlib.sha256(loader.encode()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    from experiments.automl_sequence.build_fresh_lgbm import build as append_fresh_baseline
    append_fresh_baseline()


if __name__ == "__main__":
    build()
