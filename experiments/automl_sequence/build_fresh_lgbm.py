import ast
import hashlib
import json
from pathlib import Path

from experiments.automl_sequence import build_notebooks as core


MARKER = "FRESH_LGBM_EXTENSION"


def build():
    path = core.DEST / "05_model_comparison_AutoML.ipynb"
    book = json.loads(path.read_text(encoding="utf-8"))
    book["cells"] = [c for c in book["cells"] if MARKER not in "".join(c["source"])]
    source = core.notebook_source((core.HERE / "fresh_lgbm.py").read_text(encoding="utf-8"))
    identity = hashlib.sha256(source.encode()).hexdigest()
    setup = f'''
FRESH_LGBM_EXTENSION = "setup"
import importlib.util
if importlib.util.find_spec("lightgbm") is None:
    raise RuntimeError("LightGBM is required. Install lightgbm==4.6.0 on this compute, then rerun setup and the new baseline cells only.")
fresh_lgbm = types.ModuleType("fresh_lgbm")
FRESH_LGBM_SHA256 = {identity!r}
fresh_source = {source!r}
if hashlib.sha256(fresh_source.encode()).hexdigest() != FRESH_LGBM_SHA256:
    raise ValueError("Baseline source integrity failure")
exec(fresh_source, fresh_lgbm.__dict__)
'''
    train = '''
FRESH_LGBM_EXTENSION = "fit"
fresh_data = load_stage()
try:
    fresh_saved = wh.read_artifacts(RUN_PREFIX + "_PREPARATION", wf.PREP_NAMES)
    fresh_prep = json.loads(fresh_saved["preparation.json"])
    fresh_protocol = json.loads(fresh_saved["protocol.json"])
    fresh_lock = json.loads(wh.read_artifacts(RUN_PREFIX + "_LOCK", {"lock.json"})["lock.json"])
    fresh_baseline = fresh_lgbm.fit_or_resume(fresh_data, fresh_prep, fresh_protocol,
        fresh_lock, wf, mdl, wh, RUN_PREFIX, FRESH_LGBM_SHA256)
    fresh_reports = {}
    print("Saved baseline prefix:", fresh_baseline["root"])
except BaseException:
    wf.release_inputs(fresh_data)
    del fresh_data
    raise
'''
    compare = '''
FRESH_LGBM_EXTENSION = "compare"
FRESH_LGBM_EVALUATE_TEST = False
if "fresh_data" not in globals():
    fresh_data = load_stage()
try:
    for fresh_split in (["validation", "test"] if FRESH_LGBM_EVALUATE_TEST else ["validation"]):
        fresh_table = RUN_PREFIX + ("_FINAL_TEST" if fresh_split == "test" else "_VALIDATION_COMPARISON")
        if not wh.table_exists(fresh_table):
            raise ValueError("Frozen Transformer predictions are missing for " + fresh_split + "; run the existing gated comparison before continuing")
        fresh_predictions = wh.read_artifacts(fresh_table, {"report.json"} | {a + ".npy" for a in fresh_protocol["arms"]})
        fresh_report_identity = json.loads(fresh_predictions["report.json"])
        if fresh_report_identity["lock_hash"] != wf.digest(fresh_lock) or fresh_report_identity["split"] != fresh_split:
            raise ValueError("Transformer predictions belong to a different lock or split")
        fresh_scores = {a: wf.bytes_array(fresh_predictions[a + ".npy"]) for a in fresh_protocol["arms"]}
        fresh_reports[fresh_split] = fresh_lgbm.compare(fresh_data, fresh_prep, fresh_baseline,
            fresh_scores, fresh_split, wf, mdl, cmp, wh, allow_test=FRESH_LGBM_EVALUATE_TEST,
            n_bootstrap=1000)
        print(fresh_reports[fresh_split]["scope"])
        print(fresh_reports[fresh_split]["representation"])
        display(fresh_lgbm.ranking(fresh_reports[fresh_split]))
        display(pd.DataFrame(fresh_reports[fresh_split]["intervals"]))
finally:
    wf.release_inputs(fresh_data)
    del fresh_data
'''
    rank = '''
FRESH_LGBM_EXTENSION = "ranking"
for fresh_split, fresh_result in fresh_reports.items():
    print("Lift@10% ranking:", fresh_split.upper())
    display(fresh_lgbm.ranking(fresh_result)[["Rank", "Model", "Lift@10%"]])
'''
    for text in (setup, train, compare, rank):
        cell = core.code(text)
        cell["source"] = core.notebook_source("".join(cell["source"])).splitlines(True)
        cell["id"] = "05-fresh-" + str(len(book["cells"]))
        book["cells"].append(cell)
    for cell in book["cells"]:
        ast.parse("".join(cell["source"]))
    book["metadata"]["tak861"]["fresh_lgbm_sha256"] = identity
    path.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = core.HERE / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["notebooks"]:
        if Path(entry["path"]).name == path.name:
            entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cells=len(book["cells"]))
    manifest["fresh_lgbm_extension"] = dict(source_sha256=identity,
        generator="experiments/automl_sequence/build_fresh_lgbm.py", clinical_execution=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(path)
    print("Fresh baseline source:", identity)


if __name__ == "__main__":
    build()
