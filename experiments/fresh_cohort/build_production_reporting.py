"""Append production lift reporting without changing the frozen input/training identity."""
import ast
import hashlib
import json
from pathlib import Path
import re

from experiments.fresh_cohort import build_notebooks as core


def cells():
    source = core.cleaned((core.HERE/'production_comparison.py').read_text(encoding='utf-8'))
    identity = hashlib.sha256((source+Path(__file__).read_text(encoding='utf-8')).encode()).hexdigest()
    setup = core.base.code('''
import types
prodcmp = types.ModuleType("production_comparison")
PRODUCTION_REPORTING_SOURCE = '''+repr(source)+'''
PRODUCTION_REPORTING_SHA256 = '''+repr(identity)+'''
exec(PRODUCTION_REPORTING_SOURCE, prodcmp.__dict__)
PRODUCTION_CONFIG = {
    "score_table": "DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_UNIVERSE_W_FEATURES_SCORED",
    "score_column": "SCORE", "label_column": "RESP", "score_kind": "probability", "max_rows": 2000000,
    "evidence": {"production_model_fingerprint": "", "production_training_window_end": "",
        "production_training_patient_exclusion_reference": "", "target_definition_match_reference": "",
        "historical_input_vintage_reference": "", "patient_disjoint_from_production_training": False,
        "same_target_definition": False, "historical_inputs_frozen": False}}
print("Production is the frozen scoring artifact, separate from the newly fitted static control.")
print("The configured historical score table must cover every current evaluation key; otherwise its comparison is blocked.")
''')
    body = core.base.code('''
data = load_stage()
production_reports = {}
try:
    prep, protocol = preparation()
    lock = json.loads(wh.read_artifacts(RUN_PREFIX+"_LOCK", {"lock.json"})["lock.json"])
    for split in lift_reports:
        if split == "test" and not EVALUATE_TEST:
            raise ValueError("TEST remains disabled")
        meta = data["metadata"].iloc[data["indices"][split]].reset_index(drop=True)
        table = RUN_PREFIX+("_FINAL_TEST" if split == "test" else "_VALIDATION_COMPARISON")
        payload = wh.read_artifacts(table, {"report.json"}|{arm+".npy" for arm in protocol["arms"]})
        report = json.loads(payload["report.json"])
        if report["lock_hash"] != wf.digest(lock) or report["split"] != split:
            raise ValueError("Transformer prediction identity differs")
        scores = {arm:wf.bytes_array(payload[arm+".npy"]) for arm in protocol["arms"]}
        static_payload = wh.read_artifacts(baseline["root"]+"_"+split.upper()+"_LIFT", {"report.json", "baseline.npy"})
        static_report = json.loads(static_payload["report.json"])
        if static_report["identity"] != lift_reports[split]["identity"]:
            raise ValueError("Static control prediction identity differs")
        scores[fresh_lgbm.NAME] = wf.bytes_array(static_payload["baseline.npy"])
        for name, values in scores.items():
            if wf.array_sha256(values) != static_report["identity"]["predictions"][name]:
                raise ValueError("Saved comparison predictions changed")
        result = prodcmp.run(read_query, meta, PRODUCTION_CONFIG, scores, split,
            comparison=cmp, workflow=wf, allow_test=EVALUATE_TEST, selection_locked=True, n_bootstrap=1000)
        production_reports[split] = result
        print(result["evaluation_scope"])
        print("Production benchmark:",result["benchmark"]["status"],result["benchmark"]["statement"])
        if result["production"]["status"] == "ALIGNED":
            identity = dict(lock_hash=wf.digest(lock),reporting_hash=PRODUCTION_REPORTING_SHA256,
                production_configuration_hash=wf.digest(PRODUCTION_CONFIG),
                predictions_hash=result["prediction_identity_sha256"])
            artifact = RUN_PREFIX+"_PRODUCTION_"+split.upper()+"_"+wf.digest(identity)[:12].upper()
            record = dict(identity=identity,report=result)
            if wh.table_exists(artifact):
                old = json.loads(wh.read_artifacts(artifact,{"report.json"})["report.json"])
                if old != wf.plain(record):
                    raise ValueError("Frozen production comparison changed; preserve the earlier report")
            else:
                wh.save_artifacts(artifact,{"report.json":wf.json_bytes(record)})
            print("Saved production comparison:",artifact)
        else:
            print("Production BLOCKED:",result["production"].get("reason",""))
            print("Required evidence:",result["production"].get("required",[]))
        aggregate(prodcmp.report_table(result),100)
        aggregate(prodcmp.interval_table(result),100)
        ranked = prodcmp.ranking_table(result)
        display(plots.lift_ranking(ranked.loc[ranked["Lift@10%"].notna()],split.upper()+" — "+result["benchmark"]["status"]))
finally:
    wf.release_inputs(data)
''')
    final = core.base.code('''
for split, result in production_reports.items():
    print(RUN_MODE,split.upper(),"Lift@10% ranking —",result["benchmark"]["status"])
    display(prodcmp.ranking_table(result)[["Rank","Model","Lift@10%","Status"]])
''')
    for cell in (setup,body,final):
        cell['source'] = core.cleaned(''.join(cell['source'])).splitlines(True)
    return (setup,body,final), identity


def build():
    manifest = core.build()
    path = core.DEST/'05_model_comparison_AutoML.ipynb'
    book = json.loads(path.read_text(encoding='utf-8'))
    appended, identity = cells()
    book['cells'].extend(appended)
    for i,cell in enumerate(book['cells']):
        cell['id']='05-'+str(i)
        ast.parse(''.join(cell['source']))
    raw = json.dumps(book,ensure_ascii=False,indent=1)+'\n'
    if re.search(r'\b(codex|claude|github|git|client|brian)\b',raw,re.I):
        raise ValueError('Unexpected presentation text')
    path.write_text(raw,encoding='utf-8')
    entry = next(e for e in manifest['notebooks'] if Path(e['path']).name==path.name)
    entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cells=len(book['cells']))
    manifest['production_reporting_sha256'] = identity
    manifest['production_status'] = 'Frozen-score coverage and provenance checked at runtime; unavailable production has no lift or rank'
    (core.HERE/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
