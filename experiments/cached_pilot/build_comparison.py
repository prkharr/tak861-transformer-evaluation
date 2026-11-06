"""Build an independent, embedded cache comparison and its bounded handoff cell."""
import ast
import hashlib
import json
import re
from pathlib import Path

from experiments.automl_sequence.build_notebooks import code, notebook_source

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT / 'notebooks/expanded cohort'


def selected(path, names, imports=''):
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    chunks = [imports]
    for node in tree.body:
        name = getattr(node, 'name', None)
        assigned = {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()
        if name in names or assigned.intersection(names):
            chunks.append(ast.get_source_segment(source, node))
    return '\n\n'.join(chunks) + '\n'


def local_source(name):
    source = (HERE / (name + '.py')).read_text(encoding='utf-8-sig')
    return re.sub(r'^from experiments[^\n]*\n', '', source, flags=re.M)


def sources():
    paths = {
        'inputs': ROOT/'experiments/automl_sequence/inputs.py',
        'workflow': ROOT/'experiments/automl_sequence/workflow.py',
        'comparison': ROOT/'experiments/rigorous_transformer/comparison.py',
        'warehouse': ROOT/'experiments/temporal_selection/input_helpers.py',
        'pipeline': ROOT/'experiments/fresh_cohort/pipeline.py',
    }
    result = {name: path.read_text(encoding='utf-8-sig') for name, path in paths.items()}
    result['modeling'] = selected(ROOT/'experiments/automl_sequence/modeling.py', {'sample_weights', 'class_weight'}, 'import numpy as np')
    result['raw_automl'] = selected(ROOT/'experiments/automl_sequence/raw_automl.py', {'_feature_name'}, 'import re, hashlib')
    result['settings'] = selected(ROOT/'experiments/automl_sequence/fresh_lgbm.py', {'SETTINGS'})
    result['time_lgbm'] = selected(ROOT/'experiments/challenger/time_lgbm.py', {'FEATURE_NAMES', 'ROLES', 'channels', 'temporal_columns'}, 'import re\nimport numpy as np')
    result['cache'] = (HERE/'cache.py').read_text(encoding='utf-8-sig')
    result['analysis'] = local_source('analysis')
    result['artifacts'] = local_source('artifacts')
    return {name: notebook_source(source) for name, source in result.items()}


def loader(source_map, *, handoff=False):
    wanted = ('inputs', 'time_lgbm', 'cache', 'artifacts') if handoff else tuple(source_map)
    payload = {name: source_map[name] for name in wanted}
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in payload.items()}
    text = 'import hashlib, json, types\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n'
    text += 'PILOT_SOURCES = ' + repr(payload) + '\nPILOT_HASHES = ' + repr(hashes) + '\n'
    text += '''
def pilot_module(name, **injected):
    source = PILOT_SOURCES[name]
    if hashlib.sha256(source.encode()).hexdigest() != PILOT_HASHES[name]:
        raise ValueError("Embedded implementation checksum mismatch")
    module = types.ModuleType("pilot_" + name)
    module.__dict__.update(injected)
    exec(source, module.__dict__)
    return module
pilot_inputs = pilot_module("inputs")
pilot_time = pilot_module("time_lgbm")
pilot_cache = pilot_module("cache")
pilot_cache.inputs = pilot_inputs
pilot_artifacts = pilot_module("artifacts", inputs=pilot_inputs, time_lgbm=pilot_time)
'''
    if not handoff:
        text += '''
pilot_workflow = pilot_module("workflow")
pilot_comparison = pilot_module("comparison")
pilot_modeling = pilot_module("modeling")
pilot_raw = pilot_module("raw_automl")
pilot_settings = pilot_module("settings")
pilot_analysis = pilot_module("analysis", inputs=pilot_inputs, modeling=pilot_modeling,
    raw_automl=pilot_raw, workflow=pilot_workflow, FULL_SETTINGS=pilot_settings.SETTINGS,
    comparison=pilot_comparison, cache=pilot_cache)
pilot_analysis.SOURCE_SHA256 = hashlib.sha256(json.dumps(PILOT_HASHES, sort_keys=True).encode()).hexdigest()
pilot_warehouse = pilot_module("warehouse")
pilot_pipeline = pilot_module("pipeline")
'''
    return text


def build():
    source_map = sources()
    handoff = '''
HANDOFF_ROOT = "/local_disk0/tmp/tak861_cached_pilot/handoff_v001"
RAW_SQLITE_PATH = ""
PREVIOUS_CACHE_DIRECTORY = ""
EXACT_COUNT_CHANNELS = {"study": "", "diagnosis": ""}
CHUNK_LIMIT = None
'''+loader(source_map, handoff=True)+'''
if not RAW_SQLITE_PATH:
    raise ValueError("Set RAW_SQLITE_PATH to the raw.sqlite used by the paused AutoML extraction; do not create a new extraction")
if "counts" not in globals() or "CONFIG" not in globals() or "IMPLEMENTATION_SHA256" not in globals() or "RUN_PREFIX" not in globals():
    raise ValueError("Run this handoff cell in notebook 01 after a graceful stop of extraction; its counts and configuration must remain loaded")
pilot_root = Path(HANDOFF_ROOT)
pilot_root.mkdir(parents=True, exist_ok=True)
pilot_meta = pilot_inputs.normalize_snapshots(counts["metadata"])
pilot_count_receipt = pilot_artifacts.freeze_count_channels(counts, pilot_root/"counts", EXACT_COUNT_CHANNELS)
pilot_cache_manifest = pilot_cache.freeze_cache(RAW_SQLITE_PATH, pilot_root/"cache", pilot_meta,
    chunk_limit=CHUNK_LIMIT, previous_destination=PREVIOUS_CACHE_DIRECTORY or None)
pilot_handoff_path = pilot_artifacts.save_handoff(pilot_root, pilot_cache_manifest, pilot_count_receipt,
    pilot_meta, CONFIG, IMPLEMENTATION_SHA256, RUN_PREFIX)
print("Frozen handoff:", pilot_handoff_path)
print("Completed chunks:", len(pilot_cache.load_frozen_cache(pilot_root/"cache")["chunks"]))
print("Resume the original extraction cell with its existing settings. Use notebook 07 in a separate session.")
'''
    cells = [code('''
import os
import importlib.util
sf_options = {"sfUrl":"", "sfUser":"", "sfPassword":"", "sfRole":"", "sfWarehouse":"", "sfDatabase":"", "sfSchema":""}
HANDOFF_MANIFEST = "/local_disk0/tmp/tak861_cached_pilot/handoff_v001/handoff.json"
PILOT_OUTPUT = "/local_disk0/tmp/tak861_cached_pilot/results/v001"
PREVIOUS_PREPARED = ""
ENABLE_NLP = True
MIN_FREE_GIB = 3.0
BOOTSTRAP_DRAWS = 1000
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
missing_packages = [name for name in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "psutil", "matplotlib") if importlib.util.find_spec(name) is None]
if missing_packages:
    raise RuntimeError("Required packages are unavailable: " + ", ".join(missing_packages) + ". Configure the analysis session before running; do not restart a session running extraction.")
'''), code(loader(source_map)), code('''
handoff_path = Path(HANDOFF_MANIFEST)
if not handoff_path.is_file():
    raise FileNotFoundError("Run HANDOFF_CELL.py in the paused notebook 01 session first, then resume extraction. Set HANDOFF_MANIFEST to its output.")
handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
for path_field, hash_field in (("cache_manifest", "cache_manifest_sha256"), ("count_receipt", "count_receipt_sha256")):
    if pilot_artifacts.file_hash(handoff[path_field]) != handoff[hash_field]:
        raise ValueError("Handoff artifact checksum mismatch: " + path_field)
if pilot_artifacts.file_hash(handoff["parent_metadata"]) != handoff["metadata_file_sha256"]:
    raise ValueError("Handoff metadata checksum mismatch")
parent_metadata = pilot_inputs.normalize_snapshots(pd.read_csv(handoff["parent_metadata"], dtype={"PATIENT_ID":str}))
if pilot_inputs.frame_hash(parent_metadata) != handoff["metadata_sha256"]:
    raise ValueError("Handoff population identity changed")
pilot_analysis.memory_guard(0, MIN_FREE_GIB)
prepared = pilot_analysis.prepare(handoff["cache_manifest"], PILOT_OUTPUT,
    previous_version=PREVIOUS_PREPARED or None, max_features=512)
display(pilot_analysis.population_summary(prepared["metadata"]))
print("Complete patients from a sorted request prefix; exploratory development subset, not a random sample.")
print("TEST is unused. Validation keys and feature definitions remain fixed on refresh.")
print("AutoML columns:", len(prepared["feature_names"]))
'''), code('''
timing = pilot_artifacts.timing_group(handoff["count_receipt"], prepared["metadata"])
print("Timing columns:", timing["names"])
print("Rows with complete monthly history:", int(np.isfinite(timing["values"]).all(axis=1).sum()))
report = pilot_analysis.run(prepared, timing, bootstrap=BOOTSTRAP_DRAWS, min_free_gib=MIN_FREE_GIB)
display(pd.DataFrame(report["rows"]))
print("Saved report:", str(Path(PILOT_OUTPUT)/"report.json"))
'''), code('''
nlp = None
nlp_blocker = None
if ENABLE_NLP:
    if not all(sf_options.values()):
        nlp_blocker = "Connection settings are empty; matching NLP package not loaded"
    else:
        pilot_warehouse.spark = spark
        pilot_warehouse.sf_options_dl_poc = dict(sf_options)
        package = None
        try:
            pilot_analysis.memory_guard(0, MIN_FREE_GIB)
            package = pilot_pipeline.load_nlp(parent_metadata, handoff["config"], handoff["implementation"],
                handoff["run_prefix"], pilot_inputs, pilot_workflow, pilot_warehouse)
            package["_payload_verified"] = True
            nlp = pilot_artifacts.nlp_group(package, prepared["metadata"])
        except Exception as error:
            nlp_blocker = type(error).__name__ + ": matching NLP package unavailable or provenance failed; inspect the exception locally"
            print(nlp_blocker)
        finally:
            if package is not None:
                pilot_pipeline.release_nlp(package)
    if nlp is not None:
        print("NLP encoder was fitted on full parent TRAIN patients; monthly means summarize twelve positions.")
        report = pilot_analysis.run(prepared, timing, nlp, bootstrap=BOOTSTRAP_DRAWS, min_free_gib=MIN_FREE_GIB)
if nlp_blocker:
    pilot_artifacts.atomic_json(Path(PILOT_OUTPUT)/"nlp_blocker.json", {"reason":nlp_blocker})
    print(nlp_blocker, "; completed AutoML/timing fits remain available.")
'''), code('''
import matplotlib.pyplot as plt
rows = pd.DataFrame(report["rows"])
validation = rows.loc[rows.Split.eq("VALIDATION")].sort_values("Lift@10%", ascending=False, kind="stable").copy()
validation.insert(0, "Rank", validation["Lift@10%"].rank(ascending=False, method="min").astype(int))
display(validation[["Rank","Model","Lift@10%","Captured@10%","Selected@10%","Lift@20%","Lift@30%"]])
paired = []
for arm, result in report["paired_lift"].items():
    for fraction, delta in result["lift_by_fraction"].items():
        paired.append(dict(Model=arm, Compared_with="AUTOML", **delta))
display(pd.DataFrame(paired))
display(pd.DataFrame(report["overlaps"]).T)
fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
validation.plot.barh(x="Model", y="Lift@10%", ax=axes[0], legend=False, color="#0B6E75")
axes[0].invert_yaxis()
axes[0].axvline(1, color="#777777", linestyle="--")
axes[0].set_title("Development validation: lift at 10%")
rows.pivot(index="Model", columns="Split", values="Lift@10%").plot.bar(ax=axes[1], color=["#A5B4C2", "#0B6E75"])
axes[1].set_title("TRAIN versus validation: overfitting diagnostic")
axes[1].tick_params(axis="x", labelrotation=25)
plt.show()
fig.savefig(Path(PILOT_OUTPUT)/"lift_diagnostics.png", dpi=160, bbox_inches="tight")
print(report["scope"])
print("Additional capture is measured against AutoML on identical rows; historical production scores are excluded.")
print("Blocked arms:", report["blocked_arms"])
print("Refresh: create a new handoff version with PREVIOUS_CACHE_DIRECTORY, then set a new PILOT_OUTPUT and PREVIOUS_PREPARED to this version's prepared.json.")
print("Reuse completed fits after interruption by keeping the same handoff and PILOT_OUTPUT. No extraction or other notebook is rerun here.")
''')]
    for i, cell in enumerate(cells):
        cell['id'] = '07-' + str(i+1)
    book = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python')), cells=cells)
    raw = json.dumps(book, ensure_ascii=False, indent=1) + '\n'
    if re.search(r'\b(codex|claude|github|git|client|brian)\b', raw, re.I):
        raise ValueError('Unexpected notebook presentation text')
    ast.parse(handoff)
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST/'07_cached_feature_comparison_AutoML.ipynb'
    target.write_text(raw, encoding='utf-8', newline='\n')
    (HERE/'HANDOFF_CELL.py').write_text(handoff, encoding='utf-8', newline='\n')
    return target


if __name__ == '__main__':
    print(build())
