"""Build four standalone temporal feature-selection notebooks."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
from textwrap import dedent
import zipfile
from early_stages import EARLY_HELPERS_SOURCE, early_notebook_cells

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO.parent / "outputs" / "Temporal_Feature_Selection"
MIRROR = REPO / "notebooks" / "Temporal Feature Selection"


def read(path):
    return path.read_text(encoding="utf-8-sig")


def cell(number, title, body):
    return f"# CELL {number}: {title}\n" + dedent(body).strip() + "\n"


INPUTS = read(HERE / "input_helpers.py")
HELPERS = INPUTS + '\n' + EARLY_HELPERS_SOURCE
HELPERS += '\n' + '\n'.join(read(REPO / 'src' / name) for name in ('model.py', 'metrics.py', 'reproducibility.py'))
HELPERS += '\n' + read(HERE / 'training_helpers.py') + '\n' + read(HERE / 'workflow.py')
# Shared decile calculation, embedded in full rather than importing at runtime.
source = read(REPO / 'experiments' / 'manual_features' / 'core.py')
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'rank_tables')
HELPERS += '\n' + ast.get_source_segment(source, node)
HASH = hashlib.sha256((HELPERS + read(Path(__file__)) + read(HERE / 'early_stages.py')).encode()).hexdigest()
HELPERS += f'\nIMPLEMENTATION_SHA256 = "{HASH}"\n'

CONFIG = '''
# Paste private sf_options = {...} here. No credentials are supplied.
# Runtime: Spark Snowflake connector, numpy, pandas, sklearn, torch, matplotlib.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if "sf_options" not in globals() or not isinstance(sf_options, dict) or "spark" not in globals():
    raise RuntimeError("Supply sf_options on the approved Spark runtime.")
sf_options_dl_poc = dict(sf_options)
sf_options_dl_poc.update(sfDatabase="DSVC_TAKEDA_TA_PRIVATE", sfSchema="DS_ML")
PREFIX = "TAK861_TX_READY_V63_DL_POC"
EXPERIMENT_PREFIX = PREFIX + "_TEMPORAL_SELECTION_V1"
REFERENCE_RUN_ID = "RUN_001"
RUN_ID = "P001"  # New ID for changed ranking/training settings or code; same ID in all four notebooks.
PLAN_SEED = 42
import re
if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", RUN_ID):
    raise ValueError("RUN_ID must be a short uppercase identifier.")
PREPARATION_TABLE = EXPERIMENT_PREFIX + "_PREPARATION"
SPLIT_AUDIT_TABLE = EXPERIMENT_PREFIX + "_SPLIT_AUDIT"
REFERENCE_MODEL_TABLE = PREFIX + "_MODEL_" + REFERENCE_RUN_ID
RUN_PREFIX = EXPERIMENT_PREFIX + "_" + RUN_ID
INTERNAL_TABLE = RUN_PREFIX + "_INTERNAL"
RANK_MODEL_TABLE = RUN_PREFIX + "_RANK_MODEL"
SELECTION_TABLE = RUN_PREFIX + "_FEATURE_SELECTION"
MODEL_TABLE = RUN_PREFIX + "_MODEL"
EVALUATION_TABLE = RUN_PREFIX + "_EVALUATION"
'''

early = early_notebook_cells(CONFIG, INPUTS)
preparation = early['01_tensor_initialization_DL_POC.ipynb']
splitting = early['02_patient_level_split_DL_POC.ipynb']
splitting[1] = cell(2, 'Embedded split checks and internal patient-split helpers', HELPERS)
splitting[6] += dedent('''
plan = make_internal_plan(metadata, PLAN_SEED)
validate_plan(metadata, plan)
plan_info = {"seed": PLAN_SEED, "source_hash": split_report["snapshot_manifest_sha256"],
             "records": plan.to_dict("records"), "method": "train_patient_60_20_20_latest_rank_snapshot_same_cutoff"}
save_artifacts(INTERNAL_TABLE, {"plan.json": canonical_json(plan_info).encode()})
display(plan.groupby("ROLE").agg(patients=("PATIENT_ID", "nunique"), snapshots=("RESP", "size"), positives=("RESP", "sum")))
print("Ranking snapshots:", int(plan.RANK_EVALUATE.sum()), "| Ranking positives:", int(plan.loc[plan.RANK_EVALUATE, "RESP"].sum()))
print("Outer splits unchanged. Internal fitting, stopping and ranking patients are mutually exclusive.")
''')

LOAD = '''
if "data" in globals():
    release_inputs(data)
data = load_inputs()
preparation_audit = json.loads(read_artifacts(PREPARATION_TABLE, {"preparation_report.json"})["preparation_report.json"])
split_audit = json.loads(read_artifacts(SPLIT_AUDIT_TABLE, {"split_audit.json"})["split_audit.json"])
verify_audits(data, preparation_audit, split_audit)
plan_info = json.loads(read_artifacts(INTERNAL_TABLE, {"plan.json"})["plan.json"])
if plan_info["seed"] != PLAN_SEED or plan_info["source_hash"] != data["hashes"]["snapshot_manifest_sha256"]:
    raise ValueError("Internal plan differs from this run.")
plan = validate_plan(data["metadata"], pd.DataFrame(plan_info["records"]))
if plan.to_dict("records") != make_internal_plan(data["metadata"], PLAN_SEED).to_dict("records"):
    raise ValueError("Internal plan does not reproduce the declared patient split.")
'''

training = [
cell(1, 'Connection and fixed top-50 temporal selection experiment', CONFIG + '''
TOP_K = 50
PERMUTATION_REPEATS = 5
PERMUTATION_SEED = 1042
FEATURES_PER_CHUNK = 32  # Completed chunks are saved for restart; does not change ranking.
MODEL_SETTINGS = dict(seq_len=12, d_model=128, n_heads=4, encoder_layers=2,
                      feedforward_dim=256, dropout=.2)
TRAINING_SETTINGS = dict(seed=42, epochs=20, patience=5, min_delta=1e-4, batch_size=64,
                        learning_rate=.001, weight_decay=.0001, grad_clip=1., device="auto")
# Preserve original AP checkpoint selection and training settings.
# Importance uses lift drop; AP drop breaks ranking ties. TEST is never scored here.
'''),
cell(2, 'Embedded original temporal encoder, training, permutation and storage helpers', HELPERS),
cell(3, 'Load unchanged tensor and verify the declared experiment', LOAD + '''
if type(PERMUTATION_REPEATS) is not int or PERMUTATION_REPEATS < 2:
    raise ValueError("Use at least two permutation repetitions.")
if type(FEATURES_PER_CHUNK) is not int or FEATURES_PER_CHUNK < 1:
    raise ValueError("FEATURES_PER_CHUNK must be positive.")
if type(TOP_K) is not int or not 0 < TOP_K < len(data["features"]):
    raise ValueError("TOP_K must be between 1 and the original feature count minus one.")
reference = json.loads(read_artifacts(REFERENCE_MODEL_TABLE, REFERENCE_TRAINING_ARTIFACT_NAMES)["training_summary.json"])
if any(reference["model_config"].get(k) != v for k, v in MODEL_SETTINGS.items()):
    raise ValueError("Architecture differs from original RUN_001; inspect original settings.")
if any(reference["training_settings"].get(k) != v for k, v in TRAINING_SETTINGS.items() if k != "device"):
    raise ValueError("Training settings differ from original RUN_001; inspect original settings.")
contract = {"implementation": IMPLEMENTATION_SHA256, "input_hashes": data["hashes"],
            "plan_sha256": digest_json(plan_info), "top_k": TOP_K,
            "repeats": PERMUTATION_REPEATS, "permutation_seed": PERMUTATION_SEED,
            "chunk_size": FEATURES_PER_CHUNK, "model": MODEL_SETTINGS, "training": TRAINING_SETTINGS}
fit_rows, stop_rows, ranking_rows = [plan_rows(data, plan, role) for role in ("fit", "stop", "rank")]
all_columns = list(range(len(data["features"])))
ranking_data = selected_view(data, all_columns, fit_rows, stop_rows)
print("Original tensor:", data["X"].shape, "| Final selected tensor:", (len(data["X"]), 12, TOP_K))
print("Importance scoring needs", len(all_columns) * PERMUTATION_REPEATS, "holdout inference passes; GPU recommended.")
'''),
cell(4, 'Fit the ranking encoder on internal fitting patients only', '''
model_names = {"checkpoint.pt", "summary.json", "history.csv"}
if not table_exists(RANK_MODEL_TABLE):
    artifacts = model_artifacts(ranking_data, ModelConfig(input_dim=len(all_columns), **MODEL_SETTINGS),
        TRAINING_SETTINGS, RUN_ID + "_RANK", contract)
    save_artifacts(RANK_MODEL_TABLE, artifacts)
rank_blobs = read_artifacts(RANK_MODEL_TABLE, model_names)
rank_model, rank_payload, rank_device = checked_model(rank_blobs, ranking_data, RUN_ID + "_RANK", contract)
print("Ranking encoder checkpoint chosen using internal stopping patients only.")
'''),
cell(5, 'Rank all original features by same-date whole-sequence permutation', '''
ranking_meta = data["metadata"].iloc[ranking_rows].reset_index(drop=True)
ranking_y = data["y"][ranking_rows]
unshuffled = score_rows(rank_model, data, ranking_rows, rank_device)
baseline = {"lift": top10_lift(ranking_y, unshuffled, ranking_meta),
            "ap": float(average_precision_score(ranking_y, unshuffled))}
print("Independent internal ranking baseline:", baseline)
if baseline["lift"] <= 1:
    raise ValueError("Ranking encoder has no top-decile enrichment on its independent holdout; do not trust this ranking.")
records = []
rank_contract = {"contract": contract, "checkpoint_sha256": hashlib.sha256(rank_blobs["checkpoint.pt"]).hexdigest()}
for start in range(0, len(all_columns), FEATURES_PER_CHUNK):
    end = min(start + FEATURES_PER_CHUNK, len(all_columns))
    table = RUN_PREFIX + f"_IMPORTANCE_{start:04d}"
    if table_exists(table):
        chunk = json.loads(read_artifacts(table, {"importance.json"})["importance.json"])
        if (chunk["contract"] != rank_contract or chunk["start"] != start or chunk["end"] != end
                or not np.isclose(chunk["baseline"]["lift"], baseline["lift"], rtol=0, atol=1e-10)
                or not np.isclose(chunk["baseline"]["ap"], baseline["ap"], rtol=0, atol=1e-8)):
            raise ValueError("Saved ranking chunk differs; use the matching runtime or a new RUN_ID.")
    else:
        items = []
        for feature_index in range(start, end):
            result = rank_feature(rank_model, data, ranking_rows, rank_device, feature_index,
                                  PERMUTATION_REPEATS, PERMUTATION_SEED, baseline)
            items.append(result)
            print(f"Feature {feature_index+1}/{len(all_columns)}: lift drop={result['mean_lift_drop']:.4f}", flush=True)
        chunk = {"contract": rank_contract, "start": start, "end": end, "baseline": baseline, "records": items}
        save_artifacts(table, {"importance.json": json_blob(chunk)})
    if [r["feature_index"] for r in chunk["records"]] != list(range(start, end)):
        raise ValueError("Ranking chunk indices are incomplete or reordered.")
    records.extend(chunk["records"])
ranking, selected_indices = selection_from_ranking(records, data["features"], TOP_K)
manifest = {"contract": contract, "selected_indices": selected_indices,
            "selected_features": [data["features"][i] for i in selected_indices],
            "source_features": data["features"], "ranking_baseline": baseline,
            "ranking_patients": len(ranking_rows), "ranking_positives": int(ranking_y.sum()),
            "ranking_checkpoint_sha256": rank_contract["checkpoint_sha256"],
            "selection_rule": "mean_top10_lift_drop_then_mean_AP_drop_then_original_feature_index",
            "time_steps": list(range(12)), "test_used": False, "outer_validation_used_for_ranking": False}
save_artifacts(SELECTION_TABLE, {"manifest.json": json_blob(manifest), "ranking.json": json_blob(records),
                               "ranking.csv": ranking.to_csv(index=False).encode()})
display(ranking.drop(columns="lift_drops"))
print("Selected features with nonpositive measured lift drop:", int((ranking.loc[ranking.selected, "mean_lift_drop"] <= 0).sum()))
print("Top 50 is a fixed experiment, not proof that every selected feature is relevant. Correlated features can mask importance.")
del rank_model, rank_payload, rank_blobs
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''),
cell(6, 'Retrain the original encoder on full TRAIN with selected monthly features', '''
selected_data = selected_view(data, selected_indices)
final_contract = {"experiment": contract, "selection_sha256": digest_json(manifest)}
if not table_exists(MODEL_TABLE):
    artifacts = model_artifacts(selected_data, ModelConfig(input_dim=len(selected_indices), **MODEL_SETTINGS),
                                TRAINING_SETTINGS, RUN_ID, final_contract)
    save_artifacts(MODEL_TABLE, artifacts)
final_blobs = read_artifacts(MODEL_TABLE, model_names)
model, payload, device = checked_model(final_blobs, selected_data, RUN_ID, final_contract)
summary = json.loads(final_blobs["summary.json"])
print("Selected input shape:", selected_data["X"].shape, "| Best epoch:", summary["best_epoch"])
'''),
cell(7, 'Training and validation curves, lift and deciles', '''
import matplotlib.pyplot as plt
history = pd.read_csv(io.BytesIO(final_blobs["history.csv"]))
plot_history(history, "Selected monthly features")
plt.show()
for split in ("train", "validation"):
    _, labels, scores = predict_loader(model, make_loader(selected_data, split, TRAINING_SETTINGS["batch_size"], 42), device)
    meta = data["metadata"].iloc[data["indices"][split]]
    deciles, top = rank_tables(meta, scores)
    print(split.upper(), "top-10% lift:", top10_lift(labels, scores, meta))
    display(deciles)
    display(top)
'''),
cell(8, 'Training complete; retain the frozen feature selection for evaluation', '''
print("Run notebook 04 with the same RUN_ID. No TEST scores were used for ranking or training.")
print("All 12 monthly timesteps are retained; only the feature axis was reduced.")
release_inputs(data)
del model, selected_data, ranking_data, data
'''),
]

evaluation = [
cell(1, 'Connection and matching run identifier', CONFIG),
cell(2, 'Embedded encoder, metrics and artifact verification', HELPERS),
cell(3, 'Load the frozen feature selection and verify its ranking', LOAD + '''
selection_blobs = read_artifacts(SELECTION_TABLE, {"manifest.json", "ranking.json", "ranking.csv"})
manifest = json.loads(selection_blobs["manifest.json"])
contract = manifest["contract"]
if (contract["implementation"] != IMPLEMENTATION_SHA256 or contract["input_hashes"] != data["hashes"]
        or contract["plan_sha256"] != digest_json(plan_info) or manifest["source_features"] != data["features"]
        or manifest["time_steps"] != list(range(12)) or manifest["test_used"] is not False
        or manifest["outer_validation_used_for_ranking"] is not False):
    raise ValueError("Selection provenance does not match the unchanged temporal input.")
ranking, selected_indices = selection_from_ranking(json.loads(selection_blobs["ranking.json"]), data["features"], contract["top_k"])
if selected_indices != manifest["selected_indices"] or manifest["selected_features"] != [data["features"][i] for i in selected_indices]:
    raise ValueError("Saved selected features differ from the frozen ranking.")
selected_data = selected_view(data, selected_indices)
'''),
cell(4, 'Restore the saved temporal model and threshold', '''
blobs = read_artifacts(MODEL_TABLE, {"checkpoint.pt", "summary.json", "history.csv"})
final_contract = {"experiment": contract, "selection_sha256": digest_json(manifest)}
model, payload, device = checked_model(blobs, selected_data, RUN_ID, final_contract)
threshold = payload["validation_threshold"]
print("Tensor shape:", selected_data["X"].shape, "| Best epoch:", payload["best_epoch"], "| Threshold:", threshold)
'''),
cell(5, 'Score TRAIN, VALIDATION and TEST with the saved model', '''
reports, score_frames = {}, []
for split in ("train", "validation", "test"):
    _, labels, scores = predict_loader(model, make_loader(selected_data, split, payload["training_settings"]["batch_size"], 42), device)
    meta = data["metadata"].iloc[data["indices"][split]].reset_index(drop=True)
    deciles, top = rank_tables(meta, scores)
    metrics = dict(split=split, **classification_metrics(labels, scores, threshold),
        top10_lift=top10_lift(labels, scores, meta), snapshots=len(labels), positives=int(labels.sum()))
    reports[split] = {"metrics": metrics, "deciles": deciles, "top": top}
    score_frames.append(meta[["PATIENT_ID", "END_DT", "RESP", "SPLIT"]].assign(SCORE=scores))
    print(split.upper(), "top-10% lift:", metrics["top10_lift"])
    display(deciles)
    display(top)
summary = json.loads(blobs["summary.json"])
if not np.isclose(reports["validation"]["metrics"]["average_precision"], summary["best_validation_average_precision"], atol=1e-7):
    raise ValueError("Restored model does not reproduce saved validation AP.")
metrics_frame = pd.DataFrame([item["metrics"] for item in reports.values()])
display(metrics_frame)
'''),
cell(6, 'Plot training curves and lift across all splits', '''
import matplotlib.pyplot as plt
fig = plot_history(pd.read_csv(io.BytesIO(blobs["history.csv"])), "Selected monthly features")
buffer = io.BytesIO()
fig.savefig(buffer, format="png", dpi=150)
history_png = buffer.getvalue()
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for split, report in reports.items():
    axes[0].plot(report["deciles"].decile, report["deciles"].lift, marker="o", label=split)
    axes[1].plot(report["top"].fraction * 100, report["top"].lift, marker="o", label=split)
axes[0].set(xlabel="Decile (10 is highest)", ylabel="Lift")
axes[0].invert_xaxis()
axes[1].set(xlabel="Top % of snapshots", ylabel="Lift")
for ax in axes:
    ax.legend()
    ax.grid(alpha=.3)
fig.tight_layout()
buffer = io.BytesIO()
fig.savefig(buffer, format="png", dpi=150)
lift_png = buffer.getvalue()
plt.show()
'''),
cell(7, 'Save results, selected features and exact-key predictions privately', '''
evaluation_report = {"run_id": RUN_ID, "contract": final_contract,
    "metrics": {split: item["metrics"] for split, item in reports.items()},
    "training_is_in_sample": True, "test_is_retrospective": True,
    "ranking_scope": "One latest snapshot per internal TRAIN ranking patient, permuted within exact cutoff date.",
    "limitations": ["Importance is model-specific and correlated features may mask each other.",
        "Permutation variability is not a confidence interval for future performance.",
        "Existing TEST was previously inspected; a new cohort is needed for untouched confirmation."]}
artifacts = {"evaluation.json": json_blob(evaluation_report), "metrics.csv": metrics_frame.to_csv(index=False).encode(),
    "predictions.csv": pd.concat(score_frames, ignore_index=True).to_csv(index=False).encode(),
    "selection.json": json_blob(manifest), "ranking.csv": selection_blobs["ranking.csv"],
    "training_history.png": history_png, "lift.png": lift_png}
for split, item in reports.items():
    artifacts[split + "_deciles.csv"] = item["deciles"].to_csv(index=False).encode()
    artifacts[split + "_top_k.csv"] = item["top"].to_csv(index=False).encode()
if table_exists(EVALUATION_TABLE):
    saved = read_artifacts(EVALUATION_TABLE, artifacts.keys())
    if saved["evaluation.json"] != artifacts["evaluation.json"]:
        raise ValueError("Existing evaluation differs; investigate before using a new RUN_ID.")
    print("Matching evaluation retained.")
else:
    save_artifacts(EVALUATION_TABLE, artifacts)
'''),
cell(8, 'Final lift report', '''
for split, item in reports.items():
    m = item["metrics"]
    print(f"{split.upper()}: top-10% lift={m['top10_lift']:.3f}; AP={m['average_precision']:.4f}; AUC={m['roc_auc']:.4f}")
print("Monthly sequence retained. Feature ranking used internal TRAIN holdout only.")
print("Compare with original results at the same targeting fraction; improvement is not guaranteed.")
print("Saved:", EVALUATION_TABLE)
release_inputs(data)
del model, selected_data, data
'''),
]

SPECS = [('01_tensor_initialization.ipynb', preparation), ('02_patient_level_split.ipynb', splitting),
         ('03_transformer_training.ipynb', training), ('04_transformer_evaluation.ipynb', evaluation)]


def build():
    for folder in (OUT, MIRROR):
        folder.mkdir(parents=True, exist_ok=True)
    for name, cells in SPECS:
        assert len(cells) == 8
        for i, source in enumerate(cells):
            compile(source, f'{name}:{i}', 'exec')
        payload = dict(nbformat=4, nbformat_minor=5, metadata={
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            'language_info': {'name': 'python'}}, cells=[dict(cell_type='code', id=f'temporal-{i+1}',
            metadata={}, execution_count=None, outputs=[], source=s) for i, s in enumerate(cells)])
        target = OUT / name
        target.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
        shutil.copy2(target, MIRROR / name)
    for name in ('START_HERE.md', 'VALIDATION.md'):
        if (HERE / name).exists():
            for folder in (OUT, MIRROR):
                shutil.copy2(HERE / name, folder / name)
    archive = OUT.parent / 'Temporal_Feature_Selection_4_Notebooks.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()):
            if path.suffix in {'.ipynb', '.md'}:
                z.write(path, path.name)
    print(archive)


if __name__ == '__main__':
    build()
