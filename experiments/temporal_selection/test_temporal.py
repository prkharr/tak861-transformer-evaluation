import importlib.util
import io
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_notebooks as builder

torch.set_num_threads(1)
scope = {"__name__": __name__}
exec(builder.HELPERS, scope)


def synthetic():
    rows = []
    for split, count in (("train", 240), ("validation", 40), ("test", 40)):
        for p in range(count):
            for month in ("2025-04-21", "2025-05-21"):
                rows.append([f"{split}_{p:04}", month, p % 2, split, '{}'])
    meta = pd.DataFrame(rows, columns=["PATIENT_ID", "END_DT", "RESP", "SPLIT", "SPLIT_CONFIG"])
    meta = meta.sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)
    y = meta.RESP.to_numpy(dtype=np.float32)
    rng = np.random.default_rng(55)
    X = rng.poisson(1, size=(len(meta), 12, 4)).astype(np.float32)
    X[:, :, 0] = y[:, None] * 10 + np.arange(12)[None, :] * .01
    return {"X": X, "y": y, "metadata": meta, "features": ["DX__A", "RX__B", "PX__C", "DX__D"],
            "indices": {s: np.flatnonzero(meta.SPLIT.eq(s)) for s in ("train", "validation", "test")},
            "hashes": {"snapshot_manifest_sha256": "synthetic", "model_input_sha256": "synthetic"}}


def test_internal_patient_separation_and_date_matched_donors():
    data = synthetic()
    plan = scope["make_internal_plan"](data["metadata"])
    scope["validate_plan"](data["metadata"], plan)
    ids = {r: set(plan.loc[plan.ROLE.eq(r), "PATIENT_ID"]) for r in ("fit", "stop", "rank")}
    assert not (ids["fit"] & ids["stop"] or ids["rank"] & (ids["fit"] | ids["stop"]))
    ranked = plan.loc[plan.RANK_EVALUATE].reset_index(drop=True)
    assert not ranked.PATIENT_ID.duplicated().any()
    assert ranked.END_DT.eq("2025-05-21").all()
    donors = scope["same_date_donors"](ranked, 13)
    assert np.all(donors != np.arange(len(ranked)))
    assert sorted(donors) == list(range(len(ranked)))
    assert ranked.END_DT.tolist() == ranked.iloc[donors].END_DT.tolist()
    pd.testing.assert_frame_equal(plan, scope["make_internal_plan"](data["metadata"]))
    corrupted = plan.copy()
    corrupted.loc[0, "PATIENT_ID"] = "test_0000"
    with pytest.raises(ValueError):
        scope["validate_plan"](data["metadata"], corrupted)


def test_multiple_date_groups_and_singleton_rejection():
    meta = pd.DataFrame({"PATIENT_ID": list('abcdef'), "END_DT": ['A']*3+['B']*3})
    donors = scope["same_date_donors"](meta, 42)
    assert meta.END_DT.tolist() == meta.iloc[donors].END_DT.tolist()
    meta.loc[0, 'END_DT'] = 'unique'
    with pytest.raises(ValueError):
        scope["same_date_donors"](meta, 42)


def test_whole_sequence_permutation_and_no_heldout_reads():
    data = synthetic()
    plan = scope["make_internal_plan"](data["metadata"])
    rows = scope["plan_rows"](data, plan, "rank")
    allowed = set(rows)
    original = data["X"]

    class Guard:
        def __getitem__(self, row):
            assert int(row) in allowed, "Read beyond internal ranking patients"
            return original[int(row)]

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.observed = []
        def forward(self, x):
            self.observed.append(x.cpu().numpy().copy())
            return x[:, 0, 0] - 1

    spy = Spy()
    data["X"] = Guard()
    meta = data["metadata"].iloc[rows].reset_index(drop=True)
    donors = scope["same_date_donors"](meta, 2)
    scope["score_rows"](spy, data, rows, torch.device('cpu'), 9, feature=0, donors=donors)
    observed = np.concatenate(spy.observed)
    expected = original[rows].copy()
    expected[:, :, 0] = original[rows[donors], :, 0]
    np.testing.assert_allclose(observed, np.log1p(expected), rtol=1e-6)
    np.testing.assert_array_equal(original[:, :, 0], synthetic()["X"][:, :, 0])


def test_ranking_detects_signal_and_keeps_original_feature_order():
    data = synthetic()
    rows = scope["plan_rows"](data, scope["make_internal_plan"](data["metadata"]), "rank")
    meta, y = data["metadata"].iloc[rows], data["y"][rows]
    class Signal(torch.nn.Module):
        def forward(self, x):
            return x[:, :, 0].mean(1) - 1
    model = Signal()
    p = scope["score_rows"](model, data, rows, torch.device('cpu'))
    baseline = {"lift": scope["top10_lift"](y, p, meta), "ap": scope["average_precision_score"](y, p)}
    records = [scope["rank_feature"](model, data, rows, torch.device('cpu'), i, 3, 42, baseline) for i in range(4)]
    ranked, selected = scope["selection_from_ranking"](records, data["features"], 2)
    assert ranked.iloc[0].feature_index == 0 and ranked.iloc[0].mean_lift_drop > 0
    assert all(r["mean_lift_drop"] == 0 for r in records[1:])
    view = scope["selected_view"](data, [0, 2])
    assert view["X"].shape == (len(y)*0 + len(data["metadata"]), 12, 2)
    np.testing.assert_array_equal(view["X"][7], data["X"][7][:, [0, 2]])
    with pytest.raises(ValueError):
        scope["selection_from_ranking"](records[:-1], data["features"], 2)


def test_training_roundtrip_and_no_outer_rows_during_ranking_fit():
    data = synthetic()
    plan = scope["make_internal_plan"](data["metadata"])
    fit, stop = [scope["plan_rows"](data, plan, r) for r in ('fit', 'stop')]
    original = data["X"]
    class Guard:
        shape = original.shape
        def __getitem__(self, row):
            assert int(row) in set(fit) | set(stop)
            return original[int(row)]
    data["X"] = Guard()
    view = scope["selected_view"](data, [0, 1], fit, stop)
    config = scope["ModelConfig"](input_dim=2, d_model=8, n_heads=2, encoder_layers=1, feedforward_dim=16)
    settings = dict(seed=42, epochs=2, patience=1, min_delta=1e-4, batch_size=32,
                    learning_rate=.001, weight_decay=.0001, grad_clip=1., device='cpu')
    artifacts = scope["model_artifacts"](view, config, settings, 'SYNTH', {'test': True})
    model, payload, device = scope["checked_model"](artifacts, view, 'SYNTH', {'test': True})
    history = pd.read_csv(io.BytesIO(artifacts['history.csv']))
    assert {'training_top10_lift', 'validation_top10_lift', 'training_loss'} <= set(history)
    _, y, p = scope['predict_loader'](model, scope['make_loader'](view, 'validation', 32, 42), device)
    assert np.isfinite(p).all()
    with pytest.raises(ValueError):
        scope['checked_model'](artifacts, view, 'SYNTH', {'test': False})


def test_generated_notebooks_compile_and_are_self_contained():
    for name, cells in builder.SPECS:
        assert len(cells) == 8
        for code in cells:
            compile(code, name, 'exec')
        assert 'def read_artifacts' in cells[1]
        assert 'def load_inputs' in cells[1]
        assert not any(word in '\n'.join(cells).lower() for word in ('brian', 'client'))
    assert '("train", "validation", "test")' not in '\n'.join(builder.training[3:])
    assert 'input_dim=len(selected_indices)' in '\n'.join(builder.training)


def test_training_and_evaluation_notebook_orchestration(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: plt.close('all'))
    data = synthetic()
    plan = scope['make_internal_plan'](data['metadata'])
    audit = {'snapshot_manifest_sha256': 'synthetic'}
    artifacts = {}
    def save(table, blobs):
        if table in artifacts:
            assert artifacts[table] == blobs
        artifacts[table] = blobs
    def read(table, names):
        assert set(artifacts[table]) == set(names)
        return artifacts[table]
    for cells in (builder.training, builder.evaluation):
        ns = {'__name__': __name__, 'sf_options': {}, 'spark': object(), 'display': lambda x: None}
        for index, code in enumerate(cells):
            if index == 0:
                exec(code, ns)
                ns.update(TOP_K=2, PERMUTATION_REPEATS=2, FEATURES_PER_CHUNK=2)
                if 'MODEL_SETTINGS' in ns:
                    ns['MODEL_SETTINGS'].update(d_model=8, n_heads=2, encoder_layers=1, feedforward_dim=16)
                    ns['TRAINING_SETTINGS'].update(epochs=2, patience=1, batch_size=64, device='cpu')
                artifacts.setdefault(ns['INTERNAL_TABLE'], {'plan.json': scope['json_blob']({
                    'seed': 42, 'source_hash': 'synthetic', 'records': plan.to_dict('records')})})
                artifacts.setdefault(ns['PREPARATION_TABLE'], {'preparation_report.json': b'{}'})
                artifacts.setdefault(ns['SPLIT_AUDIT_TABLE'], {'split_audit.json': b'{}'})
                if 'MODEL_SETTINGS' in ns:
                    ref = dict(model_config=ns['MODEL_SETTINGS'], training_settings=ns['TRAINING_SETTINGS'])
                    artifacts[ns['REFERENCE_MODEL_TABLE']] = {'checkpoint.pt': b'ref', 'training_history.csv': b'',
                        'training_history.png': b'', 'training_summary.json': scope['json_blob'](ref)}
            else:
                exec(code, ns)
                if index == 1:
                    ns.update(load_inputs=lambda: data, verify_audits=lambda *a: None,
                        read_artifacts=read, save_artifacts=save, table_exists=lambda t: t in artifacts,
                        release_inputs=lambda d: None)
    report = json.loads(artifacts[ns['EVALUATION_TABLE']]['evaluation.json'])
    assert set(report['metrics']) == {'train', 'validation', 'test'}
    selection = json.loads(artifacts[ns['SELECTION_TABLE']]['manifest.json'])
    assert len(selection['selected_indices']) == 2 and selection['time_steps'] == list(range(12))
    assert selection['test_used'] is False and selection['outer_validation_used_for_ranking'] is False
