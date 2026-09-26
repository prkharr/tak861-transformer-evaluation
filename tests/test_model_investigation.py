"""Offline synthetic validation of the self-contained investigation cells in notebook 04."""
import ast
import copy
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BOOK = Path(__file__).resolve().parents[1]/'notebooks/Temporal Feature Selection/04_transformer_evaluation.ipynb'


def load_scope():
    book = json.loads(BOOK.read_text(encoding='utf-8'))
    scope = {'display': lambda *args: None, 'show_network': lambda *args: None}
    # Load definitions, not configuration requiring a warehouse or execution cells.
    for cell in book['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                if isinstance(node, ast.ImportFrom) and node.module == 'IPython.display':
                    continue
                if isinstance(node, ast.Import) and any(n.name.startswith('matplotlib') for n in node.names):
                    continue  # Plotting is not part of these numerical runtime checks.
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(BOOK), 'exec'), scope)
    scope.update(TIME_ORIENTATION='0=most_recent; increasing_index=older',
                 MASK_CONVENTION='valid=True means observed; src_key_padding_mask=~valid; masked mean pooling',
                 CONSUMER_IMPLEMENTATION_SHA256='synthetic-consumer', OVERFIT_IMPLEMENTATION_SHA256='synthetic-investigation',
                 RUN_PREFIX='SYNTHETIC', RUN_ID='RTEST')
    return scope


def make_data(steps=4):
    rng = np.random.default_rng(91)
    n, f = 42, 49
    groups = np.repeat(np.arange(n//2), 2)
    y = (groups % 2).astype(np.float32)
    x = rng.normal(size=(n, steps, f)).astype(np.float32)
    x[:, :, 0] = y[:, None] + rng.normal(0, .2, (n, steps))
    valid = np.ones((n, steps), dtype=bool)
    valid[[0, 24]] = False
    valid[1::3, -1] = False
    x[~valid] = 0
    raw = x.copy()
    raw[2, 0, 2] = np.nan
    split = ['train']*24 + ['validation']*12 + ['test']*6
    meta = pd.DataFrame(dict(PATIENT_ID=[f'SYN{g}' for g in groups],
        END_DT=['2024-01-01' if i % 2 == 0 else '2024-02-01' for i in range(n)],
        RESP=y, SPLIT=split))
    return dict(X=x, raw_X=raw, y=y, valid=valid, metadata=meta,
        features=[f'FEATURE_{j}' for j in range(f)], representation='MONTHLY' if steps == 12 else 'QUARTERLY',
        indices={s: np.flatnonzero(meta.SPLIT.eq(s).to_numpy()) for s in ('train', 'validation', 'test')},
        hashes={'synthetic': 'input'+str(steps)}, preprocessor={})


def settings():
    return dict(seed=42, epochs=2, patience=1, min_delta=1e-4, batch_size=12,
                learning_rate=.001, weight_decay=0., grad_clip=1., device='cpu',
                l1_lambda=1e-6, l2_lambda=1e-4, regularization_scope='linear_and_attention_weights')


def test_architecture_counts_and_svg_are_from_actual_models():
    s = load_scope()
    for steps in (4, 12):
        m = s['ClaimsTransformer'](s['ModelConfig'](49, seq_len=steps))
        before = {k: v.clone() for k, v in m.state_dict().items()}
        spec = s['neural_architecture_spec'](m)
        assert spec['total_parameters'] == sum(p.numel() for p in m.parameters())
        assert spec['parameter_counts']['position'] == steps*128
        assert spec['head_width'] == 64 and spec['layers'] == 2
        svg = s['neural_architecture_svg'](m, 'MONTHLY <structure>', settings())
        ET.fromstring(svg)
        assert '&lt;structure&gt;' in svg and 'All-padded row' in svg
        assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_evaluation_mode_repeatability_masks_and_state():
    s, d = load_scope(), make_data()
    m = s['ClaimsTransformer'](s['ModelConfig'](49, seq_len=4, d_model=8, n_heads=2, encoder_layers=1, feedforward_dim=16))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    scores, audit = s['diagnostic_predict'](m, d, 'validation', torch.device('cpu'), 5)
    assert len(scores) == 12 and audit['FIRST_BATCH_REPEAT_MAX_LOGIT_DELTA'] == 0
    assert not m.training and all(p.grad is None for p in m.parameters())
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    with unittest.TestCase().assertRaisesRegex(ValueError, 'restricted'):
        s['diagnostic_predict'](m, d, 'test', torch.device('cpu'))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'batch size'):
        s['diagnostic_predict'](m, d, 'validation', torch.device('cpu'), 0)


def test_population_groups_and_train_only_drift_reference():
    s, d = load_scope(), make_data()
    rows = s['diagnostic_population'](d, 'validation', np.linspace(.1, .9, 12))
    groups = {r['GROUP']: r for r in rows}
    assert groups['all_snapshots']['SNAPSHOTS'] == 12
    assert groups['latest_snapshot_per_patient']['SNAPSHOTS'] == 6
    assert groups['all_padded']['SNAPSHOTS'] == 1
    assert sum(groups[g]['SNAPSHOTS'] for g in ('all_padded','partly_covered','fully_covered')) == 12
    audit = pd.DataFrame(dict(FEATURE_NAME=d['features'], VALUE_SOURCE=['TEMPORAL']*49))
    before = s['diagnostic_feature_drift'](d, audit)
    d['raw_X'][d['indices']['test']] = 123456  # Must not enter diagnostic distribution shifts.
    after = s['diagnostic_feature_drift'](d, audit)
    pd.testing.assert_frame_equal(before, after)


def test_patient_bootstrap_is_deterministic_and_paired():
    s, d = load_scope(), make_data()
    rows = d['indices']['validation']
    meta, y = d['metadata'].iloc[rows], d['y'][rows]
    baseline, perfect = np.full(len(rows), .5), y*.8+.1
    same = s['paired_patient_ap_interval'](meta, y, baseline, baseline, 40)
    gain = s['paired_patient_ap_interval'](meta, y, baseline, perfect, 40)
    assert same['AP_DELTA_CI_LOW'] == same['AP_DELTA_CI_HIGH'] == 0
    assert gain['AP_DELTA_CI_LOW'] > 0
    assert gain == s['paired_patient_ap_interval'](meta, y, baseline, perfect, 40)


def test_search_runs_actual_training_and_preserves_baseline_and_inputs():
    s = load_scope()
    memory, training_calls = {}, []
    def read(table, names):
        assert set(memory[table]) == set(names)
        return memory[table].copy()
    def save(table, artifacts):
        assert table not in memory
        memory[table] = artifacts.copy()
    s.update(table_exists=lambda table: table in memory, read_artifacts=read, save_artifacts=save)
    original_train, original_loader = s['train_run'], s['make_loader']
    def train(*args):
        training_calls.append(args[-1])
        return original_train(*args)
    def loader(data, split, *args, **kwargs):
        assert split != 'test', 'TEST inference entered validation search'
        return original_loader(data, split, *args, **kwargs)
    s.update(train_run=train, make_loader=loader)
    architecture = dict(d_model=8, n_heads=2, encoder_layers=1, feedforward_dim=16, dropout=.2)
    candidates = s['regularization_candidates'](architecture, settings())
    assert len(candidates) == 6
    for steps in (4, 12):
        data = make_data(steps)
        before_x, before_mask = data['X'].copy(), data['valid'].copy()
        table, report = s['run_regularization_search'](data, candidates, 'OTEST', 25)
        assert len(table) == 6 and table.iloc[0].VALIDATION_AP == table.VALIDATION_AP.max()
        assert report['test_inference_performed'] is False
        assert np.array_equal(data['X'], before_x) and np.array_equal(data['valid'], before_mask)
        restored = s['restore_selected_tuning_model'](data['representation'], data, candidates, 'OTEST')
        model, payload, device, _, history = restored
        audit = pd.DataFrame(dict(FEATURE_NAME=data['features'], VALUE_SOURCE=['TEMPORAL']*49))
        diagnostics = s['run_overfit_diagnostics'](model, payload, data, history, device, audit, 12)
        assert diagnostics['history'].iloc[0].BEST_VALIDATION_AP == table.iloc[0].VALIDATION_AP
        prior = len(training_calls)
        repeated, _ = s['run_regularization_search'](data, candidates, 'OTEST', 25)
        assert len(training_calls) == prior  # Verified resume without retraining.
        pd.testing.assert_frame_equal(table, repeated)
        changed = copy.deepcopy(candidates[0]); changed['training_settings']['l1_lambda'] *= 2
        with unittest.TestCase().assertRaisesRegex(ValueError, 'protocol'):
            s['restore_tuning_candidate'](data, changed, 'OTEST')
    assert len(training_calls) == 12
    assert all('_TUNE_OTEST_' in table for table in memory)


def test_embedded_training_is_unchanged_and_no_external_addon_import():
    s = json.loads(BOOK.read_text(encoding='utf-8'))
    train = json.loads((BOOK.parent/'03_transformer_training.ipynb').read_text(encoding='utf-8'))
    def functions(book):
        result = {}
        for c in book['cells']:
            if c['cell_type'] != 'code': continue
            source = ''.join(c['source'])
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef): result[node.name] = ast.dump(node)
        return result
    a, b = functions(s), functions(train)
    for name in ['train_run','regularization_penalties','seed_everything','select_validation_threshold','top10_lift']:
        assert a[name] == b[name]
    for c in s['cells']:
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']))
            assert c['outputs'] == [] and c['execution_count'] is None


def test_execution_cells_work_together_and_save_only_aggregates():
    s = load_scope()
    architecture = dict(d_model=8, n_heads=2, encoder_layers=1, feedforward_dim=16, dropout=.2)
    training = settings(); training.update(epochs=1, patience=1)
    experiments = {name: make_data(steps) for name, steps in [('MONTHLY',12), ('QUARTERLY',4)]}
    features = experiments['MONTHLY']['features']
    manifest = dict(features=features, audit=[dict(FEATURE_NAME=f, VALUE_SOURCE='TEMPORAL') for f in features])
    memory, baseline = {}, {}
    def save(table, artifacts):
        assert table not in memory
        memory[table] = artifacts.copy()
    def read(table, expected):
        assert set(memory[table]) == set(expected)
        return memory[table].copy()
    s.update(experiments=experiments, manifest=manifest, MODEL_SETTINGS=architecture, TRAINING_SETTINGS=training,
             DATASET_ID='DTEST', TUNING_ID='OGLUE', DIAGNOSTIC_ID='D001', RUN_VALIDATION_SEARCH=True,
             ANALYZE_SELECTED_FEATURES=False, BOOTSTRAP_REPEATS=25,
             SVG=lambda **kw: kw['data'], display=lambda *args: None, show_network=lambda *args: None,
             table_exists=lambda table: table in memory, save_artifacts=save, read_artifacts=read)
    # Use real checkpoints for current-model cells, with a single search candidate to bound this glue check.
    for name, data in experiments.items():
        config = s['ModelConfig'](input_dim=49, seq_len=data['X'].shape[1], **architecture)
        blob, summary, history = s['train_run'](data, config, training, 'RTEST_'+name)
        baseline[name] = (blob, summary, history)
    def restore(name, data):
        blob, summary, history = baseline[name]
        model, payload, device = s['load_verified_model'](blob, data, 'RTEST_'+name, device='cpu')
        return model, payload, device, summary, history
    s['restored_representation'] = restore
    original_candidates = s['regularization_candidates']
    s['regularization_candidates'] = lambda a, b: original_candidates(a, b)[:1]
    book = json.loads(BOOK.read_text(encoding='utf-8'))
    for suffix in ['current-model','search-run','selected-model','selected-features','save']:
        cell = next(c for c in book['cells'] if c.get('id') == 'model-investigation-'+suffix)
        tree = ast.parse(''.join(cell['source']))
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module=='IPython.display')]
        exec(compile(tree, str(BOOK), 'exec'), s)
    artifacts = memory['SYNTHETIC_INVESTIGATION_OGLUE_D001']
    report = json.loads(artifacts['investigation.json'])
    assert set(report['selected']) == {'MONTHLY','QUARTERLY'}
    assert report['test_scored_by_this_section'] is False
    assert len([f for f in artifacts if f.endswith('.svg')]) == 4
    for blob in artifacts.values():
        assert b'PATIENT_ID' not in blob and b'SYN0' not in blob
