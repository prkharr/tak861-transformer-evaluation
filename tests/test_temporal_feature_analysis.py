"""Synthetic checks of the functions embedded in the delivered evaluation notebook."""
import ast
import copy
import hashlib
import json
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score


BOOK = Path(__file__).resolve().parents[1] / 'notebooks/Temporal Feature Selection/04_transformer_evaluation.ipynb'


def source():
    return json.loads(BOOK.read_text(encoding='utf-8'))


def notebook_scope():
    book = source()
    scope = dict(np=np, pd=pd, torch=torch, average_precision_score=average_precision_score)
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    scope['require'] = require
    helper = next(c['source'] for c in book['cells'] if c.get('id') == 'feature-analysis-helpers')
    exec(compile(helper, str(BOOK), 'exec'), scope)
    return scope


class SignalOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(6.))

    def forward(self, X, valid):
        assert torch.all(X[~valid] == 0)
        signal = (X[:, :, 0] * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.weight * (signal - .5)


def make_view():
    rng = np.random.default_rng(41)
    n, t, f = 85, 4, 4
    y = np.tile([0, 1], 43)[:n].astype(np.float32)
    valid = np.ones((n, t), dtype=bool)
    valid[:60, -1] = False
    valid[60:80, 1:] = False
    valid[80:84] = False
    valid[84] = [True, False, True, False]  # singleton mask, cannot exchange
    X = rng.normal(size=(n, t, f)).astype(np.float32)
    X[:, :, 0] = y[:, None]
    X[:, :, 2] = 7.  # constant, no measurable perturbation
    X[:, :, 3] = 1 - y[:, None]  # correlated substitute, ignored by this toy model
    X[~valid] = 0
    meta = pd.DataFrame(dict(PATIENT_ID=[f'SYN{i:03}' for i in range(n)],
                             END_DT=['2025-01-20']*n, RESP=y, SPLIT=['validation']*n))
    return dict(X=X, raw=X.copy(), valid=valid, y=y, metadata=meta,
                features=['signal', 'noise', 'constant', 'inverse'], representation='QUARTERLY')


def test_permutation_identifies_model_signal_and_preserves_inputs():
    s = notebook_scope()
    view = make_view()
    before = copy.deepcopy(view)
    model = SignalOnly()
    before_weight = model.weight.detach().clone()
    rank, repeats, audit = s['feature_permutation_importance'](
        model, view, 'cpu', repeats=3, seed=42, batch_size=16, progress=False)
    by_name = rank.set_index('FEATURE_NAME')
    assert rank.iloc[0].FEATURE_NAME == 'signal'
    assert by_name.loc['signal', 'AP_DROP_MEAN'] > .25
    assert by_name.loc['noise', 'AP_DROP_MEAN'] == 0
    assert by_name.loc['inverse', 'AP_DROP_MEAN'] == 0  # correlation is not model reliance
    assert by_name.loc['constant', 'STATUS'] == 'NO_VARIATION_WITHIN_MASK_GROUPS'
    assert pd.isna(by_name.loc['constant', 'IMPORTANCE_RANK'])
    assert audit['EXCHANGEABLE_SNAPSHOTS'] == 80 and audit['ALL_PADDED_SNAPSHOTS'] == 4
    assert audit['SINGLETON_VALID_SNAPSHOTS'] == 1
    assert len(repeats) == 9
    for key in ['X', 'raw', 'valid', 'y']:
        np.testing.assert_array_equal(view[key], before[key])
    torch.testing.assert_close(model.weight, before_weight)
    assert model.weight.grad is None
    again = s['feature_permutation_importance'](model, view, 'cpu', repeats=3, seed=42,
                                               batch_size=16, progress=False)
    pd.testing.assert_frame_equal(rank, again[0])
    with unittest.TestCase().assertRaisesRegex(ValueError, 'baseline'):
        s['feature_permutation_importance'](model, view, 'cpu', expected_validation_ap=0., progress=False)


def test_mask_groups_keep_padding_and_static_trajectories():
    s, view = notebook_scope(), make_view()
    _, groups = s['feature_permutation_groups'](view['valid'])
    donor = s['feature_permutation_map'](len(view['X']), groups, 123)
    assert (donor[80:] == np.arange(80, 85)).all()
    assert np.array_equal(view['valid'], view['valid'][donor])
    changed = view['X'][donor, :, 0]
    for row in range(len(changed)):
        assert len(np.unique(changed[row, view['valid'][row]])) <= 1
    assert np.all(changed[~view['valid']] == 0)


def test_negative_ap_drops_are_retained():
    class Opposite(SignalOnly):
        def forward(self, X, valid):
            return -super().forward(X, valid)
    s, view = notebook_scope(), make_view()
    rank, _, _ = s['feature_permutation_importance'](Opposite(), view, 'cpu', repeats=3, progress=False)
    assert rank.set_index('FEATURE_NAME').loc['signal', 'AP_DROP_MEAN'] < 0


def test_no_exchangeable_rows_are_not_zero_importance():
    s, view = notebook_scope(), make_view()
    view['valid'][:] = False
    view['X'][:] = 0
    rank, _, audit = s['feature_permutation_importance'](SignalOnly(), view, 'cpu', progress=False)
    assert rank.STATUS.eq('NO_EXCHANGEABLE_TRAJECTORIES').all()
    assert rank.AP_DROP_MEAN.isna().all() and rank.IMPORTANCE_RANK.isna().all()
    assert audit['EXCHANGEABLE_SNAPSHOTS'] == 0
    json.dumps(s['feature_analysis_records'](rank), allow_nan=False)


def test_summaries_exclude_padding_and_missingness():
    s = notebook_scope()
    raw = np.array([[[2., np.nan], [4., 8.], [999., 999.]],
                    [[9., 9.], [9., 9.], [9., 9.]]], dtype=np.float32)
    valid = np.array([[True, True, False], [False, False, False]])
    mean, newest = s['feature_time_summaries'](raw, valid)
    np.testing.assert_allclose(mean[0], [3., 8.])
    np.testing.assert_allclose(newest[0], [2., 8.])
    assert np.isnan(mean[1]).all() and np.isnan(newest[1]).all()


def test_correlation_direction_redundancy_and_latest_patient_snapshot():
    s, view = notebook_scope(), make_view()
    # The first row is older than row1 for the same patient; row1's features and RESP must be selected.
    view['metadata'].loc[0, 'PATIENT_ID'] = view['metadata'].loc[1, 'PATIENT_ID']
    view['metadata'].loc[0, 'END_DT'] = '2024-12-01'
    corr, pairs, audit = s['feature_correlations'](view, minimum_rows=20)
    corr = corr.set_index('FEATURE_NAME')
    assert audit['PATIENTS'] == 84
    assert corr.loc['signal', 'N_PATIENTS_MEAN'] == 80  # 85 - 1 duplicate patient - 4 all-padded
    assert np.isclose(corr.loc['signal', 'RESP_SPEARMAN_MEAN'], 1.)
    assert np.isclose(corr.loc['inverse', 'RESP_SPEARMAN_MEAN'], -1.)
    assert corr.loc['constant', 'MEAN_STATUS'] == 'CONSTANT_INPUT'
    assert pd.isna(corr.loc['constant', 'RESP_SPEARMAN_MEAN'])
    inverse_pair = pairs.loc[pairs.FEATURE_A.eq('signal') & pairs.FEATURE_B.eq('inverse')].iloc[0]
    assert np.isclose(inverse_pair.SPEARMAN, -1.)
    rho, n, status = s['feature_spearman'](np.array([1., np.nan]), np.array([0., 1.]), 3)
    assert np.isnan(rho) and n == 1 and status == 'INSUFFICIENT_OBSERVATIONS'


def test_analysis_view_reads_only_validation():
    s, v = notebook_scope(), make_view()
    data = dict(X=v['X'], raw_X=v['raw'], valid=v['valid'], y=v['y'], metadata=v['metadata'],
                features=v['features'], representation='QUARTERLY', indices={'validation':np.arange(10)})
    out = s['feature_analysis_view'](data)
    assert len(out['X']) == 10
    out['X'][:] = 123
    assert not np.all(data['X'][:10] == 123)
    data['metadata'].loc[0, 'SPLIT'] = 'test'
    with unittest.TestCase().assertRaisesRegex(ValueError, 'VALIDATION'):
        s['feature_analysis_view'](data)


def test_saved_section_has_no_patient_rows_and_runs_end_to_end():
    s, v = notebook_scope(), make_view()
    book = source()
    cells = {c.get('id'):c['source'] for c in book['cells'] if c['cell_type']=='code'}
    data = dict(X=v['X'], raw_X=v['raw'], valid=v['valid'], y=v['y'], metadata=v['metadata'],
                features=v['features'], indices={'validation':np.arange(len(v['X']))}, hashes={'synthetic':'only'})
    experiments = {name:dict(data, representation=name) for name in ['MONTHLY','QUARTERLY']}
    def restore(name, data):
        model = SignalOnly()
        p = s['feature_analysis_predict'](model, data['X'], data['valid'], 'cpu', 128)
        return model, {'best_validation_average_precision':float(average_precision_score(data['y'], p))}, 'cpu', {}, None
    stored = {}
    s.update(experiments=experiments, restored_representation=restore, validate_model_inputs=lambda d:True,
             manifest={'audit':[{'FEATURE_NAME':f, 'VALUE_SOURCE':'V63_SNAPSHOT' if f=='signal' else 'TEMPORAL'} for f in v['features']],
                       'features':v['features'], 'input_description':'synthetic hybrid'},
             FEATURE_ANALYSIS_ID='SYNTHETIC', FEATURE_ANALYSIS_IMPLEMENTATION_SHA256='test',
             FEATURE_ANALYSIS_SETTINGS=dict(repeats=3,seed=42,batch_size=32,correlation_min_patients=20),
             array_hash=lambda a:hashlib.sha256(a.tobytes()).hexdigest(),
             digest_json=lambda a:hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest(),
             canonical_json=lambda a:json.dumps(a,sort_keys=True,allow_nan=False),
             RUN_ID='SYNTHETIC', DATASET_ID='SYNTHETIC', RUN_PREFIX='SYNTHETIC',
             table_exists=lambda _:False, save_artifacts=lambda table,artifacts:stored.update(artifacts))
    exec(cells['feature-analysis-run'],s)
    exec(cells['feature-analysis-save'],s)
    assert len(stored)==5
    report=json.loads(stored['feature_analysis.json'])
    assert report['test_used'] is False and report['split']=='validation'
    for result in report['models'].values():
        assert all('VALUE_SOURCE' in r for r in result['feature_results'])
        assert all('VALUE_SOURCE_A' in r and 'VALUE_SOURCE_B' in r for r in result['feature_pairs'])
    assert b'SYN000' not in stored['feature_analysis.json']
    for name, value in stored.items():
        assert b'PATIENT_ID' not in value and b'END_DT' not in value


def test_notebook_cells_parse_and_outputs_are_clear():
    book=source()
    for c in book['cells']:
        if c['cell_type']=='code':
            ast.parse(c['source'])
            assert not c['outputs'] and c['execution_count'] is None


def test_analysis_predict_matches_actual_transformer_for_both_grids():
    s = notebook_scope()
    code = '\n'.join(c['source'] for c in source()['cells'] if c['cell_type']=='code')
    definitions = [n for n in ast.parse(code).body if isinstance(n,ast.ClassDef)
                   and n.name in {'ModelConfig','ClaimsTransformer'}]
    model_scope = dict(torch=torch, nn=torch.nn, dataclass=dataclass)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(BOOK),'exec'),model_scope)
    for t in [4,12]:
        config=model_scope['ModelConfig'](input_dim=49,seq_len=t)
        model=model_scope['ClaimsTransformer'](config).eval()
        X=np.random.default_rng(42).normal(size=(6,t,49)).astype(np.float32)
        valid=np.ones((6,t),dtype=bool)
        valid[0]=False
        valid[1,1:]=False
        X[~valid]=0
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        result=s['feature_analysis_predict'](model,X,valid,'cpu',3)
        with torch.inference_mode():
            expected=torch.sigmoid(model(torch.from_numpy(X),torch.from_numpy(valid))).numpy()
        np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-7)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value,before[name])
        assert all(p.grad is None for p in model.parameters())
