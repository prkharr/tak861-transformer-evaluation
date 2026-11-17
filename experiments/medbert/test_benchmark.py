import copy
import json
import time
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.medbert import benchmark as b
from experiments.automl_sequence import workflow as wf, modeling as mdl
from experiments.rigorous_transformer import comparison as cmp


def fixture():
    n = 48
    split = ['train'] * 24 + ['validation'] * 16 + ['test'] * 8
    meta = pd.DataFrame(dict(PATIENT_ID=[f'p{i//2:03d}' for i in range(n)],
        END_DT=[pd.Timestamp('2024-01-31') + pd.Timedelta(days=i % 2) for i in range(n)],
        RESP=np.arange(n) % 2, SPLIT=split))
    x = np.zeros((n, 12, 4), dtype=np.float32)
    x[:, 0, 0] = 2
    x[:, 11, 1] = np.arange(n) % 2
    x[:, :, 2] = 50.5  # Continuous context must never become a token.
    x[24:40, 4, 3] = 1  # Held-out-only code must map to UNK.
    x[40:] = np.nan  # No TEST content should be read or tokenized.
    mask = np.ones((n, 12), dtype=bool)
    mask[:, 5] = False
    data = dict(metadata=meta, X_count=x, count_mask=mask, y=meta.RESP.to_numpy(),
        indices={s: np.flatnonzero(meta.SPLIT.eq(s)) for s in ('train', 'validation', 'test')},
        manifest=dict(count_contract=dict(orientation='0=newest', n_steps=12,
            date_policy='month_end_history', feature_names=['DX__a', 'PX__b', 'AGE', 'RX__c'])))
    return data


class Store:
    def __init__(self):
        self.saved = {}

    def table_exists(self, name):
        return name in self.saved

    def read_artifacts(self, name, keys):
        return {k: self.saved[name][k] for k in keys}

    def save_artifacts(self, name, value):
        if name in self.saved:
            raise ValueError('Attempted overwrite')
        self.saved[name] = value


def mocks(data):
    prep, protocol, prefix = {}, {'arms': {'COUNT': {}}}, 'SYNTHETIC'
    store = Store()
    old_scores = np.linspace(.01, .5, 16)
    runs = {'COUNT': [{'payload_hashes': {'validation.npy': 'synthetic'}, 'val_scores': old_scores}]}
    lock = dict(protocol_hash=wf.digest(protocol), preparation_hash=wf.digest(prep),
        input_manifest_hash=wf.digest(data['manifest']), models={'COUNT': {'predictor_hashes': [runs['COUNT'][0]['payload_hashes']]}})
    store.saved[prefix + '_LOCK'] = {'lock.json': wf.json_bytes(lock)}
    helper = types.SimpleNamespace(**{n: getattr(wf, n) for n in ('digest', 'array_sha256', 'array_bytes', 'bytes_array',
        'json_bytes', 'paired_lift_bootstrap')}, read_runs=lambda *a: runs)
    return prep, protocol, prefix, store, helper, lock


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.cfg = dict(b.DEFAULTS, max_train_patients=20, max_train_rows=24,
            d_model=8, heads=2, depth=1, ff=16, dropout=0., batch_size=8,
            pretrain_steps=2, epochs=1, finetune_steps_per_epoch=2,
            threads=1, bootstrap_draws=20, total_seconds=300, save_reserve=5, device='cpu')
        self.budget = b.Budget(300)

    def test_vocabulary_is_train_only_and_context_is_excluded(self):
        columns = b.fit_vocabulary(self.data, self.data['indices']['train'], self.cfg, self.budget)
        self.assertEqual(columns.tolist(), [0, 1])
        arrays, report = b.tokenize(self.data, np.array([24]), self.cfg, self.budget, columns)
        ids, month, frequency = arrays
        self.assertIn(4, ids[0])
        self.assertEqual(report['unknown_code_months'], 1)
        self.assertNotIn(6, month[0])  # unavailable month 5 has no token
        self.assertTrue(np.any((ids[0] == 3) & (month[0] > 0)))  # observed empty month
        nonpad = month[0][month[0] > 0]
        self.assertTrue(np.all(nonpad[:-1] >= nonpad[1:]))
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            b.fit_vocabulary(self.data, self.data['indices']['validation'], self.cfg, self.budget)
        with self.assertRaisesRegex(ValueError, 'TEST'):
            b.tokenize(self.data, np.array([40]), self.cfg, self.budget, columns)

    def test_subset_does_not_use_outcomes(self):
        before = b.train_subset(self.data['metadata'], dict(self.cfg, max_train_patients=3))
        changed = self.data['metadata'].copy()
        changed.RESP = 1 - changed.RESP
        after = b.train_subset(changed, dict(self.cfg, max_train_patients=3))
        np.testing.assert_array_equal(before, after)

    def test_masking_hides_counts_and_preserves_special_tokens(self):
        ids = torch.tensor([[1, 5, 6, 3, 0], [1, 6, 5, 4, 0]])
        freq = torch.ones_like(ids)
        corrupted, frequencies, labels = b.mask_codes(ids, freq, 7, torch.Generator().manual_seed(3))
        selected = labels.ne(-100)
        self.assertTrue(selected.any())
        self.assertTrue(torch.all(frequencies[selected] == 0))
        self.assertTrue(torch.all(labels[ids < len(b.SPECIAL)] == -100))
        self.assertTrue(torch.all(corrupted[ids == 0] == 0))

    def test_pretraining_preserves_classifier_head(self):
        columns = b.fit_vocabulary(self.data, self.data['indices']['train'], self.cfg, self.budget)
        arrays, _ = b.tokenize(self.data, self.data['indices']['train'], self.cfg, self.budget, columns)
        model = b.MonthlyEncoder(len(b.SPECIAL) + len(columns), self.cfg)
        head_before = {name: value.detach().clone() for name, value in model.classifier.state_dict().items()}
        embedding_before = model.code.weight.detach().clone()
        receipt = b.pretrain(model, arrays, self.cfg, self.budget, 'cpu', time.monotonic() + 30)
        self.assertGreater(receipt['steps'], 0)
        for name, value in model.classifier.state_dict().items():
            torch.testing.assert_close(value, head_before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.code.weight, embedding_before))

    def test_full_fit_persistence_and_resume(self):
        prep, protocol, prefix, store, helper, lock = mocks(self.data)
        table, report = b.run(self.data, prep, protocol, prefix, 'implementation', self.cfg,
            helper, mdl, cmp, store)
        self.assertFalse(report['test_prediction_performed'])
        self.assertEqual(report['status'], 'completed_full_validation')
        self.assertEqual(report['mlm']['steps'], 2)
        self.assertEqual(report['vocabulary'], b.SPECIAL + ['DX__a', 'PX__b'])
        self.assertEqual({r['Snapshots'] for r in report['rows']}, {16})
        self.assertEqual({r['Model'] for r in report['rows']}, {'COUNT', b.MODEL_NAME, b.CONTROL_NAME})
        self.assertIn(b.CONTROL_NAME, report['intervals'])
        self.assertEqual(store.saved[table][b.MODEL_NAME + '.npy'][:6], b'\x93NUMPY')
        _, resumed = b.run(self.data, prep, protocol, prefix, 'implementation', self.cfg,
            helper, mdl, cmp, store)
        self.assertEqual(resumed['identity'], report['identity'])
        store.saved[table]['models.pt'] += b'bad'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            b.run(self.data, prep, protocol, prefix, 'implementation', self.cfg, helper, mdl, cmp, store)

    def test_deadline_and_changed_benchmark_are_rejected(self):
        with self.assertRaises(TimeoutError):
            b.Budget(1, time.monotonic() - 10).require()
        prep, protocol, prefix, store, helper, lock = mocks(self.data)
        lock['input_manifest_hash'] = 'different'
        store.saved[prefix + '_LOCK']['lock.json'] = wf.json_bytes(lock)
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            b.frozen_benchmarks(self.data, prep, protocol, helper, store, prefix)

    def test_saved_lightgbm_is_verified_and_compared(self):
        prep, protocol, prefix, store, helper, lock = mocks(self.data)
        probabilities = np.linspace(.2, .6, 16)
        identity = dict(split='validation', baseline=dict(input_hash=wf.digest(self.data['manifest']),
            lock_hash=wf.digest(lock)), predictions={'LIGHTGBM_TRAIN_ONLY_STATIC': wf.array_sha256(probabilities)})
        table = prefix + '_FRESH_LGBM_STATIC_V1_VALIDATION_LIFT'
        store.saved[table] = {'report.json': wf.json_bytes(dict(identity=identity)), 'baseline.npy': wf.array_bytes(probabilities)}
        _, report = b.run(self.data, prep, protocol, prefix, 'implementation', self.cfg, helper, mdl, cmp, store)
        self.assertIn('LIGHTGBM_TRAIN_ONLY_STATIC', report['intervals'])
        identity['predictions']['LIGHTGBM_TRAIN_ONLY_STATIC'] = 'wrong'
        store.saved[table]['report.json'] = wf.json_bytes(dict(identity=identity))
        with self.assertRaisesRegex(ValueError, 'provenance or predictions'):
            b.frozen_benchmarks(self.data, prep, protocol, helper, store, prefix)

    def test_expired_input_load_and_incomplete_validation_never_save(self):
        prep, protocol, prefix, store, helper, lock = mocks(self.data)
        with self.assertRaises(TimeoutError):
            b.run(self.data, prep, protocol, prefix, 'implementation', self.cfg, helper, mdl, cmp, store,
                started=time.monotonic() - 400)
        self.assertEqual(set(store.saved), {prefix + '_LOCK'})
        real_predict = b.predict
        calls = [0]
        def failing_validation(*args, **kwargs):
            calls[0] += 1
            if calls[0] > 1:
                raise TimeoutError('Synthetic deadline mid-validation')
            return real_predict(*args, **kwargs)
        with patch.object(b, 'predict', side_effect=failing_validation), self.assertRaises(TimeoutError):
            b.run(self.data, prep, protocol, prefix, 'implementation', self.cfg, helper, mdl, cmp, store)
        self.assertEqual(set(store.saved), {prefix + '_LOCK'})

    def test_generated_execution_cells_with_synthetic_artifacts(self):
        from experiments.medbert.build_notebook import build
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            book = json.loads(build(Path(directory) / 'benchmark.ipynb').read_text(encoding='utf-8'))
        prep, protocol, prefix, store, helper, lock = mocks(self.data)
        released = []
        helper.release_inputs = lambda data: released.append(data is self.data)
        namespace = dict(hashlib=hashlib, types=types, json=json, np=np, pd=pd, time=time,
            MEDBERT_SETTINGS=self.cfg, BENCHMARK_STARTED=time.monotonic(),
            load_stage=lambda: self.data, preparation=lambda: (prep, protocol), RUN_PREFIX=prefix,
            wf=helper, mdl=mdl, cmp=cmp, wh=store, display=lambda value: None)
        for cell in book['cells'][2:]:
            exec(compile(''.join(cell['source']), '<generated>', 'exec'), namespace)
        self.assertEqual(released, [True])
        self.assertEqual(namespace['medbert_report']['status'], 'completed_full_validation')

    def test_generated_notebook_has_blank_credentials_and_all_code(self):
        from experiments.medbert.build_notebook import build
        with tempfile.TemporaryDirectory() as directory:
            path = build(Path(directory) / 'benchmark.ipynb')
            book = json.loads(path.read_text(encoding='utf-8'))
        self.assertTrue(all(c['cell_type'] == 'code' and not c['outputs'] for c in book['cells']))
        import ast
        source = ''.join(book['cells'][0]['source'])
        assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)]
        sf = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == 'sf_options' for t in n.targets))
        self.assertTrue(all(v == '' for v in ast.literal_eval(sf.value).values()))
        self.assertLess(source.index('BENCHMARK_STARTED'), source.index('sf_options'))
        self.assertNotIn('EVALUATE_TEST = True', ''.join(''.join(c['source']) for c in book['cells']))


if __name__ == '__main__':
    unittest.main()
