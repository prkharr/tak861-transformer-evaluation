import time
import unittest
from unittest.mock import patch
import numpy as np
from experiments.medbert import pretrained_transfer as t, benchmark as b
from experiments.medbert.test_benchmark import fixture, mocks
from experiments.automl_sequence import modeling as mdl
from experiments.rigorous_transformer import comparison as cmp

class TransferTests(unittest.TestCase):
    def test_mask_recency_and_test_exclusion(self):
        data = fixture()
        data['X_count'][0,5,0] = 999
        c,e = t.features(data, np.array([0]), [0,1], np.eye(2), b.Budget(30), 0)
        self.assertAlmostEqual(c[0,0], np.log(3), places=5)
        self.assertEqual(c[0,2], 0)
        self.assertEqual(e[0,0], 1)
        with self.assertRaisesRegex(ValueError, 'TEST'):
            t.features(data, np.array([40]), [0,1], np.eye(2), b.Budget(30), 0)

    def test_full_pipeline_with_synthetic_encoder(self):
        data = fixture()
        prep, protocol, prefix, store, wf, _ = mocks(data)
        cfg = dict(max_train_patients=20, max_train_rows=24, bootstrap_draws=20,
                   projection_dim=2, total_seconds=300, save_reserve=1)
        def fake(names, *args):
            self.assertEqual(names, ['DX__a','PX__b']) # no held-out-only RX code
            return np.eye(2, dtype='float32'), {'model_id':'synthetic_test_only'}
        with patch.object(t, 'encode', side_effect=fake):
            table, report = t.run(data, prep, protocol, prefix, 'test', cfg, b, wf, store, cmp, mdl, time.monotonic())
        self.assertEqual(report['status'], 'completed_full_validation')
        self.assertEqual(len(report['rows']), 4)
        self.assertEqual(len(report['intervals']), 2)
        for name in ['MATCHED_COUNT_LR','COUNT_PLUS_RANDOM_LR','COUNT_PLUS_BIOMEDBERT_LR']:
            p = wf.bytes_array(store.saved[table][name+'.npy'])
            self.assertEqual(p.shape, (16,))
            self.assertTrue(np.isfinite(p).all())
        with patch.object(t, 'encode', side_effect=AssertionError('Resume must not re-encode')):
            table2, report2 = t.run(data, prep, protocol, prefix, 'test', cfg, b, wf, store, cmp, mdl, time.monotonic())
        self.assertEqual(table, table2)
        self.assertEqual(report['identity'], report2['identity'])
        store.saved[table]['MATCHED_COUNT_LR.npy'] = b'corrupt'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            t.run(data, prep, protocol, prefix, 'test', cfg, b, wf, store, cmp, mdl, time.monotonic())

    def test_head_reconstruction(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(40,4)); y = (x[:,0]>0).astype(int)
        p,state,_ = t.fit_head(x,y,x,np.ones(40),t.SETTINGS)
        z = ((x-state['mean'])/state['scale']) @ state['coef'].T + state['intercept']
        np.testing.assert_allclose(p, (1/(1+np.exp(-z))).ravel())

    def test_expired_budget(self):
        with self.assertRaises(TimeoutError):
            t.features(fixture(), np.array([0]), [0], np.ones((1,2)), b.Budget(-1), 0)

if __name__ == '__main__': unittest.main()
