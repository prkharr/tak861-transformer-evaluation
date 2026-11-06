"""Synthetic checks for compact count and monthly language handoff packages."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.automl_sequence import inputs
from experiments.cached_pilot import artifacts


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.meta = inputs.normalize_snapshots(pd.DataFrame({
            'PATIENT_ID': ['001', '002', '003', '004'],
            'END_DT': ['2026-01-19'] * 4, 'RESP': [0, 1, 1, 0],
            'SPLIT': ['train', 'train', 'validation', 'test']}))
        self.names = ['DX__sleep wake disorders', 'DX__unused', 'PX__sleep study', 'RX__unused']
        x = np.zeros((4, 12, 4), dtype=np.float32)
        x[:, :, 1] = 777
        x[:, :, 3] = 888
        x[:, 0, 2] = [2, 4, 6, 8]
        x[:, 1, 0] = [3, 6, 9, 12]
        mask = np.ones((4, 12), dtype=bool)
        self.counts = dict(X=x, valid=mask, metadata=self.meta, features=self.names,
            contract=dict(orientation='0=newest', date_policy='month_end_history',
                          metadata_sha256=inputs.frame_hash(self.meta), content_sha256='synthetic'))

    def tearDown(self):
        self.tmp.cleanup()

    def nlp(self):
        names = ['sleep', 'study']
        train_requests, _ = inputs.make_requests(self.meta[self.meta.SPLIT.eq('train')],
                                                 date_policy='month_end_history')
        requests, _ = inputs.make_requests(self.meta, date_policy='month_end_history')
        membership = self.meta.groupby('PATIENT_ID').SPLIT.first().to_dict()
        request_hash = inputs.stable_hash([[i, r.PATIENT_ID, r.AS_OF_DATE.strftime('%Y-%m-%d'), membership[r.PATIENT_ID]]
            for i, r in enumerate(requests.sort_values(['PATIENT_ID', 'AS_OF_DATE']).itertuples(index=False))])
        common = dict(fit_scope='train_only', train_patients_sha256=inputs.train_patient_hash(self.meta),
                      train_document_count=len(train_requests))
        frequency = np.array([2, 3])
        fitted = dict(common, terms=names, term_order_sha256=inputs.stable_hash(names),
                      schema='monthly_tfidf_v1', training_documents=len(train_requests),
                      document_frequency=frequency.tolist(),
                      idf=(np.log((1+len(train_requests))/(1+frequency))+1).tolist(),
                      extraction_identity=dict(requests=request_hash))
        manifest = dict(common, orientation='0=newest', metadata_hash=inputs.frame_hash(self.meta),
                        feature_names=names, fitted_state=fitted, receipt=dict(common))
        x = np.zeros((4, 12, 2), np.float32)
        mask = np.zeros((4, 12), bool)
        x[0, 0] = [1, 0]
        x[0, 3] = [0, 1]
        mask[0, [0, 3]] = True
        x[1, 1] = [.6, .8]
        mask[1, 1] = True
        x[2, 6] = [1, 0]
        mask[2, 6] = True
        return dict(metadata=self.meta.copy(), manifest=manifest, feature_names=names,
                    X_sequence=x, mask=mask, _payload_verified=True)

    def test_count_subset_orientation_and_exact_values(self):
        receipt = artifacts.freeze_count_channels(self.counts, self.root / 'counts', row_block=3)
        saved = np.load(Path(receipt).parent / 'counts.npy', allow_pickle=False)
        self.assertEqual(saved.shape, (4, 12, 2))
        self.assertEqual(saved.dtype, np.float32)
        np.testing.assert_array_equal(saved[:, :, 0], self.counts['X'][:, :, 2])
        np.testing.assert_array_equal(saved[:, :, 1], self.counts['X'][:, :, 0])
        self.assertFalse((saved == 777).any())
        self.assertFalse((saved == 888).any())
        document = json.loads(Path(receipt).read_text())
        self.assertEqual(document['identity']['channel_indices'], [2, 0])
        self.assertEqual(artifacts.freeze_count_channels(self.counts, self.root / 'counts'), receipt)

    def test_count_contract_shape_mask_and_metadata_rejected(self):
        for alteration, error in (
            (lambda c: c['contract'].update(orientation='0=oldest'), 'grid'),
            (lambda c: c['contract'].update(metadata_sha256='bad'), 'metadata'),
            (lambda c: c.update(X=c['X'][:, :11]), 'shape'),
            (lambda c: c.update(valid=c['valid'].astype('int8')), 'mask')):
            changed = deepcopy(self.counts)
            alteration(changed)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                artifacts.freeze_count_channels(changed, self.root / 'invalid')

    def test_count_finite_values_and_file_corruption_fail(self):
        changed = deepcopy(self.counts)
        changed['X'][0, 0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, 'count values'):
            artifacts.freeze_count_channels(changed, self.root / 'bad')
        receipt = artifacts.freeze_count_channels(self.counts, self.root / 'counts')
        with (Path(receipt).parent / 'counts.npy').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'content changed'):
            artifacts.timing_group(receipt, self.meta)

    def test_timing_exact_key_order_and_missing_history(self):
        self.counts['valid'][1, 5] = False
        receipt = artifacts.freeze_count_channels(self.counts, self.root / 'counts')
        selected = self.meta.iloc[[2, 0, 1]].reset_index(drop=True)
        group = artifacts.timing_group(receipt, selected)
        self.assertEqual(group['metadata'].PATIENT_ID.tolist(), ['003', '001', '002'])
        self.assertEqual(group['values'].shape, (3, 9))
        np.testing.assert_allclose(group['values'][0], [0, 1, 2, 6*5.5/143, 1, 1, 3, 9*4.5/143, 1])
        np.testing.assert_allclose(group['values'][1], [0, 1, 2/3, 2*5.5/143, 1, 1, 1, 3*4.5/143, 1])
        self.assertTrue(np.isnan(group['values'][2]).all())

    def test_timing_key_label_or_split_drift_rejected(self):
        receipt = artifacts.freeze_count_channels(self.counts, self.root / 'counts')
        for column, value in [('PATIENT_ID', 'unknown'), ('RESP', 1), ('SPLIT', 'validation')]:
            changed = self.meta.iloc[:1].copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'keys, labels or split'):
                artifacts.timing_group(receipt, changed)

    def test_nlp_mean_uses_all_twelve_positions_and_key_order(self):
        package = self.nlp()
        selected = self.meta.iloc[[1, 0, 3]].reset_index(drop=True)
        group = artifacts.nlp_group(package, selected)
        expected = [[.6/12, .8/12, 1/12], [1/12, 1/12, 2/12], [0, 0, 0]]
        np.testing.assert_allclose(group['values'], expected, rtol=1e-6)
        self.assertEqual(group['values'].dtype, np.float32)
        self.assertEqual(group['names'], ['NLP_MEAN_sleep', 'NLP_MEAN_study', 'NLP_ACTIVE_DOCUMENT_FRACTION'])
        self.assertEqual(group['provenance']['fitted_on'], 'full_parent_train')

    def test_nlp_bounded_document_hash_matches_original_with_overlapping_snapshots(self):
        second = self.meta.copy()
        second['END_DT'] = pd.Timestamp('2025-12-31')
        third = self.meta.copy()
        third['END_DT'] = pd.Timestamp('2025-11-30')
        meta = inputs.normalize_snapshots(pd.concat([third, self.meta, second], ignore_index=True))
        meta['PATIENT_ID'] = meta.PATIENT_ID.replace({'002': '00\u00e9'})
        requests, _ = inputs.make_requests(meta, date_policy='month_end_history')
        train_requests, _ = inputs.make_requests(meta[meta.SPLIT.eq('train')], date_policy='month_end_history')
        membership = meta.groupby('PATIENT_ID').SPLIT.first().to_dict()
        expected = inputs.stable_hash([[i, r.PATIENT_ID, r.AS_OF_DATE.strftime('%Y-%m-%d'), membership[r.PATIENT_ID]]
            for i, r in enumerate(requests.itertuples(index=False))])
        for size in (1, 2, 3, 4, 5, 7, 100):
            with self.subTest(size=size):
                self.assertEqual(artifacts._nlp_request_identity(meta, row_block=size), (len(train_requests), expected))

    def test_nlp_float32_mean_overflow_rejected(self):
        package = self.nlp()
        package['X_sequence'] = package['X_sequence'].astype(np.float64)
        package['X_sequence'][0, 0, 0] = float(np.finfo(np.float32).max) * 100
        with self.assertRaisesRegex(ValueError, 'finite float32'):
            artifacts.nlp_group(package, self.meta)

    def test_handoff_cross_artifact_identity_and_file_hashes(self):
        count = artifacts.freeze_count_channels(self.counts, self.root / 'counts')
        cache = self.root / 'cache_manifest.json'
        document = dict(request_steps=1, parent_metadata_sha256=inputs.frame_hash(self.meta),
                        identity=dict(run_identity='run'))
        artifacts.atomic_json(cache, document)
        config = dict(raw_automl=dict(request_steps=1))
        handoff = artifacts.save_handoff(self.root / 'handoff', cache, count, self.meta, config, 'impl', 'run')
        receipt = json.loads(Path(handoff).read_text())
        self.assertEqual(receipt['cache_manifest_sha256'], artifacts.file_hash(cache))
        self.assertEqual(receipt['count_receipt_sha256'], artifacts.file_hash(count))
        for alteration in (
            lambda d: d.update(request_steps=12),
            lambda d: d.update(parent_metadata_sha256='other'),
            lambda d: d['identity'].update(run_identity='other')):
            changed = deepcopy(document)
            alteration(changed)
            artifacts.atomic_json(cache, changed)
            with self.assertRaisesRegex(ValueError, 'source identity'):
                artifacts.save_handoff(self.root / 'blocked', cache, count, self.meta, config, 'impl', 'run')
        artifacts.atomic_json(cache, document)
        with self.assertRaisesRegex(ValueError, 'request contract'):
            artifacts.save_handoff(self.root / 'blocked', cache, count, self.meta, {'raw_automl': {'request_steps':12}}, 'impl', 'run')

    def test_nlp_fitted_state_receipt_and_train_identity_required(self):
        for location, key, value in (
            ('manifest', 'fit_scope', 'all_rows'),
            ('manifest', 'orientation', '0=oldest'),
            ('manifest', 'metadata_hash', 'wrong'),
            ('fitted_state', 'train_patients_sha256', 'wrong'),
            ('receipt', 'train_document_count', 1),
            ('fitted_state', 'terms', ['study', 'sleep']),
            ('fitted_state', 'term_order_sha256', 'wrong')):
            package = self.nlp()
            target = package['manifest'] if location == 'manifest' else package['manifest'][location]
            target[key] = value
            with self.subTest(location=location, key=key), self.assertRaises(ValueError):
                artifacts.nlp_group(package, self.meta)
        package = self.nlp()
        del package['manifest']['fitted_state']
        with self.assertRaisesRegex(ValueError, 'provenance'):
            artifacts.nlp_group(package, self.meta)

    def test_nlp_source_document_hash_required(self):
        package = self.nlp()
        package['manifest']['fitted_state']['extraction_identity']['requests'] = 'different_docs'
        with self.assertRaisesRegex(ValueError, 'source document identity'):
            artifacts.nlp_group(package, self.meta)

    def test_nlp_payload_verified_idf_and_unique_names_required(self):
        for kind in ('unverified', 'idf', 'idf_nan', 'frequency', 'empty_names', 'duplicate_names'):
            package = self.nlp()
            if kind == 'unverified':
                del package['_payload_verified']
            elif kind == 'idf':
                package['manifest']['fitted_state']['idf'][0] += .1
            elif kind == 'idf_nan':
                package['manifest']['fitted_state']['idf'][0] = np.nan
            elif kind == 'frequency':
                package['manifest']['fitted_state']['document_frequency'][0] = -1
            elif kind == 'empty_names':
                package['feature_names'] = []
            else:
                package['feature_names'] = ['sleep', 'sleep']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                artifacts.nlp_group(package, self.meta)

    def test_count_receipt_contract_shape_and_export_bounds_required(self):
        for row_block in (0, -1, 9000):
            with self.subTest(row_block=row_block), self.assertRaisesRegex(ValueError, 'row_block'):
                artifacts.freeze_count_channels(self.counts, self.root / 'invalid', row_block=row_block)
        receipt = Path(artifacts.freeze_count_channels(self.counts, self.root / 'counts'))
        original = json.loads(receipt.read_text())
        for kind in ('orientation', 'shape', 'verified', 'files'):
            changed = deepcopy(original)
            if kind == 'orientation':
                changed['identity']['contract']['orientation'] = '0=oldest'
            elif kind == 'shape':
                changed['shape'] = [4, 12, 3]
            elif kind == 'verified':
                changed['verified'] = False
            else:
                changed['files'] = {}
            receipt.write_text(json.dumps(changed))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                artifacts.timing_group(receipt, self.meta)

    def test_nlp_keys_labels_split_mask_and_values_checked(self):
        for column, value in [('PATIENT_ID', 'unknown'), ('RESP', 1), ('SPLIT', 'validation')]:
            changed = self.meta.iloc[:1].copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'keys, labels or split'):
                artifacts.nlp_group(self.nlp(), changed)
        for kind in ('nan', 'negative', 'unobserved', 'mask_type', 'shape'):
            package = self.nlp()
            if kind == 'nan':
                package['X_sequence'][0, 0, 0] = np.nan
            elif kind == 'negative':
                package['X_sequence'][0, 0, 0] = -1
            elif kind == 'unobserved':
                package['X_sequence'][0, 1, 0] = .5
            elif kind == 'mask_type':
                package['mask'] = package['mask'].astype(np.int8)
            else:
                package['X_sequence'] = package['X_sequence'][:, :11]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                artifacts.nlp_group(package, self.meta)

    def test_handoff_refuses_credentials_in_nested_configuration(self):
        with self.assertRaisesRegex(ValueError, 'Credentials'):
            artifacts.save_handoff(self.root, 'cache.json', 'counts.json', self.meta,
                                   {'nested': [{'sfPassword': 'synthetic'}]}, 'impl', 'PREFIX')


if __name__ == '__main__':
    unittest.main()
