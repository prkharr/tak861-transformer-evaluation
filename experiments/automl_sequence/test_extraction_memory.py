"""Bounded extraction and exact frozen-fixture equivalence; no private data."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.automl_sequence import inputs, raw_automl as raw
from experiments.automl_sequence import test_raw_automl, test_inputs


class ExtractionMemoryTest(unittest.TestCase):
    def setUp(self):
        self.fixture=test_raw_automl.RawAutoMLTest(); self.fixture.setUp()
        self.owned=[]

    def tearDown(self):
        for value in self.owned:
            if isinstance(value,inputs.DiskPanel): value.close()
            else: inputs.release_array(value)

    def test_disk_panel_preserves_frozen_baseline_hashes_and_tensor(self):
        # Digests recorded from the untouched 2106cf1/361ee87 implementation.
        plain=raw.build_raw_automl_inputs(lambda sql:self.fixture.long,self.fixture.meta)
        with patch.object(inputs,'PANEL_MEMORY_LIMIT',0):
            disk=raw.build_raw_automl_inputs(lambda sql:self.fixture.long.sample(frac=1,random_state=19),self.fixture.meta)
        self.owned.append(disk['cached_panel'])
        self.assertIsInstance(disk['cached_panel'],inputs.DiskPanel)
        self.assertFalse(disk['cached_panel'].values.flags.writeable)
        self.assertEqual(disk['receipt']['source_content_sha256'],
            '93fa8f46e42b63a8b54dcfa4e6a9f61d5ade0e01bdf4b5b2bf6954aebcea7d43')
        expected={'panel_sha256':'a2f2d48b4a5d0a7410b01b2d41ed0f6afb4b9661ac757ddfc3f022512dadcd59',
            'key_sha256':'37c832bd2c81d242a0d8a03060b8309022a91ade6ed459e52bba8c7c36cdfef3',
            'contract_sha256':'64ceefbbddcdca8f6c950eef028827596511ca16c506e4035258eb6641c4d4af',
            'feature_specs_sha256':'a2df8f38089ad75929ea5d664ef636af45385c2089a7528853227b49b3fc8a2e'}
        for key,digest in expected.items(): self.assertEqual(disk['cached_manifest'][key],digest,key)
        self.assertEqual(plain['feature_specs'],disk['feature_specs'])
        self.assertEqual(plain['contract'],disk['contract'])
        a=inputs.assemble_inputs(self.fixture.meta,plain['cached_panel'],plain['feature_specs'],date_policy='month_end_history')
        b=inputs.assemble_inputs(self.fixture.meta,disk['cached_panel'],disk['feature_specs'],date_policy='month_end_history')
        for data in (a,b): self.owned.extend([data['X_sequence'],data['X_summary']])
        for key in ('X_sequence','X_summary','sequence_mask','summary_mask','y'):
            np.testing.assert_array_equal(a[key],b[key])
        self.assertIsInstance(b['X_sequence'],np.memmap)
        self.assertEqual(b['X_sequence'].dtype,np.dtype('float32'))
        self.assertEqual(a['hashes'],b['hashes']);self.assertEqual(a['report'],b['report'])
        self.assertEqual(b['hashes']['grid_sha256'],'2a20b1b8b7438217da77706b19d9da19dddb5dad1a9771c68f1c7e1906908946')
        req,_=inputs.make_requests(self.fixture.meta,date_policy='month_end_history')
        with patch.object(disk['cached_panel'],'iter_frames',side_effect=AssertionError('Panel rescanned')):
            reused,_=inputs.obtain_panel(req,disk['feature_specs'],disk['contract'],disk['cached_panel'],disk['cached_manifest'])
            self.assertIs(reused,disk['cached_panel'])

    def test_spark_iterator_is_bounded_and_preserves_nullable_integer_hash_type(self):
        source=self.fixture.long
        # Separate coverage and feature partitions make per-chunk inference differ
        # from whole-query pandas unless Spark nullable integer types are retained.
        ordered=pd.concat([source[source.ROW_KIND.eq('FEATURE')]]*200+
                          [source[source.ROW_KIND.eq('COVERAGE')]],ignore_index=True)
        records=[]
        for row in ordered.itertuples(index=False,name=None):
            records.append(tuple(None if pd.isna(v) else int(v) if c in ('VALUE','RECENT_VALUE','AGE','AGE_CONFLICT') else v
                                 for c,v in zip(ordered.columns,row)))
        LongType=type('LongType',(),{})
        class SparkResult:
            columns=list(ordered.columns)
            schema=SimpleNamespace(fields=[SimpleNamespace(name=c,dataType=LongType(),nullable=True)
                for c in ('VALUE','RECENT_VALUE','AGE','AGE_CONFLICT')])
            def toPandas(self): raise AssertionError('Whole-query collection is prohibited')
            def toLocalIterator(self,prefetchPartitions):
                if prefetchPartitions: raise AssertionError('Partition prefetch is prohibited')
                return iter(records)
        req,_=inputs.make_requests(self.fixture.meta)
        parts=list(raw._read_long(lambda sql:SparkResult(),req,'WITH REQUESTS AS ({request_sql})',5000))
        self.assertGreater(len(parts),1)
        self.assertLessEqual(max(map(len,parts)),4096)
        for part in parts: self.assertEqual(part.VALUE.dtype,np.dtype('float64'))
        self.assertEqual(inputs.frame_hash(pd.concat(parts,ignore_index=True)),inputs.frame_hash(ordered))

    def test_partial_disk_cache_combines_without_wide_pandas_concat(self):
        fixture=test_inputs.InputsTest();fixture.setUp()
        req,_=inputs.make_requests(fixture.metadata)
        all_panel=inputs.validate_panel(fixture.extractor(req),fixture.specs)
        first=all_panel.iloc[:3].copy()
        names=[s['name'] for s in fixture.specs]
        values=inputs.disk_array((len(first),len(names)),np.float64)
        values[:]=first[names].to_numpy();values.flush();values.flags.writeable=False
        disk=inputs.DiskPanel(first.drop(columns=names),values,names);self.owned.append(disk)
        manifest=inputs.make_panel_manifest(disk,fixture.specs,fixture.contract)
        result,receipt=inputs.obtain_panel(req,fixture.specs,fixture.contract,disk,manifest,fixture.extractor)
        self.owned.append(result)
        self.assertIsInstance(result,inputs.DiskPanel)
        self.assertEqual(inputs.frame_hash(result),inputs.frame_hash(all_panel))
        self.assertEqual(receipt['extraction_report']['cache_keys_reused'],3)

    def test_owned_memmap_is_deleted_after_release(self):
        data=inputs.disk_array((3,12,5))
        path=data.filename
        self.assertTrue(os.path.isfile(path))
        inputs.release_array(data)
        self.assertFalse(os.path.exists(path))

    def test_releasing_views_rejects_without_closing_or_deleting_owner(self):
        owner=inputs.disk_array((3,12,5));self.owned.append(owner)
        owner[:]=7
        for view in (owner[1:],owner.reshape(36,5),np.asarray(owner),np.asarray(owner)[1:]):
            with self.subTest(type=type(view).__name__),self.assertRaisesRegex(ValueError,'release the owning array'):
                inputs.release_array(view)
            self.assertFalse(owner._mmap.closed)
            self.assertTrue(os.path.isfile(owner.filename))
            self.assertEqual(float(owner[0,0,0]),7.)
        inputs.release_array(owner)
        self.assertFalse(os.path.exists(owner.filename))

    def test_readonly_reopen_transfers_ownership_and_rejects_old_handle(self):
        original=inputs.disk_array((3,12,5));original[:]=9
        owner=inputs.readonly_array(original);self.owned.append(owner)
        self.assertFalse(owner.flags.writeable)
        with self.assertRaisesRegex(ValueError,'release the owning array'):
            inputs.release_array(original)
        with self.assertRaisesRegex(ValueError,'release the owning array'):
            inputs.release_array(owner[1:])
        self.assertFalse(owner._mmap.closed)
        self.assertTrue(os.path.isfile(owner.filename))
        self.assertEqual(float(owner[0,0,0]),9.)
        inputs.release_array(owner)
        self.assertFalse(os.path.exists(owner.filename))

    def test_mutated_disk_metadata_changes_hash_and_rejects_frozen_manifest(self):
        with patch.object(inputs,'PANEL_MEMORY_LIMIT',0):
            result=raw.build_raw_automl_inputs(lambda sql:self.fixture.long,self.fixture.meta)
        panel=result['cached_panel'];self.owned.append(panel)
        original=inputs.frame_hash(panel)
        # Metadata is mutable independently of the read-only numeric matrix.
        panel.metadata.loc[0,'OBSERVATION_EVIDENCE']='changed provenance'
        changed=inputs.frame_hash(panel)
        self.assertNotEqual(changed,original)
        self.assertEqual(changed,inputs.frame_hash(pd.concat(list(panel.iter_frames()),ignore_index=True)))
        req,_=inputs.make_requests(self.fixture.meta,date_policy='month_end_history')
        with self.assertRaisesRegex(ValueError,'Cached panel mismatch: panel_sha256'):
            inputs.obtain_panel(req,result['feature_specs'],result['contract'],panel,result['cached_manifest'])
        # Coverage edits are also revalidated, even after the direct hash call.
        observed_row=np.flatnonzero(panel.IS_OBSERVED.to_numpy())[0]
        panel.metadata.loc[observed_row,'IS_OBSERVED']=False
        with self.assertRaisesRegex(ValueError,'unavailable|Unobserved|unobserved'):
            inputs.validate_panel(panel,result['feature_specs'])


if __name__=='__main__': unittest.main()
