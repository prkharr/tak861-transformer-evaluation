"""Synthetic preparation parity, integrity, resume and fixed-panel refresh checks."""
import gc
import hashlib
import io
import json
import re
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.automl_sequence import inputs, raw_automl, workflow
from experiments.cached_pilot import analysis, cache


class PilotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'extraction').mkdir()
        self.source = self.root/'extraction/raw.sqlite'
        self.template = 'WITH REQUESTS AS ({request_sql}) SELECT * FROM REQUESTS'
        rows=[]
        for n in range(120):
            split=['train','train','train','validation','test'][n%5]
            for date in ['2025-12-19','2026-01-19']:
                rows.append(dict(PATIENT_ID=f'{n:03d}',END_DT=pd.Timestamp(date),RESP=int(n%3==0),SPLIT=split))
        self.meta=pd.DataFrame(rows)
        raw_automl.inputs=inputs
        self.arrays=[]
        requests,_=inputs.make_requests(self.meta,steps=1,date_policy='month_end_history')
        with closing(sqlite3.connect(self.source)) as db, redirect_stdout(io.StringIO()):
            raw_automl._spool_long(self.reader,requests,self.template,20,
                                  set(self.meta.loc[self.meta.SPLIT.eq('train'),'PATIENT_ID']),db,
                                  run_identity='SYNTHETIC')

    def tearDown(self):
        for x in self.arrays:
            if hasattr(x,'_mmap') and x._mmap is not None:
                try:x._mmap.close()
                except Exception:pass
        gc.collect()
        self.temp.cleanup()

    def reader(self, sql):
        rows=[]
        for p,d in re.findall(r"\('([^']+)','(\d{4}-\d{2}-\d{2})'\)",sql):
            n=int(p)
            observed=n%17!=0
            rows.append(dict(PATIENT_ID=p,AS_OF_DATE=d,ROW_KIND='COVERAGE',CATEGORY=None,
                             VALUE=None,RECENT_VALUE=None,MAX_EVENT_DATE=d if observed else None,
                             AGE=40+n%30,AGE_CONFLICT=0))
            if observed:
                for cat,value,recent in [('STUDY',8+n%4,3),('SLEEP',6,2)]:
                    rows.append(dict(PATIENT_ID=p,AS_OF_DATE=d,ROW_KIND='FEATURE',CATEGORY=cat,
                                     VALUE=value,RECENT_VALUE=recent,MAX_EVENT_DATE=d,AGE=None,AGE_CONFLICT=None))
                if n%5==3:
                    rows.append(dict(PATIENT_ID=p,AS_OF_DATE=d,ROW_KIND='FEATURE',CATEGORY='VAL_ONLY',
                                     VALUE=100,RECENT_VALUE=30,MAX_EVENT_DATE=d,AGE=None,AGE_CONFLICT=None))
        return pd.DataFrame(rows)

    def prepared(self,limit=8,version='v1',previous=None,max_features=512):
        with redirect_stdout(io.StringIO()):
            manifest=cache.freeze_cache(self.source,self.root/('cache_'+version),self.meta,chunk_limit=limit)
        result=analysis.prepare(manifest,self.root/version,previous_version=previous,max_features=max_features)
        self.arrays.append(result['X_automl'])
        return result

    def group(self,prepared,names=None,nlp=False):
        meta=prepared['metadata'].iloc[::-1].reset_index(drop=True)
        names=names or [f'TIME_{n}' for n in range(9)]
        values=np.array([[int(p)%7+j/10 for j in range(len(names))] for p in meta.PATIENT_ID],np.float32)
        provenance=dict(verified=True,definition='synthetic fixed formulas')
        if nlp:provenance.update(fit_scope='train_only',fitted_on='full_parent_train',encoder_sha256='abc')
        return dict(metadata=meta,values=values,names=names,provenance=provenance)

    def run_pilot(self,p,timing=None,nlp=None):
        with patch.object(analysis,'memory_guard',return_value=50*1024**3), redirect_stdout(io.StringIO()):
            return analysis.run(p,timing or self.group(p),nlp,bootstrap=20,min_free_gib=0)

    def test_complete_patients_no_test_and_pair_cap_train_only(self):
        p=self.prepared(limit=7,max_features=4)
        self.assertFalse(p['metadata'].SPLIT.eq('test').any())
        self.assertEqual(p['metadata'].groupby('PATIENT_ID').size().unique().tolist(),[2])
        self.assertNotIn('VAL_ONLY',p['preparation']['categories'])
        self.assertEqual(len(p['preparation']['categories']),1)
        self.assertEqual(len(p['feature_names']),4)
        self.assertTrue(p['preparation']['categories'][0] in ['STUDY','SLEEP'])

    def test_raw_feature_parity_with_existing_builder(self):
        p=self.prepared()
        with patch.object(raw_automl,'raw_sql_template',return_value=self.template), redirect_stdout(io.StringIO()):
            old=raw_automl.build_raw_automl_inputs(self.reader,p['metadata'],dict(request_steps=1,max_features=512))
        panel=old['cached_panel'].rename(columns={'AS_OF_DATE':'END_DT'})
        ix=pd.MultiIndex.from_frame(panel[['PATIENT_ID','END_DT']]).get_indexer(pd.MultiIndex.from_frame(p['metadata'][['PATIENT_ID','END_DT']]))
        np.testing.assert_array_equal(p['X_automl'][:,:-1],panel.iloc[ix][p['feature_names'][:-1]].to_numpy(np.float32))
        np.testing.assert_array_equal(p['X_automl'][:,-1],panel.iloc[ix].IS_OBSERVED.to_numpy(np.float32))
        self.assertEqual([x['name'] for x in old['feature_specs']],p['feature_names'][:-1])

    def test_alignment_is_by_keys_and_rejects_changed_labels(self):
        p=self.prepared();g=self.group(p)
        a=analysis._align_group(p,g,'time');self.arrays.append(a['values'])
        np.testing.assert_array_equal(a['values'][:,0],[int(x)%7 for x in p['metadata'].PATIENT_ID])
        g['metadata'].loc[0,'RESP']=1-g['metadata'].loc[0,'RESP']
        with self.assertRaisesRegex(ValueError,'mismatch'):
            analysis._align_group(p,g,'time')

    def test_refresh_freezes_validation_and_feature_definitions(self):
        p=self.prepared(limit=7)
        report=self.run_pilot(p)
        q=self.prepared(limit=12,version='v2',previous=self.root/'v1')
        self.assertEqual(q['preparation']['evaluation_keys_sha256'],p['preparation']['evaluation_keys_sha256'])
        self.assertEqual(q['preparation']['evaluation_labels_sha256'],p['preparation']['evaluation_labels_sha256'])
        self.assertGreater(len(q['train_indices']),len(p['train_indices']))
        self.assertEqual(len(q['validation_indices']),len(p['validation_indices']))
        self.assertEqual(q['feature_names'],p['feature_names'])
        newer=self.run_pilot(q)
        self.assertEqual(report['identity']['feature_definition_hash'],newer['identity']['feature_definition_hash'])
        self.assertEqual(report['identity']['protocol_hash'],newer['identity']['protocol_hash'])

    def test_real_fixed_tree_fits_and_restart_reuses_scores(self):
        import lightgbm
        p=self.prepared()
        report=self.run_pilot(p)
        self.assertEqual(set(r['Model'] for r in report['rows']),{'AUTOML','AUTOML_TIME'})
        self.assertEqual(report['identity']['settings']['n_estimators'],300)
        self.assertNotIn('stopping_rounds',report['identity']['settings'])
        self.assertIn('AUTOML_TIME',report['paired_lift'])
        with patch.object(lightgbm.LGBMClassifier,'fit',side_effect=AssertionError('unexpected refit')):
            again=self.run_pilot(p)
        self.assertEqual(report,again)
        receipt=json.loads((self.root/'v1/report_receipt.json').read_text())
        self.assertEqual(receipt['sha256'],analysis._file_hash(self.root/'v1/report.json'))

    def test_refresh_rejects_changed_protocol_or_implementation(self):
        self.prepared(limit=7)
        for attribute,replacement in [('SETTINGS',dict(analysis.SETTINGS,n_estimators=301)),
                                      ('SOURCE_SHA256','changed-implementation')]:
            with patch.object(analysis,attribute,replacement):
                with self.assertRaisesRegex(ValueError,'frozen pilot settings or implementation'):
                    self.prepared(limit=12,version='changed_'+attribute,previous=self.root/'v1')

    def test_failed_nlp_provenance_does_not_block_base_pair(self):
        p=self.prepared();nlp=self.group(p,names=['TERM_A','ACTIVE'],nlp=True)
        nlp['provenance']['fit_scope']='all_splits'
        report=self.run_pilot(p,nlp=nlp)
        self.assertEqual(len(report['blocked_arms']),2)
        self.assertEqual(set(r['Model'] for r in report['rows']),{'AUTOML','AUTOML_TIME'})

    def test_verified_nlp_adds_two_arms_without_refitting_base(self):
        import lightgbm
        p=self.prepared();self.run_pilot(p)
        original=lightgbm.LGBMClassifier.fit;calls=[]
        def fit(model,*args,**kwargs):
            calls.append(args[0].shape[1]);return original(model,*args,**kwargs)
        with patch.object(lightgbm.LGBMClassifier,'fit',new=fit):
            report=self.run_pilot(p,nlp=self.group(p,names=['TERM_A','ACTIVE'],nlp=True))
        self.assertEqual(len(calls),2)
        self.assertEqual(set(r['Model'] for r in report['rows']),set(analysis.ARMS))

    def test_embedded_loader_two_stage_resume_and_source_hash(self):
        import lightgbm
        from experiments.cached_pilot.build_comparison import loader,sources
        namespace={}
        exec(loader(sources()),namespace)
        embedded=namespace['pilot_analysis']
        expected=hashlib.sha256(json.dumps(namespace['PILOT_HASHES'],sort_keys=True).encode()).hexdigest()
        self.assertEqual(embedded.SOURCE_SHA256,expected)
        self.assertEqual(len(expected),64)
        with patch(__name__+'.analysis',embedded):
            p=self.prepared();self.run_pilot(p)
            original=lightgbm.LGBMClassifier.fit;calls=[]
            def fit(model,*args,**kwargs):
                calls.append(args[0].shape[1]);return original(model,*args,**kwargs)
            with patch.object(lightgbm.LGBMClassifier,'fit',new=fit):
                report=self.run_pilot(p,nlp=self.group(p,names=['TERM_A','ACTIVE'],nlp=True))
            self.assertEqual(len(calls),2)
            self.assertEqual(set(r['Model'] for r in report['rows']),set(embedded.ARMS))
            self.assertEqual(p['preparation']['implementation'],expected)

    def test_saved_score_tamper_rejected(self):
        p=self.prepared();self.run_pilot(p)
        (self.root/'v1/AUTOML/validation.npy').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_pilot(p)

    def test_mutated_metadata_rejected_before_training(self):
        p=self.prepared();p['metadata'].loc[0,'RESP']=1-p['metadata'].loc[0,'RESP']
        with self.assertRaisesRegex(ValueError,'Prepared population'):self.run_pilot(p)

    def test_preparation_rerun_and_mismatch_preserve_saved_arrays(self):
        p=self.prepared()
        before=analysis._file_hash(self.root/'v1/automl.npy')
        q=self.prepared()
        self.assertEqual(p['preparation'],q['preparation'])
        with self.assertRaisesRegex(ValueError,'different preparation'):
            self.prepared(max_features=4)
        self.assertEqual(analysis._file_hash(self.root/'v1/automl.npy'),before)

    def test_partial_model_failure_resumes_completed_arm_only(self):
        import lightgbm
        p=self.prepared();original=lightgbm.LGBMClassifier.fit;calls=[]
        def fail_second(model,*args,**kwargs):
            calls.append(args[0].shape[1])
            if len(calls)==2:raise InterruptedError('synthetic interruption')
            return original(model,*args,**kwargs)
        with patch.object(lightgbm.LGBMClassifier,'fit',new=fail_second):
            with self.assertRaises(InterruptedError):self.run_pilot(p)
        saved=analysis._file_hash(self.root/'v1/AUTOML/validation.npy')
        self.assertTrue((self.root/'v1/report.json').exists())
        self.assertFalse((self.root/'v1/AUTOML_TIME/receipt.json').exists())
        calls=[]
        def count(model,*args,**kwargs):
            calls.append(args[0].shape[1]);return original(model,*args,**kwargs)
        with patch.object(lightgbm.LGBMClassifier,'fit',new=count):self.run_pilot(p)
        self.assertEqual(len(calls),1)
        self.assertEqual(saved,analysis._file_hash(self.root/'v1/AUTOML/validation.npy'))

    def test_memory_guard_pauses_without_modifying_sources(self):
        with patch.dict('sys.modules',{'psutil':SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=1))}):
            with self.assertRaisesRegex(MemoryError,'Pause analysis'):
                analysis.memory_guard(min_free_gib=2)

    def test_deterministic_tie_overlap(self):
        y=np.array([1,0,1,0,1,0,1,0,1,0])
        result=analysis._overlap(y,np.ones(10)*.5,np.ones(10)*.5)
        self.assertEqual(result['positive_shared'],1)
        self.assertEqual(result['additional_captures'],0)


if __name__=='__main__': unittest.main()
