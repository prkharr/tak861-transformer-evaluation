import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd

from experiments.evidence_challenger import pipeline as p
from experiments.automl_sequence import modeling as mdl, workflow as wf
from experiments.challenger import time_lgbm as timing
from experiments.rigorous_transformer import comparison as cmp


class Warehouse:
    def __init__(self): self.tables={}
    def _has_artifact_jvm(self): return True
    def table_exists(self,t): return t in self.tables
    def save_artifacts(self,t,b):
        if t in self.tables: raise ValueError('Overwrite')
        self.tables[t]=dict(b)
    def read_artifacts(self,t,n):
        if set(n)!=set(self.tables[t]): raise ValueError('Missing or unexpected saved artifacts')
        return {k:self.tables[t][k] for k in n}


def fixture():
    rng=np.random.default_rng(22); n=180
    y=np.tile([0,1],n//2)
    summary=rng.normal(size=(n,1,4)).astype('float32'); summary[:,0,0]+=y
    counts=rng.poisson(1,size=(n,12,4)).astype('float32'); counts[y==1,:,0]+=1
    meta=pd.DataFrame(dict(PATIENT_ID=['P%03d'%i for i in range(n)],END_DT=pd.to_datetime(['2026-01-01']*n),
                           SPLIT=['train']*120+['validation']*60))
    names=['PX__LEVEL_1_CPT_SLEEP_STUDY','DX__Sleep wake disorders','DX__A','RX__B']
    manifest=dict(panel_manifest={'contract':{'fit_scope':'deterministic_raw'}},
        count_contract=dict(orientation='0=newest',date_policy='month_end_history',feature_names=names),
        feature_specs=[{'kind':'continuous'}]*4)
    data=dict(metadata=meta,manifest=manifest,y=y,X_summary=summary,X_count=counts,
        X_nlp=rng.random((n,12,3)).astype('float32'),summary_mask=np.ones((n,1),bool),
        count_mask=np.ones((n,12),bool),nlp_mask=np.ones((n,12),bool),
        nlp_manifest={'fit_scope':'train_only','orientation':'0=newest'},feature_names=['A','B','C','D'],
        indices={'train':np.arange(120),'validation':np.arange(120,180)})
    prep={}; protocol=dict(experiment_phase='SUMMARY',identity={'input_manifest_hash':wf.digest(manifest)},preparation_hash=wf.digest(prep))
    pilot=SimpleNamespace(frozen_benchmarks=lambda *args:({'NLP_FUSION':np.linspace(.01,.5,60)},'test-lock'))
    return data,prep,protocol,pilot


class PipelineTests(unittest.TestCase):
    def test_end_to_end_resume_and_report(self):
        data,prep,protocol,pilot=fixture(); wh=Warehouse()
        small={k:v[:1] for k,v in p.grids().items()}
        cfg=dict(seconds=300,threads=1,device='cpu',seeds=[42],epochs=2,lgb_rounds=5,
                 selection_sizes=[2,4],bootstrap=20,batch_size=64)
        args=(data,prep,protocol,'TEST','test-source',cfg,{'study':'','diagnosis':''},mdl,wf,wh,timing,pilot)
        real_fit=p.fit
        def guarded_fit(*args,**kwargs):
            if args[-1] is not None:
                self.assertIsNone(args[7], 'Outer/final labels were passed into a fixed-round fit')
            return real_fit(*args,**kwargs)
        with patch.object(p,'grids',return_value=small), patch.object(p,'fit',side_effect=guarded_fit):
            table,report=p.run(*args)
            self.assertEqual(report['status'],'COMPLETE_DEVELOPMENT')
            self.assertEqual(len(report['rows']),5)
            cfg['seconds']=600
            with patch.object(p,'fit',side_effect=AssertionError('Resume refit')):
                table2,report2=p.run(*args)
                # Simulate interruption after all fits but before final publication.
                wh.tables.pop(table)
                table3,report3=p.run(*args)
            self.assertEqual(table,table2); self.assertEqual(report,report2)
            self.assertEqual(table,table3); self.assertEqual(report,report3)
        output=p.report_saved(table,wf,wh,cmp,draws=20)
        self.assertEqual(len(output['paired_intervals_vs_frozen_nlp']),4)
        self.assertIn('lift_by_fraction',output['paired_intervals_vs_frozen_nlp']['OOF_WEIGHTED_BLEND'])
        wh.tables[table]['bundle.zip']=b'bad'
        with self.assertRaisesRegex(ValueError,'Corrupt'): p.report_saved(table,wf,wh,cmp,draws=20)

    def test_disjoint_inner_and_budget(self):
        rows=np.arange(80); groups=np.repeat(np.arange(40),2); y=np.tile([0,1],40)
        a,b=p.inner_split(rows,y,groups,42)
        self.assertFalse(set(groups[a])&set(groups[b]))
        budget=p.Budget(-1)
        with self.assertRaises(TimeoutError): budget.check()

    def test_duplicate_keys_and_parent_endpoints(self):
        data,prep,protocol,_=fixture()
        data['metadata'].loc[1,'PATIENT_ID']=data['metadata'].loc[0,'PATIENT_ID']
        with self.assertRaisesRegex(ValueError,'Duplicate'): p.audit(data,prep,protocol,wf,timing,{'study':'','diagnosis':''})
        y=np.tile([0,1],20); w,curve=p.blend_weight(y,y,1-y,mdl)
        self.assertEqual(curve[0][0],0); self.assertEqual(curve[-1][0],1)
        self.assertGreater(w,.5)

if __name__=='__main__': unittest.main()
