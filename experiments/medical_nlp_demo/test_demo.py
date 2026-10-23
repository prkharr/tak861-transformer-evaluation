import json
import tempfile
import unittest
from pathlib import Path
from decimal import Decimal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS

from . import build_demo as builder


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(np=np, pd=pd, sparse=sparse, CountVectorizer=CountVectorizer,
                       TfidfTransformer=TfidfTransformer, ENGLISH_STOP_WORDS=ENGLISH_STOP_WORDS)
        exec(builder.HELPERS, self.ns)
        self.cfg = dict(max_source_rows=200000, max_patients=100, max_terms=256,
                        cohort='D.S.C', medical='D.S.M', reference='D.S.R', sample_seed=42, months=12,
                        event_id_column='MEDICAL_EVENT_ID')
        self.frame = pd.DataFrame([
            ['p1','2026-10-20',0,'DX','A','sleep disorder',1,2,0],
            ['p1','2026-10-20',11,'DX','B','chronic pain',1,3,0],
            ['p2','2026-10-20',0,'DX','B','chronic pain',1,1,0],
            ['p2','2026-10-20',11,'PX','C',None,0,1,0],
            ['p3','2026-10-20',None,None,None,None,None,None,None]],
            columns=['PATIENT_ID','END_DT','STEP','FAMILY','CODE','DESCRIPTION',
                     'DESCRIPTION_VARIANTS','EVENT_COUNT','NULL_EVENT_ID_ROWS'])

    def run_demo(self, frame=None):
        return self.ns['make_nlp_sequence'](self.frame if frame is None else frame,self.cfg)

    def test_real_temporal_positions_and_no_claim_retention(self):
        result = self.run_demo()
        self.assertEqual(result['X_sequence'].shape[:2],(3,12))
        x = result['X_sequence']
        terms = list(result['terms'])
        self.assertGreater(x[0,0,terms.index('sleep')],0)
        self.assertEqual(x[0,11,terms.index('sleep')],0)
        self.assertGreater(x[0,11,terms.index('pain')],0)
        self.assertTrue((x[2]==0).all())
        self.assertTrue((x[:,1:11]==0).all())
        self.assertEqual(result['coverage'].iloc[11].patients_with_dx_px_activity,2)
        self.assertEqual(result['coverage'].iloc[11].patients_with_mapped_description,1)
        self.assertAlmostEqual(result['audit']['mapped_code_event_fraction'],6/7)
        self.assertEqual(x.dtype,np.float32)

    def test_ranking_and_direct_text_equivalence(self):
        r = self.run_demo()
        texts = ['']*36
        texts[0] = 'sleep disorder '*2
        texts[11] = 'chronic pain '*3
        texts[12] = 'chronic pain'
        direct = r['vectorizer'].transform(texts)[:,r['selected_term_indices']]
        # Repeated full text introduces cross-description bigrams, which are deliberately
        # absent from the dictionary-based representation. Check unigrams only.
        code_counts = np.zeros((36,len(r['terms'])),dtype=float)
        for i,desc,n in [(0,'sleep disorder',2),(11,'chronic pain',3),(12,'chronic pain',1)]:
            code_counts[i] = r['vectorizer'].transform([desc])[:,r['selected_term_indices']].toarray()[0]*n
        expected = TfidfTransformer(sublinear_tf=True).fit_transform(code_counts).toarray()
        np.testing.assert_allclose(r['X_sequence'].reshape(36,-1),expected,rtol=1e-6,atol=1e-7)
        self.assertTrue(r['ranking'].mean_tfidf.is_monotonic_decreasing)
        np.testing.assert_allclose(direct.toarray()[:,[i for i,t in enumerate(r['terms']) if ' ' not in t]],
            code_counts[:,[i for i,t in enumerate(r['terms']) if ' ' not in t]])

    def test_reordering_and_decimal_sources(self):
        base = self.run_demo()
        frame = self.frame.sample(frac=1,random_state=7).copy()
        frame['EVENT_COUNT'] = frame.EVENT_COUNT.map(lambda x: Decimal(str(x)) if pd.notna(x) else None)
        result = self.run_demo(frame)
        np.testing.assert_array_equal(base['terms'],result['terms'])
        np.testing.assert_array_equal(base['X_sequence'],result['X_sequence'])

    def test_negation_preserved_and_partial_months(self):
        f = self.frame.copy()
        f.loc[0,'DESCRIPTION'] = 'disease without complication'
        f.loc[[1,2],'DESCRIPTION'] = 'disease with complication'
        f.loc[f.PATIENT_ID=='p2','END_DT'] = '2026-10-31'
        r = self.run_demo(f)
        self.assertIn('without complication',r['terms'])
        self.assertIn('with complication',r['terms'])
        self.assertEqual(r['audit']['snapshots_with_partial_final_month'],2)

    def test_frozen_transform_does_not_refit(self):
        original = self.run_demo()
        state = original['feature_state']
        idf_before = state['tfidf'].idf_.copy()
        f = self.frame.copy()
        f['END_DT'] = '2026-11-20'
        reused = self.ns['make_nlp_sequence'](f,self.cfg,frozen_state=state)
        np.testing.assert_array_equal(reused['X_sequence'],original['X_sequence'])
        np.testing.assert_array_equal(state['tfidf'].idf_,idf_before)
        self.assertEqual(list(reused['terms']),list(original['terms']))
        f.loc[0,'DESCRIPTION'] = 'changed code description'
        with self.assertRaises(ValueError):
            self.ns['make_nlp_sequence'](f,self.cfg,frozen_state=state)

    def test_medical_type_and_stage_numbers_preserved(self):
        f = self.frame.copy()
        f.loc[0,'DESCRIPTION'] = 'Narcolepsy type 1 with cataplexy'
        f.loc[[1,2],'DESCRIPTION'] = 'Narcolepsy type 2 without cataplexy'
        r = self.run_demo(f)
        self.assertIn('type 1',r['terms'])
        self.assertIn('type 2',r['terms'])
        self.assertIn('without cataplexy',r['terms'])

    def test_reject_invalid_input(self):
        for column,value in [('STEP',12),('STEP',.5),('EVENT_COUNT',np.inf),('NULL_EVENT_ID_ROWS',1)]:
            f = self.frame.copy()
            f.loc[0,column] = value
            with self.assertRaises(ValueError):
                self.run_demo(f)
        with self.assertRaises(ValueError):
            self.run_demo(pd.concat([self.frame,self.frame.iloc[[0]]]))
        self.cfg['max_source_rows'] = 1
        with self.assertRaises(ValueError):
            self.run_demo()

    def test_missing_descriptions_and_multiple_snapshot_stop(self):
        f = self.frame.copy()
        f['DESCRIPTION'] = None
        with self.assertRaises(ValueError):
            self.run_demo(f)
        f = self.frame.copy()
        f.loc[1,'END_DT'] = '2026-09-20'
        with self.assertRaises(ValueError):
            self.run_demo(f)

    def test_sql_compiles_and_is_label_free(self):
        import sqlglot
        query = self.ns['source_sql'](self.cfg)
        sqlglot.parse_one(query,read='snowflake')
        self.assertNotIn('RESP',query)
        self.assertIn("DATEADD(month,-11,DATE_TRUNC('month',END_DT))",query)
        self.assertIn('LIMIT 200001',query)
        self.cfg['medical']='bad;DROP TABLE x'
        with self.assertRaises(ValueError):
            self.ns['source_sql'](self.cfg)

    def test_notebook_outputs_and_export(self):
        import hashlib, time, os, joblib, platform, importlib.metadata
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        builder.build()
        nb = json.loads(builder.DEST.read_text())
        self.assertEqual(len(nb['cells']),7)
        self.assertTrue(all(c['cell_type']=='code' and not c['outputs'] for c in nb['cells']))
        for c in nb['cells']:
            compile(''.join(c['source']),'cell','exec')
        with tempfile.TemporaryDirectory() as d:
            ns = {**self.ns, 'medical_aggregates':self.frame,'CONFIG':{**self.cfg,'output_directory':d},
                  'display':lambda *args:None,'plt':plt,'os':os,'json':json,'time':time,'joblib':joblib,'hashlib':hashlib,
                  'platform':platform,'importlib':importlib,
                  'query_sha256':'synthetic','started':time.perf_counter()}
            for source in [builder.MODEL,builder.PLOT,builder.SAVE,builder.EXPORT]:
                exec(source,ns)
            self.assertEqual(len(list(Path(d).iterdir())),6)
            self.assertNotIn('PATIENT_ID',(Path(d)/'nlp_feature_ranking.csv').read_text())
            bundle = Path(d)/'private_feature_package'
            stored = np.load(bundle/'sequential_features.npz',allow_pickle=False)
            specs = pd.read_csv(bundle/'ordered_feature_list.csv')
            self.assertEqual(list(stored['feature_names']),list(specs.feature_name))
            np.testing.assert_array_equal(stored['X_sequence'],ns['demo']['X_sequence'])
            contract = json.loads((bundle/'feature_contract.json').read_text())
            for name,sha in contract['files_sha256'].items():
                self.assertEqual(hashlib.sha256((bundle/name).read_bytes()).hexdigest(),sha)
            saved_state = joblib.load(bundle/'frozen_tfidf_state.joblib')
            restored = self.ns['make_nlp_sequence'](self.frame,self.cfg,frozen_state=saved_state)
            np.testing.assert_array_equal(stored['X_sequence'],restored['X_sequence'])
            stored.close()


if __name__=='__main__':
    unittest.main()
