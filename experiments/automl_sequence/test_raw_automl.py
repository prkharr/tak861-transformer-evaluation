"""Synthetic raw SQL/long-form adapter tests; never production execution."""
import unittest
import numpy as np
import pandas as pd

try:
    from . import inputs, raw_automl as raw
except ImportError:
    import inputs
    import raw_automl as raw


class RawAutoMLTest(unittest.TestCase):
    def setUp(self):
        raw.inputs=inputs
        self.meta=pd.DataFrame({'PATIENT_ID':['train','heldout'],
            'END_DT':pd.to_datetime(['2026-10-21']*2),'RESP':[1,0],'SPLIT':['train','test']})
        req,_=inputs.make_requests(self.meta,date_policy='month_end_history')
        rows=[]
        for patient,date in req.itertuples(index=False,name=None):
            observed=date>=pd.Timestamp('2026-02-28')
            rows.append(dict(PATIENT_ID=patient,AS_OF_DATE=date,ROW_KIND='COVERAGE',CATEGORY=None,
                             VALUE=None,RECENT_VALUE=None,MAX_EVENT_DATE=date if observed else None,
                             AGE=40,AGE_CONFLICT=0))
            if observed:
                for cat,value,recent in [('KNOWN',10,4),('HELDOUT_ONLY' if patient=='heldout' else 'TRAIN_ONLY',3,1)]:
                    rows.append(dict(PATIENT_ID=patient,AS_OF_DATE=date,ROW_KIND='FEATURE',CATEGORY=cat,
                                     VALUE=value,RECENT_VALUE=recent,MAX_EVENT_DATE=date,AGE=None,AGE_CONFLICT=None))
        self.long=pd.DataFrame(rows)

    def test_sql_is_readonly_raw_and_uses_recovered_windows(self):
        sql=raw.raw_sql_template()
        self.assertTrue(sql.startswith('WITH REQUESTS AS ({request_sql})'))
        for forbidden in ('RESP','VALUE_B','VALUE_P','QUALIFY','CALL ','CREATE TABLE','RND'):
            self.assertNotIn(forbidden,sql)
        for expected in ("DATEADD(MONTH,-12,R.AS_OF_DATE)","DATEADD(MONTH,-4,R.AS_OF_DATE)",
                         "DATEADD(MONTH,-1,R.AS_OF_DATE)","UPPER(CURR_IND)='Y'",
                         "COUNT(DISTINCT ENTITY_KEY)","('FINAL','STANDALONE')",
                         'PRIMARY_SPECIALTY','BILL_TYPE_CODE','RX_CHANNEL_','CLAIMS_NRX'):
            self.assertIn(expected,sql)
        self.assertEqual(sql.count('{request_sql}'),1)

    def test_source_rx_paid_population_and_all_medical_dx_counts(self):
        sql=raw.raw_sql_template()
        self.assertIn('DX_BASE AS (SELECT * FROM MED)',sql)
        self.assertIn("SELECT * FROM RX_PHARM WHERE NULLIF(NDC11,'') IS NOT NULL",sql)
        rx_tokens=[line for line in sql.splitlines()
                   if ' AS CATEGORY,' in line and (' FROM RX_BASE WHERE ' in line or ' FROM RX_MAPPED WHERE ' in line)]
        self.assertTrue(rx_tokens)
        for line in rx_tokens:
            if "'CLAIMS_RX_'||TRANSACTION_RESULT AS CATEGORY" in line:
                self.assertTrue(line.endswith('WHERE TRUE'))
            else:
                self.assertIn("UPPER(TRANSACTION_RESULT)='PAID'",line)

    def test_train_vocabulary_actual_history_missing_mask_and_recent_pct(self):
        calls=[]
        def read(sql):
            calls.append(sql)
            self.assertNotIn('RESP',sql)
            return self.long
        out=raw.build_raw_automl_inputs(read,self.meta)
        self.assertEqual(len(calls),1)
        self.assertEqual(out['receipt']['original_train_candidate_features'],5)
        self.assertFalse(any(s.get('raw_category')=='HELDOUT_ONLY' for s in out['feature_specs']))
        panel=out['cached_panel']
        specs=out['feature_specs']
        count=next(s['name'] for s in specs if s.get('raw_category')=='KNOWN' and s.get('measure')=='count')
        pct=next(s['name'] for s in specs if s.get('raw_category')=='KNOWN' and s.get('measure')=='recent_pct')
        self.assertTrue(panel.loc[panel.IS_OBSERVED,count].eq(10).all())
        self.assertTrue(panel.loc[panel.IS_OBSERVED,pct].eq(40).all())
        self.assertTrue(panel.loc[~panel.IS_OBSERVED,[s['name'] for s in specs]].isna().all().all())
        bundle=inputs.assemble_inputs(self.meta,panel,specs,date_policy='month_end_history')
        self.assertEqual(bundle['X_sequence'].shape,(2,12,5))
        self.assertFalse(bundle['sequence_mask'][:,-1].any())
        self.assertEqual(out['contract']['fit_scope'],'deterministic_raw')
        self.assertFalse(out['receipt']['source_availability_verified'])

    def test_resource_cap_is_audited_and_never_uses_labels(self):
        out=raw.build_raw_automl_inputs(lambda sql:self.long,self.meta,{'max_features':3})
        other=self.meta.copy();other['RESP']=1-other.RESP
        out2=raw.build_raw_automl_inputs(lambda sql:self.long,other,{'max_features':3})
        self.assertEqual(out['feature_specs'],out2['feature_specs'])
        self.assertEqual(len(out['feature_specs']),3)
        self.assertEqual(len(out['receipt']['removed_feature_candidates']),2)
        self.assertIn('AML_AGE_YEARS',[s['name'] for s in out['feature_specs']])

    def test_duplicate_missing_or_future_source_rows_fail(self):
        malformed=[pd.concat([self.long,self.long.iloc[:1]],ignore_index=True),self.long.iloc[1:].copy()]
        future=self.long.copy();future.loc[0,'MAX_EVENT_DATE']=pd.Timestamp('2030-01-01');malformed.append(future)
        too_old=self.long.copy();too_old.loc[0,'MAX_EVENT_DATE']=pd.Timestamp('2020-01-01');malformed.append(too_old)
        for frame in malformed:
            with self.subTest(length=len(frame)),self.assertRaises(ValueError):
                raw.build_raw_automl_inputs(lambda sql:frame,self.meta)

    def test_query_identifiers_and_configuration_reject_injection(self):
        with self.assertRaises(ValueError):raw.raw_sql_template({'source_schema':'X.Y;CALL BAD()'})
        with self.assertRaises(ValueError):raw.raw_sql_template({'max_features':0})


if __name__=='__main__':unittest.main()
