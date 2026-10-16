"""Synthetic revised-cohort contracts, not a live warehouse certificate."""
import unittest
import numpy as np
import pandas as pd
from experiments.automl_sequence import client_universe as c, fresh_counts as f, inputs, modeling


class ClientUniverseTests(unittest.TestCase):
    def setUp(self):
        c.inputs, c.fresh_counts = inputs, f
        self.old = pd.DataFrame(dict(PATIENT_ID=['01','02','03','04','05','06'],
            END_DT=['2026-06-21']*6, RESP=[0,1]*3, SPLIT=['train']*2+['validation']*2+['test']*2))

    def test_added_positive_snapshots_keep_patient_membership(self):
        added = self.old[self.old.RESP.eq(1)].copy(); added.END_DT='2026-07-21'
        new = pd.concat([self.old, added], ignore_index=True)
        result, audit = c.align_universe(new, self.old)
        self.assertEqual(audit['new_positives'],6)
        self.assertEqual(audit['positive_snapshot_ratio'],2.)
        self.assertEqual(audit['added_snapshots'],3)
        self.assertEqual(audit['new_positive_patients'],3)
        self.assertTrue(result.groupby('PATIENT_ID').SPLIT.nunique().eq(1).all())
        self.assertEqual(result.set_index('PATIENT_ID').loc['04','SPLIT'].tolist(),['validation']*2)

    def test_changed_labels_and_duplicate_keys_stop(self):
        new=self.old.copy();new.loc[0,'RESP']=1
        with self.assertRaisesRegex(ValueError,'changes RESP'):c.align_universe(new,self.old)
        with self.assertRaisesRegex(ValueError,'Duplicate'):c.align_universe(pd.concat([self.old,self.old]),self.old)

    def test_new_patients_hash_stable_and_old_ids_never_migrate(self):
        new = pd.concat([self.old,pd.DataFrame([dict(PATIENT_ID='new001',END_DT='2026-07-21',RESP=1)])],ignore_index=True)
        a, receipt = c.align_universe(new,self.old)
        b, _ = c.align_universe(new.iloc[::-1],self.old)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(receipt['new_patients'],1)
        self.assertEqual(a[a.PATIENT_ID.eq('01')].SPLIT.iloc[0],'train')

    def test_exact_table_discovery_and_ambiguity(self):
        rows=pd.DataFrame([['DB','SC',c.TABLE_NAME]],columns=['TABLE_CATALOG','TABLE_SCHEMA','TABLE_NAME'])
        self.assertEqual(c.resolve_universe(lambda q:rows),'DB.SC.'+c.TABLE_NAME)
        for bad in (rows.iloc[:0],pd.concat([rows,rows.assign(TABLE_SCHEMA='OTHER')])):
            with self.assertRaisesRegex(ValueError,'exactly one'):c.resolve_universe(lambda q:bad)
        with self.assertRaisesRegex(ValueError,'requested'):c.resolve_universe(None,'DB.SC.OLD_TABLE')

    def test_removed_or_reformatted_patient_keys_stop(self):
        new=self.old.copy(); new.PATIENT_ID='prefix'+new.PATIENT_ID
        with self.assertRaisesRegex(ValueError,'removes prior'): c.align_universe(new,self.old)
        with self.assertRaisesRegex(ValueError,'removes prior'): c.align_universe(self.old.iloc[1:],self.old)

    def test_cohort_comparison_ignores_string_storage_dtype(self):
        a=self.old.copy(); b=self.old.copy()
        a.PATIENT_ID=a.PATIENT_ID.astype(object)
        b.PATIENT_ID=b.PATIENT_ID.astype('string')
        self.assertTrue(f.same_cohort(a,b))
        b.loc[0,'RESP']=1
        self.assertFalse(f.same_cohort(a,b))

    def test_sparse_dictionary_requires_explicit_reviewed_floor(self):
        mapping=pd.DataFrame({'FEATURE_NAME':['DX__A','PX__B','RX__C','DX__D']})
        observed=mapping.iloc[:3]
        with self.assertRaisesRegex(ValueError,'inactivity'): f.audit_feature_overlap(observed,mapping)
        self.assertEqual(f.audit_feature_overlap(observed,mapping,.30)['missing_feature_fraction'],.25)

    def test_medical_count_coverage_mismatch_fails(self):
        data=dict(count_contract={'feature_names':['DX__A','RX__B']}, X_count=np.ones((2,12,2)),
                  summary_mask=np.array([[True],[False]]),y=np.array([0,1]))
        with self.assertRaisesRegex(ValueError,'Medical count activity'):c.check_activity_alignment(data)
        data['X_count'][1,:,0]=0
        self.assertEqual(c.check_activity_alignment(data)['medical_active_snapshots'],1)

    def test_transform_current_position_is_not_counted_twice(self):
        seq=np.array([[[10.],[0.],[0.]]])
        out=modeling._fit_transform(seq[:,:1],seq,np.ones((1,3),bool),['x'],['continuous'])
        self.assertAlmostEqual(out['center'][0],10/3)

    def test_source_sql_uses_requested_universe_and_event_dates(self):
        contract=dict(sources=dict(f.DEFAULT_SOURCES,cohort='DB.SC.'+c.TABLE_NAME,dx_reference='DB.SC.CCSR'),
                      family_column='C_TYPE',rx_values=['rx'],px_values=['px'],ndc_column='NDC11',policy=dict(f.POLICY))
        contract['contract_sha256']=f._hash(contract)
        sql=f.monthly_count_query(contract)
        self.assertIn('FROM DB.SC.'+c.TABLE_NAME,sql)
        self.assertNotIn('MODEL_DATA',sql)
        self.assertNotIn('RESP',sql)
        self.assertIn("DATEDIFF('MONTH',e.EVENT_DATE,s.END_DT)",sql)
        self.assertIn('e.EVENT_DATE <= s.END_DT',sql)


if __name__=='__main__':unittest.main()
