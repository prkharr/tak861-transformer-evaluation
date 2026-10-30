"""Execute the generated relational logic on synthetic sources in a local SQL engine."""
import unittest

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

from experiments.fresh_cohort import cohort_binding as binding


class CohortSQLTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(candidate='DB.SC.CANDIDATE',basket='DB.SC.BASKET',medical='DB.SC.MEDICAL',months=48)
        self.candidate = pd.DataFrame([
            (p, '2026-03-31', 0) for p in ('day0','day1','day90','day91','prior','unobserved','positive','unknown')]
            + [('lower','2022-03-31',0),('inside','2022-04-01',0),('future','2026-04-01',0)]
            + [('overlap',d,0) for d in ('2026-01-31','2026-02-28','2026-03-31')],
            columns=['PATIENT_ID','END_DT','RESP'])
        self.candidate.END_DT = pd.to_datetime(self.candidate.END_DT)
        self.basket = pd.DataFrame([
            ('day0','2026-03-31',False),('day1','2026-04-01',False),('day90','2026-06-29',False),
            ('day91','2026-06-30',False),('prior','2019-01-01',False),('prior','2026-04-01',False),
            ('positive','2026-04-15',False),('overlap','2026-04-01',False),('unknown','2026-04-01',None),
            ('horizon','2026-06-29',True)],columns=['PATIENT_ID','FILL_DATE','IS_GENERIC'])
        self.basket.FILL_DATE = pd.to_datetime(self.basket.FILL_DATE)
        self.medical = pd.DataFrame([('day91','2026-04-01'),('inside','2022-04-02'),('lower','2022-04-01'),
            ('unknown','2026-04-01'),('horizon','2026-06-29')],columns=['PATIENT_ID','SERVICE_DATE'])
        self.medical.SERVICE_DATE = pd.to_datetime(self.medical.SERVICE_DATE)
        self.db = duckdb.connect()
        for name in ('candidate','basket','medical'):
            self.db.register(name, getattr(self,name))

    def tearDown(self):
        self.db.close()

    def execute(self, sql):
        tree = sqlglot.parse_one(sql, read='snowflake')
        self.assertIsInstance(tree, exp.Select)
        for table in tree.find_all(exp.Table):
            if table.catalog:
                table.set('catalog',None); table.set('db',None)
        return self.db.execute(tree.sql(dialect='duckdb')).fetchdf()

    def test_exact_first_event_outcomes_retention_and_reused_windows(self):
        query, audit = binding.queries(self.plan)
        d = self.execute(query)
        a = self.execute(audit).iloc[0]
        d = binding.validate_decisions(d, self.candidate, a)
        by = d.set_index('PATIENT_ID')
        self.assertEqual(int(by.loc['day0','RETAIN']),0)
        self.assertEqual(int(by.loc['prior','RETAIN']),0)
        for p in ('day1','day90','positive'):
            self.assertEqual(int(by.loc[p,'RESP']),1)
            self.assertEqual(int(by.loc[p,'RETAIN']),1)
        self.assertEqual(int(by.loc['positive','OBSERVED_NEGATIVE']),0)
        self.assertEqual(int(by.loc['day91','RESP']),0)
        self.assertEqual(int(by.loc['day91','RETAIN']),1)
        self.assertEqual(int(by.loc['unobserved','RETAIN']),0)
        self.assertTrue(by.loc['overlap','RESP'].eq(1).all())
        self.assertEqual(len(by.loc['overlap']),3)
        self.assertEqual(int(a.POSITIVES),6)
        self.assertEqual(int(a.NULL_CLASSIFICATION_ROWS),1)

    def test_48_month_exclusive_lower_inclusive_upper_and_observed_horizon(self):
        query, audit = binding.queries(self.plan)
        d = self.execute(query)
        a = self.execute(audit).iloc[0]
        self.assertEqual(pd.Timestamp(a.UPPER_END),pd.Timestamp('2026-03-31'))
        self.assertEqual(pd.Timestamp(a.LOWER_EXCLUSIVE),pd.Timestamp('2022-03-31'))
        self.assertNotIn('lower',set(d.PATIENT_ID))
        self.assertNotIn('future',set(d.PATIENT_ID))
        self.assertIn('inside',set(d.PATIENT_ID))
        self.assertIn('day1',set(d.PATIENT_ID))
        self.assertIn('MIN(b.FILL_DATE)',query)
        self.assertNotIn('ROW_NUMBER',query.upper())
        self.assertNotIn('LIMIT',query.upper())

    def test_label_boundary_and_unexplained_row_changes_rejected(self):
        query, audit = binding.queries(self.plan)
        d = self.execute(query)
        a = self.execute(audit).iloc[0]
        tampered = d.copy(); tampered.loc[tampered.PATIENT_ID.eq('day0'),'RESP']=1
        with self.assertRaisesRegex(ValueError,'boundary'):
            binding.validate_decisions(tampered,self.candidate,a)
        with self.assertRaisesRegex(ValueError,'snapshot keys'):
            binding.validate_decisions(d.iloc[1:],self.candidate,a)


if __name__=='__main__':
    unittest.main()
