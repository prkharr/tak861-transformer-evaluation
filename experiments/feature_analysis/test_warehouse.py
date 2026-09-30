"""Synthetic contract checks; no warehouse credentials or patient data required."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location('feature_warehouse', Path(__file__).with_name('warehouse.py'))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


def population():
    split = pd.DataFrame([
        ('p1', '2024-01-31', 0, 'train'),
        ('p1', '2024-02-29', 1, 'train'),
        ('p2', '2024-02-29', 0, 'train'),
        ('p3', '2024-02-29', 1, 'validation'),
        ('p4', '2024-02-29', 0, 'test'),
    ], columns=['PATIENT_ID', 'END_DT', 'RESP', 'SPLIT'])
    return split.drop(columns='SPLIT'), split


def train_rows():
    return w.metadata_contract(*population())[1]


def inventory():
    types = {'PID': 'VARCHAR', 'ASOF': 'DATE', 'AVAILABLE': 'TIMESTAMP_NTZ',
             'VALUE': 'FLOAT', 'CATEGORY': 'VARCHAR', 'EVENT_ID': 'VARCHAR', 'STATUS': 'VARCHAR'}
    return pd.DataFrame([
        {'TABLE_CATALOG': 'DB', 'TABLE_SCHEMA': 'SC', 'TABLE_NAME': table,
         'COLUMN_NAME': col, 'DATA_TYPE': dtype}
        for table in ['RAW_SNAPSHOT', 'EVENTS'] for col, dtype in types.items()
    ])


def extra_contract(kind='snapshot_numeric'):
    common = dict(table='DB.SC.RAW_SNAPSHOT', kind=kind, patient_column='PID',
                  date_column='ASOF', availability_column='AVAILABLE',
                  review_note='Reviewed raw values, exact snapshot grain and timestamp availability.',
                  uses_resp_encoding=False, available_at_cutoff_verified=True)
    if kind == 'snapshot_numeric':
        common['columns'] = ['VALUE']
    else:
        common.update(table='DB.SC.EVENTS', category_column='CATEGORY',
                      event_id_columns=['EVENT_ID'], no_event_means_zero_verified=True,
                      equals_filters={'STATUS': ['paid']})
    return common


class WarehouseContractTests(unittest.TestCase):
    def test_latest_train_snapshot_and_frozen_population(self):
        snapshots, split = population()
        frozen, train = w.metadata_contract(snapshots.sample(frac=1, random_state=2), split,
                                            expected=(5, 4, 2))
        self.assertEqual(len(frozen), 5)
        self.assertEqual(train.PATIENT_ID.tolist(), ['p1', 'p2'])
        self.assertEqual(train.END_DT.tolist(), ['2024-02-29'] * 2)
        self.assertEqual(train.RESP.tolist(), [1, 0])

    def test_metadata_rejects_leakage_duplicate_null_and_label_mismatch(self):
        snapshots, split = population()
        bad_split = split.copy()
        bad_split.loc[0, 'SPLIT'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'Patient split leakage'):
            w.metadata_contract(snapshots, bad_split)
        with self.assertRaisesRegex(ValueError, 'Duplicate snapshot'):
            w.metadata_contract(pd.concat([snapshots, snapshots.iloc[:1]]), split)
        bad = snapshots.copy()
        bad.loc[0, 'RESP'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Null/empty'):
            w.metadata_contract(bad, split)
        bad = snapshots.copy()
        bad.loc[0, 'RESP'] = 1
        with self.assertRaisesRegex(ValueError, 'labels do not match'):
            w.metadata_contract(bad, split)

    def test_metadata_rejects_fractional_labels_and_timestamp_cutoffs(self):
        snapshots, split = population()
        snapshots['RESP'] = snapshots.RESP.astype(float)
        snapshots.loc[0, 'RESP'] = 0.5
        with self.assertRaisesRegex(ValueError, 'binary'):
            w.metadata_contract(snapshots, split)
        snapshots, split = population()
        snapshots.loc[0, 'END_DT'] = '2024-01-31 12:00:00'
        with self.assertRaises(ValueError):
            w.metadata_contract(snapshots, split)

    def test_feature_map_recovers_order_without_inventing_categories(self):
        mapping = pd.DataFrame({'FEATURE_INDEX': [1, 0, 2],
                                'FEATURE_NAME': ['DX__B', 'RX__A', 'PX__C'],
                                'FEATURE_COLUMN': ['F0001', 'F0000', 'F0002']})
        names, aliases = w.feature_map_contract(mapping, {'RX': 1, 'DX': 1, 'PX': 1})
        self.assertEqual(names, ['RX__A', 'DX__B', 'PX__C'])
        self.assertEqual(aliases, ['F0000', 'F0001', 'F0002'])
        mapping.loc[0, 'FEATURE_COLUMN'] = 'F9999'
        with self.assertRaisesRegex(ValueError, 'aliases/order'):
            w.feature_map_contract(mapping)

    def claims(self):
        result = train_rows().drop(columns='SPLIT').copy()
        for column, value in {'N_ROWS': 12, 'N_STEPS': 12, 'MIN_STEP': 0,
                              'MAX_STEP': 11, 'BAD_STEPS': 0, 'BAD_LABELS': 0,
                              'BAD_VALUES': 0}.items():
            result[column] = value
        result['F0000_M12'] = [9.0, 0.0]
        return result

    def test_claim_values_and_lineage(self):
        X, lineage = w.assemble_claim_features(self.claims().iloc[::-1], train_rows(),
                                              ['RX__A'], ['F0000'], 'DB.SC.COUNTS')
        self.assertEqual(X['RX__A__SUM_M12'].tolist(), [9.0, 0.0])
        self.assertEqual(lineage.SOURCE_COLUMN.tolist(), ['F0000'])
        self.assertIn('NOT_RECOVERED', lineage.LINEAGE_STATUS.iloc[0])

    def test_claims_reject_bad_months_labels_and_counts(self):
        for column, value in [('N_ROWS', 13), ('N_STEPS', 11), ('BAD_STEPS', 1),
                              ('BAD_LABELS', 1), ('BAD_VALUES', 1),
                              ('F0000_M12', -1), ('F0000_M12', np.nan),
                              ('F0000_M12', np.inf)]:
            with self.subTest(column=column, value=value):
                bad = self.claims()
                bad.loc[0, column] = value
                with self.assertRaises(ValueError):
                    w.assemble_claim_features(bad, train_rows(), ['RX__A'], ['F0000'], 'DB.SC.COUNTS')

    def test_claim_query_train_only_and_month_checks(self):
        sql = w.claim_query('DB.SC.COUNTS', 'DB.SC.SPLIT', ['F0000'], (3, 12))
        self.assertIn("WHERE SPLIT = 'train'", sql)
        self.assertIn('PARTITION BY PATIENT_ID ORDER BY END_DT DESC', sql)
        self.assertIn('COUNT(DISTINCT t.TIME_STEP)', sql)
        self.assertIn('t.TIME_STEP < 3', sql)
        self.assertIn('t.TIME_STEP < 12', sql)
        self.assertIn('t."F0000" IS NULL OR t."F0000" < 0', sql)
        for windows in [(0,), (13,), (12, 12)]:
            with self.assertRaises(ValueError):
                w.claim_query('DB.SC.COUNTS', 'DB.SC.SPLIT', ['F0000'], windows)

    def test_extras_require_explicit_lineage_and_availability(self):
        w.validate_extra_contract(extra_contract(), inventory())
        for update in [dict(uses_resp_encoding=True), dict(available_at_cutoff_verified=False),
                       dict(availability_column=None), dict(columns=['CATEGORY']),
                       dict(columns=['RESP']), dict(table='DB.SC.SHAP_VALUES')]:
            with self.subTest(update=update):
                c = extra_contract()
                c.update(update)
                with self.assertRaises(ValueError):
                    w.validate_extra_contract(c, inventory())
        c = extra_contract()
        c.update(availability_column=None, historical_snapshot_certified=True)
        w.validate_extra_contract(c, inventory())

    def test_snapshot_missing_is_retained_and_duplicate_grain_fails(self):
        frame = train_rows()[['PATIENT_ID', 'END_DT']].copy()
        frame['SOURCE_MATCH'] = ['p1', None]
        frame['V0'] = [0.0, np.nan]
        X, lineage, audit = w.gather_extras(lambda sql: frame.copy(), [extra_contract()],
                                           inventory(), train_rows(), 'DB.SC.SPLIT')
        self.assertEqual(X.iloc[0, 0], 0.0)
        self.assertTrue(np.isnan(X.iloc[1, 0]))
        self.assertEqual(audit.MATCHED_PATIENTS.tolist(), [1])
        self.assertIn('retain NaN', lineage.MISSING_RULE.iloc[0])
        doubled = pd.concat([frame, frame.iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'multiplied snapshot rows'):
            w.gather_extras(lambda sql: doubled, [extra_contract()], inventory(), train_rows(), 'DB.SC.SPLIT')

    def test_extra_sql_cutoffs_and_event_dedup(self):
        c = extra_contract('event_categories')
        sql = w.event_extra_query(c, 'DB.SC.SPLIT')
        self.assertIn("WHERE SPLIT = 'train'", sql)
        self.assertIn('SELECT DISTINCT', sql)
        self.assertIn('s."EVENT_ID" AS "E0"', sql)
        self.assertIn('s."ASOF" < DATEADD(day, 1, k.END_DT)', sql)
        self.assertIn('s."AVAILABLE" < DATEADD(day, 1, k.END_DT)', sql)
        self.assertIn("DATEADD(month, -11, DATE_TRUNC('month', k.END_DT))", sql)
        self.assertIn("CAST(s.\"STATUS\" AS VARCHAR) IN ('paid')", sql)
        snapshot_sql = w.snapshot_extra_query(extra_contract(), 'DB.SC.SPLIT')
        self.assertIn('s."ASOF" = k.END_DT', snapshot_sql)
        self.assertIn('s."AVAILABLE" < DATEADD(day, 1, k.END_DT)', snapshot_sql)

    def test_snapshot_filters_cannot_be_silently_ignored(self):
        c = extra_contract()
        c['equals_filters'] = {'STATUS': ['paid']}
        try:
            w.validate_extra_contract(c, inventory())
        except ValueError:
            # Explicitly rejecting unsupported snapshot filters is also safe.
            return
        sql = w.snapshot_extra_query(c, 'DB.SC.SPLIT')
        self.assertIn('s."STATUS"', sql)
        self.assertIn("'paid'", sql)

    def events(self):
        return pd.DataFrame([
            ('p1', '2024-02-29', 'RX_A', 3, 0),
            ('p1', '2024-02-29', 'DX_B', 1, 0),
        ], columns=['PATIENT_ID', 'END_DT', 'CATEGORY', 'VALUE', 'BAD_KEYS'])

    def test_event_categories_preserve_counts_and_certified_zero(self):
        X, lineage, audit = w.gather_extras(lambda sql: self.events(), [extra_contract('event_categories')],
                                           inventory(), train_rows(), 'DB.SC.SPLIT')
        by_category = dict(zip(lineage.BASE_FEATURE, lineage.FEATURE))
        self.assertEqual(X[by_category['RX_A']].tolist(), [3.0, 0.0])
        self.assertEqual(X[by_category['DX_B']].tolist(), [1.0, 0.0])
        self.assertEqual(audit.MATCHED_PATIENTS.tolist(), [1])

    def test_event_contract_rejects_unknown_keys_null_identity_and_category_overflow(self):
        for column, value in [('PATIENT_ID', 'test_patient'), ('BAD_KEYS', 1), ('CATEGORY', None)]:
            with self.subTest(column=column):
                bad = self.events()
                bad.loc[0, column] = value
                with self.assertRaises(ValueError):
                    w.gather_extras(lambda sql: bad, [extra_contract('event_categories')],
                                    inventory(), train_rows(), 'DB.SC.SPLIT')
        with self.assertRaisesRegex(ValueError, 'vocabulary exceeds'):
            w.gather_extras(lambda sql: self.events(), [extra_contract('event_categories')],
                            inventory(), train_rows(), 'DB.SC.SPLIT', max_event_categories=1)

    def test_event_counts_reject_negative_missing_fractional_and_nonfinite_values(self):
        for value in [-1.0, np.nan, np.inf, 1.5]:
            with self.subTest(value=value):
                bad = self.events()
                bad['VALUE'] = bad.VALUE.astype(float)
                bad.loc[0, 'VALUE'] = value
                with self.assertRaises(ValueError):
                    w.gather_extras(lambda sql: bad, [extra_contract('event_categories')],
                                    inventory(), train_rows(), 'DB.SC.SPLIT')

    def test_duplicate_source_contract_does_not_duplicate_candidate(self):
        frame = train_rows()[['PATIENT_ID', 'END_DT']].copy()
        frame['SOURCE_MATCH'], frame['V0'] = ['p1', 'p2'], [1.0, 2.0]
        with self.assertRaisesRegex(ValueError, 'Duplicate candidate identity'):
            w.gather_extras(lambda sql: frame, [extra_contract(), copy.deepcopy(extra_contract())],
                            inventory(), train_rows(), 'DB.SC.SPLIT')

    def test_identifiers_are_quoted_and_fqn_rejects_injection(self):
        self.assertEqual(w.qi('odd"name'), '"odd""name"')
        self.assertEqual(w.literal("O'Brien"), "'O''Brien'")
        with self.assertRaises(ValueError):
            w.qtable('DB.SC.T;DROP TABLE T')


if __name__ == '__main__':
    unittest.main(verbosity=2)
