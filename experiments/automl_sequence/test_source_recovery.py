"""Synthetic read-only GET_DDL adapter tests, not warehouse execution."""
import hashlib
import unittest

import pandas as pd

try:
    from . import source_recovery as sr
except ImportError:
    import source_recovery as sr


def ddl(signature, body="return 'evidence only'", qualified=True):
    declaration = signature if qualified else signature.split('.')[-1]
    return "CREATE OR REPLACE PROCEDURE " + declaration + "\nRETURNS VARCHAR LANGUAGE PYTHON AS $$\n" + body + "\n$$;\n"


class SparkDouble:
    def __init__(self, text): self.text = text
    def toPandas(self): return pd.DataFrame({"unusual_connector_column": [self.text]})


class SourceRecoveryTest(unittest.TestCase):
    def test_supplied_signatures_only_readonly_and_stable_source_hashes(self):
        queries = []
        def read(sql):
            queries.append(sql)
            self.assertTrue(sql.startswith("SELECT GET_DDL('PROCEDURE', '"))
            self.assertTrue(sql.endswith("') AS PROCEDURE_DDL"))
            signature = sql.split("'")[3]
            self.assertIn(signature, sr.DEFAULT_PROCEDURES.values())
            self.assertNotIn(';', sql)
            return SparkDouble(ddl(signature))
        result = sr.recover_procedure_sources(read)
        self.assertEqual(len(queries), 3)
        self.assertEqual(set(result['ddl_texts']), set(sr.DEFAULT_PROCEDURES))
        receipt = result['receipt']
        for row in receipt['procedures']:
            text = result['ddl_texts'][row['name']]
            self.assertEqual(row['sha256'], hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(row['utf8_bytes'], len(text.encode()))
            self.assertEqual(row['lines'], len(text.splitlines()))
        self.assertEqual(receipt['completeness'], 'not_certified')
        self.assertFalse(receipt['approved_for_feature_binding'])
        self.assertFalse(receipt['source_executed'])
        other = sr.recover_procedure_sources(read, dict(reversed(list(sr.DEFAULT_PROCEDURES.items()))))
        self.assertEqual(receipt['aggregate_sha256'], other['receipt']['aggregate_sha256'])

    def test_explicit_omissions_flagged_without_source_in_receipt(self):
        signature = sr.DEFAULT_PROCEDURES['FEATURE_CREATION_PROC']
        text = ddl(signature, "# [... DX/PX/RX SQL omitted ...]\n# [remaining code truncated]\nreturn 'not executed'", qualified=False)
        result = sr.recover_procedure_sources(lambda sql: pd.DataFrame({'DDL': [text]}), {'FEATURE_CREATION_PROC': signature})
        row = result['receipt']['procedures'][0]
        self.assertTrue(row['explicit_omission_marker_detected'])
        self.assertEqual(row['omission_marker_count'], 2)
        self.assertEqual(row['omission_marker_lines'], [3, 4])
        self.assertNotIn('DX/PX/RX', str(result['receipt']))

    def test_empty_null_nonscalar_or_wrong_procedure_fails(self):
        signature = sr.DEFAULT_PROCEDURES['FEATURE_CREATION_PROC']
        bad = [pd.DataFrame({'DDL': []}), pd.DataFrame({'DDL': [None]}),
               pd.DataFrame({'DDL': [' ']}), pd.DataFrame({'DDL': [42]}),
               pd.DataFrame({'A': ['x'], 'B': ['y']}), pd.DataFrame({'DDL': ['x', 'y']}),
               pd.DataFrame({'DDL': ['not SQL']}),
               pd.DataFrame({'DDL': [ddl('OTHER.SCHEMA.WRONG(VARCHAR)')]})]
        for frame in bad:
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                sr.recover_procedure_sources(lambda sql: frame, {'FEATURE_CREATION_PROC': signature})

    def test_hash_match_then_mutation_fails(self):
        signature = sr.DEFAULT_PROCEDURES['FEATURE_CREATION_PROC']
        text = ddl(signature)
        expected = {'FEATURE_CREATION_PROC': hashlib.sha256(text.encode()).hexdigest()}
        result = sr.recover_procedure_sources(lambda sql: SparkDouble(text), {'FEATURE_CREATION_PROC': signature}, expected)
        self.assertTrue(result['receipt']['procedures'][0]['expected_sha256_verified'])
        with self.assertRaisesRegex(ValueError, 'SHA-256 changed'):
            sr.recover_procedure_sources(lambda sql: SparkDouble(text + '\n'), {'FEATURE_CREATION_PROC': signature}, expected)

    def test_bad_requests_and_hashes_rejected_before_any_query(self):
        called = []
        invalid = [({'FEATURE_CREATION_PROC': "A.B.FEATURE_CREATION_PROC(VARCHAR);CALL X()"}, None),
                   ({'FEATURE_CREATION_PROC': "A.B.FEATURE_CREATION_PROC('VARCHAR')"}, None),
                   ({'DIFFERENT': sr.DEFAULT_PROCEDURES['FEATURE_CREATION_PROC']}, None),
                   ({}, None),
                   (sr.DEFAULT_PROCEDURES, {'FEATURE_CREATION_PROC': 'bad'}),
                   (sr.DEFAULT_PROCEDURES, {'NOT_REQUESTED': 'a'*64})]
        for procedures, hashes in invalid:
            with self.subTest(procedures=procedures), self.assertRaises(ValueError):
                sr.recover_procedure_sources(lambda sql: called.append(sql), procedures, hashes)
        self.assertEqual(called, [])

    def test_connector_failure_does_not_expose_private_error(self):
        def fail(sql): raise RuntimeError('private connection detail should not be echoed')
        with self.assertRaises(RuntimeError) as cm:
            sr.recover_procedure_sources(fail)
        self.assertNotIn('private connection', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
