"""Memory-path regression tests; synthetic inputs only."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from experiments.automl_sequence import modeling as m


def fixture():
    rng = np.random.default_rng(9341)
    n, width = 72, 17
    x = rng.normal(size=(n, 12, width)).astype(np.float32)
    x[:, :, 0] = rng.poisson(3, (n, 12))
    x[:, :, 1] = rng.integers(0, 2, (n, 12))
    x[2, 3, 2] = np.nan
    valid = rng.random((n, 12)) > .1
    valid[:, 0] = True
    x[~valid] = np.nan
    return x, valid, np.tile([0, 1], n//2), np.arange(n).astype(str), [f'f{j}' for j in range(width)], ['count', 'binary']+['continuous']*(width-2)


class ModelingMemoryTests(unittest.TestCase):
    def test_feature_major_disk_statistics_match_original_order_exactly(self):
        x, valid, y, groups, names, kinds = fixture()
        expected = m._fit_transform(x[:, :1], x, valid, names, kinds)
        with patch.object(m, 'MEMORY_MAP_THRESHOLD', 0), patch.object(m, 'MEMORY_ROW_BLOCK', 7), patch.object(m, 'MEMORY_FEATURE_BLOCK', 3):
            actual = m._fit_transform(x[:, :1], x, valid, names, kinds)
        self.assertEqual(actual, expected)

    def test_release_view_rejected_without_invalidating_owner_subprocess(self):
        # Isolate the mmap lifetime regression: an accidental close can produce
        # an access violation instead of a catchable Python exception.
        script = textwrap.dedent('''
            import os
            import numpy as np
            from experiments.automl_sequence import modeling as m
            m.MEMORY_MAP_THRESHOLD = 0
            owner = m.allocate_array((4, 3, 2))
            sentinel = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
            owner[:] = sentinel
            path = owner.filename
            views = [owner[:, :1], np.asarray(owner), owner.reshape(4, 6), np.asarray(owner)[:, ::2]]
            for view in views:
                try:
                    m.release_array(view)
                except ValueError as exc:
                    assert 'release the owning array' in str(exc)
                else:
                    raise AssertionError('View release must reject cleanup ownership')
                assert os.path.exists(path)
                np.testing.assert_array_equal(owner, sentinel)
                np.testing.assert_array_equal(view, np.asarray(view).copy())
            m.release_array(None)
            ordinary = np.arange(4)
            m.release_array(ordinary)
            np.testing.assert_array_equal(ordinary, np.arange(4))
            views.clear()
            del view
            m.release_array(owner)
            m.release_array(owner)  # Idempotent owner cleanup.
            assert not os.path.exists(path)
            print('owner sentinel remains usable after every rejected view release')
        ''')
        # Discovery runners may change cwd and inject dependency paths only in
        # sys.path. Give the isolated interpreter the same import environment.
        environment = dict(os.environ)
        environment['PYTHONPATH'] = os.pathsep.join(
            str(Path(path).resolve()) for path in sys.path if path)
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=30, env=environment,
                                cwd=Path(__file__).resolve().parents[2])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('owner sentinel remains usable', result.stdout)

    def test_featurewise_state_matches_frozen_2106cf1_preparation(self):
        # Derived from the untouched HEAD module using the same fixture/runtime.
        # Wall-clock metadata is frozen to compare the complete hashed preparation.
        x, valid, y, groups, names, kinds = fixture()
        with patch('time.perf_counter', return_value=100.):
            prep = m.prepare_features(x[:, :1], x, y, groups, names,
                                      valid_train=valid, feature_kinds=kinds)
        self.assertEqual(prep['preparation_sha256'],
                         '02d43ed3db03db61ef06333b5d4044e978cf984f617be88cfbfd7cacf39da0b6')

    def test_forced_disk_transform_proxy_and_split_have_identical_values(self):
        x, valid, y, groups, names, kinds = fixture()
        state = m._fit_transform(x[:, :1], x, valid, names, kinds)
        prep = dict(transform=state, selected_indices=[3, 0, 16, 1])
        dense = m.transform_inputs(x, prep, valid)
        rows = np.arange(0, len(x), 2)
        with patch.object(m, 'MEMORY_MAP_THRESHOLD', 0), patch.object(m, 'MEMORY_ROW_BLOCK', 7), patch.object(m, 'MEMORY_FEATURE_BLOCK', 3):
            out = m.transform_inputs(x, prep, valid[rows], row_indices=rows)
            self.assertIsInstance(out, np.memmap)
            path = out.filename
            np.testing.assert_array_equal(out, dense[rows])
            proxy = m._proxy(out[:, :1], out, valid[rows])
            expected = m._proxy(dense[rows, :1], dense[rows], valid[rows])
            np.testing.assert_array_equal(proxy, expected)
            m.release_array(proxy); m.release_array(expected); m.release_array(out)
            self.assertFalse(os.path.exists(path))

    def test_disk_correlation_uses_all_eligible_rows(self):
        rng = np.random.default_rng(81)
        summary = rng.normal(size=(100, 1, 19)).astype(np.float32)
        summary[:, :, 3] = 1
        eligible = [0, 1, 3, 5, 9, 14, 18]
        with patch.object(m, 'MEMORY_MAP_THRESHOLD', 0):
            active, corr = m._correlation(summary, eligible)
        matrix = summary[:, 0, eligible].astype(np.float64)
        expected_active = np.flatnonzero(np.ptp(matrix, axis=0) > 0)
        np.testing.assert_array_equal(active, expected_active)
        np.testing.assert_array_equal(corr, np.corrcoef(matrix[:, active], rowvar=False))

    def test_preparation_split_gather_preserves_full_result(self):
        x, valid, y, groups, names, kinds = fixture()
        rows = np.arange(0, len(x))
        with patch('time.perf_counter', return_value=100.):
            ordinary = m.prepare_features(x[:, :1], x, y, groups, names, valid_train=valid, feature_kinds=kinds)
            with patch.object(m, 'MEMORY_MAP_THRESHOLD', 0):
                mapped = m.prepare_features(x[:, :1], x, y, groups, names, valid_train=valid,
                                            feature_kinds=kinds, row_indices=rows)
        self.assertEqual(mapped, ordinary)


# The production projection may omit a source feature, but source semantics
# still apply to every column when validating a split.
class UnselectedContractTests(unittest.TestCase):
    def test_unselected_count_is_still_validated(self):
        x, valid, y, groups, names, kinds = fixture()
        state=m._fit_transform(x[:, :1], x, valid, names, kinds)
        x[0,0,0]=-1
        with self.assertRaisesRegex(ValueError,'count semantics'):
            m.transform_inputs(x, dict(transform=state,selected_indices=[2]), valid)

    def test_split_validation_reads_selected_rows_and_keeps_full_input_audit(self):
        x, valid, y, groups, names, kinds = fixture()
        state = m._fit_transform(x[:, :1], x, valid, names, kinds)
        prep = dict(transform=state, selected_indices=[2])
        rows = np.arange(0, len(x), 2)
        expected = m.transform_inputs(x[rows], prep, valid[rows])
        x[1, 0, 2] = np.inf  # An unused split must not be reread per fold.
        actual = m.transform_inputs(x, prep, valid, row_indices=rows)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, 'real and finite'):
            m._array(x, valid, allow_empty=True)  # Full-source audit still stops.
        x[0, 0, 0] = -1  # Unselected source feature inside this split still stops.
        with self.assertRaisesRegex(ValueError, 'count semantics'):
            m.transform_inputs(x, prep, valid, row_indices=rows)


if __name__ == '__main__':
    unittest.main()
