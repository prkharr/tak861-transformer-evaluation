import contextlib
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sql_runtime = ROOT.parent/'work'/'sql_audit'
sys.path.insert(0,str(sql_runtime))
from experiments.medical_nlp_demo import build_demo

out = ROOT/'experiments'/'medical_nlp_demo'
build_demo.build()
started = time.perf_counter()
with (out/'validation.log').open('w',encoding='utf-8') as stream:
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        suite = unittest.defaultTestLoader.loadTestsFromName('experiments.medical_nlp_demo.test_demo')
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
    elapsed_seconds=round(time.perf_counter()-started,3),python=platform.python_version(),
    interpreter=sys.executable,sql_parser_runtime=str(sql_runtime),
    packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib','sqlglot']},
    notebook_sha256=hashlib.sha256(build_demo.DEST.read_bytes()).hexdigest(),
    private_source_execution=False,ten_minute_runtime_verified=False,
    scope='Synthetic temporal/TF-IDF/frozen-state/export tests; Snowflake SQL parsed locally, not executed')
(out/'local_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
if not result.wasSuccessful():
    print((out/'validation.log').read_text())
    raise SystemExit(1)
