"""Reproduce local synthetic verification; never connects to source systems."""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest

from experiments.fresh_cohort import build_notebooks


RUNTIME_MODULES = (
    "test_artifact_streaming", "test_client_universe", "test_count_loader",
    "test_extraction_memory", "test_fresh_lgbm", "test_inputs", "test_lift_reporting",
    "test_modeling", "test_modeling_memory", "test_raw_automl", "test_raw_resume",
    "test_reliance", "test_source_recovery", "test_workflow", "test_workflow_memory",
)


def main():
    here = Path(__file__).resolve().parent
    sql_dependencies = build_notebooks.base.REPO.parent/'work'/'sql_audit'
    if sql_dependencies.is_dir():
        sys.path.insert(0, str(sql_dependencies))
    plot_dependencies = build_notebooks.base.REPO.parent/'work'/'feature_analysis_runtime'
    if plot_dependencies.is_dir():
        sys.path.append(str(plot_dependencies))
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['experiments.automl_sequence.'+p.stem
        for p in sorted(build_notebooks.base.HERE.glob('test_*.py'))])
    suite.addTests(loader.loadTestsFromNames(['experiments.fresh_cohort.'+p.stem for p in sorted(here.glob('test_*.py'))]))
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    manifest = json.loads((here/"build_manifest.json").read_text())
    manifest['status'] = 'SYNTHETICALLY_VERIFIED_RUNTIME_AUDIT_REQUIRED' if result.wasSuccessful() else 'LOCAL_VALIDATION_FAILED'
    (here/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    versions = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "lightgbm", "sqlglot", "duckdb", "matplotlib"):
        module = importlib.import_module(name)
        versions[name] = {"version": module.__version__, "path": module.__file__}
    source_paths = list(here.glob("*.py")) + [build_notebooks.base.HERE/(name+".py") for name in
        ("raw_automl", "modeling", "workflow", "fresh_lgbm", "build_notebooks")]
    record = dict(status="PASS" if result.wasSuccessful() else "FAIL", clinical_execution=False,
        delivery_status="SYNTHETICALLY_VERIFIED_RUNTIME_AUDIT_REQUIRED", tests_run=result.testsRun,
        failures=[{"test": str(t), "traceback": text} for t, text in result.failures],
        errors=[{"test": str(t), "traceback": text} for t, text in result.errors],
        skipped=[{"test": str(t), "reason": reason} for t, reason in result.skipped],
        elapsed_seconds=time.perf_counter()-start, interpreter=sys.executable, python=platform.python_version(),
        packages=versions, python_paths=sys.path,
        command="python -m experiments.fresh_cohort.validate_local",
        implementation_sha256=manifest["implementation_sha256"], notebooks=manifest["notebooks"],
        source_sha256={str(p.relative_to(build_notebooks.base.REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source_paths},
        scope=["Full original delivery and shared runtime discovery", "full fresh_cohort test discovery in the same process",
               "generated five-notebook execution with synthetic source/store doubles",
               "frozen twelve-anchor content/hash regression", "TRAIN-only fitting and validation perturbation",
               "checkpoint tampering, mapping drift and exact partial-month cutoff checks"],
        limitations=["No clinical records queried", "Spark aggregation/warehouse I/O substituted in integration fixture",
                     "Original delivery verified against pinned published source; new delivery against current source",
                     "Synthetic fixture uses reduced training/bootstrap settings; delivered defaults are asserted separately",
                     "No production-size memory or runtime measurement", "Runtime population counts measured and frozen at binding; original eligibility producer and source completeness unverified",
                     "Snowflake SQL parsed and transpiled to DuckDB for synthetic execution; warehouse execution still required"])
    (here/"local_validation.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
