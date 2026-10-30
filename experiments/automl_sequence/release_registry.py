"""Pinned identities of published notebook releases, read back from Git blobs.

The published notebooks embed the module sources of a specific commit. Later
edits to the shared modules must not make the published delivery look stale or
silently rebuild it, so release tests compare the embedded sources against the
pinned commit's blobs rather than the working tree. Nothing here is skipped
when Git is unavailable: a missing blob is a test failure.
"""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

EMBEDDED_PATHS = {
    **{name: "experiments/automl_sequence/" + name + ".py"
       for name in ("inputs", "count_loader", "modeling", "workflow", "reliance",
                    "source_recovery", "raw_automl", "fresh_counts", "client_universe")},
    "comparison": "experiments/rigorous_transformer/comparison.py",
    "warehouse": "experiments/temporal_selection/input_helpers.py",
}

RELEASES = {
    "R20261023_OLD_B": {
        "commit": "3f0116f06935cc7230913a0d39c99983d05ca9b1",
        "notebook_folder": "notebooks/AutoML Sequential Features",
        "builder": "experiments/automl_sequence/build_notebooks.py",
        "manifest": "experiments/automl_sequence/build_manifest.json",
        "implementation_sha256": "b0bfd8af64692fc623232116197fd1d64848c7529088eacd4343f5be0c41ed83",
        "source_sha256": "ed97c874b2040d9c91d2de7d6cf844b90d8e4f74b0520f46e83878ba5b3d3cb9",
        "reporting_sha256": "9cbf44dafe8cbf9a6c12f57f21887a94e790c7613e4c12178378ac2e214f9548",
        "existing_score_loader_sha256": "9e072d9bfd63dd7aa2898d25556309f4cba5d801f93df68bba0dcc7c5aa8dab4",
        "notebooks": {
            "01_tensor_initialization_AutoML.ipynb": "13ddf053934e2158424a93efcc3abf78bcc224fdbfac5b40a52bef2dc68711ae",
            "02_patient_level_split_AutoML.ipynb": "5cd987c37ddcddf31337bbaf4132cbe9353469f8cb480b269cb4dec72af9b6d4",
            "03_transformer_training_AutoML.ipynb": "d58364ffdb6eeb4d98a70956ea5e0fe137fbc4e1cf594956749dd7660013409f",
            "04_transformer_evaluation_AutoML.ipynb": "e8bd00e626ba99bb5b7621c54e98ab4cb14eca69d98af6e40aff8ddbe4401f95",
            "05_model_comparison_AutoML.ipynb": "f8fc264168bfc5cc2eed8417d0297e8bee8b6f003a72f43bd299a58f221d37ab",
        },
    },
}

CURRENT_RELEASE = "R20261023_OLD_B"


def published_bytes(commit, path):
    """Exact blob bytes at the pinned commit; raises if Git or the blob is unavailable."""
    result = subprocess.run(["git", "-c", "safe.directory=" + REPO.as_posix(), "show", commit + ":" + path],
                            cwd=REPO, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise AssertionError("Pinned release blob unavailable: " + commit[:12] + ":" + path)
    return result.stdout


def published_text(commit, path):
    return published_bytes(commit, path).decode("utf-8-sig")


def published_sources(release):
    """The embedded-source dictionary as the pinned builder would have read it."""
    return {name: published_text(release["commit"], path) for name, path in EMBEDDED_PATHS.items()}


def pinned_notebook_source(release):
    """notebook_source() exactly as defined in the pinned builder blob."""
    text = published_text(release["commit"], release["builder"])
    prelude = text.split("\ndef build(", 1)[0]
    namespace = {"__file__": str(REPO / release["builder"]), "__name__": "pinned_builder"}
    exec(compile(prelude, "pinned_builder", "exec"), namespace)
    if "notebook_source" not in namespace:
        raise AssertionError("Pinned builder at " + release["commit"][:12] + " does not define notebook_source")
    return namespace["notebook_source"], text


def embedded_assignments(helper_source):
    found = {}
    for node in ast.parse(helper_source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in {"EMBEDDED_SOURCES", "EMBEDDED_SOURCE_SHA256", "IMPLEMENTATION_SHA256"}:
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found


def verify_release_identity(release):
    """Recompute source and implementation digests from pinned blobs; return the stripped sources."""
    strip, builder_text = pinned_notebook_source(release)
    stripped = {name: strip(source) for name, source in published_sources(release).items()}
    source_sha256 = hashlib.sha256(json.dumps(stripped, sort_keys=True).encode()).hexdigest()
    implementation = hashlib.sha256((source_sha256 + builder_text).encode()).hexdigest()
    if source_sha256 != release["source_sha256"]:
        raise AssertionError("Pinned blobs do not reproduce the published source_sha256")
    if implementation != release["implementation_sha256"]:
        raise AssertionError("Pinned blobs do not reproduce the published implementation_sha256")
    return stripped, builder_text


RELEASES["R20261030_NEW_SUMMARY_CORE"] = {'commit': 'd3eac3a4a986d9c3d3234282dc7699bbdeb9f35a', 'notebook_folder': 'notebooks/Fresh Cohort AutoML NLP', 'manifest': 'experiments/fresh_cohort/core_release_manifest.json', 'implementation_sha256': '9eb0620c84a7363ef0d53269a75eef203a4ce9737c2e944abc68450589b5e17a', 'source_sha256': 'bb11202d8b793ce0c0a439f3639c57b3c896aaa02dff11ca315e0c0d47ce55fb', 'notebooks': {'01_tensor_initialization_AutoML.ipynb': 'a888b54be6f87841440beb7fc2fd69f75779b0ae2854db5273a315caac58d407', '02_patient_level_split_AutoML.ipynb': '8cfc9de2f0a15bf03ca8871a9ddfc1dc7888c76346b91944373a7216c1415303', '03_transformer_training_AutoML.ipynb': '1b8d536e51c568defefdb280b08b4003a5059a182c92125de7a2f3678e774344', '04_transformer_evaluation_AutoML.ipynb': 'ebab517835256e911f391f202d5e167cc14486014b4e3f7e8dd0afda927055b7'}, 'stage': '01_04', 'local_build_notebooks': {'01_tensor_initialization_AutoML.ipynb': '488a9b797222e0de018d84d8bc3978a767cf62a2394d157f0399dab0650732f9', '02_patient_level_split_AutoML.ipynb': '83827062f503b460d35f3128f144cb28cd5980efeb183abfa911c4f8369b2dcc', '03_transformer_training_AutoML.ipynb': '262d039e00bf079770ca24d36896bcca53d966c1f9f5a31e56e256e96f39b954', '04_transformer_evaluation_AutoML.ipynb': 'ab661b3c579665112043a71f20aa2eab058a4d3e05ecabd917e0f89c9a6ee2dc'}, 'serialization': 'Published LF blobs; local Windows build CRLF bytes recorded separately'}
