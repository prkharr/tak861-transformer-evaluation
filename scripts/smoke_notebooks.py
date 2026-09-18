"""Execute notebooks 01-03 in fresh kernels using synthetic inputs only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A new local directory for synthetic inputs and executed notebooks.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scratch = args.output_dir.expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(456)
    features = [f"{domain}__SYNTHETIC_{index}" for domain in ("DX", "PX", "RX") for index in range(2)]
    patients = ["NA", "NULL", "N/A"] + [f"SYNTHETIC_NOTEBOOK_{n:04d}" for n in range(3, 80)]
    rows = []
    for patient, patient_id in enumerate(patients):
        for snapshot, end_date in enumerate(("2025-01-31", "2025-02-28")):
            label = int(patient % 2 == 0 and snapshot == 1)
            for step in range(12):
                values = np.zeros(6) if step == 3 else rng.poisson(1, 6)
                rows.append([patient_id, end_date, label, step, *values])
    monthly = pd.DataFrame(rows, columns=["PATIENT_ID", "END_DT", "RESP", "TIME_STEP", *features])
    monthly["PATIENT_ID"] = pd.Categorical(monthly.PATIENT_ID, categories=patients + ["SYNTHETIC_UNUSED"])
    monthly.sample(frac=1, random_state=22).to_parquet(scratch / "monthly.parquet", index=False)

    cfg = json.loads((root / "config.example.json").read_text(encoding="utf-8"))
    cfg.update(input={"mode": "monthly_parquet", "path": "monthly.parquet", "feature_order_json": None},
               bundle_dir="bundle", manifest_path="split/snapshot_manifest.csv", run_dir="run", report_dir="report")
    cfg["expected"] = {"shape": [160, 12, 6], "patients": 80, "resp1": 40,
                       "domain_counts": {"RX": 2, "DX": 2, "PX": 2}}
    for flag in ("feature_order_verified", "snapshot_semantics_verified", "predictor_cutoff_verified", "outcome_events_excluded"):
        cfg["source_review"][flag] = True
    cfg["source_review"]["review_reference"] = "Synthetic software test only; no clinical or upstream validation."
    cfg["model"].update(d_model=8, n_heads=2, encoder_layers=1, feedforward_dim=16, dropout=0.0)
    cfg["training"].update(epochs=2, patience=1, batch_size=16, device="cpu", print_progress=False)
    config_path = scratch / "config.local.json"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.environ["TAK861_CONFIG"] = str(config_path)

    kernel_dir = scratch / "kernels" / "synthetic-smoke"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Synthetic notebook smoke", "language": "python",
        "env": {"TAK861_CONFIG": config_path.as_posix()},
    }), encoding="utf-8")

    paths = sorted((root / "notebooks" / "Transformer Encoder").glob("0[123]*.ipynb"))
    if len(paths) != 3:
        raise RuntimeError("Expected exactly notebooks 01, 02 and 03.")
    # Rerun 01 once to exercise frozen bundle/manifest/audit reuse before training.
    for index, path in enumerate([paths[0], *paths]):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None and not cell.outputs
                compile(cell.source, str(path), "exec")
        nb.cells.insert(0, nbformat.v4.new_code_cell("import torch\ntorch.set_num_threads(1)"))
        manager = KernelManager(kernel_name="synthetic-smoke",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
        NotebookClient(nb, km=manager, timeout=180,
                       resources={"metadata": {"path": str(path.parent)}}).execute(cleanup_kc=True)
        nbformat.write(nb, scratch / f"{index}_{path.stem}.executed.ipynb")
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert "SYNTHETIC_NOTEBOOK_" not in json.dumps(cell.outputs)
        print(f"PASS: {path.name}" + (" (reuse)" if index == 1 else ""), flush=True)

    assert len(list((scratch / "report").glob("*.png"))) == 6
    assert len(list((scratch / "report").glob("*.csv"))) == 5
    metadata = json.loads((scratch / "run" / "training_metadata.json").read_text(encoding="utf-8"))
    assert metadata["test_inference_performed"] is False
    assert set(metadata["class_counts"]) == {"TRAIN", "VALIDATION"}
    for path in (scratch / "report").iterdir():
        if path.is_file():
            assert b"SYNTHETIC_NOTEBOOK_" not in path.read_bytes()
            if path.suffix == ".csv":
                assert not {"PATIENT_ID", "END_DT", "P_RESP1"}.intersection(pd.read_csv(path).columns)
    print("PASS: fresh kernels, frozen-split reuse, checkpoint-to-TEST evaluation, six charts, aggregate reports.")


if __name__ == "__main__":
    main()
