"""Machine-local configuration; no patient data paths or credentials in source."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    path = Path(path or os.environ.get("TAK861_CONFIG", root / "config.local.json")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Create a local copy of config.example.json, configure approved paths and the upstream review, then set TAK861_CONFIG if needed.")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = {"input", "bundle_dir", "manifest_path", "run_dir", "report_dir", "expected", "source_review", "split", "model", "training"}
    if not required.issubset(cfg):
        raise ValueError("Pipeline configuration is incomplete; use config.example.json as the schema.")
    if cfg["input"].get("mode") not in {"monthly_parquet", "in_memory", "bundle"}:
        raise ValueError("input.mode must be monthly_parquet, in_memory, or bundle.")

    def resolve(value):
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (path.parent / candidate).resolve()

    for key in ("bundle_dir", "manifest_path", "run_dir", "report_dir"):
        cfg[key] = resolve(cfg[key])
    for key in ("path", "feature_order_json"):
        if cfg["input"].get(key):
            cfg["input"][key] = resolve(cfg["input"][key])
    if cfg["input"]["mode"] == "monthly_parquet" and not cfg["input"].get("path"):
        raise ValueError("Monthly Parquet mode requires input.path.")
    # A report/run must not overwrite the frozen input or another stage's files.
    destinations = [cfg[k] for k in ("bundle_dir", "run_dir", "report_dir")]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Bundle, training run, and evaluation report directories must be distinct.")
    return cfg
