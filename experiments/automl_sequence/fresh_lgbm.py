import hashlib
import json
import time

import numpy as np
import pandas as pd


NAME = "LIGHTGBM_TRAIN_ONLY_STATIC"
SETTINGS = dict(seeds=[42, 142, 242], n_estimators=1000, learning_rate=.05,
                num_leaves=31, max_depth=-1, min_child_samples=20,
                colsample_bytree=.8, subsample=.8, subsample_freq=1,
                reg_lambda=1., n_jobs=8, stopping_rounds=100,
                weighting="patient", balance="sqrt", primary_fraction=.1)


def design(data, prep, split, mdl):
    idx = data["indices"][split]
    z = mdl.transform_inputs(data["X_summary"], prep["automl"],
                             valid=data["summary_mask"], row_indices=idx)
    try:
        x = np.empty((len(idx), z.shape[-1] + 1), dtype=np.float32)
        x[:, :-1] = z[:, 0]
        x[:, -1] = np.asarray(data["summary_mask"])[idx, 0]
    finally:
        mdl.release_array(z)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite baseline design")
    return x


def validate(data):
    meta = data["metadata"]
    if meta.duplicated(["PATIENT_ID", "END_DT"]).any():
        raise ValueError("Duplicate snapshot keys")
    if meta.groupby("PATIENT_ID").SPLIT.nunique().max() != 1:
        raise ValueError("Patient overlap between splits")
    if not np.array_equal(meta.RESP.to_numpy(), data["y"]):
        raise ValueError("Metadata label mismatch")
    for split in ("train", "validation", "test"):
        if not np.array_equal(data["indices"][split], np.flatnonzero(meta.SPLIT.eq(split))):
            raise ValueError("Split indices differ from ordered metadata")
    for split in ("train", "validation"):
        if set(np.unique(data["y"][data["indices"][split]])) != {0, 1}:
            raise ValueError("Both classes required for training and validation")


def scores_ok(scores, n):
    if scores.shape != (n,) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid positive-class probabilities")


def fit_or_resume(data, prep, protocol, lock, wf, mdl, wh, prefix, source_hash, settings=None):
    import lightgbm as lgb
    cfg = dict(SETTINGS if settings is None else settings)
    validate(data)
    if (prep["automl"].get("selection_used_test") is not False
            or prep["automl"].get("selection_used_external_validation") is not False):
        raise ValueError("TRAIN-only feature-selection provenance is required")
    if (lock["protocol_hash"] != wf.digest(protocol) or lock["preparation_hash"] != wf.digest(prep)
            or lock["input_manifest_hash"] != wf.digest(data["manifest"])):
        raise ValueError("Frozen preparation/input/protocol mismatch")
    persisted = json.loads(wh.read_artifacts(prefix + "_LOCK", {"lock.json"})["lock.json"])
    if persisted != lock:
        raise ValueError("Persisted Transformer lock changed")
    ti, vi = data["indices"]["train"], data["indices"]["validation"]
    xt, xv = design(data, prep, "train", mdl), design(data, prep, "validation", mdl)
    yt, yv = data["y"][ti], data["y"][vi]
    weights = mdl.sample_weights(data["metadata"].PATIENT_ID.to_numpy()[ti], cfg["weighting"])
    weights = weights * np.where(yt == 1, mdl.class_weight(yt, weights, cfg["balance"]), 1.)
    names = list(prep["automl"]["selected_names"]) + ["AUTOML_CURRENT_AVAILABLE"]
    identity = dict(source_hash=source_hash, settings=cfg, preparation_hash=wf.digest(prep),
                    input_hash=wf.digest(data["manifest"]), lock_hash=wf.digest(lock),
                    train_design_hash=wf.array_sha256(xt), validation_design_hash=wf.array_sha256(xv),
                    features=names, lightgbm_version=lgb.__version__, numpy_version=np.__version__)
    root = prefix + "_FRESH_LGBM_STATIC_V1"
    artifact_names = {"receipt.json", "model.txt", "validation.npy"}
    models, predictions, receipts = [], [], []
    for seed in cfg["seeds"]:
        table = root + "_SEED_" + str(seed)
        seed_identity = dict(identity, seed=seed)
        if wh.table_exists(table):
            blob = wh.read_artifacts(table, artifact_names)
            receipt = json.loads(blob["receipt.json"])
            if receipt["identity"] != seed_identity:
                raise ValueError("Baseline identity changed; do not overwrite an existing fit")
            for key in ("model.txt", "validation.npy"):
                if hashlib.sha256(blob[key]).hexdigest() != receipt["payload_hashes"][key]:
                    raise ValueError("Baseline artifact integrity failure")
            booster = lgb.Booster(model_str=blob["model.txt"].decode())
            score = wf.bytes_array(blob["validation.npy"])
            if not np.allclose(booster.predict(xv), score, rtol=0, atol=1e-12):
                raise ValueError("Resumed model predictions differ")
            print("Resumed LightGBM seed", seed)
        else:
            params = {k: v for k, v in cfg.items() if k not in
                      {"seeds", "stopping_rounds", "weighting", "balance", "primary_fraction"}}
            estimator = lgb.LGBMClassifier(**params, objective="binary", metric="None",
                                         random_state=seed, deterministic=True, force_col_wise=True, verbosity=-1)
            def validation_lift(y, p):
                return "lift10", mdl.lift_at_fraction(y, p, cfg["primary_fraction"]), True
            start = time.monotonic()
            estimator.fit(xt, yt, sample_weight=weights, eval_set=[(xv, yv)],
                          eval_metric=validation_lift, callbacks=[
                              lgb.early_stopping(cfg["stopping_rounds"], first_metric_only=True, verbose=False)])
            if list(estimator.classes_) != [0, 1]:
                raise ValueError("Unexpected class order")
            model_bytes = estimator.booster_.model_to_string(num_iteration=estimator.best_iteration_).encode()
            booster = lgb.Booster(model_str=model_bytes.decode())
            score = booster.predict(xv)
            scores_ok(score, len(vi))
            blob = {"model.txt": model_bytes, "validation.npy": wf.array_bytes(score)}
            receipt = dict(identity=seed_identity, best_iteration=estimator.best_iteration_,
                           seconds=time.monotonic() - start,
                           payload_hashes={k: hashlib.sha256(v).hexdigest() for k, v in blob.items()})
            blob["receipt.json"] = wf.json_bytes(receipt)
            wh.save_artifacts(table, blob)
            print("Completed LightGBM seed", seed, "best iteration", receipt["best_iteration"],
                  "seconds", round(receipt["seconds"], 1))
        scores_ok(score, len(vi))
        models.append(booster)
        predictions.append(score)
        receipts.append(receipt)
    return dict(models=models, validation_scores=np.mean(predictions, axis=0),
                identity=identity, receipts=receipts, root=root)


def compare(data, prep, baseline, transformer_scores, split, wf, mdl, cmp, wh,
            allow_test=False, n_bootstrap=1000):
    if split not in ("validation", "test") or (split == "test" and not allow_test):
        raise ValueError("TEST requires explicit authorization after fitting is locked")
    if set(transformer_scores) != {"COUNT", "STATIC_FUSION", "TEMPORAL_FUSION"}:
        raise ValueError("All three frozen Transformer predictions required")
    idx = data["indices"][split]
    meta = data["metadata"].iloc[idx].reset_index(drop=True)
    y = data["y"][idx]
    if split == "validation":
        score = baseline["validation_scores"]
    else:
        x = design(data, prep, split, mdl)
        score = np.mean([model.predict(x) for model in baseline["models"]], axis=0)
        del x
    scores = dict(transformer_scores, **{NAME: score})
    for values in scores.values():
        scores_ok(np.asarray(values), len(idx))
    identity = dict(baseline=baseline["identity"], split=split, n_bootstrap=n_bootstrap,
                    models=[r["payload_hashes"] for r in baseline["receipts"]],
                    predictions={k: wf.array_sha256(v) for k, v in scores.items()})
    table = baseline["root"] + "_" + split.upper() + "_LIFT"
    if wh.table_exists(table):
        cached = wh.read_artifacts(table, {"report.json", "baseline.npy"})
        stored = json.loads(cached["report.json"])
        if stored["identity"] != identity:
            raise ValueError("Saved comparison identity mismatch")
        if not np.array_equal(wf.bytes_array(cached["baseline.npy"]), score):
            raise ValueError("Saved baseline probabilities changed")
        return stored
    lifts = {name: {float(r["fraction"]): r["lift"] for r in
              cmp.binary_metrics(y, values, top_fractions=(.1, .2, .3))["topk"]}
             for name, values in scores.items()}
    rows, intervals = [], []
    for name, values in scores.items():
        row = dict(Model=name, Split=split, Snapshots=len(y), Positives=int(y.sum()))
        for f in (.1, .2, .3):
            row["Lift@" + str(int(f * 100)) + "%"] = lifts[name][f]
        row["Delta@10%"] = lifts[name][.1] - lifts[NAME][.1]
        row["Relative_delta@10%"] = row["Delta@10%"] / lifts[NAME][.1] if lifts[NAME][.1] else None
        rows.append(row)
        if name != NAME:
            pair = wf.paired_lift_bootstrap(y, values, score, meta.PATIENT_ID.to_numpy(),
                                            fractions=(.1, .2, .3), n_bootstrap=n_bootstrap, seed=20261013)
            for fraction, interval in pair["lift_by_fraction"].items():
                intervals.append(dict(Model=name, Reference=NAME, Fraction=fraction,
                                      Reliable=pair["interval_reliable"], **interval))
    stored = dict(identity=identity, rows=rows, intervals=intervals,
                  scope="Validation checkpoint selection conditions these intervals" if split == "validation" else
                  "Retrospective TEST; already inspected; no independent generalization claim",
                  representation="Static selected AutoML summary plus availability; no count or historical tokens",
                  fit_scope="TRAIN-only saved feature preparation and fitting; validation early stopping on lift10")
    wh.save_artifacts(table, {"report.json": wf.json_bytes(stored), "baseline.npy": wf.array_bytes(score)})
    return stored


def ranking(report):
    table = pd.DataFrame(report["rows"])
    table.insert(0, "Rank", table["Lift@10%"].rank(method="min", ascending=False).astype(int))
    return table.sort_values(["Rank", "Model"], kind="stable").reset_index(drop=True)
