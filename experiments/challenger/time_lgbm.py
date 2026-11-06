"""Paired development experiment on frozen expanded-cohort snapshot inputs."""
import hashlib
import json
import re

import numpy as np
import pandas as pd


FEATURE_NAMES = [signal + "_" + suffix for signal in ("STUDY", "SLEEP_WAKE")
                 for suffix in ("MONTHS_SINCE_LAST", "ACTIVE_MONTHS", "RECENT3_MINUS_PRIOR9_MONTHLY_MEAN",
                                "MONTHLY_SLOPE")] + ["STUDY_M0_5_AND_SLEEP_WAKE_M0_2"]
ROLES = {"study": ("PX", {"sleep study", "sleep studies", "level 1 cpt sleep study"}),
         "diagnosis": ("DX", {"sleep wake disorder", "sleep wake disorders"})}


def channels(names, exact):
    if len(set(names)) != len(names) or set(exact) != set(ROLES):
        raise ValueError("Invalid count dictionary or channel configuration")
    result = []
    for role, (family, labels) in ROLES.items():
        options = [n for n in names if n.startswith(family + "__")]
        if exact[role]:
            matches = [n for n in options if n == exact[role]]
        else:
            matches = [n for n in options if re.sub("[^a-z0-9]+", " ", n.split("__", 1)[1].lower()).strip() in labels]
        if len(matches) != 1:
            candidates = sorted(n for n in options if "sleep" in n.lower())
            raise ValueError(f"Need one verified {family} count category for {role}. "
                             f"Set its exact name using the source dictionary. Candidates: {candidates}")
        result.append(names.index(matches[0]))
    return result


def temporal_columns(counts, masks, indices, column_indices):
    """Read two monthly channels per row block. Position zero ends at snapshot cutoff."""
    if counts.ndim != 3 or counts.shape[1] != 12 or masks.shape != counts.shape[:2] or masks.dtype != bool:
        raise ValueError("Expected twelve monthly count positions and a boolean position mask")
    out = np.full((len(indices), len(FEATURE_NAMES)), np.nan, np.float32)
    newest_first = np.arange(12, dtype=float)
    slope_weights = (5.5 - newest_first) / 143.
    for start in range(0, len(indices), 1024):
        idx = indices[start:start+1024]
        complete = masks[idx].all(axis=1)
        block = out[start:start+len(idx)]
        recent = []
        for j, col in enumerate(column_indices):
            x = counts[idx, :, col]
            if not np.isfinite(x).all() or (x < 0).any():
                raise ValueError("Invalid count values")
            active = x > 0
            last = np.where(active.any(axis=1), active.argmax(axis=1), 12)
            block[:, j*4:j*4+4] = np.column_stack((last, active.sum(axis=1),
                x[:, :3].mean(axis=1)-x[:, 3:].mean(axis=1), x @ slope_weights))
            recent.append(active[:, :6 if j == 0 else 3].any(axis=1))
        block[:, -1] = recent[0] & recent[1]
        block[~complete] = np.nan
    return out


def augment_static(static, extra, mdl):
    design = mdl.allocate_array((len(static), static.shape[1] + extra.shape[1]), np.float32)
    try:
        for start in range(0, len(static), 1024):
            rows = slice(start, start + 1024)
            design[rows, :static.shape[1]] = static[rows]
            design[rows, static.shape[1]:] = extra[rows]
        return design
    except BaseException:
        mdl.release_array(design)
        raise


def _seed_scores(xt, xv, y, weights, seed, settings, identity, table, wf, wh):
    import lightgbm as lgb
    seed_identity = dict(identity, seed=seed)
    files = {"model.txt", "validation.npy", "receipt.json"}
    if wh.table_exists(table):
        saved = wh.read_artifacts(table, files)
        receipt = json.loads(saved["receipt.json"])
        if receipt["identity"] != seed_identity:
            raise ValueError("Resume identity differs")
        for name in ("model.txt", "validation.npy"):
            if hashlib.sha256(saved[name]).hexdigest() != receipt["hashes"][name]:
                raise ValueError("Corrupt seed artifact")
        return wf.bytes_array(saved["validation.npy"])
    params = {k:v for k,v in settings.items() if k not in {"seeds","primary_fraction","weighting","balance","stopping_rounds"}}
    model = lgb.LGBMClassifier(**params, objective="binary", metric="None", random_state=seed,
                               deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(xt, y, sample_weight=weights)
    if list(model.classes_) != [0,1]:
        raise ValueError("Invalid positive class")
    scores = model.booster_.predict(xv)
    files = {"model.txt": model.booster_.model_to_string().encode(), "validation.npy": wf.array_bytes(scores)}
    files["receipt.json"] = wf.json_bytes(dict(identity=seed_identity,
        hashes={name:hashlib.sha256(value).hexdigest() for name,value in files.items()}))
    wh.save_artifacts(table, files)
    return scores


def run(data, prep, protocol, parent_prefix, parent_impl, source_hash, exact, baseline, mdl, cmp, wf, wh,
        *, settings=None, bootstrap=1000):
    import lightgbm as lgb
    baseline.validate(data)
    cfg = dict(baseline.SETTINGS if settings is None else settings)
    manifest = data["manifest"]
    if (protocol["identity"]["input_manifest_hash"] != wf.digest(manifest)
            or protocol["identity"]["implementation_hash"] != parent_impl
            or protocol["identity"]["config_hash"] != manifest["configuration_hash"]
            or protocol["preparation_hash"] != wf.digest(prep)):
        raise ValueError("Mismatch between frozen inputs, preparation and protocol")
    if any(prep["automl"].get(k) is not False for k in ("selection_used_test","selection_used_external_validation")):
        raise ValueError("TRAIN-only static feature selection is required")
    contract = manifest["count_contract"]
    if contract["orientation"] != "0=newest" or contract["date_policy"] != "month_end_history":
        raise ValueError("Input time grid or orientation differs from the monthly history contract")
    names = contract["feature_names"]
    chosen = channels(names, exact)
    static_names = list(prep["automl"]["selected_names"]) + ["AUTOML_CURRENT_AVAILABLE"]
    identity = dict(source_hash=source_hash, parent_implementation=parent_impl, parent_prefix=parent_prefix,
                    protocol_hash=wf.digest(protocol), lightgbm=lgb.__version__, numpy=np.__version__,
                    channel_names=[names[c] for c in chosen], static_names=static_names,
                    temporal_names=FEATURE_NAMES, settings=cfg, rounds="fixed TRAIN fit; no early stopping",
                    budget=cfg["n_estimators"], bootstrap=bootstrap)
    prefix = parent_prefix + "_TIME_LGBM_" + wf.digest(identity)[:12].upper()
    ti, vi = (data["indices"][s] for s in ("train","validation"))
    patients = data["metadata"].PATIENT_ID.to_numpy()
    y = data["y"]
    train_weights = mdl.sample_weights(patients[ti], cfg["weighting"])
    train_weights = train_weights * np.where(y[ti] == 1, mdl.class_weight(y[ti], train_weights, cfg["balance"]), 1.)
    extra_t, extra_v = (temporal_columns(data["X_count"], data["count_mask"], idx, chosen) for idx in (ti, vi))
    observed_train = np.isfinite(extra_t[:, -1])
    if np.unique(extra_t[observed_train, -1]).size != 2:
        raise ValueError("No TRAIN variation in the prespecified interaction; verify mapped source coverage")
    support = {split: {"complete_history": int(np.isfinite(extra[:, -1]).sum()),
                       "flagged": int(np.count_nonzero(extra[:, -1] == 1))}
               for split, extra in (("train", extra_t), ("validation", extra_v))}
    static_t = baseline.design(data, prep, "train", mdl)
    static_v = baseline.design(data, prep, "validation", mdl)
    predictions = {}
    for arm in ("STATIC", "STATIC_PLUS_TIME"):
        xt = xv = None
        try:
            if arm == "STATIC":
                xt, xv = static_t, static_v
            else:
                xt = augment_static(static_t, extra_t, mdl)
                xv = augment_static(static_v, extra_v, mdl)
                del static_t, static_v, extra_t, extra_v
            arm_identity = dict(identity, arm=arm, train_design_sha256=wf.array_sha256(xt),
                                validation_design_sha256=wf.array_sha256(xv))
            results = []
            for seed in cfg["seeds"]:
                table = prefix + "_" + arm + "_" + str(seed)
                score = _seed_scores(xt, xv, y[ti], train_weights, seed, cfg, arm_identity, table, wf, wh)
                baseline.scores_ok(score, len(vi))
                results.append(score)
            predictions[arm] = np.mean(results, axis=0)
        finally:
            if arm == "STATIC_PLUS_TIME":
                mdl.release_array(xt)
                mdl.release_array(xv)
    meta = data["metadata"].iloc[vi][["PATIENT_ID","END_DT","RESP","SPLIT"]].reset_index(drop=True)
    rows = []
    for arm, scores in predictions.items():
        row = {"Model":arm, "Split":"VALIDATION", "Snapshots":len(vi), "Patients":meta.PATIENT_ID.nunique(),
               "Positive_snapshots":int(y[vi].sum())}
        for frac in (.1,.2,.3):
            m = cmp.topk_metrics(y[vi],scores,frac)
            suffix = str(round(frac*100))+"%"
            row.update({"Lift@"+suffix:m["lift"], "Captured@"+suffix:m["selected_positives"],
                        "Selected@"+suffix:m["selected"]})
        rows.append(row)
    pair = wf.paired_lift_bootstrap(y[vi],predictions["STATIC_PLUS_TIME"],predictions["STATIC"],patients[vi],
                                   n_bootstrap=bootstrap,seed=20261013)
    report = dict(identity=identity, support=support, rows=rows, paired_lift=pair, comparison="STATIC_PLUS_TIME minus STATIC",
                  positives_gained_at_10=rows[1]["Captured@10%"]-rows[0]["Captured@10%"],
                  scope="Expanded patient-disjoint development validation; out-of-time confirmation is pending",
                  scores_hashes={a:wf.array_sha256(s) for a,s in predictions.items()})
    payload = {"report.json":wf.json_bytes(report), "metadata.csv":meta.to_csv(index=False,lineterminator="\n").encode()}
    payload.update({a+".npy":wf.array_bytes(s) for a,s in predictions.items()})
    wh.save_artifacts(prefix+"_VALIDATION",payload)
    return prefix+"_VALIDATION", report
