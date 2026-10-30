"""Audit gates and streamed feature-package persistence for expanded populations."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def validate_evidence(config):
    evidence = config.get("cohort_evidence", {})
    required = ("eligibility_reference", "first_initiation_reference",
                "negative_followup_reference", "effective_cutoff_reference", "source_snapshot_reference")
    missing = [key for key in required if not isinstance(evidence.get(key), str) or not evidence[key].strip()]
    if missing:
        raise ValueError("Cohort audit binding is incomplete: " + ", ".join(missing))
    lineage = evidence.get("lineage_status", "reviewed_producer")
    if lineage == "candidate_as_delivered_development":
        if (evidence.get("candidate_labels_accepted") is not True or evidence.get("automatic_relabel") is not False
                or not isinstance(evidence.get("unresolved_rules"), list) or not evidence["unresolved_rules"]):
            raise ValueError("Candidate development requires explicit acceptance and recorded unresolved rules; no automatic relabeling")
    elif lineage == "bounded_reconstruction_development":
        if evidence.get('automatic_relabel') is not True or not evidence.get('unresolved_rules'):
            raise ValueError('Reconstruction requires explicit label authorization and unresolved lineage record')
        if not evidence.get('reconstruction_sha256') or evidence.get('candidate_labels_accepted') is not False:
            raise ValueError('Reconstruction must identify its own query, not claim original producer verification')
    elif lineage != "reviewed_producer":
        raise ValueError("Unknown cohort lineage status")
    sha = evidence.get("producer_sha256")
    if lineage == "reviewed_producer" or sha is not None:
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
            raise ValueError("The reviewed population producer needs its actual SHA-256 fingerprint")
    if evidence.get("retention_rule") != "all_eligible_preinitiation_snapshots":
        raise ValueError("Every retained positive snapshot must independently satisfy its forward outcome window")
    if evidence.get("outcome_days") != 90 or evidence.get("outcome_boundary") != "cutoff < first_event <= cutoff + 90 days":
        raise ValueError("Forward 90-day first-initiation contract must be explicit")
    if not config.get("client_universe", {}).get("expected_population"):
        raise ValueError("Bind independent aggregate and per-split population counts before extraction")
    if config.get("experiment_phase") not in ("SUMMARY", "TEMPORAL"):
        raise ValueError("Declare SUMMARY or TEMPORAL before running")
    if evidence.get("historical_availability") != "retrospective_latest_sources":
        raise ValueError("This implementation does not prove historical ingestion-time availability")
    return dict(evidence)


def load_metadata(read_query, read_sf, config, population):
    validate_evidence(config)
    cfg = config["client_universe"]
    name = population.resolve_universe(read_query, cfg["table"], cfg["table_name"])
    prior = population.frame(read_sf("PATIENT_SPLIT"))
    incoming = population.frame(read_query("SELECT PATIENT_ID, END_DT, RESP FROM " + name))
    metadata, audit = population.align_universe(incoming, prior, cfg.get("split_seed", 42), change_policy=cfg.get('change_policy'))
    population.assert_expected_population(audit, cfg["expected_population"])
    return metadata, audit


NLP_NAMES = {"sequence.npy", "mask.npy", "manifest.json"}


def nlp_identity(metadata, config, implementation, inp, wf):
    return dict(metadata_hash=inp.frame_hash(metadata), configuration_hash=wf.digest(config),
                implementation_hash=implementation, fit_scope="train_only", orientation="0=newest")


def save_nlp(result, metadata, config, implementation, prefix, inp, wf, wh):
    if inp.frame_hash(result["metadata"]) != inp.frame_hash(metadata):
        raise ValueError("NLP output reordered or changed the population")
    names = list(result["feature_names"])
    if not names or len(set(names)) != len(names):
        raise ValueError("NLP feature order must be nonempty and unique")
    x, mask = result["X_sequence"], result["mask"]
    if x.shape != (len(metadata), 12, len(names)) or mask.shape != (len(metadata), 12) or mask.dtype != bool:
        raise ValueError("NLP output must be N x 12 x F with explicit boolean activity")
    for start in range(0, len(metadata), 256):
        a, m = x[start:start+256], mask[start:start+256]
        if not np.isfinite(a).all() or (a < 0).any() or (a[~m] != 0).any():
            raise ValueError("NLP values must be finite nonnegative; inactive documents must remain zero")
    identity = nlp_identity(metadata, config, implementation, inp, wf)
    fitted, receipt = result["fitted_state"], result["receipt"]
    expected_patients = inp.train_patient_hash(metadata)
    train_meta = metadata.loc[metadata.SPLIT.eq("train")]
    train_requests, _ = inp.make_requests(train_meta, date_policy="month_end_history")
    all_requests, _ = inp.make_requests(metadata, date_policy="month_end_history")
    membership = metadata.groupby("PATIENT_ID").SPLIT.first().to_dict()
    ordered = all_requests.sort_values(["PATIENT_ID", "AS_OF_DATE"]).reset_index(drop=True)
    request_hash = inp.stable_hash([[i, row.PATIENT_ID, row.AS_OF_DATE.strftime("%Y-%m-%d"), membership[row.PATIENT_ID]]
                                   for i, row in enumerate(ordered.itertuples(index=False))])
    for evidence in (fitted, receipt):
        if (evidence.get("fit_scope") != "train_only" or evidence.get("train_patients_sha256") != expected_patients
                or evidence.get("train_document_count") != len(train_requests)):
            raise ValueError("NLP fitted state does not match the frozen TRAIN patient/document scope")
    if fitted.get("terms") != names or fitted.get("term_order_sha256") != inp.stable_hash(names):
        raise ValueError("NLP fitted vocabulary differs from its output columns")
    if fitted.get("extraction_identity", {}).get("requests") != request_hash:
        raise ValueError("NLP fitted state belongs to different ordered monthly documents or split assignments")
    manifest = dict(identity, feature_names=names, fitted_state=result["fitted_state"],
                    receipt=result["receipt"], feature_order_sha256=wf.digest(names),
                    source_note="Current reference descriptions; retrospective event-date-bounded features",
                    ranking_kind="TRAIN snapshot-position mean TF-IDF salience; not predictive importance",
                    train_patients_sha256=expected_patients, train_document_count=len(train_requests))
    with tempfile.TemporaryDirectory(prefix="monthly_nlp_write_") as directory:
        files = {"sequence.npy": Path(directory)/"sequence.npy", "mask.npy": Path(directory)/"mask.npy"}
        np.save(files["sequence.npy"], x, allow_pickle=False)
        np.save(files["mask.npy"], mask, allow_pickle=False)
        manifest["payload_hashes"] = {key: wf.file_sha256(path) for key, path in files.items()}
        wh.save_artifacts(prefix+"_NLP_INPUTS", dict(files, **{"manifest.json": wf.json_bytes(manifest)}))
    return manifest


def load_nlp(metadata, config, implementation, prefix, inp, wf, wh):
    owner = wf._InputDirectory()
    try:
        payload = wh.read_artifacts(prefix+"_NLP_INPUTS", NLP_NAMES,
                                    disk_names={"sequence.npy", "mask.npy"}, output_dir=owner.name)
        manifest = json.loads(payload["manifest.json"])
        identity = nlp_identity(metadata, config, implementation, inp, wf)
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("NLP checkpoint population, configuration or implementation mismatch")
        arrays = {}
        for key, name in (("X_sequence", "sequence.npy"), ("mask", "mask.npy")):
            if wf.file_sha256(payload[name]) != manifest["payload_hashes"][name]:
                raise ValueError("NLP payload content changed")
            arrays[key] = np.load(payload[name], mmap_mode="r", allow_pickle=False)
            owner.mappings.append(arrays[key]._mmap)
        return dict(arrays, metadata=metadata, feature_names=manifest["feature_names"],
                    manifest=manifest, _owner=owner)
    except BaseException:
        owner.cleanup()
        raise


def release_nlp(package):
    owner = package.pop("_owner", None)
    if owner is not None:
        owner.cleanup()


def nlp_ranking(package):
    """Streaming TRAIN salience without displaying patient identifiers or histories."""
    x = package["X_sequence"]
    rows = np.flatnonzero(package["metadata"].SPLIT.eq("train").to_numpy())
    sums, nonzero = np.zeros(x.shape[-1]), np.zeros(x.shape[-1], dtype=np.int64)
    for start in range(0, len(rows), 256):
        block = x[rows[start:start+256]]
        sums += block.sum(axis=(0, 1), dtype=np.float64)
        nonzero += (block > 0).sum(axis=(0, 1))
    out = pd.DataFrame(dict(Feature=package["feature_names"],
                           Mean_TFIDF=sums/max(1, len(rows)*12), Nonzero_positions=nonzero))
    out = out.sort_values(["Mean_TFIDF", "Feature"], ascending=[False, True], kind="stable").reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out)+1))
    return out
