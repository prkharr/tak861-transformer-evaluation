"""Metadata-only source discovery, embedded verbatim in the delivery notebook.

A candidate is a column to REVIEW, never automatic permission to join it or use
it as a predictor. Catalog visibility does not prove SELECT access or timing.
"""
import re

import pandas as pd


def known_sources():
    """Recovered source leads, with unresolved qualifications kept unresolved."""
    private = "DSVC_TAKEDA_TA_PRIVATE.DS_ML."
    v63 = private + "TAK861_TX_READY_V63_"
    claims = "DSVC_TAKEDA_FULLMAP_PLAID_PROD.COHORT_1009719."
    rows = [
        (v63 + "DL_POC_FEATURE_MAP", "saved_vocabulary", "VERIFIED_CONTRACT", "FEATURE_INDEX, FEATURE_NAME, FEATURE_COLUMN; aliases F0000..F1027; original category-building SQL not recovered."),
        (v63 + "DL_POC_TENSOR_MONTHLY", "saved_monthly_counts", "VERIFIED_CONTRACT", "PATIENT_ID, END_DT, RESP, TIME_STEP and mapped numeric aliases; 12 steps, 0 newest. Zero activity is not independently verified observation coverage."),
        (v63 + "DL_POC_SNAPSHOTS", "frozen_population", "VERIFIED_CONTRACT", "23,151 patient/date snapshots; 12,447 patients; 1,345 positive snapshot labels."),
        (v63 + "DL_POC_PATIENT_SPLIT", "frozen_split", "VERIFIED_CONTRACT", "PATIENT_ID, END_DT, RESP, SPLIT, SPLIT_CONFIG; do not select features using validation or test."),
        (v63 + "MODEL_DATA", "encoded_snapshot_features", "ENCODING_REVIEW_REQUIRED", "Exact patient/date alignment previously passed. Numeric/Binary business encoding and upstream feature selection may depend on RESP; raw TRAIN-safe provenance is not established."),
        (v63 + "MODEL_TYPE", "configured_feature_list", "METADATA_ONLY", "Read FEATURES at runtime for the configured 49-feature list. Do not substitute manually transcribed names."),
        (v63 + "FINAL_MODEL", "fitted_feature_summary", "METADATA_ONLY", "Feature descriptions/encoding parameters; not raw feature observations."),
        (v63 + "FEATURES_SUMMARY", "feature_summary", "METADATA_ONLY", "Inventory/lineage evidence; not a predictor source."),
        (v63 + "UNIVERSE_W_FEATURES", "long_feature_values", "ENCODING_REVIEW_REQUIRED", "Observed PATIENT_ID, END_DT, RESP, CAT, VALUE, VALUE_B, DATA_TYPE, PNT_CNT, RESP_CNT. Need category-grain uniqueness, raw-value and encoding provenance before pivoting."),
        (v63 + "MANUAL_FEATURES", "engineered_snapshot_candidates", "ALIGNMENT_REVIEW_REQUIRED", "Prior exact-key audit matched 0 of 23,151 snapshot keys. Do not repair with patient-only joins or shifted dates."),
        (v63 + "CUSTOM_PLAN_FTS", "plan_snapshot_candidates", "TIMING_REVIEW_REQUIRED", "Recover exact patient/date grain, formula, cutoff and historical plan data before use."),
        (v63 + "CUSTOM_SPECIALIST_VISITS_FTS", "specialist_snapshot_candidates", "TIMING_REVIEW_REQUIRED", "Observed NEURO_VISIT_RECENCY_PCT, NEURO_VISITS, NUM_NEURO_SPECIALISTS; formulas and as-of lookup provenance require review."),
        (v63 + "CUSTOM_PT_HCP_ATOPEN_METRICS", "patient_provider_metrics", "TIMING_REVIEW_REQUIRED", "Verify source metric vintage and historical window before use."),
        (v63 + "TMP_CUSTOM_HCP_ATOPEN_METRICS", "provider_metrics", "FUTURE_WINDOW_BLOCKED", "NPI and START_DT/END_DT; observed single window 2025-08 to 2026-08 is unsuitable for the historical V63 cutoffs. Do not join regenerated future values."),
        (v63 + "TMP_CUSTOM_HCP_ATOPEN_NOZOLP", "provider_metrics_intermediate", "TIMING_REVIEW_REQUIRED", "Source lead only; do not assume an as-of historical metric exists."),
        (v63 + "TMP_CUSTOM_HCP_ATOPEN_WITHZOLP", "provider_metrics_intermediate", "TIMING_REVIEW_REQUIRED", "Source lead only; do not assume an as-of historical metric exists."),
        (v63 + "SKIPGRAM_SEQ_EXT", "undated_sequence_patterns", "SNAPSHOT_AUDIT_REQUIRED", "Snapshot patterns lack dated historical reconstruction. Require cutoff and encoding audit; never manufacture monthly history."),
        (v63 + "NT1_PT_DX_INFO", "diagnosis_history_candidates", "TIMING_REVIEW_REQUIRED", "FIRST/LAST_NT1/NT2 and NUM_NT1/NT2_DX leads; lifetime summaries can include post-cutoff records."),
        (v63 + "ZC_AT_GENERIC_PLAN_CONTROL", "plan_access_candidates", "TIMING_REVIEW_REQUIRED", "GENERICS_NTNL_PCT, GENERICS_PLAN_PCT; historical formulas unverified."),
        (private + "ENROLLMENT_20260825", "observation_intervals", "VERIFIED_SCHEMA", "PATIENT_ID, START_DATE, END_DATE; use coverage as an explicit availability contract, not a label or an automatic feature."),
        (private + "HCP_LOOKUP", "provider_dimension", "ASOF_REVIEW_REQUIRED", "Correct recovered schema is DS_ML. HCP_NPI joins RENDERING_NPI; current specialties do not prove historical specialty availability."),
        ("MEDICAL_EVENTS_20260825", "dated_medical_claims", "QUALIFY_AT_RUNTIME", "Dated table name supplied; do not guess database/schema. Resolve from accessible metadata; event grain, code mappings, dedup and availability need explicit contracts."),
        ("PHARMACY_EVENTS_20260825", "dated_pharmacy_claims", "QUALIFY_AT_RUNTIME", "Dated table name supplied; resolve database/schema from metadata. Paid/reversed status, dedup and fill availability require explicit contracts."),
        (claims + "MEDICAL_EVENTS_LATEST", "current_medical_claims", "VINTAGE_REVIEW_REQUIRED", "SERVICE_DATE is an event date, not proof of record availability. LATEST need not reproduce the 20260825 extract."),
        (claims + "PHARMACY_EVENTS_LATEST", "current_pharmacy_claims", "VINTAGE_REVIEW_REQUIRED", "FILL_DATE, NDC11 and DAYS_SUPPLY are source leads. Never substitute current claims silently for a frozen vintage."),
        (claims + "PATIENT_DEMOGRAPHICS_LATEST", "demographic_dimension", "ASOF_REVIEW_REQUIRED", "YEAR_OF_BIRTH supports approximate year-based age only after patient-key and historical availability checks."),
        (claims + "PROVIDERS_LATEST", "provider_dimension", "ASOF_REVIEW_REQUIRED", "Current provider specialty/type is not an as-of dimension."),
        (claims + "PLANS_LATEST", "plan_dimension", "ASOF_REVIEW_REQUIRED", "Current insurance group/segment is not an as-of dimension."),
        ("DSVC_TAKEDA_TA_PRIVATE.DS_ML_PROD.DX_PX_RX_PLAID", "code_reference", "REFERENCE_REVIEW_REQUIRED", "Diagnosis/procedure/drug categories, descriptions and hierarchy; mapping multiplicity must be verified before counting."),
        ("DSVC_TAKEDA_TA_PRIVATE.DS_ML_PROD.DX_PX_RX_PLAID_ANNUAL_CNT", "upstream_category_counts", "REFERENCE_REVIEW_REQUIRED", "Inclusion-lineage lead only; original 1,028-category threshold/filter SQL is not recovered."),
        ("DSVC_TAKEDA_TA_PRIVATE.DS_ML_PROD.ALL_PROCEDURES_SIMPLE", "procedure_reference", "REFERENCE_REVIEW_REQUIRED", "ICD-10-PCS decomposition; not a substitute for ordered visit patterns."),
        ("TAK861.NARCOLEPSY_MARKET_BASKET_CODES", "drug_basket_reference", "QUALIFY_AT_RUNTIME", "Database qualification not verified; resolve live metadata before use."),
    ]
    return pd.DataFrame(rows, columns=["source", "role", "status", "notes"])


# Callable alias lets the notebook display a fresh DataFrame without mutable state.
KNOWN_SOURCES = known_sources


def classify_inventory(columns):
    """Classify each INFORMATION_SCHEMA column without reading patient values.

    The returned candidate flag means eligible for CONTRACT REVIEW only. Every
    row has included_by_default=False. The saved tensor is loaded by its separate
    FEATURE_MAP adapter, never by inferring numeric field semantics here.
    """
    required = ["TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"]
    missing = set(required) - set(columns.columns)
    if missing:
        raise ValueError("Inventory lacks required metadata columns: " + repr(sorted(missing)))
    out = columns.copy().reset_index(drop=True)
    if out[required].isna().any().any():
        raise ValueError("Inventory contains null table/column/type identifiers.")
    if out.duplicated(required[:4]).any():
        raise ValueError("Inventory contains duplicate fully qualified column keys.")
    table_keys = required[:3]
    schemas = {
        tuple(str(x).upper() for x in key): set(group.COLUMN_NAME.astype(str).str.upper())
        for key, group in out.groupby(table_keys, sort=False)
    }
    numeric = {"NUMBER", "DECIMAL", "NUMERIC", "INT", "INTEGER", "BIGINT", "SMALLINT", "FLOAT", "FLOAT4", "FLOAT8", "DOUBLE", "DOUBLE PRECISION", "REAL", "BOOLEAN"}
    target_names = {"RESP", "TARGET", "LABEL", "OUTCOME", "RESPONSE", "RESP_CNT", "POSITIVE_COUNT", "Y_TRUE"}
    identifier_names = {"PATIENT_ID", "ID", "OMNI_ID", "OMNL_ID", "NPI", "HCP_NPI", "RENDERING_NPI", "PRESCRIBER_NPI", "CLAIM_ID", "EVENT_ID", "PLAN_ID", "MEMBER_ID", "SOURCE_ID", "RUN_ID", "DATASET_ID", "TIME_STEP", "FEATURE_INDEX", "RN", "RND"}
    decisions = []
    for row in out.itertuples(index=False):
        t, c, d = str(row.TABLE_NAME).upper(), str(row.COLUMN_NAME).upper(), str(row.DATA_TYPE).upper().split("(", 1)[0]
        names = schemas[(str(row.TABLE_CATALOG).upper(), str(row.TABLE_SCHEMA).upper(), t)]
        snapshot = {"PATIENT_ID", "END_DT"}.issubset(names)
        events = "PATIENT_ID" in names and bool(names & {"SERVICE_DATE", "FILL_DATE", "EVENT_DATE"})
        candidate, kind, reason, adapter = False, "unclassified", "Requires a verified feature contract.", "manual_contract"
        if c in target_names or c.startswith(("TARGET_", "OUTCOME_", "RESP_")):
            kind, reason, adapter = "target_or_target_summary", "Target and target-derived statistics are not predictor columns.", "exclude"
        elif c in identifier_names or c.endswith("_ID") or c.endswith("_NPI") or c in {"SPLIT", "SPLIT_CONFIG"}:
            kind, reason, adapter = "identifier_or_assignment", "Use for keys, grouping or source audit only.", "exclude"
        elif d.startswith(("DATE", "TIME")) or c.endswith(("_DT", "_DATE", "_TIMESTAMP")):
            kind, reason, adapter = "date_or_availability", "Retain as alignment/cutoff metadata; a reviewed derived recency may be a separate feature.", "date_contract"
        elif any(x in t for x in ("SHAP", "SCORED", "PREDICTION", "EVALUATION", "CHECKPOINT", "_MODEL_RUN_", "FEATURE_ANALYSIS")) or any(x in c for x in ("SHAP", "PREDICT", "Y_PROB", "Y_SCORE", "COEFFICIENT", "IMPORTANCE")) or c in {"SCORE", "PROBABILITY", "PROB", "PNT_CNT"}:
            kind, reason, adapter = "model_output_or_fit_summary", "Model outputs, fitted summaries and population summary fields cannot become raw predictors.", "exclude"
        elif t.endswith(("MODEL_TYPE", "FINAL_MODEL", "FEATURES_SUMMARY", "FEATURE_MAP", "PATIENT_SPLIT")):
            kind, reason, adapter = "configuration_or_summary", "Source lineage/configuration only.", "metadata_only"
        elif t.endswith("TMP_CUSTOM_HCP_ATOPEN_METRICS"):
            kind, reason, adapter = "future_window_blocked", "Observed metric window is after historical V63 cutoffs; a new valid historical source is required.", "blocked"
        elif t.endswith(("MODEL_DATA", "UNIVERSE_W_FEATURES")):
            candidate, kind, reason, adapter = True, "encoded_snapshot_review", "Review raw/encoded value definition, RESP-dependent transformation fit population, exact grain and cutoff before inclusion.", "reviewed_snapshot_or_long_pivot"
        elif "SKIPGRAM" in t:
            candidate, kind, reason, adapter = True, "undated_snapshot_review", "Undated patterns require snapshot cutoff and encoding audit; not a temporal sequence source.", "reviewed_snapshot"
        elif t.endswith("TENSOR_MONTHLY") and re.fullmatch(r"F[0-9]{4}", c):
            candidate, kind, reason, adapter = True, "saved_claim_count", "Recover name/order from FEATURE_MAP and validate complete 12-step nonnegative counts; do not infer alias meaning.", "saved_tensor_feature_map"
        elif events:
            candidate, kind, reason, adapter = True, "event_attribute_review", "Requires event grain, date/availability, code mapping, count/dedup and coverage contract; categorical code fields are included in this inventory.", "reviewed_event_aggregate"
        elif snapshot and d in numeric:
            candidate, kind, reason, adapter = True, "numeric_snapshot_review", "Requires unique PATIENT_ID+END_DT, exact cohort alignment, feature semantics and cutoff/encoding audit.", "reviewed_snapshot"
        elif snapshot:
            candidate, kind, reason, adapter = True, "categorical_snapshot_review", "Requires exact snapshot grain and TRAIN-fitted category encoding; raw strings are not automatically numeric.", "reviewed_snapshot_encoder"
        elif "PATIENT_ID" in names or any(x in t for x in ("LOOKUP", "REFERENCE", "PLAID", "PROVIDERS", "PLANS", "MARKET_BASKET")):
            candidate, kind, reason, adapter = True, "dimension_or_reference_review", "Requires explicit relation keys, uniqueness and historical validity; never join arbitrary fields by patient alone.", "reviewed_dimension"
        decisions.append((candidate, kind, reason, adapter, False))
    additions = pd.DataFrame(decisions, columns=["candidate", "kind", "exclusion_reason", "proposed_adapter", "included_by_default"])
    return pd.concat([out, additions], axis=1)
