"""Bind Brian's revised universe without reusing old-cohort tensors.

Only dictionary and patient assignments are inherited. Claims are reaggregated
for every new universe key; the source procedure's RESP is never regenerated.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd

inputs = fresh_counts = count_loader = None  # standalone notebook injection
TABLE_NAME = "TAK861_TX_READY_PATIENT_UNIVERSE_DL"


def frame(value):
    result = value.copy() if isinstance(value, pd.DataFrame) else value.toPandas()
    result.columns = [str(c).upper() for c in result.columns]
    if not result.columns.is_unique:
        raise ValueError("Duplicate source columns")
    return result


def resolve_universe(read_query, explicit=None):
    if explicit:
        name = fresh_counts._fqn(explicit)
        if name.rsplit('.', 1)[-1] != TABLE_NAME:
            raise ValueError("Expected Brian's requested patient universe")
        return name
    found = frame(read_query(
        "SELECT TABLE_CATALOG,TABLE_SCHEMA,TABLE_NAME FROM "
        "DSVC_TAKEDA_TA_PRIVATE.INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_NAME='" + TABLE_NAME + "'"))
    names = sorted(set('.'.join(map(str, r)) for r in
                       found[['TABLE_CATALOG', 'TABLE_SCHEMA', 'TABLE_NAME']].itertuples(index=False, name=None)))
    if len(names) != 1:
        raise ValueError("Set CONFIG['client_universe']['table'] to the verified fully qualified table; "
                         "metadata discovery requires exactly one match, found " + repr(names))
    return fresh_counts._fqn(names[0])


def align_universe(new, prior, seed=42):
    """Preserve every known patient's partition; hash only genuinely new IDs."""
    new, prior = frame(new), frame(prior)
    new['SPLIT'] = 'train'
    new = inputs.normalize_snapshots(new).sort_values(['PATIENT_ID', 'END_DT']).reset_index(drop=True)
    prior = inputs.normalize_snapshots(prior).sort_values(['PATIENT_ID', 'END_DT']).reset_index(drop=True)
    membership = prior.groupby('PATIENT_ID').SPLIT.first().to_dict()
    if prior.groupby('PATIENT_ID').SPLIT.nunique().gt(1).any():
        raise ValueError("Prior patient membership crosses splits")
    def assignment(patient):
        if patient in membership:
            return membership[patient]
        value = int(hashlib.sha256((str(seed) + ':' + patient).encode()).hexdigest()[:16], 16) / 2**64
        return 'train' if value < .7 else 'validation' if value < .85 else 'test'
    new['SPLIT'] = new.PATIENT_ID.map(assignment)
    keys = ['PATIENT_ID', 'END_DT']
    overlap = prior.merge(new, on=keys, how='outer', suffixes=('_old', '_new'), indicator=True)
    shared = overlap._merge.eq('both')
    if overlap._merge.eq('left_only').any():
        raise ValueError("Client universe removes prior snapshot keys; verify patient-key format and producer scope before proceeding")
    changed = int((shared & overlap.RESP_old.ne(overlap.RESP_new)).sum())
    if changed:
        raise ValueError("Client universe changes RESP on existing snapshot keys; verify producer rules before proceeding")
    for split in ('train', 'validation', 'test'):
        if set(new.loc[new.SPLIT.eq(split), 'RESP']) != {0, 1}:
            raise ValueError("Revised universe needs both classes in " + split)
    receipt = dict(old_snapshots=len(prior), new_snapshots=len(new),
                   old_positives=int(prior.RESP.sum()), new_positives=int(new.RESP.sum()),
                   positive_snapshot_ratio=float(new.RESP.sum()/prior.RESP.sum()) if prior.RESP.sum() else None,
                   old_positive_patients=int(prior.loc[prior.RESP.eq(1), 'PATIENT_ID'].nunique()),
                   new_positive_patients=int(new.loc[new.RESP.eq(1), 'PATIENT_ID'].nunique()),
                   added_snapshots=int(overlap._merge.eq('right_only').sum()),
                   removed_snapshots=int(overlap._merge.eq('left_only').sum()), changed_shared_labels=changed,
                   new_patients=int((~new.PATIENT_ID.isin(membership)).groupby(new.PATIENT_ID).any().sum()),
                   client_statement='approximately 2.5x positive class; measured, not assumed',
                   split_policy='inherit all known patients; new IDs SHA256(seed:patient) 70/15/15',
                   seed=seed, cohort_sha256=inputs.frame_hash(new),
                   prior_membership_sha256=inputs.stable_hash(membership))
    receipt['population_by_split'] = [dict(universe=label, split=split, snapshots=len(part),
        patients=int(part.PATIENT_ID.nunique()), positives=int(part.RESP.sum()),
        positive_rate=float(part.RESP.mean()), first_date=str(part.END_DT.min().date()),
        last_date=str(part.END_DT.max().date()))
        for label, population in [('old',prior),('new',new)] for split,part in population.groupby('SPLIT')]
    receipt['snapshot_changes_by_split'] = [dict(split=split,
        added=int((overlap._merge.eq('right_only') & overlap.SPLIT_new.eq(split)).sum()),
        removed=int((overlap._merge.eq('left_only') & overlap.SPLIT_old.eq(split)).sum()),
        added_positive_snapshots=int((overlap._merge.eq('right_only') & overlap.SPLIT_new.eq(split) & overlap.RESP_new.eq(1)).sum()))
        for split in ('train','validation','test')]
    receipt['snapshot_multiplicity_by_patient_class'] = [dict(universe=label,patient_class=int(cls),
        patients=len(part),mean_snapshots=float(part['size'].mean()),max_snapshots=int(part['size'].max()))
        for label,population in [('old',prior),('new',new)]
        for cls,part in population.groupby('PATIENT_ID').RESP.agg(['max','size']).groupby('max')]
    return new, receipt


def load_client_counts(spark, read_query, read_sf, config):
    """Fresh raw counts with inherited 1028-column dictionary and patient splits."""
    name = resolve_universe(read_query, config.get('table'))
    prior = frame(read_sf('PATIENT_SPLIT'))
    new = frame(read_query('SELECT PATIENT_ID,END_DT,RESP FROM ' + name))
    metadata, audit = align_universe(new, prior, config.get('split_seed', 42))
    mapping = frame(read_sf('FEATURE_MAP')).sort_values('FEATURE_INDEX').reset_index(drop=True)
    if (len(mapping) != 1028 or mapping.FEATURE_INDEX.tolist() != list(range(1028))
            or mapping.FEATURE_COLUMN.tolist() != [f'F{i:04d}' for i in range(1028)]
            or not mapping.FEATURE_NAME.is_unique):
        raise ValueError("Inherited count dictionary must contain ordered 1028 unique features")
    mapping['FAMILY'] = mapping.FEATURE_NAME.str.split('__').str[0]
    if set(mapping.FAMILY) != {'DX', 'PX', 'RX'}:
        raise ValueError("Count dictionary naming differs from recovered FAMILY__category contract")
    sources = dict(config.get('sources', {}), cohort=name)
    contract = fresh_counts.resolve_source_contract(read_query, sources=sources,
        common_scope=config.get('common_scope'), family_column=config.get('family_column'))
    built = fresh_counts.build_source_frames(spark, read_query, contract,
        reference_map=mapping, prepared_metadata=metadata,
        max_missing_feature_fraction=config.get('max_missing_feature_fraction', .10))
    try:
        counts = count_loader.load_count_inputs(lambda suffix: built[suffix], name, 'fresh_client_universe_v1')
        counts['contract']['client_universe_audit'] = audit
        counts['contract']['fresh_source_manifest'] = built['manifest']
        counts['contract']['inherited_dictionary_provenance'] = 'Names/order retained from previous FEATURE_MAP; historical vocabulary fitting scope not independently established'
        counts['contract']['historical_count_value_equivalence'] = 'Not claimed: new raw aggregation and current LATEST sources; prior tensor values not reused'
        counts['contract']['provenance_status'] = 'Fresh aggregation for client universe; current references, historical availability unresolved'
        return counts
    finally:
        built['cached_aggregates'].unpersist(blocking=False)


def check_activity_alignment(data):
    """Medical count activity requires raw medical AutoML coverage at step zero.

    RX-only disagreements can reflect final-status filters, so are reported only.
    This is a consistency check, not an enrollment assertion or arbitrary floor.
    """
    names = data['count_contract']['feature_names']
    ix = [i for i, n in enumerate(names) if n.startswith(('DX__', 'PX__'))]
    active = np.any(data['X_count'][:, :, ix] > 0, axis=(1, 2)) if ix else np.zeros(len(data['y']), bool)
    missing = ~data['summary_mask'][:, 0]
    if np.any(active & missing):
        raise ValueError("Medical count activity but no current AutoML event coverage; inspect patient keys/source alignment")
    return dict(medical_active_snapshots=int(active.sum()), current_automl_observed=int((~missing).sum()),
                snapshots=len(missing), medical_coverage_mismatches=int((active & missing).sum()))
