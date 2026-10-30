"""Bound candidate eligibility, reconstruct forward outcomes, freeze the measured population."""
import copy
import hashlib
import json
import re

import pandas as pd


KEYS = ['PATIENT_ID', 'END_DT']


def identifier(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Z_][A-Z0-9_$]*(\.[A-Z_][A-Z0-9_$]*){2}', name):
        raise ValueError('Use a fully qualified uppercase source identifier')
    return name


def queries(plan):
    candidate, basket, medical = [identifier(plan[k]) for k in ('candidate', 'basket', 'medical')]
    months = plan['months']
    if type(months) is not int or not 1 <= months <= 120:
        raise ValueError('Snapshot months must be an explicit integer from 1 to 120')
    # This is an observed date-range bound, not proof of completeness or enrollment.
    common = f"""WITH source_end AS (
 SELECT LEAST((SELECT MAX(FILL_DATE)::DATE FROM {basket}),
              (SELECT MAX(SERVICE_DATE)::DATE FROM {medical})) AS OBSERVED_THROUGH
), bounds AS (
 SELECT LEAST(MAX(c.END_DT)::DATE, DATEADD(DAY,-90,s.OBSERVED_THROUGH)) AS UPPER_END,
        s.OBSERVED_THROUGH
 FROM {candidate} c CROSS JOIN source_end s GROUP BY s.OBSERVED_THROUGH
), bounded AS (
 SELECT c.PATIENT_ID, c.END_DT::DATE AS END_DT, c.RESP AS ORIGINAL_RESP
 FROM {candidate} c CROSS JOIN bounds b
 WHERE c.END_DT > DATEADD(MONTH,-{months},b.UPPER_END) AND c.END_DT <= b.UPPER_END
), patients AS (SELECT DISTINCT PATIENT_ID FROM bounded), first_event AS (
 SELECT b.PATIENT_ID, MIN(b.FILL_DATE)::DATE AS FIRST_EVENT
 FROM {basket} b JOIN patients p ON b.PATIENT_ID=p.PATIENT_ID
 WHERE b.IS_GENERIC=FALSE GROUP BY b.PATIENT_ID
), labeled AS (
 SELECT c.*, f.FIRST_EVENT,
        IFF(f.FIRST_EVENT > c.END_DT AND f.FIRST_EVENT <= DATEADD(DAY,90,c.END_DT),1,0) AS RESP,
        IFF(f.FIRST_EVENT <= c.END_DT,1,0) AS PRIOR_INITIATION
 FROM bounded c LEFT JOIN first_event f ON c.PATIENT_ID=f.PATIENT_ID
), followup AS (
 SELECT DISTINCT c.PATIENT_ID,c.END_DT
 FROM labeled c JOIN {medical} m ON c.PATIENT_ID=m.PATIENT_ID
      AND m.SERVICE_DATE > c.END_DT AND m.SERVICE_DATE <= DATEADD(DAY,90,c.END_DT)
 WHERE c.RESP=0 AND c.PRIOR_INITIATION=0
 AND m.SERVICE_DATE > (SELECT MIN(END_DT) FROM bounded)
 AND m.SERVICE_DATE <= (SELECT DATEADD(DAY,90,UPPER_END) FROM bounds)
), decisions AS (
 SELECT c.*, IFF(o.PATIENT_ID IS NOT NULL,1,0) AS OBSERVED_NEGATIVE,
        IFF(c.PRIOR_INITIATION=0 AND (c.RESP=1 OR o.PATIENT_ID IS NOT NULL),1,0) AS RETAIN
 FROM labeled c LEFT JOIN followup o ON c.PATIENT_ID=o.PATIENT_ID AND c.END_DT=o.END_DT
)"""
    population = common + '\nSELECT * FROM decisions ORDER BY PATIENT_ID,END_DT'
    audit = common + f"""
SELECT (SELECT COUNT(*) FROM {candidate}) AS CANDIDATE_SNAPSHOTS,
 (SELECT COUNT(*) FROM bounded) AS BOUNDED_SNAPSHOTS,
 (SELECT OBSERVED_THROUGH FROM bounds) AS OBSERVED_THROUGH,
 (SELECT UPPER_END FROM bounds) AS UPPER_END,
 (SELECT DATEADD(MONTH,-{months},UPPER_END) FROM bounds) AS LOWER_EXCLUSIVE,
 COUNT(*) AS SNAPSHOTS, COUNT(DISTINCT PATIENT_ID) AS PATIENTS, COALESCE(SUM(RESP),0) AS POSITIVES,
 COUNT(DISTINCT IFF(RESP=1,PATIENT_ID,NULL)) AS POSITIVE_PATIENTS,
 COALESCE(SUM(IFF(ORIGINAL_RESP=0 AND RESP=1,1,0)),0) AS POSITIVES_GAINED,
 COALESCE(SUM(IFF(ORIGINAL_RESP=1 AND RESP=0,1,0)),0) AS POSITIVES_LOST,
 (SELECT COUNT(*) FROM {basket} b JOIN patients p ON b.PATIENT_ID=p.PATIENT_ID WHERE b.IS_GENERIC IS NULL) AS NULL_CLASSIFICATION_ROWS,
 (SELECT COUNT(*) FROM {basket} b JOIN patients p ON b.PATIENT_ID=p.PATIENT_ID WHERE b.IS_GENERIC=FALSE AND b.FILL_DATE IS NULL) AS NULL_ADVANCED_DATE_ROWS,
 (SELECT COUNT(*) FROM decisions WHERE PRIOR_INITIATION=1) AS PRIOR_INITIATION_EXCLUSIONS,
 (SELECT COUNT(*) FROM decisions WHERE PRIOR_INITIATION=0 AND RESP=0 AND OBSERVED_NEGATIVE=0) AS UNOBSERVED_NEGATIVE_EXCLUSIONS
FROM decisions WHERE RETAIN=1"""
    return population, audit


def change_policy(incoming, prior, inp):
    overlap = prior.merge(incoming, on=KEYS, how='outer', suffixes=('_old','_new'), indicator=True)
    removed = overlap.loc[overlap._merge.eq('left_only')].sort_values(KEYS)
    changed = overlap.loc[overlap._merge.eq('both') & overlap.RESP_old.ne(overlap.RESP_new)].sort_values(KEYS)
    return dict(removed_keys_sha256=inp.stable_hash([[r.PATIENT_ID,r.END_DT.strftime('%Y-%m-%d')]
                for r in removed.itertuples(index=False)]),
                changed_labels_sha256=inp.stable_hash([[r.PATIENT_ID,r.END_DT.strftime('%Y-%m-%d'),int(r.RESP_old),int(r.RESP_new)]
                for r in changed.itertuples(index=False)]),
                reason='Authorized bounded snapshot window, known missing historical keys, first-event days1..90 and prior-initiation/negative-activity rules')


def expected_population(metadata):
    return dict(snapshots=len(metadata), patients=int(metadata.PATIENT_ID.nunique()), positives=int(metadata.RESP.sum()),
        per_split={s:dict(snapshots=len(g),patients=int(g.PATIENT_ID.nunique()),positives=int(g.RESP.sum()))
                   for s,g in metadata.groupby('SPLIT')})


def validate_decisions(decisions, candidate, row):
    d = decisions.copy()
    d['END_DT'] = pd.to_datetime(d.END_DT)
    d['FIRST_EVENT'] = pd.to_datetime(d.FIRST_EVENT)
    if d.duplicated(KEYS).any() or d[KEYS+['RESP','ORIGINAL_RESP','PRIOR_INITIATION','OBSERVED_NEGATIVE','RETAIN']].isna().any().any():
        raise ValueError('Invalid or duplicate bounded decision rows')
    lower, upper = pd.Timestamp(row.LOWER_EXCLUSIVE), pd.Timestamp(row.UPPER_END)
    expected = candidate.loc[candidate.END_DT.gt(lower) & candidate.END_DT.le(upper), KEYS+['RESP']]
    joined = expected.merge(d, on=KEYS, how='outer', indicator=True, suffixes=('_candidate',''))
    if not joined._merge.eq('both').all() or not joined.RESP_candidate.eq(joined.ORIGINAL_RESP).all():
        raise ValueError('Bounded decisions differ from candidate snapshot keys or original labels')
    positives = d.FIRST_EVENT.gt(d.END_DT) & d.FIRST_EVENT.le(d.END_DT+pd.Timedelta(days=90))
    prior = d.FIRST_EVENT.le(d.END_DT)
    if not d.RESP.eq(positives.astype(int)).all() or not d.PRIOR_INITIATION.eq(prior.astype(int)).all():
        raise ValueError('First-initiation label or prior-treatment boundary differs from days1..90 contract')
    retain = ~prior & (positives | d.OBSERVED_NEGATIVE.eq(1))
    if not d.RETAIN.eq(retain.astype(int)).all() or not d.OBSERVED_NEGATIVE.isin([0,1]).all():
        raise ValueError('Outcome observability retention rule differs')
    return d


def prior_changes(prior, candidate, decisions, incoming):
    p = prior.merge(candidate[KEYS], on=KEYS, how='left', indicator='candidate_match')
    p = p.merge(decisions[KEYS+['PRIOR_INITIATION','OBSERVED_NEGATIVE','RETAIN']], on=KEYS, how='left')
    p = p.merge(incoming[KEYS+['RESP']], on=KEYS, how='left', suffixes=('_old','_new'))
    p['REASON'] = 'retained'
    p.loc[p.candidate_match.eq('left_only'),'REASON'] = 'missing_from_candidate'
    p.loc[p.candidate_match.eq('both') & p.RETAIN.isna(),'REASON'] = 'outside_snapshot_or_observed_horizon_window'
    p.loc[p.PRIOR_INITIATION.eq(1),'REASON'] = 'prior_initiation'
    p.loc[p.PRIOR_INITIATION.eq(0) & p.RETAIN.eq(0),'REASON'] = 'unobserved_negative'
    if not p.RESP_new.isna().eq(p.REASON.ne('retained')).all():
        raise ValueError('An old snapshot removal has no authorized reconstruction reason')
    summary = []
    for split,g in p.groupby('SPLIT'):
        for reason,part in g.groupby('REASON'):
            summary.append(dict(split=split,reason=reason,snapshots=len(part),
                labels_0_to_1=int((part.RESP_old.eq(0)&part.RESP_new.eq(1)).sum()),
                labels_1_to_0=int((part.RESP_old.eq(1)&part.RESP_new.eq(0)).sum())))
    return summary


def bind(read_query, read_sf, write_population, config, implementation, prefix, inp, pop, wf, wh, *, create=False):
    """Restore the same population contract in every stage; create only in stage 01."""
    base_hash = wf.digest(config)
    audit_table = prefix+'_COHORT_AUDIT'
    if wh.table_exists(audit_table):
        saved = json.loads(wh.read_artifacts(audit_table, {'cohort.json'})['cohort.json'])
        if saved['base_configuration_hash'] != base_hash or saved['implementation_hash'] != implementation:
            raise ValueError('Frozen cohort belongs to different configuration or implementation; use a new run identifier')
        config.clear(); config.update(saved['bound_configuration'])
        return saved
    if not create:
        raise ValueError('Run notebook 01 population binding before this stage')
    plan = config['cohort_plan']
    query, audit_query = queries(plan)
    for table, required in ((plan['candidate'], {'PATIENT_ID','END_DT','RESP'}),
                            (plan['basket'], {'PATIENT_ID','FILL_DATE','IS_GENERIC'}),
                            (plan['medical'], {'PATIENT_ID','SERVICE_DATE'})):
        columns = {str(c).upper() for c in read_query('SELECT * FROM '+identifier(table)+' WHERE 1=0').columns}
        if not required <= columns:
            raise ValueError('Required source columns missing in '+table+': '+repr(sorted(required-columns)))
    prior = inp.normalize_snapshots(pop.frame(read_sf('PATIENT_SPLIT'))).sort_values(KEYS).reset_index(drop=True)
    candidate = pop.frame(read_query('SELECT PATIENT_ID,END_DT,RESP FROM '+plan['candidate']))
    candidate['SPLIT'] = 'train'
    candidate = inp.normalize_snapshots(candidate).sort_values(KEYS).reset_index(drop=True)
    absent = prior.merge(candidate[KEYS], on=KEYS, how='left', indicator=True)
    missing = absent.loc[absent._merge.eq('left_only')]
    if len(missing) > plan['max_missing_original_snapshots']:
        raise ValueError('Unfiltered candidate exceeds the authorized 16 missing historical keys: '+str(len(missing)))
    candidate_hash = inp.frame_hash(candidate)
    source_audit = pop.frame(read_query(audit_query))
    if len(source_audit) != 1 or source_audit[['UPPER_END','OBSERVED_THROUGH']].isna().any().any():
        raise ValueError('Source date range cannot support a bounded 90-day outcome')
    row = source_audit.iloc[0]
    if int(row.NULL_CLASSIFICATION_ROWS) or int(row.NULL_ADVANCED_DATE_ROWS):
        raise ValueError('Basket has unclassified therapy flags or missing advanced-fill dates; inspect aggregate audit before labeling')
    decisions = validate_decisions(pop.frame(read_query(query)), candidate, row)
    incoming = decisions.loc[decisions.RETAIN.eq(1), KEYS+['RESP']].copy()
    incoming['SPLIT'] = 'train'
    incoming = inp.normalize_snapshots(incoming).sort_values(KEYS).reset_index(drop=True)
    if not len(incoming):
        raise ValueError('No eligible snapshots in the configured window')
    totals = expected_population(incoming)
    for key in ('snapshots','patients','positives'):
        if totals[key] != int(row[key.upper()]):
            raise ValueError('Source aggregate and population extraction disagree: '+key)
    if int(row.CANDIDATE_SNAPSHOTS) != len(candidate):
        raise ValueError('Candidate population changed during audit')
    policy = change_policy(incoming, prior, inp)
    exclusions = prior_changes(prior, candidate, decisions, incoming)
    metadata, audit = pop.align_universe(incoming, prior, config['client_universe'].get('split_seed',42), change_policy=policy)
    expected = expected_population(metadata)
    target = identifier('DSVC_TAKEDA_TA_PRIVATE.DS_ML.'+prefix+'_POPULATION')
    if not wh.table_exists(target.split('.')[-1]):
        write_population(target, metadata)
    restored = pop.frame(read_query('SELECT PATIENT_ID,END_DT,RESP FROM '+target))
    restored, restored_audit = pop.align_universe(restored, prior, config['client_universe'].get('split_seed',42), change_policy=policy)
    pop.assert_expected_population(restored_audit, expected)
    if inp.frame_hash(metadata) != inp.frame_hash(restored):
        raise ValueError('Persisted population differs from the audited keys and labels')
    config['client_universe'].update(table=target,table_name=target.split('.')[-1],expected_population=expected,change_policy=policy)
    config['cohort_evidence'].update(lineage_status='bounded_reconstruction_development', automatic_relabel=True,
        candidate_labels_accepted=False, reconstruction_sha256=hashlib.sha256(query.encode()).hexdigest())
    serial_audit = json.loads(source_audit.to_json(orient='records',date_format='iso'))[0]
    saved = dict(base_configuration_hash=base_hash,implementation_hash=implementation,bound_configuration=copy.deepcopy(config),
        population_audit=audit,source_audit=serial_audit,candidate_metadata_sha256=candidate_hash,
        prior_changes_by_split=exclusions,expected_population_origin='measured_at_binding_reconciled_with_SQL_aggregates',
        test_scope='Inherited TEST patients plus hashed new patients; retrospective with reconstructed labels; disabled by default',
        missing_original_snapshots=len(missing),missing_original_keys_sha256=inp.stable_hash(
            [[r.PATIENT_ID,r.END_DT.strftime('%Y-%m-%d')] for r in missing.sort_values(KEYS).itertuples(index=False)]),
        metadata_sha256=inp.frame_hash(metadata),query=query,audit_query=audit_query,
        original_producer_verified=False,source_completeness_verified=False,
        outcome_note='First event in available designated basket history; source maxima bound calendar horizon only, not completeness or enrollment',
        population_scope='Existing candidate eligibility only; a wider filter cannot create dates or people absent from that candidate')
    wh.save_artifacts(audit_table, {'cohort.json': wf.json_bytes(saved)})
    return saved
