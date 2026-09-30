"""Warehouse helpers embedded verbatim in the self-contained feature notebook.

No warehouse connection is opened when this file is imported. Read functions are
injected so the contracts and SQL can be checked without private data.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def qi(name):
    require(isinstance(name, str) and bool(name) and '\x00' not in name,
            'Invalid SQL identifier.')
    return '"' + name.replace('"', '""') + '"'


def qtable(name):
    parts = name.split('.')
    require(len(parts) == 3 and all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', p) for p in parts),
            'Use a fully qualified DATABASE.SCHEMA.TABLE name.')
    return '.'.join(qi(p) for p in parts)


def literal(value):
    require(isinstance(value, str), 'SQL category literals must be strings.')
    return "'" + value.replace("'", "''") + "'"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(',', ':')).encode()).hexdigest()


def metadata_contract(snapshots, split, expected=None):
    def clean(frame, with_split=False):
        fields = ['PATIENT_ID', 'END_DT', 'RESP'] + (['SPLIT'] if with_split else [])
        out = frame[fields].copy()
        require(not out.empty and not out.isna().any().any(), 'Null/empty snapshot metadata.')
        require(out.PATIENT_ID.map(lambda v: isinstance(v, str) and bool(v)).all(),
                'PATIENT_ID must retain nonempty string identifiers.')
        dates = pd.to_datetime(out.END_DT, errors='raise')
        require(dates.dt.tz is None and dates.eq(dates.dt.normalize()).all(),
                'Snapshot dates must be dates without time zones or intraday times.')
        out['END_DT'] = dates.dt.strftime('%Y-%m-%d')
        require(out.RESP.isin([0, 1]).all(), 'RESP must be binary before conversion.')
        out['RESP'] = out.RESP.astype(int)
        require(not out.duplicated(['PATIENT_ID', 'END_DT']).any(), 'Duplicate snapshot keys.')
        return out.sort_values(['PATIENT_ID', 'END_DT']).reset_index(drop=True)
    source, frozen = clean(snapshots), clean(split, True)
    require(source.equals(frozen[['PATIENT_ID', 'END_DT', 'RESP']]),
            'Saved split and snapshot population/labels do not match exactly.')
    require(set(frozen.SPLIT) == {'train', 'validation', 'test'}, 'Unexpected split labels.')
    require(frozen.groupby('PATIENT_ID').SPLIT.nunique().max() == 1, 'Patient split leakage.')
    if expected:
        observed = (len(source), source.PATIENT_ID.nunique(), int(source.RESP.sum()))
        require(observed == tuple(expected), f'Frozen population changed: {observed}.')
    # One independent analysis row per patient. This is not the all-snapshot evaluation population.
    train = (frozen.loc[frozen.SPLIT.eq('train')].sort_values(['PATIENT_ID', 'END_DT'])
             .drop_duplicates('PATIENT_ID', keep='last').reset_index(drop=True))
    require(set(train.RESP) == {0, 1}, 'Latest TRAIN snapshots must contain both classes.')
    return frozen, train


def feature_map_contract(mapping, expected_groups=None):
    ordered = mapping.sort_values('FEATURE_INDEX').reset_index(drop=True)
    require(ordered.FEATURE_INDEX.tolist() == list(range(len(ordered))), 'Invalid feature indices.')
    names, aliases = ordered.FEATURE_NAME.tolist(), ordered.FEATURE_COLUMN.tolist()
    require(bool(names) and len(set(names)) == len(names), 'Empty or duplicate feature names.')
    require(all(isinstance(n, str) and re.match(r'^(RX|DX|PX)__.+', n) for n in names),
            'Unexpected original claim category name.')
    require(aliases == [f'F{i:04d}' for i in range(len(names))], 'Feature aliases/order changed.')
    groups = pd.Series([n.split('__', 1)[0] for n in names]).value_counts().to_dict()
    if expected_groups:
        require(groups == expected_groups, f'Original vocabulary changed: {groups}.')
    return names, aliases


def train_cte(split_table):
    # This SQL must agree with metadata_contract: latest TRAIN snapshot per patient.
    return f'''analysis_keys AS (
      SELECT PATIENT_ID, END_DT, RESP FROM {qtable(split_table)}
      WHERE SPLIT = 'train'
      QUALIFY ROW_NUMBER() OVER (PARTITION BY PATIENT_ID ORDER BY END_DT DESC) = 1
    )'''


def inventory_query(database, schemas):
    require(bool(schemas), 'Configure at least one schema.')
    return f'''SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME,
      DATA_TYPE, ORDINAL_POSITION, IS_NULLABLE
      FROM {qi(database)}.INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA IN ({', '.join(literal(s) for s in schemas)})
      ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION'''


def claim_query(tensor_table, split_table, aliases, windows=(12,)):
    require(bool(windows) and len(set(windows)) == len(windows)
            and all(isinstance(w, int) and 1 <= w <= 12 for w in windows), 'Invalid windows.')
    aggregates = []
    for w in windows:
        for a in aliases:
            aggregates.append(f'SUM(IFF(t.TIME_STEP < {w}, t.{qi(a)}, 0)) AS {qi(a + "_M" + str(w))}')
    invalid = ' OR '.join(f't.{qi(a)} IS NULL OR t.{qi(a)} < 0' for a in aliases)
    return f'''WITH {train_cte(split_table)}
    SELECT k.PATIENT_ID, k.END_DT, k.RESP,
      COUNT(t.TIME_STEP) AS N_ROWS, COUNT(DISTINCT t.TIME_STEP) AS N_STEPS,
      MIN(t.TIME_STEP) AS MIN_STEP, MAX(t.TIME_STEP) AS MAX_STEP,
      SUM(IFF(t.TIME_STEP != FLOOR(t.TIME_STEP), 1, 0)) AS BAD_STEPS,
      SUM(IFF(t.RESP IS NULL OR t.RESP != k.RESP, 1, 0)) AS BAD_LABELS,
      SUM(IFF({invalid}, 1, 0)) AS BAD_VALUES,
      {', '.join(aggregates)}
    FROM analysis_keys k LEFT JOIN {qtable(tensor_table)} t
      ON t.PATIENT_ID = k.PATIENT_ID AND t.END_DT = k.END_DT
    GROUP BY k.PATIENT_ID, k.END_DT, k.RESP
    ORDER BY k.PATIENT_ID, k.END_DT'''


def align_rows(frame, train):
    frame = frame.copy()
    require(frame.PATIENT_ID.map(lambda v: isinstance(v, str)).all(), 'Source patient IDs changed type.')
    dates = pd.to_datetime(frame.END_DT, errors='raise')
    require(dates.dt.tz is None and dates.eq(dates.dt.normalize()).all(), 'Invalid source snapshot date.')
    frame['END_DT'] = dates.dt.strftime('%Y-%m-%d')
    require(not frame.duplicated(['PATIENT_ID', 'END_DT']).any(), 'Source multiplied snapshot rows.')
    require(len(frame) == len(train), 'Source returned extra or missing snapshot rows.')
    keys = train[['PATIENT_ID', 'END_DT']].merge(frame, how='left', on=['PATIENT_ID', 'END_DT'],
                                               validate='one_to_one', indicator=True)
    require(keys['_merge'].eq('both').all(), 'Source did not align with all analysis keys.')
    return keys.drop(columns='_merge')


def assemble_claim_features(frame, train, names, aliases, tensor_table, windows=(12,)):
    data = align_rows(frame, train)
    require(np.array_equal(data.RESP.to_numpy(), train.RESP.to_numpy()), 'Claim labels changed.')
    require(data.N_ROWS.eq(12).all() and data.N_STEPS.eq(12).all()
            and data.MIN_STEP.eq(0).all() and data.MAX_STEP.eq(11).all()
            and data.BAD_STEPS.eq(0).all(), 'Incomplete/duplicate/non-integer monthly rows.')
    require(data.BAD_LABELS.eq(0).all() and data.BAD_VALUES.eq(0).all(),
            'Monthly labels, nulls or negative counts violate the source contract.')
    values, lineage = {}, []
    for w in windows:
        for name, alias in zip(names, aliases):
            feature = f'{name}__SUM_M{w:02d}'
            values[feature] = pd.to_numeric(data[f'{alias}_M{w}'], errors='raise').to_numpy(dtype=float)
            lineage.append({'FEATURE': feature, 'BASE_FEATURE': name, 'SOURCE_TABLE': tensor_table,
                            'SOURCE_COLUMN': alias, 'FAMILY': name.split('__', 1)[0],
                            'AGGREGATION': 'SUM', 'WINDOW_MONTHS': w, 'MISSING_RULE': 'complete saved monthly counts',
                            'LINEAGE_STATUS': 'SAVED_COUNTS_UPSTREAM_SQL_NOT_RECOVERED',
                            'POINT_IN_TIME_STATUS': 'event/ingestion cutoff requires original build audit'})
    X = pd.DataFrame(values)
    require(np.isfinite(X.to_numpy()).all() and X.ge(0).all().all(), 'Nonfinite/negative claim totals.')
    return X, pd.DataFrame(lineage)


def validate_extra_contract(contract, inventory):
    required = {'table', 'kind', 'patient_column', 'date_column', 'review_note', 'availability_column'}
    require(required.issubset(contract), 'Extra source is missing its explicit join/time/lineage contract.')
    require(contract['kind'] in {'snapshot_numeric', 'event_categories'}, 'Unknown adapter.')
    require(isinstance(contract['review_note'], str) and len(contract['review_note'].strip()) >= 20,
            'Record the reviewed source grain, lookback, counting and encoding evidence.')
    table = contract['table']
    qtable(table)
    require(not re.search(r'(SHAP|SCORED|PREDICT|FINAL_MODEL|FEATURES_SUMMARY|MODEL_TYPE|ATOPEN|SKIPGRAM)',
                          table.upper()), 'Model outputs or known unresolved vintages cannot be candidate inputs.')
    require(contract.get('uses_resp_encoding') is False, 'Source must explicitly rule out RESP-fitted encoding.')
    require(contract.get('available_at_cutoff_verified') is True, 'Verify source availability before the cutoff.')
    available = contract['availability_column']
    require(bool(available) or contract.get('historical_snapshot_certified') is True,
            'Use an availability timestamp, or explicitly certify historical snapshot availability.')
    if contract['kind'] == 'event_categories':
        require(bool(available), 'Event sources require a record-availability timestamp.')
        require(contract.get('category_column') and contract.get('event_id_columns'),
                'Event source requires category and stable event-key columns.')
        require(contract.get('no_event_means_zero_verified') is True,
                'Event zero-fill requires reviewed observation/coverage semantics.')
    else:
        require(bool(contract.get('columns')), 'List reviewed numeric snapshot columns explicitly.')
        require(not contract.get('equals_filters'),
                'Snapshot equals_filters are unsupported; use an audited unique snapshot view.')
        require(not any(re.search(r'(^|_)(RESP|TARGET|LABEL|PREDICTION|SCORE|RND|NPI|PATIENT_ID)($|_)', c.upper())
                        for c in contract['columns']), 'Identifier/outcome/score candidate blocked.')
    catalog = inventory.assign(FQN=inventory.TABLE_CATALOG + '.' + inventory.TABLE_SCHEMA + '.' + inventory.TABLE_NAME)
    table_cols = catalog.loc[catalog.FQN.eq(table)]
    columns = set(table_cols.COLUMN_NAME)
    needed = [contract['patient_column'], contract['date_column']]
    if available:
        needed.append(available)
    needed += contract.get('columns', []) + contract.get('event_id_columns', [])
    if contract.get('category_column'):
        needed.append(contract['category_column'])
    needed += list(contract.get('equals_filters', {}))
    require(set(needed).issubset(columns), 'Configured columns not in the accessible inventory: ' + repr(set(needed)-columns))
    # Numeric snapshot adapter preserves raw numeric values; arbitrary text is not coerced to zero.
    if contract['kind'] == 'snapshot_numeric':
        numeric = r'^(NUMBER|DECIMAL|NUMERIC|FLOAT|DOUBLE|REAL|INT|BIGINT|SMALLINT|BOOLEAN)'
        dtypes = table_cols.set_index('COLUMN_NAME').DATA_TYPE
        require(all(re.match(numeric, str(dtypes[c])) for c in contract['columns']),
                'Snapshot adapter accepts numeric/boolean columns only.')
    return contract


def availability_predicate(c, alias='s'):
    if not c['availability_column']:
        return 'TRUE'
    # Snapshot cutoff is a calendar date, inclusive through that day's end.
    return f'{alias}.{qi(c["availability_column"])} < DATEADD(day, 1, k.END_DT)'


def snapshot_extra_query(c, split_table):
    selected = ', '.join(f's.{qi(column)} AS {qi("V" + str(i))}' for i, column in enumerate(c['columns']))
    return f'''WITH {train_cte(split_table)}
    SELECT k.PATIENT_ID, k.END_DT, s.{qi(c['patient_column'])} AS SOURCE_MATCH, {selected}
    FROM analysis_keys k LEFT JOIN {qtable(c['table'])} s
    ON s.{qi(c['patient_column'])} = k.PATIENT_ID
      AND s.{qi(c['date_column'])} = k.END_DT AND {availability_predicate(c)}
    ORDER BY k.PATIENT_ID, k.END_DT'''


def event_extra_query(c, split_table, window_months=12):
    require(isinstance(window_months, int) and 1 <= window_months <= 12, 'Invalid event window.')
    ids = ', '.join(f's.{qi(v)} AS {qi("E" + str(i))}' for i, v in enumerate(c['event_id_columns']))
    nonnull_ids = ' AND '.join(f's.{qi(v)} IS NOT NULL' for v in c['event_id_columns'])
    conditions = []
    for col, vals in c.get('equals_filters', {}).items():
        require(isinstance(vals, list) and bool(vals), 'Event filters require nonempty value lists.')
        conditions.append(f'CAST(s.{qi(col)} AS VARCHAR) IN ({", ".join(literal(str(v)) for v in vals)})')
    filters = ' AND '.join(conditions) or 'TRUE'
    # DISTINCT is on the explicitly reviewed event identity *and category*, not an arbitrary row.
    return f'''WITH {train_cte(split_table)}, observations AS (
    SELECT DISTINCT k.PATIENT_ID, k.END_DT,
      CAST(s.{qi(c['category_column'])} AS VARCHAR) AS CATEGORY, {ids},
      IFF({nonnull_ids}, 0, 1) AS BAD_EVENT_KEY
    FROM analysis_keys k JOIN {qtable(c['table'])} s
      ON s.{qi(c['patient_column'])} = k.PATIENT_ID
      AND s.{qi(c['date_column'])} >= DATEADD(month, -{window_months-1}, DATE_TRUNC('month', k.END_DT))
      AND s.{qi(c['date_column'])} < DATEADD(day, 1, k.END_DT)
      AND {availability_predicate(c)}
    WHERE {filters})
    SELECT PATIENT_ID, END_DT, CATEGORY, COUNT(*) AS VALUE,
      SUM(BAD_EVENT_KEY) AS BAD_KEYS FROM observations GROUP BY 1,2,3'''


def gather_extras(read_query, contracts, inventory, train, split_table, max_event_categories=2000):
    blocks, rows, audits = [], [], []
    for position, raw in enumerate(contracts):
        c = validate_extra_contract(raw, inventory)
        prefix = 'EXTRA_' + hashlib.sha256(c['table'].encode()).hexdigest()[:12].upper()
        if c['kind'] == 'snapshot_numeric':
            frame = align_rows(read_query(snapshot_extra_query(c, split_table)), train)
            block = {}
            for i, column in enumerate(c['columns']):
                name = prefix + '__' + column
                block[name] = pd.to_numeric(frame['V' + str(i)], errors='raise').to_numpy(dtype=float)
                rows.append({'FEATURE': name, 'BASE_FEATURE': column, 'SOURCE_TABLE': c['table'],
                             'SOURCE_COLUMN': column, 'FAMILY': 'SNAPSHOT', 'AGGREGATION': 'EXACT_KEY',
                             'WINDOW_MONTHS': None, 'MISSING_RULE': 'retain NaN; train-fold imputation',
                             'LINEAGE_STATUS': c['review_note'], 'POINT_IN_TIME_STATUS': 'reviewed contract'})
            blocks.append(pd.DataFrame(block))
            audits.append({'TABLE': c['table'], 'CANDIDATES': len(block),
                           'MATCHED_PATIENTS': int(frame.SOURCE_MATCH.notna().sum())})
        else:
            frame = read_query(event_extra_query(c, split_table))
            require(not frame.empty, 'Reviewed event source returned no analysis events: ' + c['table'])
            require(frame.BAD_KEYS.eq(0).all(), 'Null event identity: cannot deduplicate safely.')
            require(frame.CATEGORY.notna().all(), 'Null event category requires an explicit mapping policy.')
            counts = pd.to_numeric(frame.VALUE, errors='raise').to_numpy(dtype=float)
            require(np.isfinite(counts).all() and (counts >= 0).all()
                    and np.equal(counts, np.floor(counts)).all(),
                    'Event counts must be finite nonnegative integers, never missing.')
            frame['END_DT'] = pd.to_datetime(frame.END_DT).dt.strftime('%Y-%m-%d')
            require(not frame.duplicated(['PATIENT_ID', 'END_DT', 'CATEGORY']).any(), 'Duplicate event aggregates.')
            keys = set(map(tuple, train[['PATIENT_ID', 'END_DT']].to_numpy()))
            require(set(map(tuple, frame[['PATIENT_ID', 'END_DT']].to_numpy())).issubset(keys), 'Extra source keys.')
            categories = sorted(frame.CATEGORY.unique())
            require(len(categories) <= max_event_categories,
                    'Category vocabulary exceeds limit; supply a reviewed category mapping rather than truncate.')
            wide = frame.pivot(index=['PATIENT_ID', 'END_DT'], columns='CATEGORY', values='VALUE')
            wide = wide.reindex(pd.MultiIndex.from_frame(train[['PATIENT_ID', 'END_DT']])).fillna(0)
            block = {}
            for category in categories:
                name = prefix + '__CAT_' + hashlib.sha256(category.encode()).hexdigest()[:16].upper() + '__SUM_M12'
                block[name] = wide[category].to_numpy(dtype=float)
                rows.append({'FEATURE': name, 'BASE_FEATURE': category, 'SOURCE_TABLE': c['table'],
                             'SOURCE_COLUMN': c['category_column'], 'FAMILY': 'EVENT_CATEGORY',
                             'AGGREGATION': 'COUNT_DISTINCT_REVIEWED_EVENT_ID', 'WINDOW_MONTHS': 12,
                             'MISSING_RULE': 'zero if no events; source coverage explicitly certified',
                             'LINEAGE_STATUS': c['review_note'], 'POINT_IN_TIME_STATUS': 'event and availability dates <= cutoff'})
            blocks.append(pd.DataFrame(block))
            audits.append({'TABLE': c['table'], 'CANDIDATES': len(block),
                           'MATCHED_PATIENTS': int(frame.PATIENT_ID.nunique())})
    X = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=train.index)
    require(not X.columns.duplicated().any(), 'Duplicate candidate identity in extra source contracts.')
    require(not np.isinf(X.to_numpy(dtype=float)).any(), 'Infinite extra-source values.')
    return X, pd.DataFrame(rows), pd.DataFrame(audits)
