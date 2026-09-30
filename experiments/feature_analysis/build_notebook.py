"""Build the standalone delivery; helper source is embedded, never imported at runtime."""
import hashlib
import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DESTINATION = REPO / 'notebooks' / 'feature analysis' / '01_gather_and_select_features.ipynb'


def build():
    cells = []

    def markdown(text):
        cells.append({'cell_type': 'markdown', 'metadata': {},
                      'id': f'feature-analysis-{len(cells):02d}', 'source': dedent(text).strip() + '\n'})

    def code(text):
        source = dedent(text).strip() + '\n'
        compile(source, 'feature_analysis_cell', 'exec')
        cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
                      'id': f'feature-analysis-{len(cells):02d}', 'source': source})

    components = {name: (HERE / name).read_text(encoding='utf-8')
                  for name in ('source_catalog.py', 'warehouse.py', 'selection.py')}
    implementation_sha = hashlib.sha256('\n'.join(components.values()).encode()).hexdigest()
    markdown('''
        # Gather features and compare three selection strategies

        **Deliverable:** three separate ranked tables—`FILTER`, `WRAPPER`, and `EMBEDDED`—with selected flags,
        rejected-feature reasons, source lineage, patient support and fold selection stability. This notebook
        contains every runtime helper. No other notebook or repository module is required to run it.

        **Run order:** configuration → accessible-column inventory → frozen TRAIN population → original claims
        features → reviewed additional sources → three selection methods → comparison → three Snowflake tables.
        Each code cell is preceded by an explanation of its purpose and outputs. Use an approved Databricks
        Python environment with the Spark Snowflake connector, pandas, NumPy, SciPy, scikit-learn and matplotlib.
        Provide your existing private Snowflake connection options in cell 1; no credentials belong in this file.

        ## What was recovered

        | Input | Recovered contract |
        |---|---|
        | Original automated features | 42 `RX__`, 490 `DX__`, 496 `PX__` = 1,028 claim-category count features |
        | Monthly representation | Twelve complete rows per snapshot; timestep 0 newest, 11 oldest; nonnegative raw counts |
        | Saved source family | `DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_DL_POC_` plus `FEATURE_MAP`, `TENSOR_MONTHLY`, `SNAPSHOTS`, `PATIENT_SPLIT` |
        | Frozen population | 23,151 snapshots, 12,447 patients, 1,345 positive snapshot labels |
        | Analysis population here | One latest snapshot per **TRAIN** patient; no holdout feature values or scores are read |
        | Feature representation here | Default: sum each original count over the twelve saved monthly bins, giving 1,028 tabular candidates |
        | Source extension | Inventory every column visible in the configured schemas; load additional values only through explicit reviewed source contracts |

        The original raw-code mappings, deduplication and vocabulary-threshold SQL were **not recovered**. We do
        not regenerate or claim to reverse-engineer them. The saved counts are the starting point. Original
        event/ingestion cutoffs and RESP construction remain upstream audit limitations. This notebook reads
        source inputs and the fixed split, never old model checkpoints, predictions or performance results.

        The 49 V63 business features are a **reference definition list**, not automatically a subset of these
        1,028 categories. Existing `MODEL_DATA` encoding may depend on RESP; its values are excluded by default.
        Provider metrics with a future window, undated patterns and model outputs are not silently added.

        **Interpretation:** these are retrospective feature-screening experiments. Repeatedly tuning against
        these cross-validation results makes them development evidence. Independent later-period confirmation
        is needed before claiming a generalization improvement; selection alone cannot prove overfitting is fixed.
    ''')
    markdown('''
        ## 1. Configure connection, discovery scope and source contracts

        **Why:** source scope, feature windows and selection budgets must be explicit before looking at scores.
        The original saved 1,028-feature route runs without additional source contracts. Discovery covers all
        tables/views visible to the role in the listed schemas, not every account/database automatically.
        Add approved databases/schemas to `DISCOVERY_SCOPES` when needed. Metadata visibility does not guarantee
        permission to read values. A missing scope is reported rather than described as empty.

        `SUM_WINDOWS=(12,)` ranks annual totals. Setting `(3, 6, 12)` creates 3,084 count-window candidates and
        distinguishes recent activity, at higher runtime/memory cost. Windows are saved calendar bins ending at
        END_DT, not exactly 90/180/365 days. The newest bin may be partial. We do not sum already-log-transformed
        values. Median imputation, optional log1p and scaling are fitted later inside each training fold.

        **Adding more sources:** first run the inventory cells. For each source you can substantiate, add a
        contract below and rerun from this cell. `snapshot_numeric` needs unique exact patient/date rows, an
        explicit numeric column list and audited availability/encoding. `event_categories` needs event date,
        ingestion/availability timestamp, category, stable event IDs, transaction filters and a verified zero
        interpretation. It counts distinct reviewed event IDs per category over twelve calendar bins. It does
        not guess a medical vocabulary, historical provider join, categorical snapshot encoder or long-value
        pivot. These sources stay visible as pending work in the inventory.

        Example contract structure (replace names with **verified inventory names**; do not enable unchanged):

        ```python
        # snapshot_numeric:
        # {"table": "DATABASE.SCHEMA.REVIEWED_RAW_SNAPSHOT_VIEW", "kind": "snapshot_numeric",
        #  "patient_column": "PATIENT_ID", "date_column": "END_DT", "columns": ["RAW_FEATURE"],
        #  "availability_column": "AVAILABLE_AT", "uses_resp_encoding": False,
        #  "available_at_cutoff_verified": True, "review_note": "Reference to reviewed grain, SQL and cutoff evidence"}
        # event_categories:
        # {"table": "DATABASE.SCHEMA.REVIEWED_EVENTS", "kind": "event_categories",
        #  "patient_column": "PATIENT_ID", "date_column": "EVENT_DATE", "availability_column": "AVAILABLE_AT",
        #  "category_column": "REVIEWED_CATEGORY", "event_id_columns": ["EVENT_ID", "LINE_ID"],
        #  "equals_filters": {"TRANSACTION_STATUS": ["PAID"]}, "uses_resp_encoding": False,
        #  "available_at_cutoff_verified": True, "no_event_means_zero_verified": True,
        #  "review_note": "Reference to reviewed counting, deduplication, observation and availability evidence"}
        ```

        For an exact historical snapshot without an availability column, use `availability_column=None` only
        with `historical_snapshot_certified=True` and documented evidence. Missing snapshot matches remain NaN.
        Enrollment absence must not be relabeled as zero activity. This notebook does not broadcast snapshots
        into a sequence or apply padding.

        **Output:** a fresh run identifier and configuration summary. `WRITE_RESULTS=True` creates three new
        aggregate feature tables at the end. Existing tables are never overwritten. Intermediate patient-level
        feature values remain in the private session and are not displayed or exported.
    ''')
    code('''
        import json
        import hashlib
        import time
        import uuid
        from datetime import datetime, timezone
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import matplotlib.pyplot as plt

        # Supply private connection options here using your normal secret-backed setup.
        # Example variable name only: sf_options = {...}; never save secret values in the notebook.
        connection = globals().get('sf_options_dl_poc', globals().get('sf_options'))
        if not isinstance(connection, dict):
            raise RuntimeError('Provide your existing private sf_options or sf_options_dl_poc dictionary.')
        sf_options_analysis = dict(connection)
        sf_options_analysis.update(sfDatabase='DSVC_TAKEDA_TA_PRIVATE', sfSchema='DS_ML')
        if 'spark' not in globals():
            raise RuntimeError('Run on Databricks with the approved Spark Snowflake connector.')

        SOURCE_ROOT = 'DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63'
        SAVED_PREFIX = SOURCE_ROOT + '_DL_POC'
        TABLES = {name: SAVED_PREFIX + '_' + name for name in
                  ('FEATURE_MAP', 'SNAPSHOTS', 'PATIENT_SPLIT', 'TENSOR_MONTHLY')}
        DISCOVERY_SCOPES = [
            {'database': 'DSVC_TAKEDA_TA_PRIVATE', 'schemas': ['DS_ML', 'DS_ML_PROD']},
            {'database': 'DSVC_TAKEDA_FULLMAP_PLAID_PROD', 'schemas': ['COHORT_1009719']},
        ]
        EXPECTED_POPULATION = (23151, 12447, 1345)
        EXPECTED_GROUPS = {'RX': 42, 'DX': 490, 'PX': 496}
        SUM_WINDOWS = (12,)
        EXTRA_SOURCE_CONTRACTS = []  # Review inventory first; examples are documented above.
        MAX_CANDIDATES = 6000
        MAX_EVENT_CATEGORIES = 2000
        SELECTION_CONFIG = {
            'seed': 42, 'n_splits': 3,
            'min_nonzero_patients': 10, 'min_observed_patients': 20,
            'max_missing_fraction': 0.95,
            'filter_max_features': 100, 'redundancy_abs_spearman': 0.95,
            'wrapper_shortlist': 100, 'wrapper_max_features': 50, 'wrapper_rfe_step': 0.25,
            'embedded_max_features': 100, 'elasticnet_C': 0.1, 'elasticnet_l1_ratio': 0.5,
            'linear_max_iter': 2000, 'tree_estimators': 128,
            'tree_max_depth': 6, 'tree_min_samples_leaf': 15, 'n_jobs': 2,
            'log1p_nonnegative': True,
        }
        RUN_ID = 'R' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6].upper()
        OUTPUT_PREFIX = SOURCE_ROOT + '_FEATURE_ANALYSIS_' + RUN_ID
        WRITE_RESULTS = True

        def read_sql(query):
            return (spark.read.format('snowflake').options(**sf_options_analysis)
                    .option('query', query).load().toPandas())

        print('Run:', RUN_ID)
        print('Versions:', {'numpy': np.__version__, 'pandas': pd.__version__,
                            'scipy': scipy.__version__, 'sklearn': sklearn.__version__})
        print('Additional source contracts:', len(EXTRA_SOURCE_CONTRACTS))
        print('Only TRAIN feature values are loaded; frozen split metadata is audited across all splits.')
    ''')
    markdown('''
        ## 2. Load source inventory and extraction helpers

        **Why:** explicit helpers validate keys, dates, labels, category order and monthly completeness before
        scores can hide data errors. SQL joins preserve exact patient/date identity. Saved count extraction is
        pushed into Snowflake, returning roughly 8,712 rows × 1,028 columns rather than a full monthly tensor.
        Values are summed on the server; no trained model artifact is loaded. All helper code is embedded here.

        **Output:** function definitions only. `known_sources()` contains 32 recovered source leads and the
        limitations of each. An unqualified dated table is resolved from live metadata rather than assigned
        an invented schema. `LATEST` sources do not silently replace the 20260825 vintage.
    ''')
    code(components['source_catalog.py'] + '\n\n' + components['warehouse.py'] +
         f'\n\nIMPLEMENTATION_SHA256 = "{implementation_sha}"\nprint("Source helpers loaded.")')
    markdown('''
        ## 3. Inventory all accessible columns and identify pending source work

        **Why:** this is the broad feature-discovery step. It includes numeric and categorical candidates,
        metadata, model outputs and reference tables, while clearly separating them. A candidate flag means
        **review for inclusion**, not a feature already used in training. IDs, RESP and output scores cannot
        become predictors. Catalog classification is conservative guidance, not an automatic proof of safety.

        **Outputs:** (1) schema discovery status, (2) recovered table leads with live resolutions,
        (3) a summary by table/kind, and (4) full `feature_inventory`. Nonvisible source leads and unsupported
        dimensions/long-form/categorical sources remain explicit. No patient records are shown. If a needed
        source lies elsewhere, add its database/schema to `DISCOVERY_SCOPES` and rerun. When a database is
        accessible but a schema returns no metadata, the result says `NO_VISIBLE_COLUMNS`, not "no features".
    ''')
    code('''
        inventory_parts, discovery_status = [], []
        for scope in DISCOVERY_SCOPES:
            for schema in scope['schemas']:
                try:
                    part = read_sql(inventory_query(scope['database'], [schema]))
                    status = 'DISCOVERED' if len(part) else 'NO_VISIBLE_COLUMNS'
                    inventory_parts.append(part)
                    discovery_status.append({'DATABASE': scope['database'], 'SCHEMA': schema,
                                             'STATUS': status, 'COLUMNS': len(part)})
                except Exception as error:
                    # Exception text may contain connection details; keep the UI aggregate-only.
                    discovery_status.append({'DATABASE': scope['database'], 'SCHEMA': schema,
                                             'STATUS': 'QUERY_FAILED_' + type(error).__name__, 'COLUMNS': 0})
        require(any(len(p) for p in inventory_parts), 'No columns were discovered. Check role and configured scopes.')
        inventory = pd.concat(inventory_parts, ignore_index=True).drop_duplicates(
            ['TABLE_CATALOG', 'TABLE_SCHEMA', 'TABLE_NAME', 'COLUMN_NAME'])
        feature_inventory = classify_inventory(inventory)
        feature_inventory['SOURCE_TABLE'] = (feature_inventory.TABLE_CATALOG + '.' +
                                            feature_inventory.TABLE_SCHEMA + '.' + feature_inventory.TABLE_NAME)
        visible_tables = sorted(feature_inventory.SOURCE_TABLE.unique())
        recovered_sources = known_sources()
        recovered_sources['VISIBLE_MATCHES'] = recovered_sources.source.map(
            lambda source: json.dumps([t for t in visible_tables if t == source or t.endswith('.' + source)]))
        table_inventory = (feature_inventory.groupby(['SOURCE_TABLE', 'kind'], dropna=False)
                           .agg(COLUMNS=('COLUMN_NAME', 'size'), REVIEW_CANDIDATES=('candidate', 'sum')).reset_index())
        display(pd.DataFrame(discovery_status))
        display(recovered_sources)
        display(table_inventory)
        display(feature_inventory)
    ''')
    markdown('''
        ## 4. Recover the feature vocabulary and freeze the analysis population

        **Why:** name/order alignment prevents one feature being mistaken for another. We check that snapshot
        keys/labels equal the saved split, and no patient crosses splits. We then take the latest snapshot per
        TRAIN patient **without looking at its label**. Each person contributes one independent analysis row;
        this avoids giving frequent visitors more weight and makes stratified folds patient-disjoint.

        **Output:** aggregate split counts, analysis patient/positive counts and the RX/DX/PX vocabulary sizes.
        Analysis prevalence can differ from the full snapshot prevalence; all AP/lift results below use the
        analysis population. Existing VALIDATION and TEST feature values are never loaded. The original
        feature-order hash stored with the split must match. Counts alone would not prove identity.
    ''')
    code('''
        feature_map = read_sql('SELECT FEATURE_INDEX, FEATURE_NAME, FEATURE_COLUMN FROM ' + qtable(TABLES['FEATURE_MAP']))
        base_features, aliases = feature_map_contract(feature_map, EXPECTED_GROUPS)
        snapshots = read_sql('SELECT PATIENT_ID, END_DT, RESP FROM ' + qtable(TABLES['SNAPSHOTS']))
        saved_split = read_sql('SELECT PATIENT_ID, END_DT, RESP, SPLIT, SPLIT_CONFIG FROM ' + qtable(TABLES['PATIENT_SPLIT']))
        frozen_metadata, train_metadata = metadata_contract(snapshots, saved_split, EXPECTED_POPULATION)
        require(saved_split.SPLIT_CONFIG.nunique() == 1, 'Inconsistent frozen split configuration.')
        split_contract = json.loads(saved_split.SPLIT_CONFIG.iloc[0])
        original_feature_hash = hashlib.sha256(json.dumps(base_features, ensure_ascii=False).encode()).hexdigest()
        require(split_contract.get('feature_order_sha256') == original_feature_hash
                and split_contract.get('n_timesteps') == 12, 'Feature mapping differs from the frozen split.')
        display(frozen_metadata.groupby('SPLIT').agg(
            SNAPSHOTS=('RESP', 'size'), PATIENTS=('PATIENT_ID', 'nunique'), POSITIVES=('RESP', 'sum')).reset_index())
        display(pd.DataFrame([{'ANALYSIS_PATIENTS': len(train_metadata),
                               'POSITIVE_PATIENTS': int(train_metadata.RESP.sum()),
                               'PREVALENCE': float(train_metadata.RESP.mean()),
                               'REPRESENTATION': 'one latest snapshot per TRAIN patient'}]))
        display(pd.Series([name.split('__', 1)[0] for name in base_features]).value_counts().rename('FEATURES').to_frame())
    ''')
    markdown('''
        ## 5. Gather the original 1,028 claim-category features

        **Why:** the saved `FEATURE_MAP` resolves aliases `F0000…F1027`. For each latest TRAIN snapshot,
        sum the original raw counts over the configured bins. The query audits twelve distinct steps,
        step bounds, labels, nulls and negative counts. It fails on missing or duplicate months; it never
        manufactures missing rows, pads inputs or forward-fills values.

        **Output:** shape, category counts and `feature_lineage`. `RX__category__SUM_M12` means the sum of that
        saved category across timesteps 0–11. This simple first experiment does not preserve visit order,
        within-year changes, unique provider identities or exact day-level recency. Use extra windows or a
        reviewed event adapter for those hypotheses. Zero counts preserve the source convention; they do not
        independently establish insurance coverage. No label-based grouping or outlier removal happens here.
    ''')
    code('''
        monthly_table_columns = set(feature_inventory.loc[
            feature_inventory.SOURCE_TABLE.eq(TABLES['TENSOR_MONTHLY']), 'COLUMN_NAME'])
        require(monthly_table_columns == set(['PATIENT_ID', 'END_DT', 'RESP', 'TIME_STEP'] + aliases),
                'Monthly source columns do not match the saved feature map.')
        require(len(base_features) * len(SUM_WINDOWS) <= MAX_CANDIDATES, 'Configured windows exceed candidate budget.')
        claim_aggregates = read_sql(claim_query(TABLES['TENSOR_MONTHLY'], TABLES['PATIENT_SPLIT'], aliases, SUM_WINDOWS))
        X_claims, feature_lineage = assemble_claim_features(
            claim_aggregates, train_metadata, base_features, aliases, TABLES['TENSOR_MONTHLY'], SUM_WINDOWS)
        del claim_aggregates
        print('Claim candidate matrix:', X_claims.shape)
        print('Raw matrix memory (MiB):', round(X_claims.memory_usage(deep=True).sum() / 2**20, 1))
        display(feature_lineage)
    ''')
    markdown('''
        ## 6. Gather additional reviewed features and assemble the candidate pool

        **Why:** arbitrary joins can multiply claims or attach future values. Snapshot adapters enforce exact
        patient/date keys and retain missing values as NaN; event adapters use both event date and availability
        timestamp and explicit event-key deduplication. Candidate names include source identity; similarly named
        columns from different tables are not assumed to be equivalent. Exact duplicate vectors are handled
        later using only each fold's fitting data.

        **Outputs:** loaded-source coverage and the final candidate matrix shape. With no extra contracts,
        only the original saved claims features are included. The discovered candidate pool is broader than
        the loaded pool; that distinction is intentional and shown explicitly. Availability declarations in
        the configuration must refer to actual reviewed evidence—they do not verify themselves.
    ''')
    code('''
        X_extra, extra_lineage, extra_audit = gather_extras(
            read_sql, EXTRA_SOURCE_CONTRACTS, inventory, train_metadata, TABLES['PATIENT_SPLIT'], MAX_EVENT_CATEGORIES)
        X = pd.concat([X_claims, X_extra], axis=1)
        feature_lineage = pd.concat([feature_lineage, extra_lineage], ignore_index=True)
        require(X.columns.is_unique and len(X.columns) <= MAX_CANDIDATES, 'Duplicate names or candidate budget exceeded.')
        require(feature_lineage.FEATURE.tolist() == X.columns.tolist(), 'Candidate lineage order mismatch.')
        require(not np.isinf(X.to_numpy(dtype=float)).any(), 'Infinite candidate values.')
        y = train_metadata.RESP.to_numpy(dtype=int)
        require(train_metadata.PATIENT_ID.is_unique, 'Analysis must have one row per TRAIN patient.')
        analysis_fingerprint = fingerprint(train_metadata[['PATIENT_ID', 'END_DT', 'RESP']].to_dict('records'))
        value_hash = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes()).hexdigest()
        input_fingerprint = fingerprint({'population': analysis_fingerprint, 'values': value_hash,
                                         'features': list(X.columns)})
        display(pd.DataFrame([{'DISCOVERED_COLUMNS': len(feature_inventory),
                               'COLUMNS_REQUIRING_REVIEW': int(feature_inventory.candidate.sum()),
                               'LOADED_CLAIM_CANDIDATES': X_claims.shape[1],
                               'LOADED_EXTRA_CANDIDATES': X_extra.shape[1], 'TOTAL_CANDIDATES': X.shape[1]}]))
        if len(extra_audit):
            display(extra_audit)
        else:
            print('No additional source contracts configured. Other sources remain inventoried, not loaded.')
    ''')
    markdown('''
        ## 7. Compare the 49 configured feature names with the claims vocabulary

        **Why:** this makes the mapping question explicit without assuming the 49 are a subset of the 1,028.
        Read the current V63 `MODEL_TYPE.FEATURES` metadata only. Match exact names to original category names
        and discovered columns; no feature values, coefficient summaries or outcome-informed encodings are added.

        **Output:** `business_feature_crosswalk`. A name match is a lead, **not a verified formula match**.
        Ratios, sequences, age, distinct-provider counts and plan measures need their own definitions. A read
        failure is reported and does not invalidate the independent original-count feature analysis.
    ''')
    code('''
        business_feature_crosswalk = pd.DataFrame()
        try:
            configuration = read_sql('SELECT FEATURES FROM ' + qtable(SOURCE_ROOT + '_MODEL_TYPE'))
            require(len(configuration) == 1, 'Expected one feature configuration row.')
            configured = configuration.FEATURES.iloc[0]
            if isinstance(configured, str):
                configured = json.loads(configured)
            require(isinstance(configured, (list, tuple, np.ndarray)), 'Unrecognized FEATURES encoding.')
            require(all(isinstance(f, str) for f in configured), 'Invalid configured feature names.')
            crosswalk_rows = []
            for name in configured:
                matches = [f for f in base_features if name in (f, f.split('__', 1)[1])]
                locations = feature_inventory.loc[feature_inventory.COLUMN_NAME.eq(name), 'SOURCE_TABLE'].unique().tolist()
                crosswalk_rows.append({'BUSINESS_FEATURE': name, 'CLAIM_NAME_MATCHES': json.dumps(matches),
                                       'COLUMN_LOCATIONS': json.dumps(locations),
                                       'STATUS': 'NAME_MATCH_DEFINITION_UNVERIFIED' if matches else 'NO_EXACT_CLAIM_NAME_MATCH',
                                       'INCLUDED_FROM_MODEL_DATA': False})
            business_feature_crosswalk = pd.DataFrame(crosswalk_rows)
            display(business_feature_crosswalk)
            print('Configured business feature names:', len(configured))
        except Exception as error:
            print('Business-name crosswalk unavailable:', type(error).__name__, '; original-count analysis can proceed.')
    ''')
    markdown('''
        ## 8. Define the three selection methods and TRAIN-fold evaluation

        **Why:** rankings must be learned separately from the rows used to assess them. Every fold refits
        support/missingness checks, exact-duplicate removal, medians, transformations, scaling and selectors
        using its fitting patients only. The final lists are refitted on all TRAIN analysis patients.

        | Strategy | Implementation | Read its output as |
        |---|---|---|
        | Filter | Mutual information (MI) ranking, capped at 100; remove retained-pair absolute Spearman ≥0.95 | Marginal associations after simple quality/redundancy screening |
        | Wrapper | MI shortlist of 100, then L2-logistic recursive feature elimination (RFE), down to 50 | Features useful jointly to this bounded linear search; features outside the shortlist were not tested by RFE |
        | Embedded | Elastic-net logistic + constrained ExtraTrees; mean of their within-method importance ranks, capped at 100 | An explicit equal-weight ranking heuristic that exposes both component scores |

        **L1/L2:** elastic-net logistic uses `elasticnet_C=0.1` (smaller means stronger overall regularization)
        and `elasticnet_l1_ratio=0.5` (L1/L2 mixture). Pure L1 is ratio 1. The wrapper and common evaluator use
        L2 logistic. These are **fixed initial settings, not tuned optimal values**. Tuning C, mixture or feature
        counts later needs an inner training search or separate validation design; do not report the best
        repeatedly tried fold score as untouched performance.

        MI treats integer-valued features as discrete and other features as continuous; imputation occurs on
        the fitting partition. High-cardinality discrete MI can be biased. Spearman and binary-presence phi are
        supplementary descriptions. Presence chi-square p-values are exploratory, unadjusted for multiple
        testing, and unreliable with small expected cells; they are not selection gates. Weak marginal features
        can still matter through interactions, which the constrained tree can sometimes detect. Tree impurity
        importance can favor high-cardinality features and divide credit among correlated predictors.

        **Output:** function definitions only. No feature is manually assigned priority. Ranks compare features
        within a strategy, not numeric score magnitudes across strategies. No patient or high-count outlier is
        deleted. Missingness indicators and semantic feature groups are not automatically engineered here.
    ''')
    code(components['selection.py'] + '\nprint("Selection helpers loaded.")')
    markdown('''
        ## 9. Run selection and compare against all eligible features

        **Why:** compare all three selected subsets with the full quality-eligible feature pool on identical
        TRAIN folds and the same L2-logistic evaluator. This isolates feature-list usefulness for a fixed model;
        it does not measure Transformer, LightGBM or the client's fitted model performance. The wrapper uses
        fewer features by default, so this is not an equal-feature-count contest; change budgets explicitly for
        that experiment. Selection and the scorer never see the scoring fold during fit.

        **Outputs:** elapsed runtime and fold diagnostics. AP is `average_precision_score`; the no-skill AP
        reference is positive prevalence. Top-10% lift = precision in the highest-scoring tenth / fold prevalence.
        Boundary ties receive fractional inclusion instead of arbitrary row-order tie-breaking. Fold standard
        deviations reflect variation, not confidence intervals. Nonconverged fits must be resolved before using
        coefficients as a final feature recommendation. Runtime depends on feature windows and source expansion.
    ''')
    code('''
        started = time.perf_counter()
        selection_result = select_all(X, y, SELECTION_CONFIG)
        diagnostics = selection_result['diagnostics']
        display(diagnostics)
        print('Selection runtime (minutes):', round((time.perf_counter() - started) / 60, 2))
        print('Candidates:', X.shape[1], '| independent TRAIN patients:', X.shape[0])
    ''')
    markdown('''
        ## 10. Display the three ranked feature tables

        **Why:** keep the complete audit instead of discarding rejected columns. Each table has one row per
        candidate, `SELECTED`, `RANK`, quality and selection reasons, source table/column, support and fold
        stability. Use `SELECTED=True` to obtain that strategy's final list. Unranked/unevaluated columns retain
        a null rank and an explicit reason. An arbitrary top-100 is a computation budget, not a proven optimum.

        **How to read trends:** high rank with consistent fold selection and broad positive/negative support is
        a stronger follow-up candidate than a high rank supported by a few patients. `FOLD_SELECTION_FREQUENCY`
        is a stability fraction, **not a probability of correctness**. High redundancy suggests interchangeable
        information, not automatically a useless clinical concept. A negative elastic-net coefficient is a
        conditional association, not evidence that changing the feature would change the outcome.

        The embedded table exposes elastic-net and tree scores separately; a zero L1 coefficient may coexist
        with tree importance. RFE coefficients are comparable only after the fitted scaling and conditional on
        its selected set. No current score is represented as evidence that earlier overfitting is resolved.
    ''')
    code('''
        feature_tables = {}
        for strategy in ('filter', 'wrapper', 'embedded'):
            table = selection_result[strategy].merge(feature_lineage.rename(columns={'FEATURE': 'FEATURE_NAME'}),
                                                     on='FEATURE_NAME', how='left', validate='one_to_one')
            require(len(table) == X.shape[1] and table.SOURCE_TABLE.notna().all(), 'Incomplete ranking lineage.')
            table['RUN_ID'] = RUN_ID
            table['INPUT_SHA256'] = input_fingerprint
            table['IMPLEMENTATION_SHA256'] = IMPLEMENTATION_SHA256
            feature_tables[strategy] = table
            print(strategy.upper(), '| selected:', int(table.SELECTED.sum()), '| audited:', len(table))
            display(table)
        filter_features = feature_tables['filter']
        wrapper_features = feature_tables['wrapper']
        embedded_features = feature_tables['embedded']
    ''')
    markdown('''
        ## 11. Visualize comparison, selected-feature overlap and stability

        **Why:** fewer inputs help only if out-of-fold usefulness and stability remain acceptable. Compare AP
        and lift with `all_eligible`, then inspect overlap. Disagreement can arise from correlated substitutes,
        interactions or sampling variability; it does not establish that one method is wrong.

        **Outputs:** mean TRAIN-fold AP/lift with fold standard deviations, pairwise selected-set overlap,
        and top selected-feature stability bars. Error bars are descriptive fold variability, not confidence
        intervals. If a subset performs worse, do not call feature reduction an improvement. With three folds,
        stability has only four possible values (0, 1/3, 2/3, 1), so interpret it cautiously.
    ''')
    code('''
        comparison = diagnostics.groupby('STRATEGY').agg(
            AP_MEAN=('AVERAGE_PRECISION', 'mean'), AP_FOLD_SD=('AVERAGE_PRECISION', 'std'),
            LIFT_MEAN=('TOP10_LIFT', 'mean'), LIFT_FOLD_SD=('TOP10_LIFT', 'std'),
            SELECTED_MEAN=('SELECTED_FEATURES', 'mean')).reset_index()
        display(comparison)
        chosen = {method: set(table.loc[table.SELECTED, 'FEATURE_NAME']) for method, table in feature_tables.items()}
        overlap = pd.DataFrame({a: {b: len(chosen[a] & chosen[b]) for b in chosen} for a in chosen})
        display(overlap.rename_axis('SELECTED_SET'))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        axes[0].bar(comparison.STRATEGY, comparison.AP_MEAN, yerr=comparison.AP_FOLD_SD.fillna(0), capsize=4)
        axes[0].axhline(y.mean(), color='grey', linestyle='--', label='Analysis positive prevalence')
        axes[0].set(title='TRAIN-fold AP: fixed L2-logistic evaluator', ylabel='Average precision')
        axes[0].legend()
        axes[1].bar(comparison.STRATEGY, comparison.LIFT_MEAN, yerr=comparison.LIFT_FOLD_SD.fillna(0), capsize=4)
        axes[1].axhline(1, color='grey', linestyle='--')
        axes[1].set(title='TRAIN-fold top-10% lift', ylabel='Lift')
        plt.show()
        for method, table in feature_tables.items():
            top = table.loc[table.SELECTED].head(15).iloc[::-1]
            if top.empty:
                print(method, ': no features selected; inspect quality/selection reasons.')
                continue
            fig, ax = plt.subplots(figsize=(13, max(4, 0.32 * len(top))), layout='constrained')
            ax.barh(top.FEATURE_NAME, top.FOLD_SELECTION_FREQUENCY)
            ax.set(xlim=(0, 1.05), xlabel='Fraction of TRAIN folds selecting feature',
                   title=method.upper() + ': stability of highest-ranked selected features')
            plt.show()
    ''')
    markdown('''
        ## 12. Save three new Snowflake feature tables and verify read-back

        **Why:** downstream experiments need actual selected lists and audit rows, not a screenshot of top bars.
        This cell creates `{OUTPUT_PREFIX}_FILTER`, `_WRAPPER`, `_EMBEDDED`. Each holds the full candidate audit,
        selected flags, lineage, input fingerprints, settings, runtime versions and aggregate fold diagnostics.
        No patient IDs, per-patient values or predictions are written. Tables are private project artifacts.

        **Output:** exact fully qualified table names, verified row counts and selected-feature counts. Creation
        uses `errorifexists`; all three names are preflighted before writing. If a connector error interrupts
        saving, partial new tables may exist. Rerun with a fresh `RUN_ID`; the code never deletes source/results.
        Save/read-back verification compares feature names, ranks and selected flags. `WRITE_RESULTS=False`
        keeps the same three tables as DataFrames in the session and performs no warehouse writes.

        **Next experiment:** freeze these definitions, compare the three lists with the full-feature control
        in the intended downstream model using equal tuning budgets, and evaluate on a separately reserved
        population. Do not reuse a global preselected list inside supposedly independent TRAIN cross-validation;
        re-fit the selector within each training fold, as implemented here.
    ''')
    code('''
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType

        run_manifest = {
            'run_id': RUN_ID, 'input_sha256': input_fingerprint,
            'implementation_sha256': IMPLEMENTATION_SHA256,
            'source_tables': TABLES, 'sum_windows': list(SUM_WINDOWS),
            'extra_source_contracts': EXTRA_SOURCE_CONTRACTS,
            'selection_config': selection_result['config'],
            'analysis_unit': 'latest snapshot per frozen TRAIN patient',
            'analysis_patients': len(train_metadata), 'analysis_positives': int(y.sum()),
            'candidate_count': X.shape[1],
            'upstream_limitations': ['Original claim category/deduplication SQL not recovered',
                                    'Original event/ingestion cutoff and label construction not independently verified'],
            'diagnostics': diagnostics.to_dict('records'),
            'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                         'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        }
        run_manifest_json = json.dumps(run_manifest, sort_keys=True, allow_nan=False)
        destinations = {method: OUTPUT_PREFIX + '_' + method.upper() for method in feature_tables}

        def to_spark_report(frame):
            fields, converters = [], []
            for column in frame.columns:
                dtype = frame[column].dtype
                if pd.api.types.is_bool_dtype(dtype):
                    kind, convert = BooleanType(), bool
                elif pd.api.types.is_integer_dtype(dtype):
                    kind, convert = LongType(), int
                elif pd.api.types.is_numeric_dtype(dtype):
                    kind, convert = DoubleType(), float
                else:
                    kind, convert = StringType(), str
                fields.append(StructField(str(column), kind, True))
                converters.append(convert)
            records = [tuple(None if pd.isna(value) else convert(value)
                             for value, convert in zip(row, converters))
                       for row in frame.itertuples(index=False, name=None)]
            return spark.createDataFrame(records, StructType(fields))

        saved_results = []
        if WRITE_RESULTS:
            for destination in destinations.values():
                database, schema, table_name = destination.split('.')
                qtable(destination)
                existing = read_sql(f'SELECT TABLE_NAME FROM {qi(database)}.INFORMATION_SCHEMA.TABLES '
                                    f'WHERE TABLE_SCHEMA = {literal(schema)} AND TABLE_NAME = {literal(table_name)}')
                require(existing.empty, 'Output already exists; use a fresh RUN_ID: ' + destination)
            for method, destination in destinations.items():
                exported = feature_tables[method].copy()
                exported['RUN_MANIFEST_JSON'] = run_manifest_json
                (to_spark_report(exported).write.format('snowflake').options(**sf_options_analysis)
                 .option('dbtable', destination).mode('errorifexists').save())
                observed = read_sql('SELECT FEATURE_NAME, RANK, SELECTED, RUN_ID FROM ' + qtable(destination))
                expected = exported[['FEATURE_NAME', 'RANK', 'SELECTED', 'RUN_ID']].sort_values('FEATURE_NAME').reset_index(drop=True)
                observed = observed.sort_values('FEATURE_NAME').reset_index(drop=True)
                pd.testing.assert_frame_equal(expected, observed, check_dtype=False)
                saved_results.append({'STRATEGY': method, 'TABLE': destination, 'ROWS_VERIFIED': len(observed),
                                      'SELECTED_FEATURES': int(exported.SELECTED.sum()), 'STATUS': 'SAVED_AND_VERIFIED'})
        else:
            saved_results = [{'STRATEGY': method, 'TABLE': destination, 'ROWS_VERIFIED': len(feature_tables[method]),
                              'SELECTED_FEATURES': int(feature_tables[method].SELECTED.sum()),
                              'STATUS': 'DATAFRAME_ONLY_NOT_WRITTEN'} for method, destination in destinations.items()]
        display(pd.DataFrame(saved_results))
    ''')
    markdown('''
        ## References and scope

        - [scikit-learn feature selection](https://scikit-learn.org/stable/modules/feature_selection.html): filter, RFE and embedded methods.
        - [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): classification penalties and regularization controls.
        - [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): fit preprocessing/selection only on training data.

        This is a feature discovery and selection notebook, not a deployment or a reproduction of previous
        model results. Table availability and real rankings are established only when it runs in your private
        environment. The committed copy has cleared outputs and contains no credentials or patient data.
    ''')
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'title': 'Gather and select features'},
        'nbformat': 4, 'nbformat_minor': 5}
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return DESTINATION


if __name__ == '__main__':
    print(build())
