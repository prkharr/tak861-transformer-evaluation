import ast
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'notebooks' / 'Medical NLP Demo' / '01_medical_claims_NLP_sequence_demo.ipynb'

CONFIG = '''
import time, os, json, hashlib, platform, importlib.metadata
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from IPython.display import display
import matplotlib.pyplot as plt
import joblib

sf_options = {
    "sfUrl": "", "sfUser": "", "sfPassword": "", "sfRole": "",
    "sfWarehouse": "", "sfDatabase": "", "sfSchema": ""
}
CONFIG = {
    "cohort": "DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_MODEL_DATA",
    "medical": "DSVC_TAKEDA_FULLMAP_PLAID_PROD.COHORT_1009719.MEDICAL_EVENTS_LATEST",
    "reference": "DSVC_TAKEDA_TA_PRIVATE.DS_ML_PRD.DX_PX_RX_PLAID",
    "max_patients": 100, "max_terms": 256, "max_source_rows": 200000,
    "sample_seed": 42, "months": 12,
    "event_id_column": "MEDICAL_EVENT_ID",
    "output_directory": "/local_disk0/tmp/medical_nlp_sequence_demo"
}
missing = [k for k,v in sf_options.items() if not str(v).strip()]
if missing:
    raise ValueError("Fill sf_options locally before execution: " + ", ".join(missing))
if "spark" not in globals():
    raise RuntimeError("Run this notebook in Databricks with an active Spark session.")
started = time.perf_counter()
print("Exploratory medical-description NLP: no outcome labels, model training or predictive evaluation.")
print("At most", CONFIG["max_patients"], "patients; 12 monthly steps; latest snapshot per patient.")
print("Runtime target is approximately 10 minutes, conditional on warehouse availability and scan time.")
'''

HELPERS = '''
def read_sf(query):
    return spark.read.format("snowflake").options(**sf_options).option("query", query).load()

def source_sql(cfg):
    import re
    for name in ("cohort", "medical", "reference"):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*\\.[A-Za-z_][A-Za-z0-9_$]*\\.[A-Za-z_][A-Za-z0-9_$]*", cfg[name]):
            raise ValueError("Invalid source identifier")
    for name in ("max_patients", "max_terms", "max_source_rows", "sample_seed"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError("Configuration values must be positive integers")
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', cfg['event_id_column']):
        raise ValueError('Invalid event ID column name')
    if cfg["months"] != 12 or cfg["max_patients"] > 1000:
        raise ValueError("This bounded demonstration requires 12 steps and at most 1000 patients")
    if cfg['max_terms']>1024 or cfg['max_source_rows']>200000:
        raise ValueError("Demonstration memory budget: at most 1024 terms and 200000 aggregate rows")
    return f"""
WITH latest AS (
    SELECT TO_VARCHAR(PATIENT_ID) AS PATIENT_ID, MAX(END_DT::DATE) AS END_DT
    FROM {cfg['cohort']} WHERE PATIENT_ID IS NOT NULL AND END_DT IS NOT NULL
    GROUP BY PATIENT_ID
), chosen AS (
    SELECT PATIENT_ID, END_DT,
           DATEADD(month,-11,DATE_TRUNC('month',END_DT))::DATE AS START_DT
    FROM latest
    QUALIFY ROW_NUMBER() OVER (ORDER BY SHA2(PATIENT_ID || '|{cfg['sample_seed']}',256), PATIENT_ID) <= {cfg['max_patients']}
), med AS (
    SELECT p.PATIENT_ID,p.END_DT,p.START_DT,m.{cfg['event_id_column']} AS MEDICAL_EVENT_ID,
           m.SERVICE_DATE::DATE AS SERVICE_DATE,m.DIAGNOSIS_CODES,m.PROCEDURE_CODE
    FROM {cfg['medical']} m JOIN chosen p ON TO_VARCHAR(m.PATIENT_ID) = p.PATIENT_ID
    WHERE m.SERVICE_DATE >= p.START_DT AND m.SERVICE_DATE < DATEADD(day,1,p.END_DT)
      AND m.SERVICE_DATE >= (SELECT MIN(START_DT) FROM chosen)
      AND m.SERVICE_DATE < (SELECT DATEADD(day,1,MAX(END_DT)) FROM chosen)
), events AS (
    SELECT m.PATIENT_ID,m.END_DT,m.START_DT,m.MEDICAL_EVENT_ID,m.SERVICE_DATE,
           'DX' AS FAMILY,REPLACE(UPPER(TRIM(d.VALUE)),'.','') AS CODE
    FROM med m,LATERAL SPLIT_TO_TABLE(COALESCE(m.DIAGNOSIS_CODES,''),'|') d
    WHERE NULLIF(TRIM(d.VALUE),'') IS NOT NULL
    UNION ALL
    SELECT PATIENT_ID,END_DT,START_DT,MEDICAL_EVENT_ID,SERVICE_DATE,
           'PX' AS FAMILY,UPPER(TRIM(TO_VARCHAR(PROCEDURE_CODE))) AS CODE
    FROM med WHERE NULLIF(TRIM(TO_VARCHAR(PROCEDURE_CODE)),'') IS NOT NULL
), code_map AS (
    SELECT UPPER(TRIM(C_TYPE)) AS FAMILY,
           CASE WHEN LOWER(TRIM(C_TYPE))='dx' THEN REPLACE(UPPER(TRIM(TO_VARCHAR(CODE))),'.','')
                ELSE UPPER(TRIM(TO_VARCHAR(CODE))) END AS CODE,
           MIN(NULLIF(TRIM(TO_VARCHAR("DESC")),'')) AS DESCRIPTION,
           COUNT(DISTINCT NULLIF(TRIM(TO_VARCHAR("DESC")),'')) AS DESCRIPTION_VARIANTS
    FROM {cfg['reference']}
    WHERE CURR_IND='Y' AND LOWER(TRIM(C_TYPE)) IN ('dx','px')
    GROUP BY 1,2
), aggregated AS (
    SELECT e.PATIENT_ID,e.END_DT,DATEDIFF(month,e.START_DT,e.SERVICE_DATE) AS STEP,
           e.FAMILY,e.CODE,r.DESCRIPTION,COALESCE(r.DESCRIPTION_VARIANTS,0) AS DESCRIPTION_VARIANTS,
           COUNT(DISTINCT e.MEDICAL_EVENT_ID) AS EVENT_COUNT,
           SUM(IFF(e.MEDICAL_EVENT_ID IS NULL,1,0)) AS NULL_EVENT_ID_ROWS
    FROM events e LEFT JOIN code_map r ON e.FAMILY=r.FAMILY AND e.CODE=r.CODE
    GROUP BY 1,2,3,4,5,6,7
)
SELECT p.PATIENT_ID,p.END_DT,a.STEP,a.FAMILY,a.CODE,a.DESCRIPTION,
       a.DESCRIPTION_VARIANTS,a.EVENT_COUNT,a.NULL_EVENT_ID_ROWS
FROM chosen p LEFT JOIN aggregated a ON p.PATIENT_ID=a.PATIENT_ID AND p.END_DT=a.END_DT
LIMIT {cfg['max_source_rows']+1}
"""

def make_nlp_sequence(frame, cfg, frozen_state=None):
    f = frame.copy()
    f.columns = [str(c).upper() for c in f.columns]
    required = {'PATIENT_ID','END_DT','STEP','FAMILY','CODE','DESCRIPTION',
                'DESCRIPTION_VARIANTS','EVENT_COUNT','NULL_EVENT_ID_ROWS'}
    if not required.issubset(f.columns) or f.empty:
        raise ValueError("Missing source fields or empty demonstration cohort")
    if len(f) > cfg['max_source_rows']:
        raise ValueError("Source row budget exceeded; reduce max_patients and retry. No truncated result was used")
    f['PATIENT_ID'] = f.PATIENT_ID.astype(str)
    f['END_DT'] = pd.to_datetime(f.END_DT).dt.normalize()
    meta = f[['PATIENT_ID','END_DT']].drop_duplicates().sort_values(['PATIENT_ID','END_DT']).reset_index(drop=True)
    if meta.PATIENT_ID.duplicated().any() or len(meta)>cfg['max_patients']:
        raise ValueError("Expected one latest snapshot per selected patient")
    rows = f[f.CODE.notna()].copy()
    for c in ['STEP','EVENT_COUNT','NULL_EVENT_ID_ROWS','DESCRIPTION_VARIANTS']:
        rows[c] = pd.to_numeric(rows[c], errors='raise')
    if not np.isfinite(rows[['STEP','EVENT_COUNT','NULL_EVENT_ID_ROWS','DESCRIPTION_VARIANTS']].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite source values")
    if (rows.NULL_EVENT_ID_ROWS != 0).any():
        raise ValueError("Medical events without IDs detected; cannot validate distinct-event counts")
    if not rows.STEP.between(0,11).all() or (rows.STEP%1!=0).any() or (rows.EVENT_COUNT<=0).any():
        raise ValueError("Invalid monthly step or event count")
    if rows.duplicated(['PATIENT_ID','END_DT','STEP','FAMILY','CODE']).any():
        raise ValueError("Duplicate monthly code aggregates")
    index = {(r.PATIENT_ID,r.END_DT): i for i,r in meta.iterrows()}
    rows['DOC'] = [index[(r.PATIENT_ID,r.END_DT)]*12+int(r.STEP) for r in rows.itertuples()]
    n_docs = len(meta)*12
    code_observed = np.zeros(n_docs,dtype=bool)
    code_observed[rows.DOC.to_numpy(dtype=int)] = True
    mapped = rows[rows.DESCRIPTION.notna() & rows.DESCRIPTION.fillna('').str.strip().ne('')].copy()
    if mapped.empty:
        raise ValueError("No mapped descriptions; cannot build a text-feature demonstration")
    mapped['KEY'] = mapped.FAMILY.astype(str)+'|'+mapped.CODE.astype(str)
    descriptions = mapped[['KEY','DESCRIPTION']].drop_duplicates().sort_values('KEY')
    if descriptions.KEY.duplicated().any():
        raise ValueError("Description mapping changed within extraction")
    code_index = {k:i for i,k in enumerate(descriptions.KEY)}
    dc = sparse.coo_matrix((mapped.EVENT_COUNT.to_numpy(dtype=np.float32),
        (mapped.DOC.to_numpy(dtype=int), mapped.KEY.map(code_index).to_numpy(dtype=int))),
        shape=(n_docs,len(descriptions)), dtype=np.float32).tocsr()
    stop_words = sorted(set(ENGLISH_STOP_WORDS)-{'no','not','nor','without','neither','never','with'})
    if frozen_state is None:
        vectorizer = CountVectorizer(lowercase=True,stop_words=stop_words,ngram_range=(1,2),
            token_pattern=r'(?u)\\b\\w+\\b',dtype=np.float32)
        code_terms = vectorizer.fit_transform(descriptions.DESCRIPTION)
        doc_terms = (dc @ code_terms).tocsr()
        all_terms = vectorizer.get_feature_names_out()
        frequency = np.asarray(doc_terms.sum(axis=0)).ravel()
        order = np.lexsort((all_terms,-frequency))[:cfg['max_terms']]
        terms = all_terms[order]
        term_counts = doc_terms[:,order].tocsr()
        tfidf = TfidfTransformer(norm='l2',sublinear_tf=True)
        transformed = tfidf.fit_transform(term_counts)
        fit_scope = 'entire bounded demonstration cohort; not a predictive evaluation'
    else:
        if frozen_state['schema_version'] != 'medical_nlp_v1':
            raise ValueError('Unknown frozen representation schema')
        vectorizer,tfidf = frozen_state['vectorizer'],frozen_state['tfidf']
        order = np.asarray(frozen_state['selected_term_indices'],dtype=int)
        terms = vectorizer.get_feature_names_out()[order]
        if list(terms) != list(frozen_state['terms']):
            raise ValueError('Frozen feature order is inconsistent')
        old_descriptions = frozen_state['code_descriptions']
        for r in descriptions.itertuples():
            if r.KEY in old_descriptions and old_descriptions[r.KEY] != r.DESCRIPTION:
                raise ValueError('Description changed for a previously fitted code; review mapping version')
        code_terms = vectorizer.transform(descriptions.DESCRIPTION)
        term_counts = (dc @ code_terms)[:,order].tocsr()
        transformed = tfidf.transform(term_counts)
        fit_scope = 'frozen transform; original fit scope: '+frozen_state['fit_scope']
    x = transformed.astype(np.float32).toarray().reshape(len(meta),12,len(terms))
    if not np.isfinite(x).all() or (x<0).any():
        raise ValueError("Invalid NLP tensor values")
    presence = (term_counts.toarray()>0).reshape(x.shape)
    monthly_mean = x.mean(axis=0)
    ranking = pd.DataFrame({'term':terms,'mean_tfidf':x.mean(axis=(0,1)),
        'patient_coverage_pct':100*presence.any(axis=1).mean(axis=0),
        'month_coverage_pct':100*presence.mean(axis=(0,1)),
        'mean_absolute_monthly_change':np.abs(np.diff(x,axis=1)).mean(axis=(0,1)),
        'recent_minus_early_tfidf':x[:,-3:,:].mean(axis=(0,1))-x[:,:3,:].mean(axis=(0,1))})
    ranking = ranking.sort_values(['mean_tfidf','term'],ascending=[False,True]).reset_index(drop=True)
    ranking.insert(0,'rank',np.arange(1,len(ranking)+1))
    mapped_mask = np.zeros(n_docs,dtype=bool)
    mapped_mask[mapped.DOC.to_numpy(dtype=int)] = True
    coverage = pd.DataFrame({'step':np.arange(12),'relative_month':np.arange(-11,1),
        'patients_with_dx_px_activity':code_observed.reshape(-1,12).sum(axis=0),
        'patients_with_mapped_description':mapped_mask.reshape(-1,12).sum(axis=0),
        'patients_with_selected_terms':presence.any(axis=2).sum(axis=0)})
    total_mentions = float(rows.EVENT_COUNT.sum())
    audit = {'patients':len(meta),'monthly_positions':n_docs,'source_aggregate_rows':len(rows),
        'mapped_code_event_fraction':float(mapped.EVENT_COUNT.sum()/total_mentions) if total_mentions else 0,
        'codes_with_multiple_descriptions':int(rows.loc[rows.DESCRIPTION_VARIANTS>1,['FAMILY','CODE']].drop_duplicates().shape[0]),
        'tensor_shape':list(x.shape),'tensor_bytes':int(x.nbytes),
        'patients_with_temporal_variation':int((np.abs(np.diff(x,axis=1))>1e-7).any(axis=(1,2)).sum()),
        'snapshots_with_partial_final_month':int((~meta.END_DT.dt.is_month_end).sum()),
        'step_orientation':'array index 0 oldest (-11 months); index 11 newest (snapshot month)',
        'representation':'claim-count-weighted description unigrams/bigrams, sublinear TF-IDF, L2 per patient-month',
        'ranking':'mean TF-IDF across all selected patient-months; descriptive, not predictive importance',
        'fit_scope':fit_scope,
        'mapping':'current reference; lexicographically first description for ambiguous codes',
        'zero_semantics':'no selected textual term; not proof of complete history or absence of disease',
        'source_vintage':'LATEST events and current reference; ingestion-as-of not reconstructed'}
    return {'X_sequence':x,'metadata_private':meta,'terms':terms,'ranking':ranking,
            'coverage':coverage,'monthly_mean':monthly_mean,'audit':audit,'vectorizer':vectorizer,
            'tfidf':tfidf,'selected_term_indices':order,
            'code_activity_mask':code_observed.reshape(-1,12),'mapped_description_mask':mapped_mask.reshape(-1,12),
            'feature_state': frozen_state if frozen_state is not None else {
                'schema_version':'medical_nlp_v1','vectorizer':vectorizer,'tfidf':tfidf,
                'selected_term_indices':order,'terms':list(terms),'fit_scope':fit_scope,
                'code_descriptions':dict(zip(descriptions.KEY,descriptions.DESCRIPTION))}}
'''

EXTRACT = '''
source_query = source_sql(CONFIG)
query_sha256 = hashlib.sha256(source_query.encode()).hexdigest()
if globals().get("cached_query_sha256") == query_sha256 and "medical_aggregates" in globals():
    print("Reusing this session's completed source extraction.")
else:
    for source, required in [(CONFIG['cohort'], {'PATIENT_ID','END_DT'}),
        (CONFIG['medical'], {'PATIENT_ID','SERVICE_DATE',CONFIG['event_id_column'].upper(),'DIAGNOSIS_CODES','PROCEDURE_CODE'}),
        (CONFIG['reference'], {'CODE','C_TYPE','DESC','CURR_IND'})]:
        actual = {c.upper() for c in read_sf('SELECT * FROM '+source+' WHERE 1=0').columns}
        if not required.issubset(actual):
            raise ValueError('Source schema lacks required columns: '+', '.join(sorted(required-actual)))
    print("Starting one bounded medical extraction; no pharmacy query or neural-network training.")
    extracted = read_sf(source_query).toPandas()
    if len(extracted)>CONFIG['max_source_rows']:
        raise ValueError("Source row budget exceeded; reduce max_patients and retry. Result not accepted")
    medical_aggregates = extracted
    cached_query_sha256 = query_sha256
    del extracted
print("Source aggregate rows:",len(medical_aggregates),"Elapsed seconds:",round(time.perf_counter()-started,1))
'''

MODEL = '''
demo = make_nlp_sequence(medical_aggregates, CONFIG)
display(pd.DataFrame([demo['audit']]))
display(demo['coverage'])
print("Chronological steps: -11 months through current snapshot month (which may be partial).")
print("Each step contains that month's events only; these are not rolling 12-month AutoML summaries.")
print("This is NLP over mapped diagnosis/procedure descriptions, not clinical-note NLP or AutoML.")
print("Zeros do not establish observation coverage. Terms and IDF are fitted only for this descriptive demo.")
display(demo['ranking'].head(30))
'''

PLOT = '''
top_terms = demo['ranking'].term.head(15).tolist()
positions = [int(np.flatnonzero(demo['terms']==term)[0]) for term in top_terms]
sequence_summary = pd.DataFrame(demo['monthly_mean'][:,positions],
    index=pd.Index(np.arange(-11,1),name='relative_month'),columns=top_terms)
display(sequence_summary)
fig, axes = plt.subplots(1,2,figsize=(16,7),constrained_layout=True)
heat = axes[0].imshow(sequence_summary.to_numpy().T,aspect='auto',cmap='viridis')
axes[0].set_yticks(np.arange(len(top_terms)),top_terms)
axes[0].set_xticks(np.arange(12),np.arange(-11,1))
axes[0].set_xlabel('Month relative to snapshot; 0 may be partial')
axes[0].set_title('Actual monthly medical-description NLP sequences')
fig.colorbar(heat,ax=axes[0],label='Mean TF-IDF across demonstration patients')
top = demo['ranking'].head(15).iloc[::-1]
axes[1].barh(top.term,top.mean_tfidf)
axes[1].set_xlabel('Mean TF-IDF (descriptive salience)')
axes[1].set_title('Text feature ranking — no predictive interpretation')
display(fig)
plt.close(fig)
'''

SAVE = '''
os.makedirs(CONFIG['output_directory'],exist_ok=True)
demo['ranking'].to_csv(os.path.join(CONFIG['output_directory'],'nlp_feature_ranking.csv'),index=False)
demo['coverage'].to_csv(os.path.join(CONFIG['output_directory'],'monthly_coverage.csv'),index=False)
sequence_summary.to_csv(os.path.join(CONFIG['output_directory'],'aggregate_monthly_tfidf.csv'))
fig.savefig(os.path.join(CONFIG['output_directory'],'nlp_sequence_and_ranking.png'),dpi=150,bbox_inches='tight')
receipt = {**demo['audit'],'configuration':CONFIG,'query_sha256':query_sha256,
    'elapsed_seconds':round(time.perf_counter()-started,1)}
with open(os.path.join(CONFIG['output_directory'],'demo_receipt.json'),'w') as handle:
    json.dump(receipt,handle,indent=2)
print('Aggregate outputs saved to:',CONFIG['output_directory'])
print('Completed in seconds:',receipt['elapsed_seconds'])
print('No lift or model-performance result was calculated. The separate sequence experiment is unchanged.')
'''

EXPORT = '''
feature_list = pd.DataFrame({'feature_index':np.arange(len(demo['terms'])),
    'feature_name':['NLP__'+term for term in demo['terms']], 'term':demo['terms'],
    'dtype':'float32','representation':'monthly_tfidf'})
feature_list = feature_list.merge(demo['ranking'][['term','rank','mean_tfidf']],on='term',validate='one_to_one')
feature_list = feature_list.sort_values('feature_index').reset_index(drop=True)
display(feature_list)
private_path = os.path.join(CONFIG['output_directory'],'private_feature_package')
os.makedirs(private_path,exist_ok=True)
feature_list.to_csv(os.path.join(private_path,'ordered_feature_list.csv'),index=False)
np.savez_compressed(os.path.join(private_path,'sequential_features.npz'),
    X_sequence=demo['X_sequence'],relative_month=np.arange(-11,1),
    feature_names=feature_list.feature_name.to_numpy(dtype=str),
    code_activity_mask=demo['code_activity_mask'],mapped_description_mask=demo['mapped_description_mask'])
demo['metadata_private'].assign(row_index=np.arange(len(demo['metadata_private']))).to_csv(
    os.path.join(private_path,'patient_snapshot_index_PRIVATE.csv'),index=False)
joblib.dump(demo['feature_state'],os.path.join(private_path,'frozen_tfidf_state.joblib'))
with open(os.path.join(private_path,'medical_nlp_transform.py'),'w') as handle:
    handle.write(__TRANSFORM_SOURCE__)
contract = {
    'schema_version':'medical_nlp_v1','shape':list(demo['X_sequence'].shape),
    'axes':['patient_snapshot','month_oldest_to_newest','ordered_feature'],
    'window':'11 prior calendar months plus current calendar month through END_DT inclusive',
    'keys':['PATIENT_ID','END_DT'],'source_configuration':CONFIG,'fit_scope':demo['audit']['fit_scope'],
    'feature_order':'ordered_feature_list.csv feature_index; not ranking order',
    'masks':'code_activity_mask and mapped_description_mask indicate detected activity, not enrollment coverage',
    'transform':'make_nlp_sequence(new_aggregates, CONFIG, frozen_state=loaded_state); no refit',
    'transformer_adapter':'join exact PATIENT_ID/END_DT, align time orientation, use feature width as input dimension',
    'evaluation_requirement':'refit vocabulary, term selection and IDF on TRAIN patients only before predictive comparison',
    'compatibility':'new input representation; not a drop-in replacement for the existing AutoML/count artifact contract',
    'private_files':'patient_snapshot_index_PRIVATE.csv and sequential_features.npz remain in the authorized environment',
    'local_storage':'ephemeral until copied to an approved persistent location'
}
contract['python_version'] = platform.python_version()
contract['package_versions'] = {name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','joblib']}
contract['files_sha256'] = {}
for name in ['ordered_feature_list.csv','sequential_features.npz','patient_snapshot_index_PRIVATE.csv','frozen_tfidf_state.joblib','medical_nlp_transform.py']:
    file_hash = hashlib.sha256()
    with open(os.path.join(private_path,name),'rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            file_hash.update(block)
    contract['files_sha256'][name] = file_hash.hexdigest()
with open(os.path.join(private_path,'feature_contract.json'),'w') as handle:
    json.dump(contract,handle,indent=2)
print('Reusable sequential feature package saved privately:',private_path)
print('Tensor column order matches feature_index. Keep patient-level files private; share aggregate charts only.')
print('The existing Transformer experiment was not changed. A new input adapter and TRAIN-only refit are needed for evaluation.')
'''
EXPORT = EXPORT.replace('__TRANSFORM_SOURCE__', repr(
    'import numpy as np\nimport pandas as pd\nfrom scipy import sparse\n'
    'from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS\n'
    + HELPERS))

def build():
    cells = []
    for source in [CONFIG, HELPERS, EXTRACT, MODEL, PLOT, SAVE, EXPORT]:
        source = dedent(source).strip()+'\n'
        ast.parse(source)
        cells.append(dict(cell_type='code',id='medical-nlp-'+str(len(cells)+1),metadata={},execution_count=None,outputs=[],source=source.splitlines(True)))
    book = dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(json.dumps(book,indent=1)+'\n',encoding='utf-8')
    print(DEST)

if __name__=='__main__':
    build()
