"""Frozen-cache development comparisons with bounded features and resumable fits."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.automl_sequence import inputs, modeling, raw_automl, workflow
from experiments.automl_sequence.fresh_lgbm import SETTINGS as FULL_SETTINGS
from experiments.rigorous_transformer import comparison
from experiments.cached_pilot import cache

SCHEMA = 'cached_pilot_report_v1'
SOURCE_SHA256 = 'cached_pilot_analysis_v1'
SETTINGS = dict(FULL_SETTINGS, seeds=[42], n_estimators=300, n_jobs=2)
SETTINGS.pop('stopping_rounds', None)
KEYS = ['PATIENT_ID', 'END_DT']
ARMS = {'AUTOML': ['automl'], 'AUTOML_TIME': ['automl', 'time'],
        'AUTOML_NLP': ['automl', 'nlp'], 'AUTOML_TIME_NLP': ['automl', 'time', 'nlp']}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_bytes(workflow.json_bytes(value))
    os.replace(tmp, path)


def _atomic_bytes(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_bytes(value)
    os.replace(tmp, path)


def _file_hash(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _records(meta):
    out = meta.copy()
    out['END_DT'] = out.END_DT.dt.strftime('%Y-%m-%d')
    return out.to_dict('records')


def _key_hash(meta):
    return workflow.digest([[str(p), pd.Timestamp(d).strftime('%Y-%m-%d')]
                            for p, d in meta[KEYS].itertuples(index=False, name=None)])


def population_summary(metadata):
    rows = []
    for split, frame in [('all', metadata)] + list(metadata.groupby('SPLIT', sort=True)):
        rows.append(dict(Split=split.upper(), Snapshots=len(frame), Patients=frame.PATIENT_ID.nunique(),
                         Positive_snapshots=int(frame.RESP.sum()),
                         Positive_patients=frame.loc[frame.RESP.eq(1), 'PATIENT_ID'].nunique(),
                         First_date=str(frame.END_DT.min().date()), Last_date=str(frame.END_DT.max().date())))
    return pd.DataFrame(rows)


def _load_preparation(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'prepared.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('schema') != 'cached_pilot_preparation_v1':
        raise ValueError('Unsupported prior preparation')
    unsigned = {k:v for k,v in result.items() if k != 'sha256'}
    if result.get('sha256') != workflow.digest(unsigned):
        raise ValueError('Preparation receipt hash mismatch')
    return result


def _assert_classes(meta):
    for split in ('train', 'validation'):
        frame = meta.loc[meta.SPLIT.eq(split)]
        if len(frame) == 0 or set(frame.RESP.unique()) != {0, 1}:
            raise ValueError(f'{split}: complete-patient pilot needs both classes; wait for more committed chunks')
    if meta.SPLIT.eq('test').any():
        raise ValueError('TEST must remain unused')


def _read_complete_metadata(destination):
    complete = cache.complete_patient_metadata(destination)
    return inputs.normalize_snapshots(complete)


def prepare(cache_manifest, output_root, *, previous_version=None, max_features=512):
    """Freeze complete TRAIN patients and an initial fixed development panel."""
    if type(max_features) is not int or max_features < 3:
        raise ValueError('Expected a pair-preserving vocabulary cap of at least three')
    manifest = cache.load_frozen_cache(cache_manifest)
    directory = Path(cache_manifest)
    if directory.is_file():
        directory = directory.parent
    db_path = Path(manifest.get('analysis_db_path', directory / manifest.get('analysis_db', 'analysis.sqlite')))
    meta = _read_complete_metadata(manifest)
    meta = meta.loc[meta.SPLIT.isin(['train', 'validation'])].reset_index(drop=True)
    previous = _load_preparation(previous_version) if previous_version is not None else None
    if previous is not None:
        if previous.get('settings') != SETTINGS or previous.get('implementation') != SOURCE_SHA256:
            raise ValueError('Refresh changed frozen pilot settings or implementation; start a separate experiment')
        if previous['source_identity'] != manifest['identity'] or previous['parent_metadata_sha256'] != manifest['parent_metadata_sha256']:
            raise ValueError('Refresh must use the same parent source, metadata, labels and split')
        if previous['max_features'] != max_features:
            raise ValueError('Refresh changed the vocabulary cap')
        old_val = inputs.normalize_snapshots(pd.DataFrame(previous['validation_records']))
        available = pd.MultiIndex.from_frame(meta[KEYS])
        index = available.get_indexer(pd.MultiIndex.from_frame(old_val[KEYS]))
        if (index < 0).any() or _records(meta.iloc[index].reset_index(drop=True)) != _records(old_val):
            raise ValueError('Frozen validation population or labels changed')
        old_train = inputs.normalize_snapshots(pd.DataFrame(previous['training_records']))
        ti = available.get_indexer(pd.MultiIndex.from_frame(old_train[KEYS]))
        if (ti < 0).any() or _records(meta.iloc[ti].reset_index(drop=True)) != _records(old_train):
            raise ValueError('Refresh lost or changed prior TRAIN rows')
        meta = pd.concat([meta.loc[meta.SPLIT.eq('train')].sort_values(KEYS,kind='stable'), old_val], ignore_index=True)
    else:
        # Canonical order makes all comparisons independent of source/export iteration order.
        meta = meta.sort_values(KEYS, kind='stable').reset_index(drop=True)
    _assert_classes(meta)
    train = meta.SPLIT.eq('train').to_numpy()
    val = meta.SPLIT.eq('validation').to_numpy()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_identity = dict(manifest['identity'])
    with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        train_patients = set(meta.loc[train, 'PATIENT_ID'])
        support = {}
        for category, patient in db.execute("SELECT DISTINCT c,p FROM raw WHERE k='FEATURE' ORDER BY c,p"):
            if patient in train_patients:
                support[category] = support.get(category, 0) + 1
        if previous:
            categories = list(previous['categories'])
        else:
            categories = sorted(support, key=lambda name: (-support[name], name))[:(max_features-1)//2]
        specs = sorted([dict(name=raw_automl._feature_name(category, measure), category=category, measure=measure)
                        for category in categories for measure in ('count', 'recent_pct')]
                       + [dict(name='AML_AGE_YEARS', category='AGE', measure='age')], key=lambda s:s['name'])
        names = [item['name'] for item in specs] + ['AUTOML_CURRENT_AVAILABLE']
        if len(set(names)) != len(names):
            raise ValueError('Feature name collision')
        if previous and names != previous['feature_names']:
            raise ValueError('Refresh changed frozen feature definitions')
        lookup = {(p, d.strftime('%Y-%m-%d')):i for i,(p,d) in enumerate(meta[KEYS].itertuples(index=False,name=None))}
        x_path = output / 'automl.npy'
        tmp_path = output / 'automl.partial.npy'
        x = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=np.float32, shape=(len(meta), len(names)))
        x[:] = np.nan
        x[:, -1] = 0
        seen = set()
        age_col = names.index('AML_AGE_YEARS')
        for p,d,e,a,ac in db.execute("SELECT p,d,e,a,ac FROM raw WHERE k='COVERAGE' ORDER BY p,d"):
            row = lookup.get((p,d))
            if row is None:
                continue
            if row in seen:
                raise ValueError('Duplicate coverage key')
            seen.add(row)
            if e is not None:
                event = pd.Timestamp(e)
                cutoff = pd.Timestamp(d)
                if not cutoff-pd.DateOffset(months=12) <= event <= cutoff:
                    raise ValueError('Coverage event outside feature window')
                x[row, :-1] = 0
                x[row, -1] = 1
                if a is not None and (not np.isfinite(a) or a < 0):
                    raise ValueError('Invalid raw age')
                x[row, age_col] = np.nan if a is None or (ac is not None and ac != 0) else a
        if len(seen) != len(meta):
            raise ValueError('Incomplete requested coverage in frozen cache')
        by_category = {}
        for j,spec in enumerate(specs):
            if spec['measure'] != 'age':
                by_category.setdefault(spec['category'], {})[spec['measure']] = j
        for category, columns in by_category.items():
            cursor = db.execute("SELECT p,d,v,r,e FROM raw WHERE k='FEATURE' AND c=? ORDER BY p,d", (category,))
            while True:
                batch = cursor.fetchmany(4096)
                if not batch:
                    break
                matched=[(lookup[(p,d)],d,v,r,e) for p,d,v,r,e in batch if (p,d) in lookup]
                if not matched: continue
                part=pd.DataFrame(matched,columns=['i','d','v','r','e'])
                rows=part.i.to_numpy(dtype=np.int64)
                annual=part.v.to_numpy(dtype=float);recent=part.r.to_numpy(dtype=float)
                if (not np.isfinite(annual).all() or not np.isfinite(recent).all()
                    or (annual <= 0).any() or (recent < 0).any() or (recent > annual).any()
                    or (annual != np.floor(annual)).any() or (recent != np.floor(recent)).any()
                    or (x[rows,-1] != 1).any()):
                    raise ValueError('Invalid raw counts or unobserved feature row')
                cutoff=pd.to_datetime(part.d);event=pd.to_datetime(part.e)
                if event.isna().any() or (event > cutoff).any() or (event < cutoff-pd.DateOffset(months=12)).any():
                    raise ValueError('Feature event outside cutoff')
                x[rows,columns['count']]=annual
                x[rows,columns['recent_pct']]=np.where((annual>6)&(recent>2),np.floor(100.*recent/annual+.5),0.)
        x.flush()
        x._mmap.close()
        del x
        gc.collect()
    x = np.load(tmp_path, mmap_mode='r', allow_pickle=False)
    if any(np.isinf(x[start:start+4096]).any() for start in range(0,len(x),4096)):
        raise ValueError('Float32 conversion overflow')
    val_hash = workflow.array_sha256(x[val])
    if previous and val_hash != previous['validation_automl_sha256']:
        raise ValueError('Frozen validation feature values changed')
    receipt = dict(schema='cached_pilot_preparation_v1', source_identity=source_identity,
                   parent_metadata_sha256=manifest['parent_metadata_sha256'],
                   frozen_chunks=manifest['chunks'], cache_manifest_sha256=_file_hash(directory/'manifest.json'),
                   max_features=max_features, categories=categories, feature_names=names, feature_specs=specs,
                   category_support_scope='complete pilot TRAIN patients only; labels unused',
                   support_at_creation={c:int(support.get(c,0)) for c in categories},
                   training_records=_records(meta.loc[train]), validation_records=_records(meta.loc[val]),
                   training_keys_sha256=_key_hash(meta.loc[train]), evaluation_keys_sha256=_key_hash(meta.loc[val]),
                   evaluation_labels_sha256=workflow.digest(meta.loc[val,'RESP'].tolist()),
                   validation_automl_sha256=val_hash, automl_sha256=workflow.array_sha256(x),
                   previous_version=None if previous is None else previous['sha256'],
                   sampling='completed patient prefix of sorted extraction requests; not a random sample',
                   missingness='unobserved source window stays NaN; observed absent category is zero',
                   settings=SETTINGS, implementation=SOURCE_SHA256)
    receipt['sha256'] = workflow.digest(receipt)
    existing = output/'prepared.json'
    if existing.exists() and _load_preparation(existing) != receipt:
        x._mmap.close()
        raise ValueError('Output version already contains a different preparation; choose a new version folder')
    x._mmap.close()
    if existing.exists():
        saved=np.load(x_path,mmap_mode='r',allow_pickle=False)
        try:
            if workflow.array_sha256(saved) != receipt['automl_sha256']:
                raise ValueError('Saved AutoML matrix hash mismatch')
        finally:saved._mmap.close()
        tmp_path.unlink()
    else:
        os.replace(tmp_path, x_path)
    x = np.load(x_path, mmap_mode='r', allow_pickle=False)
    _atomic_json(existing, receipt)
    _atomic_bytes(output/'metadata.csv', meta.to_csv(index=False, lineterminator='\n').encode())
    return dict(metadata=meta, X_automl=x, feature_names=names, train_indices=np.flatnonzero(train),
                validation_indices=np.flatnonzero(val), preparation=receipt, path=str(output),
                previous_version=None if previous_version is None else str(previous_version))


def _align_group(prepared, group, name):
    if not isinstance(group, dict) or group.get('provenance',{}).get('verified') is not True:
        raise ValueError(f'{name}: verified provenance is required')
    provenance = group['provenance']
    if name == 'nlp' and (provenance.get('fit_scope') != 'train_only'
                         or provenance.get('fitted_on') not in {'full_parent_train','pilot_train'}):
        raise ValueError('NLP encoder fitting scope must be verified TRAIN only')
    source = inputs.normalize_snapshots(group['metadata'])
    target = prepared['metadata']
    indices = pd.MultiIndex.from_frame(source[KEYS]).get_indexer(pd.MultiIndex.from_frame(target[KEYS]))
    if (indices < 0).any() or _records(source.iloc[indices].reset_index(drop=True)) != _records(target):
        raise ValueError(f'{name}: missing keys, labels or split mismatch')
    values = group['values']
    names = list(group['names'])
    if values.ndim != 2 or values.shape != (len(source),len(names)) or len(set(names)) != len(names) or not names:
        raise ValueError(f'{name}: invalid feature matrix or names')
    output = Path(prepared['path'])/(name+'.aligned.npy')
    x = np.lib.format.open_memmap(output, mode='w+', dtype=np.float32, shape=(len(target),len(names)))
    for start in range(0,len(target),2048):
        block = np.asarray(values[indices[start:start+2048]], dtype=np.float32)
        if np.isinf(block).any():
            raise ValueError(f'{name}: infinite feature value')
        x[start:start+len(block)] = block
    x.flush()
    return dict(values=x, names=names, provenance=provenance,
                evaluation_sha256=workflow.array_sha256(x[prepared['validation_indices']]),
                full_sha256=workflow.array_sha256(x))


def memory_guard(required_bytes=0, min_free_gib=2):
    import psutil
    available = int(psutil.virtual_memory().available)
    needed = int(required_bytes) + int(min_free_gib*1024**3)
    if available < needed:
        raise MemoryError(f'Pause analysis to preserve extraction: available {available/1024**3:.2f} GiB; '
                          f'need {needed/1024**3:.2f} GiB including reserve')
    return available


def _design(groups, group_names, rows, path):
    width = sum(len(groups[name]['names']) for name in group_names)
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows),width))
    for start in range(0,len(rows),2048):
        ix = rows[start:start+2048]
        col = 0
        for name in group_names:
            values = groups[name]['values']
            out[start:start+len(ix),col:col+values.shape[1]] = values[ix]
            col += values.shape[1]
    out.flush()
    return out


def _check_scores(score, n):
    if np.asarray(score).shape != (n,) or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError('Expected aligned finite probabilities')


def _fit_arm(prepared, groups, arm, *, min_free_gib):
    import lightgbm as lgb
    root = Path(prepared['path'])/arm
    root.mkdir(exist_ok=True)
    ti,vi = prepared['train_indices'], prepared['validation_indices']
    group_names = ARMS[arm]
    width = sum(len(groups[n]['names']) for n in group_names)
    memory_guard((len(ti)+len(vi))*width*4*6, min_free_gib)
    identity = dict(arm=arm, preparation_sha256=prepared['preparation']['sha256'],
                    groups={n:dict(names=groups[n]['names'],full_sha256=groups[n]['full_sha256'],
                                   provenance=groups[n]['provenance']) for n in group_names},
                    settings=SETTINGS, implementation=SOURCE_SHA256,
                    lightgbm_version=lgb.__version__, numpy_version=np.__version__)
    receipt_path = root/'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['identity'] != identity:
            raise ValueError('Saved arm identity differs; use a new version folder')
        for name, sha in receipt['files'].items():
            if _file_hash(root/name) != sha:
                raise ValueError('Saved arm artifact hash mismatch')
        return {s:np.load(root/(s+'.npy'),allow_pickle=False) for s in ('train','validation')}, receipt
    xt = xv = None
    started = time.perf_counter()
    try:
        xt = _design(groups,group_names,ti,root/'train_design.npy')
        xv = _design(groups,group_names,vi,root/'validation_design.npy')
        meta = prepared['metadata']
        y = meta.RESP.to_numpy(dtype=np.int8)
        w = modeling.sample_weights(meta.PATIENT_ID.to_numpy()[ti], SETTINGS['weighting'])
        w = w*np.where(y[ti] == 1, modeling.class_weight(y[ti],w,SETTINGS['balance']),1.)
        params = {k:v for k,v in SETTINGS.items() if k not in {'seeds','primary_fraction','weighting','balance'}}
        model = lgb.LGBMClassifier(**params, objective='binary',metric='None',random_state=42,
                                   deterministic=True,force_col_wise=True,verbosity=-1)
        model.fit(xt,y[ti],sample_weight=w)
        if list(model.classes_) != [0,1]:
            raise ValueError('Unexpected class order')
        predictions = {}
        for split,design in [('train',xt),('validation',xv)]:
            scores = np.empty(len(design),np.float64)
            for start in range(0,len(design),4096):
                memory_guard(0,min_free_gib)
                scores[start:start+4096] = model.booster_.predict(design[start:start+4096],num_threads=2)
            _check_scores(scores,len(design))
            predictions[split] = scores
            _atomic_bytes(root/(split+'.npy'), workflow.array_bytes(scores))
        _atomic_bytes(root/'model.txt',model.booster_.model_to_string().encode())
        receipt = dict(identity=identity,elapsed_seconds=time.perf_counter()-started,
                       files={n:_file_hash(root/n) for n in ('train.npy','validation.npy','model.txt')})
        _atomic_json(receipt_path,receipt)
        return predictions,receipt
    finally:
        for array in (xt,xv):
            if array is not None and hasattr(array,'_mmap'):
                array._mmap.close()
        gc.collect()


def _overlap(y,a,b):
    k = int(np.ceil(.1*len(y)))
    positive = np.asarray(y) == 1
    sa = set(np.lexsort((np.arange(len(y)),-np.asarray(a)))[:k])
    sb = set(np.lexsort((np.arange(len(y)),-np.asarray(b)))[:k])
    positives = set(np.flatnonzero(positive))
    return dict(capacity=k,positive_gained=len((sa-sb)&positives),
                positive_lost=len((sb-sa)&positives),positive_shared=len(sa&sb&positives),
                selected_shared=len(sa&sb),additional_captures=len(sa&positives)-len(sb&positives))


def _report(prepared, predictions, groups, receipts, blocked, bootstrap):
    meta=prepared['metadata']
    rows=[]
    for arm,by_split in predictions.items():
        for split,idx in [('train',prepared['train_indices']),('validation',prepared['validation_indices'])]:
            m=meta.iloc[idx]
            score=by_split[split]
            _check_scores(score,len(m))
            row=dict(Model=arm,Split=split.upper(),Snapshots=len(m),Patients=m.PATIENT_ID.nunique(),
                     Positive_snapshots=int(m.RESP.sum()))
            for fraction in (.1,.2,.3):
                metric=comparison.topk_metrics(m.RESP.to_numpy(),score,fraction)
                suffix=str(round(fraction*100))+'%'
                row.update({'Lift@'+suffix:metric['lift'],'Captured@'+suffix:metric['selected_positives'],
                            'Selected@'+suffix:metric['selected']})
            rows.append(row)
    vi=prepared['validation_indices']
    val=meta.iloc[vi]
    pairs={};overlap={}
    if 'AUTOML' in predictions:
        baseline=predictions['AUTOML']['validation']
        for arm in predictions:
            if arm=='AUTOML':continue
            scores=predictions[arm]['validation']
            pairs[arm]=workflow.paired_lift_bootstrap(val.RESP.to_numpy(),scores,baseline,val.PATIENT_ID.to_numpy(),
                                                    n_bootstrap=bootstrap,seed=20261013)
            overlap[arm]=_overlap(val.RESP.to_numpy(),scores,baseline)
    p=prepared['preparation']
    report=dict(schema=SCHEMA,identity=dict(version=Path(prepared['path']).name,
                evaluation_keys_sha256=p['evaluation_keys_sha256'],evaluation_labels_sha256=p['evaluation_labels_sha256'],
                training_keys_sha256=p['training_keys_sha256'],preparation_hash=p['sha256'],settings=SETTINGS,
                train_snapshots=len(prepared['train_indices']),validation_snapshots=len(vi),
                feature_names={n:g['names'] for n,g in groups.items()},
                feature_definition_hash=workflow.digest({n:dict(names=g['names'],provenance=g['provenance']) for n,g in groups.items()}),
                protocol_hash=workflow.digest(dict(settings=SETTINGS,implementation=SOURCE_SHA256)),
                source_identity=p['source_identity'],parent_metadata_sha256=p['parent_metadata_sha256']),
                rows=rows,paired_lift=pairs,overlaps=overlap,blocked_arms=blocked,
                population=population_summary(meta).to_dict('records'),
                completed_chunks=len(p['frozen_chunks']),fit_receipts=receipts,
                provenance={n:g['provenance'] for n,g in groups.items()},
                scope='Exploratory development validation on a nonrandom completed-patient prefix; TEST unused',
                limitations=['Repeated development validation is not an independent confirmation',
                             'Latest source tables and refreshed checkpoints do not establish historical availability',
                             'Intervals condition on fitted scores and do not include refit variation'],
                metric_rule='Snapshot lift; exact ceil(fraction*N), descending scores, frozen input-order ties')
    root=Path(prepared['path'])
    _atomic_json(root/'report.json',report)
    _atomic_json(root/'report_receipt.json',dict(schema=SCHEMA,sha256=_file_hash(root/'report.json')))
    return report


def run(prepared, timing, nlp=None, *, bootstrap=1000, min_free_gib=2):
    """Run static/timing first. A second call may add verified NLP without refitting them."""
    if bootstrap < 20:
        raise ValueError('At least twenty bootstrap draws are required')
    memory_guard(0,min_free_gib)
    p=prepared['preparation']
    if _load_preparation(Path(prepared['path'])/'prepared.json') != p:
        raise ValueError('Preparation changed since construction')
    meta=inputs.normalize_snapshots(prepared['metadata'])
    if (prepared['feature_names'] != p['feature_names']
        or _records(meta.loc[meta.SPLIT.eq('train')]) != p['training_records']
        or _records(meta.loc[meta.SPLIT.eq('validation')]) != p['validation_records']
        or meta.SPLIT.eq('test').any()
        or not np.array_equal(prepared['train_indices'],np.flatnonzero(meta.SPLIT.eq('train')))
        or not np.array_equal(prepared['validation_indices'],np.flatnonzero(meta.SPLIT.eq('validation')))
        or workflow.array_sha256(prepared['X_automl']) != p['automl_sha256']):
        raise ValueError('Prepared population, split indices or AutoML values changed')
    groups={'automl':dict(values=prepared['X_automl'],names=prepared['feature_names'],
                          full_sha256=p['automl_sha256'],evaluation_sha256=p['validation_automl_sha256'],
                          provenance=dict(verified=True,fit_scope='pilot_train_frequency',cap=p['max_features']))}
    groups['time']=_align_group(prepared,timing,'time')
    blocked={}
    if nlp is None:
        blocked={arm:'Verified matching NLP artifacts unavailable' for arm in ARMS if 'nlp' in ARMS[arm]}
    else:
        try:
            groups['nlp']=_align_group(prepared,nlp,'nlp')
        except (ValueError,FileNotFoundError) as exc:
            blocked={arm:str(exc) for arm in ARMS if 'nlp' in ARMS[arm]}
    definition_path=Path(prepared['path'])/'groups.json'
    definitions={n:dict(names=g['names'],provenance=g['provenance'],evaluation_sha256=g['evaluation_sha256'])
                 for n,g in groups.items()}
    if prepared['previous_version'] is not None:
        prior_path=Path(prepared['previous_version'])
        if prior_path.is_file():prior_path=prior_path.parent
        prior_groups=json.loads((prior_path/'groups.json').read_text())
        for name,current in definitions.items():
            if name not in prior_groups or prior_groups[name] != current:
                raise ValueError('Refresh changed frozen feature definitions or validation values')
    if definition_path.exists():
        old=json.loads(definition_path.read_text())
        for name,value in old.items():
            if name in definitions and definitions[name]!=value:
                raise ValueError('Feature group changed within a version')
        definitions=dict(old,**definitions)
    _atomic_json(definition_path,definitions)
    predictions={};receipts={}
    report=None
    for arm,group_names in ARMS.items():
        if arm in blocked:continue
        score,receipt=_fit_arm(prepared,groups,arm,min_free_gib=min_free_gib)
        predictions[arm]=score;receipts[arm]=receipt
        report=_report(prepared,predictions,groups,receipts,blocked,bootstrap)
        print(arm,'saved; validation Lift@10%:',next(r['Lift@10%'] for r in report['rows']
                                                   if r['Model']==arm and r['Split']=='VALIDATION'))
    return report
