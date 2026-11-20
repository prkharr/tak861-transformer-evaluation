"""Resumable development challenger on existing frozen SUMMARY tensors.

All label-dependent selection uses TRAIN only. The reused NLP representation
was fitted without labels on all TRAIN documents: OOF estimates are conditional
on this representation, not a fully inductive vocabulary-refitting experiment.
"""
import hashlib
import io
import itertools
import json
import time
import importlib.metadata
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit


DEFAULTS = dict(seconds=28800, seeds=[42, 142, 242], folds=3, threads=8,
                device='auto', bootstrap=2000, selection_sizes=[16, 32, 64, 128, 256, 512, 1028],
                lgb_rounds=1000, epochs=30, patience=5, batch_size=256)
LIMITATIONS = [
    'Development validation has already influenced prior experiments; it is not an untouched holdout.',
    'Frozen NLP vocabulary and IDF used all TRAIN documents without labels. OOF is conditional on this representation.',
    'Snapshot captures are not unique-patient interventions or HCP targeting performance.',
    'Conditional patient bootstrap excludes model-selection and retraining uncertainty.',
    'Production comparison is disabled. No improvement over production is established.',
    'The cooperative budget cannot interrupt warehouse transfers or feature-selection library calls.',
    'Blend weights use seed-42, per-fold-tuned OOF models; final predictions average three seeds with a globally selected configuration.',
    'The neural challenger retains the fusion architecture but uses a simplified fixed-learning-rate AdamW schedule.',
]


class Budget:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def check(self):
        if time.monotonic() >= self.end:
            raise TimeoutError('Run budget reached. Rerun with identical settings to reuse completed stages.')


def grids():
    trees = [dict(num_leaves=n, min_child_samples=m, learning_rate=r)
             for n, m, r in itertools.product([16, 31], [20, 100, 200], [.03, .05])]
    neural = [dict(d_model=d, dropout=p, weight_decay=w)
              for d, p, w in itertools.product([64, 128], [.1, .2], [.001, .01])]
    return dict(STATIC_LGBM=trees, TIME_LGBM=trees, NLP_FUSION=neural)


def audit(data, prep, protocol, wf, timing, exact):
    meta = data['metadata']
    if protocol.get('experiment_phase') != 'SUMMARY':
        raise ValueError('Use existing SUMMARY tensors and preparation.')
    if (protocol['identity']['input_manifest_hash'] != wf.digest(data['manifest'])
            or protocol['preparation_hash'] != wf.digest(prep)):
        raise ValueError('Input/preparation identity mismatch')
    if meta.duplicated(['PATIENT_ID', 'END_DT']).any() or meta.groupby('PATIENT_ID').SPLIT.nunique().gt(1).any():
        raise ValueError('Duplicate snapshot keys or patient overlap')
    contract = data['manifest']['panel_manifest']['contract']
    if contract['fit_scope'] != 'deterministic_raw':
        raise ValueError('Raw deterministic AutoML tensors are required for fold-local supervised selection')
    cc = data['manifest']['count_contract']
    if cc['orientation'] != '0=newest' or cc['date_policy'] != 'month_end_history':
        raise ValueError('Unverified timing orientation')
    selected = timing.channels(cc['feature_names'], exact)
    nc = data.get('nlp_manifest', {})
    if 'X_nlp' not in data or nc.get('fit_scope') != 'train_only' or nc.get('orientation') != '0=newest':
        raise ValueError('Aligned TRAIN-only NLP tensors are required')
    for split in ('train', 'validation'):
        rows = data['indices'][split]
        if not len(rows) or set(np.unique(data['y'][rows])) != {0, 1}:
            raise ValueError('Both classes required in ' + split)
    return dict(status='READY_DEVELOPMENT_CONDITIONAL_NLP',
                population=[dict(split=s, snapshots=len(data['indices'][s]),
                    patients=meta.iloc[data['indices'][s]].PATIENT_ID.nunique(),
                    positive_snapshots=int(data['y'][data['indices'][s]].sum()))
                    for s in ('train', 'validation')],
                timing_channels=[cc['feature_names'][i] for i in selected],
                limitations=LIMITATIONS, production_enabled=False, test_scoring_enabled=False)


def inner_split(rows, y, groups, seed):
    # Retry only to obtain both classes; never optimize a score when splitting.
    for attempt in range(100):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed+attempt).split(rows, groups=groups[rows]))
        if all(set(np.unique(y[rows[t]])) == {0, 1} for t in (a, b)):
            return rows[a], rows[b]
    raise ValueError('Cannot create a two-class patient-separated inner split')


def read_bundle(table, wh):
    # The warehouse API requires the complete artifact-name set, not a subset.
    blob=wh.read_artifacts(table, {'bundle.zip'})['bundle.zip']
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)) or 'receipt.json' not in names:
                raise ValueError('Invalid bundle manifest')
            receipt=json.loads(archive.read('receipt.json'))
            if set(names)!=set(receipt['hashes'])|{'receipt.json'}:
                raise ValueError('Unexpected bundle files')
            blobs={k:archive.read(k) for k in receipt['hashes']}
        if any(hashlib.sha256(blobs[k]).hexdigest()!=h for k,h in receipt['hashes'].items()):
            raise ValueError('Payload checksum mismatch')
    except (ValueError,KeyError,zipfile.BadZipFile) as exc:
        raise ValueError('Corrupt saved stage: '+table) from exc
    return receipt,blobs


class Store:
    def __init__(self, prefix, identity, wf, wh):
        self.wf, self.wh = wf, wh
        self.identity = identity
        self.prefix = prefix + '_CHALLENGER_' + wf.digest(identity)[:12].upper()

    def table(self, stage):
        return self.prefix + '_' + stage

    def read(self, stage):
        if not self.wh.table_exists(self.table(stage)):
            return None
        receipt,blobs = read_bundle(self.table(stage),self.wh)
        if receipt['identity'] != self.identity or receipt['stage'] != stage:
            raise ValueError('Resume identity mismatch')
        return blobs

    def save(self, stage, blobs):
        receipt = dict(identity=self.identity, stage=stage,
                       hashes={k:hashlib.sha256(v).hexdigest() for k,v in blobs.items()})
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED) as archive:
            for name,blob in sorted(blobs.items()): archive.writestr(name,blob)
            archive.writestr('receipt.json',self.wf.json_bytes(receipt))
        self.wh.save_artifacts(self.table(stage), {'bundle.zip':out.getvalue()})


def preparation(data, rows, cfg, mdl):
    y, g = data['y'][rows], data['metadata'].PATIENT_ID.to_numpy()[rows]
    settings = dict(candidate_sizes=cfg['selection_sizes'], n_splits=3, seed=42)
    a = mdl.prepare_features(data['X_summary'], data['X_summary'], y, g, data['feature_names'],
        valid_train=data['summary_mask'], row_indices=rows, sequence_steps=1,
        feature_kinds=[s['kind'] for s in data['manifest']['feature_specs']], config=settings)
    c = mdl.prepare_features(data['X_count'][:, :1], data['X_count'], y, g,
        data['manifest']['count_contract']['feature_names'], valid_train=data['count_mask'],
        row_indices=rows, feature_kinds=['count']*data['X_count'].shape[-1], config=settings)
    return dict(automl=a, count=c)


def design(data, rows, prep, family, mdl, timing, selected):
    if family != 'NLP_FUSION':
        a = mdl.transform_inputs(data['X_summary'], prep['automl'], data['summary_mask'], row_indices=rows)
        try:
            result = np.column_stack([a[:, 0], data['summary_mask'][rows, 0].astype(np.float32)])
        finally:
            mdl.release_array(a)
        if family == 'TIME_LGBM':
            result = np.column_stack([result, timing.temporal_columns(data['X_count'], data['count_mask'], rows, selected)])
        return result, None
    a = mdl.transform_inputs(data['X_count'], prep['count'], data['count_mask'], row_indices=rows)
    result=mdl.allocate_array((len(rows),a.shape[1],a.shape[-1]+data['X_nlp'].shape[-1]+1),np.float32)
    mask = np.asarray(data['count_mask'][rows], bool)
    try:
        for start in range(0,len(rows),256):
            end=min(start+256,len(rows)); ids=rows[start:end]; k=a.shape[-1]
            result[start:end,:,:k]=a[start:end]
            result[start:end,:,k:-1]=data['X_nlp'][ids]
            result[start:end,:,-1]=data['nlp_mask'][ids]
            block=result[start:end]; block[~mask[start:end]]=0
    except BaseException:
        mdl.release_array(result)
        raise
    finally:
        mdl.release_array(a)
    if not mask.any(axis=1).all():
        mdl.release_array(result)
        raise ValueError('NLP fusion requires at least one valid count month per row')
    return result, mask


def fit(family, x, mask, y, groups, xe, me, ye, params, cfg, mdl, budget, seed, fixed=None):
    """Inner-stop fit or fixed-round refit. Final/outer labels never select rounds."""
    import torch
    budget.check()
    started = time.monotonic()
    weights = mdl.sample_weights(groups, 'patient')
    positive_weight = mdl.class_weight(y, weights, 'sqrt')
    if family != 'NLP_FUSION':
        import lightgbm as lgb
        def deadline(env):
            budget.check()
        deadline.order=5
        deadline.before_iteration=True
        callbacks = [deadline]
        kwargs = {}
        if fixed is None:
            def lift_metric(yt, pred):
                return 'lift10', mdl.lift_at_fraction(yt, pred, .1), True
            kwargs = dict(eval_set=[(xe, ye)], eval_metric=lift_metric)
            callbacks.append(lgb.early_stopping(40, first_metric_only=True, verbose=False))
        model = lgb.LGBMClassifier(**params, n_estimators=int(fixed or cfg['lgb_rounds']),
            objective='binary', metric='None', random_state=seed, n_jobs=cfg['threads'],
            verbosity=-1, deterministic=True, force_col_wise=True)
        model.fit(x, y, sample_weight=weights*np.where(y == 1, positive_weight, 1.), callbacks=callbacks, **kwargs)
        rounds = int(fixed or model.best_iteration_ or cfg['lgb_rounds'])
        return dict(scores=model.booster_.predict(xe), train_scores=model.booster_.predict(x),
                    rounds=rounds, model=model.booster_.model_to_string().encode(),
                    seconds=time.monotonic()-started, history=[])
    from torch import nn
    mdl.seed_run(seed)
    torch.set_num_threads(cfg['threads'])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device']=='auto' else cfg['device']
    model_cfg = dict(params, heads=4, depth=1, ff=128, device=device, threads=cfg['threads'])
    model = mdl.make_encoder(x.shape[-1], x.shape[1], model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=params['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(positive_weight,device=device))
    rng = np.random.default_rng(seed)
    def predict(xx, mm):
        model.eval(); out=[]
        with torch.no_grad():
            for start in range(0, len(xx), cfg['batch_size']):
                budget.check()
                out.append(torch.sigmoid(model(torch.as_tensor(xx[start:start+cfg['batch_size']],device=device),
                    torch.as_tensor(mm[start:start+cfg['batch_size']],device=device))).cpu().numpy())
        return np.concatenate(out)
    best, state, stale, rounds, history = -np.inf, None, 0, 0, []
    for epoch in range(1, int(fixed or cfg['epochs'])+1):
        model.train(); order=rng.permutation(len(y)); loss_sum=0.
        for start in range(0, len(y), cfg['batch_size']):
            budget.check(); ix=order[start:start+cfg['batch_size']]
            opt.zero_grad(set_to_none=True)
            pred=model(torch.as_tensor(x[ix],device=device),torch.as_tensor(mask[ix],device=device))
            loss=(loss_fn(pred,torch.as_tensor(y[ix],dtype=torch.float32,device=device))*torch.as_tensor(weights[ix],device=device)).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite neural loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
            loss_sum += loss.item()*len(ix)
        score = None if fixed is not None else mdl.lift_at_fraction(ye,predict(xe,me),.1)
        history.append(dict(epoch=epoch, loss=loss_sum/len(y), inner_lift=score))
        if fixed is not None or score > best:
            best=score if score is not None else best; rounds=epoch; stale=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if fixed is None and stale>=cfg['patience']: break
    model.load_state_dict(state)
    out=io.BytesIO(); torch.save(dict(state=state, config=model_cfg, input_shape=list(x.shape[1:])),out)
    return dict(scores=predict(xe,me), train_scores=predict(x,mask), rounds=rounds,
                model=out.getvalue(), seconds=time.monotonic()-started, history=history)


def blend_weight(y, a, b, mdl):
    values=[(float(w),mdl.lift_at_fraction(y,w*a+(1-w)*b,.1)) for w in np.linspace(0,1,21)]
    # Prespecified tie: prefer smaller NLP weight, including parent endpoints.
    return max(values,key=lambda t:(t[1],-t[0]))[0], values


def metrics(y, scores, patients, mdl):
    result=mdl.quick_metrics(y,scores)
    order=np.argsort(-scores,kind='stable')
    for fraction in (.1,.2,.3):
        selected=order[:int(np.ceil(fraction*len(y)))]; positives=selected[y[selected]==1]
        suffix=str(int(fraction*100))
        result['Lift@'+suffix+'%']=mdl.lift_at_fraction(y,scores,fraction)
        result['Captured@'+suffix+'%']=len(positives)
        result['Unique_positive_patients@'+suffix+'%']=len(np.unique(patients[positives]))
    return result


def run(data, parent_prep, protocol, prefix, source_hash, settings, exact, mdl, wf, wh, timing, pilot):
    import torch
    if set(settings)-set(DEFAULTS): raise ValueError('Unknown challenger settings')
    cfg=dict(DEFAULTS,**settings)
    for k in ('seconds','folds','threads','bootstrap','lgb_rounds','epochs','patience','batch_size'):
        if type(cfg[k]) is not int or cfg[k]<1: raise ValueError('Invalid setting: '+k)
    if cfg['folds']<3 or cfg['bootstrap']<20: raise ValueError('At least 3 folds and 20 bootstrap draws required')
    if not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']): raise ValueError('Unique seeds required')
    if cfg['device']=='auto': cfg['device']='cuda' if torch.cuda.is_available() else 'cpu'
    budget=Budget(cfg['seconds'])
    audited=audit(data,parent_prep,protocol,wf,timing,exact)
    refs, lock=pilot.frozen_benchmarks(data,parent_prep,protocol,wf,wh,prefix)
    if 'NLP_FUSION' not in refs:
        raise ValueError('The saved NLP_FUSION incumbent is required before starting the challenger')
    if not wh._has_artifact_jvm():
        raise ValueError('Use classic Spark compute with JVM access for atomic resumable artifact publication')
    identity=dict(source=source_hash, parent=wf.digest(protocol), inputs=wf.digest(data['manifest']),
        config={k:v for k,v in cfg.items() if k!='seconds'}, channels=audited['timing_channels'], lock=lock,
        versions={p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','torch','lightgbm')})
    store=Store(prefix,identity,wf,wh)
    complete=store.read('COMPLETE')
    if complete is not None:
        return store.table('COMPLETE'), json.loads(complete['report.json'])
    y=np.asarray(data['y']); groups=data['metadata'].PATIENT_ID.to_numpy()
    ti,vi=(np.asarray(data['indices'][s]) for s in ('train','validation'))
    selected=timing.channels(data['manifest']['count_contract']['feature_names'],exact)
    families=grids(); oof={f:np.full(len(ti),np.nan) for f in families}; tuning={f:[] for f in families}
    def getprep(stage,rows):
        saved=store.read(stage)
        if saved:
            if not np.array_equal(wf.bytes_array(saved['rows.npy']),rows): raise ValueError('Preparation rows changed')
            return json.loads(saved['prep.json'])
        print('Preparing',stage,'on',len(rows),'TRAIN snapshots',flush=True)
        budget.check(); started=time.monotonic(); p=preparation(data,rows,cfg,mdl)
        store.save(stage,{'prep.json':wf.json_bytes(p),'rows.npy':wf.array_bytes(rows),
                         'timing.json':wf.json_bytes({'seconds':time.monotonic()-started})})
        return p
    def stagefit(stage,family,train,evaluate,p,params,seed,fixed=None):
        saved=store.read(stage)
        if saved:
            if not np.array_equal(wf.bytes_array(saved['train_rows.npy']),train) or not np.array_equal(wf.bytes_array(saved['score_rows.npy']),evaluate):
                raise ValueError('Fit or scoring rows changed')
            return json.loads(saved['fit.json']),wf.bytes_array(saved['scores.npy'])
        print('Fitting',stage,params,'fixed rounds:',fixed,flush=True)
        budget.check()
        x=xe=None
        try:
            x,m=design(data,train,p,family,mdl,timing,selected)
            xe,me=design(data,evaluate,p,family,mdl,timing,selected)
            result=fit(family,x,m,y[train],groups[train],xe,me,None if fixed else y[evaluate],params,cfg,mdl,budget,seed,fixed)
        finally:
            mdl.release_array(x); mdl.release_array(xe)
        record={k:result[k] for k in ('rounds','seconds','history')}
        record.update(params=params, seed=seed, fixed_rounds=fixed,
            train_lift=mdl.lift_at_fraction(y[train],result['train_scores'],.1))
        if fixed is None: record['inner_lift']=mdl.lift_at_fraction(y[evaluate],result['scores'],.1)
        store.save(stage,{'fit.json':wf.json_bytes(record),'scores.npy':wf.array_bytes(result['scores']),
            'model.bin':result['model'],'train_rows.npy':wf.array_bytes(train),'score_rows.npy':wf.array_bytes(evaluate)})
        print('Saved',stage,'rounds:',record['rounds'],'seconds:',round(record['seconds'],1),flush=True)
        return record,result['scores']
    splitter=StratifiedGroupKFold(cfg['folds'],shuffle=True,random_state=42)
    for fold,(fitpos,holdpos) in enumerate(splitter.split(ti,y[ti],groups[ti])):
        outer,hold=ti[fitpos],ti[holdpos]
        if set(groups[outer])&set(groups[hold]): raise ValueError('Patient leakage')
        inner,stop=inner_split(outer,y,groups,142+fold)
        ip=getprep('F%d_INNER_PREP'%fold,inner)
        op=getprep('F%d_OUTER_PREP'%fold,outer)
        for family,grid in families.items():
            records=[]
            for n,params in enumerate(grid):
                record,_=stagefit('F%d_%s_T%02d'%(fold,family,n),family,inner,stop,ip,params,42)
                records.append(record)
            tuning[family].append(records)
            best=max(range(len(records)),key=lambda k:(records[k]['inner_lift'],-k))
            _,scores=stagefit('F%d_%s_OOF'%(fold,family),family,outer,hold,op,grid[best],42,records[best]['rounds'])
            oof[family][holdpos]=scores
    if any(not np.isfinite(a).all() for a in oof.values()): raise ValueError('Incomplete OOF coverage')
    weight,curve=blend_weight(y[ti],oof['NLP_FUSION'],oof['TIME_LGBM'],mdl)
    chosen={}
    for family,grid in families.items():
        best=max(range(len(grid)),key=lambda k:(np.mean([r[k]['inner_lift'] for r in tuning[family]]),-k))
        chosen[family]=dict(params=grid[best],rounds=max(1,int(np.median([r[best]['rounds'] for r in tuning[family]]))))
    # Freeze model choices and blend before any external validation prediction.
    selection=dict(chosen=chosen,nlp_blend_weight=weight,curve=[list(v) for v in curve],
        oof_scope=LIMITATIONS[1],selection_split='TRAIN only')
    if store.read('SELECTION') is None:
        store.save('SELECTION',dict({'selection.json':wf.json_bytes(selection)},
            **{f+'.npy':wf.array_bytes(a) for f,a in oof.items()}))
    finalprep=getprep('FINAL_PREP',ti); preds={}; fit_records={}
    for family in families:
        values=[]; fit_records[family]=[]
        for seed in cfg['seeds']:
            record,score=stagefit('FINAL_%s_S%d'%(family,seed),family,ti,vi,finalprep,
                chosen[family]['params'],seed,chosen[family]['rounds'])
            record=dict(record,validation_lift=mdl.lift_at_fraction(y[vi],score,.1))
            record['train_validation_gap']=record['train_lift']-record['validation_lift']
            fit_records[family].append(record); values.append(score)
        preds[family+'_CHALLENGER']=np.mean(values,axis=0)
    preds['OOF_WEIGHTED_BLEND']=weight*preds['NLP_FUSION_CHALLENGER']+(1-weight)*preds['TIME_LGBM_CHALLENGER']
    combined=dict(refs,**preds)
    rows=[dict(Model=name,**metrics(y[vi],score,groups[vi],mdl)) for name,score in combined.items()]
    incumbent=set(np.argsort(-refs['NLP_FUSION'],kind='stable')[:int(np.ceil(.1*len(vi)))])
    overlap=[]
    for name,scores in preds.items():
        selected_set=set(np.argsort(-scores,kind='stable')[:int(np.ceil(.1*len(vi)))])
        new=selected_set-incumbent; lost=incumbent-selected_set
        overlap.append(dict(Model=name,shared_top10_snapshots=len(selected_set&incumbent),
            additional_positive_snapshots=sum(int(y[vi[i]]) for i in new),
            lost_positive_snapshots=sum(int(y[vi[i]]) for i in lost)))
    report=dict(status='COMPLETE_DEVELOPMENT',identity=identity,audit=audited,selection=selection,
        rows=rows,fit_records=fit_records,overlap_vs_nlp=overlap,limitations=LIMITATIONS,
        oof_metrics={f:metrics(y[ti],a,groups[ti],mdl) for f,a in oof.items()},
        note='OOF metrics used for blending are not independent performance estimates.')
    blobs={'report.json':wf.json_bytes(report),'y.npy':wf.array_bytes(y[vi]),
           'patients.json':wf.json_bytes(groups[vi].tolist()),
           'keys.csv':data['metadata'].iloc[vi][['PATIENT_ID','END_DT']].to_csv(index=False).encode()}
    blobs.update({name+'.npy':wf.array_bytes(a) for name,a in combined.items()})
    store.save('COMPLETE',blobs)
    return store.table('COMPLETE'),report


def report_saved(table, wf, wh, cmp, draws=2000):
    receipt,blobs=read_bundle(table,wh)
    if receipt['stage']!='COMPLETE': raise ValueError('Not a complete benchmark')
    report=json.loads(blobs['report.json'])
    y=wf.bytes_array(blobs['y.npy']); patients=np.asarray(json.loads(blobs['patients.json']))
    scores={r['Model']:wf.bytes_array(blobs[r['Model']+'.npy']) for r in report['rows']}
    reference='NLP_FUSION'
    if reference not in scores: raise ValueError('Frozen NLP incumbent unavailable')
    intervals={name:wf.paired_lift_bootstrap(y,s,scores[reference],patients,n_bootstrap=draws,seed=20261120)
        for name,s in scores.items() if name.endswith('_CHALLENGER') or name=='OOF_WEIGHTED_BLEND'}
    report['paired_intervals_vs_frozen_nlp']=intervals
    additional={}
    if 'LIGHTGBM_TRAIN_ONLY_STATIC' in scores:
        for name in ('NLP_FUSION_CHALLENGER','TIME_LGBM_CHALLENGER','OOF_WEIGHTED_BLEND'):
            additional[name+' minus LIGHTGBM_TRAIN_ONLY_STATIC']=wf.paired_lift_bootstrap(y,scores[name],scores['LIGHTGBM_TRAIN_ONLY_STATIC'],patients,n_bootstrap=draws,seed=20261120)
    additional['TIME_LGBM_CHALLENGER minus STATIC_LGBM_CHALLENGER']=wf.paired_lift_bootstrap(y,scores['TIME_LGBM_CHALLENGER'],scores['STATIC_LGBM_CHALLENGER'],patients,n_bootstrap=draws,seed=20261120)
    report['additional_paired_lift_intervals']=additional
    return report
