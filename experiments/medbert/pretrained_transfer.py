"""Frozen biomedical TEXT encoder transfer; explicitly not structured Med-BERT."""
import hashlib
import io
import re
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

MODEL_ID = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
REVISION = 'e1354b7a3a09615f6aba48dfad4b7a613eef7062'
WEIGHT_SHA256 = 'ad7bbb66376cfd6b2db3447192b034efe016337cbef135c35c411fd61b13c193'
EXPECTED_HASHES = {'pytorch_model.bin': WEIGHT_SHA256,
    'config.json':'56bab767c02bc792897638feb8addc06d483d9c73769a5f554bbabea1f5507a1',
    'vocab.txt':'79489a52be45e6fa033521e8ce8e4f62aedc0a742ee2aa6fc04667e5b0b1454d'}
SETTINGS = dict(seed=42, max_train_patients=2000, max_train_rows=25000,
    total_seconds=3300, save_reserve=120, bootstrap_draws=200,
    embedding_batch=16, projection_dim=64, max_iter=150, C=0.1,
    code_prefixes=['DX__', 'PX__', 'RX__'], device='auto', threads=4)

def description(name):
    # Use actual frozen dictionary wording; never invent ICD descriptions.
    return re.sub(r'\s+', ' ', re.sub(r'[_]+', ' ', name)).strip()

def encode(names, cfg, budget, local_path=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from huggingface_hub import snapshot_download
    from pathlib import Path
    torch.set_num_threads(cfg['threads'])
    budget.require(cfg['save_reserve'])
    path = local_path or snapshot_download(MODEL_ID, revision=REVISION,
        allow_patterns=['config.json', 'tokenizer_config.json', 'vocab.txt',
                        'pytorch_model.bin', 'README.md', 'LICENSE.md'])
    path = Path(path)
    # Both online and local mirrors are verified against independently pinned hashes.
    hashes = {}
    for name in ('config.json', 'vocab.txt', 'pytorch_model.bin'):
        h = hashlib.sha256()
        with (path / name).open('rb') as f:
            for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
        hashes[name] = h.hexdigest()
    if hashes != EXPECTED_HASHES:
        raise ValueError('Weights/config/vocabulary differ from the published pinned checkpoint')
    print('Verified pretrained checkpoint:', MODEL_ID, REVISION, flush=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model, info = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True,
        trust_remote_code=False, output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Checkpoint is incomplete; refusing random-weight fallback: ' + str(info))
    device = 'cuda' if cfg['device'] == 'auto' and torch.cuda.is_available() else ('cpu' if cfg['device'] == 'auto' else cfg['device'])
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; select auto or CPU')
    model.to(device).eval()
    print('Encoding', len(names), 'feature descriptions locally on', device, flush=True)
    vectors = []
    for start in range(0, len(names), cfg['embedding_batch']):
        budget.require(cfg['save_reserve'])
        texts = [description(n) for n in names[start:start+cfg['embedding_batch']]]
        batch = tok(texts, padding=True, truncation=True, max_length=64, return_tensors='pt', return_special_tokens_mask=True)
        valid = (batch['attention_mask'] & ~batch.pop('special_tokens_mask').bool()).to(device)
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.inference_mode():
            hidden = model.base_model(**batch).last_hidden_state
            pooled = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(-1)
        vectors.append(pooled.cpu().numpy())
    receipt = dict(model_id=MODEL_ID, revision=REVISION, hashes=hashes, device=device,
                   encoder_frozen=True, loading_info=info)
    del model
    if device == 'cuda': torch.cuda.empty_cache()
    return np.concatenate(vectors), receipt

def features(data, rows, columns, embeddings, budget, reserve):
    """All TRAIN-observed eligible channels, two recency windows; no top-five truncation."""
    if data['metadata'].iloc[rows].SPLIT.eq('test').any():
        raise ValueError('TEST feature generation is disabled')
    counts, semantic = [], []
    for start in range(0, len(rows), 256):
        budget.require(reserve)
        idx = rows[start:start+256]
        x = np.asarray(data['X_count'][idx], dtype=np.float32)[:, :, columns]
        mask = data['count_mask'][idx]
        if not np.isfinite(x).all() or (x < 0).any(): raise ValueError('Invalid counts')
        x = np.where(mask[:, :, None], x, 0)
        cb, eb = [], []
        for lo, hi in ((0, 3), (3, 12)):
            c = np.log1p(x[:, lo:hi].sum(1))
            cb.append(c)
            eb.append(c @ embeddings / np.maximum(c.sum(1, keepdims=True), 1e-8))
        counts.append(np.concatenate(cb + [mask.astype(np.float32)], axis=1))
        semantic.append(np.concatenate(eb, axis=1))
    return np.concatenate(counts), np.concatenate(semantic)

def fit_head(x, y, xv, weights, cfg):
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=cfg['C'], max_iter=cfg['max_iter'], solver='lbfgs', random_state=cfg['seed'])
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(scaler.transform(x), y, sample_weight=weights)
    p = model.predict_proba(scaler.transform(xv))[:, 1]
    state = dict(mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_, intercept=model.intercept_)
    return p, state, dict(iterations=model.n_iter_.tolist(), warnings=[str(m.message) for m in messages])

def run(data, prep, protocol, prefix, source_hash, cfg, pilot, wf, wh, cmp, mdl, started, local_path=None):
    cfg = dict(SETTINGS, **cfg)
    for name in ('max_train_patients','max_train_rows','bootstrap_draws','embedding_batch',
                 'projection_dim','max_iter','threads'):
        if type(cfg[name]) is not int or cfg[name] < 1: raise ValueError('Positive integer required: '+name)
    if not 0 < cfg['total_seconds'] <= 3600 or cfg['C'] <= 0:
        raise ValueError('Require positive C and runtime at most one hour')
    if not cfg['code_prefixes'] or any(p not in ('DX__','PX__','RX__') for p in cfg['code_prefixes']):
        raise ValueError('Use explicit coded feature prefixes only')
    budget = pilot.Budget(cfg['total_seconds'], started)
    names = pilot.validate_data(data)
    tr = pilot.train_subset(data['metadata'], cfg)
    va = data['indices']['validation']
    existing, lock_hash = pilot.frozen_benchmarks(data, prep, protocol, wf, wh, prefix)
    columns = pilot.fit_vocabulary(data, tr, cfg, budget)
    vocab = [names[i] for i in columns]
    identity = dict(source_hash=source_hash, cfg=cfg, input_hash=wf.digest(data['manifest']), lock_hash=lock_hash,
        checkpoint=dict(model_id=MODEL_ID, revision=REVISION, hashes=EXPECTED_HASHES),
        train_rows=wf.array_sha256(tr), validation_rows=wf.array_sha256(va), vocabulary=wf.digest(vocab))
    table = prefix + '_BIOMEDBERT_TEXT_V1_' + wf.digest(identity)[:12].upper()
    if wh.table_exists(table):
        import json
        report = json.loads(wh.read_artifacts(table, {'report.json'})['report.json'])
        expected = {'heads_and_embeddings.npz', 'MATCHED_COUNT_LR.npy',
                    'COUNT_PLUS_RANDOM_LR.npy', 'COUNT_PLUS_BIOMEDBERT_LR.npy'}
        if report.get('identity') != identity or set(report.get('payload_hashes', {})) != expected:
            raise ValueError('Saved report identity/payload manifest mismatch')
        saved = wh.read_artifacts(table, expected)
        if any(hashlib.sha256(saved[k]).hexdigest() != report['payload_hashes'][k] for k in expected):
            raise ValueError('Saved payload checksum mismatch')
        print('Reusing verified completed artifact:', table, flush=True)
        return table, report
    print('Frozen inputs loaded. TRAIN subset:', len(tr), '; full validation:', len(va), flush=True)
    embeddings, receipt = encode(vocab, cfg, budget, local_path)
    # Fixed Gaussian projection reduces compute, fitted on neither labels nor validation.
    rng = np.random.default_rng(cfg['seed'])
    projection = rng.normal(size=(embeddings.shape[1], cfg['projection_dim'])).astype('float32') / np.sqrt(cfg['projection_dim'])
    embedding_mean = embeddings.mean(0, keepdims=True)
    e = (embeddings - embedding_mean) @ projection
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
    random_e = rng.normal(size=e.shape).astype('float32')
    random_e /= np.maximum(np.linalg.norm(random_e, axis=1, keepdims=True), 1e-8)
    ct, et = features(data, tr, columns, np.concatenate([e, random_e], 1), budget, cfg['save_reserve'])
    cv, ev = features(data, va, columns, np.concatenate([e, random_e], 1), budget, cfg['save_reserve'])
    d = cfg['projection_dim']
    sem = np.r_[0:d, 2*d:3*d]; rand = np.r_[d:2*d, 3*d:4*d]
    arms = {'MATCHED_COUNT_LR': (ct, cv),
            'COUNT_PLUS_RANDOM_LR': (np.c_[ct, et[:, rand]], np.c_[cv, ev[:, rand]]),
            'COUNT_PLUS_BIOMEDBERT_LR': (np.c_[ct, et[:, sem]], np.c_[cv, ev[:, sem]])}
    yt, yv = data['y'][tr], data['y'][va]
    if set(np.unique(yt)) != {0, 1}: raise ValueError('TRAIN sample has one class')
    weights = mdl.sample_weights(data['metadata'].iloc[tr].PATIENT_ID.to_numpy(), 'patient')
    scores, states, fits = {}, {}, {}
    for name, (x, xv) in arms.items():
        budget.require(cfg['save_reserve'])
        print('Fitting:', name, '; feature dimensions:', x.shape[1], flush=True)
        scores[name], states[name], fits[name] = fit_head(x, yt, xv, weights, cfg)
        budget.require(cfg['save_reserve'])
    records = []
    for name, p in dict(existing, **scores).items():
        m = cmp.binary_metrics(yv, p, top_fractions=(.1,.2,.3))
        row = dict(Model=name, Snapshots=len(va), Positives=int(yv.sum()),
                   AP=m['average_precision'], ROC_AUC=m['roc_auc'], Log_loss=m['log_loss'],
                   Scope='matched bounded TRAIN' if name in scores else 'existing full TRAIN')
        for k in m['topk']:
            row['Lift@'+str(int(k['fraction']*100))+'%'] = k['lift']
            row['Captured@'+str(int(k['fraction']*100))+'%'] = k['selected_positives']
        records.append(row)
    intervals = {}
    for ref in ('MATCHED_COUNT_LR', 'COUNT_PLUS_RANDOM_LR', 'NLP_FUSION'):
        budget.require(cfg['save_reserve'])
        if ref in dict(existing, **scores):
            intervals[ref] = wf.paired_lift_bootstrap(yv, scores['COUNT_PLUS_BIOMEDBERT_LR'],
                dict(existing, **scores)[ref], data['metadata'].iloc[va].PATIENT_ID.to_numpy(),
                n_bootstrap=cfg['bootstrap_draws'], seed=cfg['seed'])
    payload = {n+'.npy':wf.array_bytes(p) for n,p in scores.items()}
    saved = dict(embeddings=embeddings, embedding_mean=embedding_mean, projection=projection, random_embeddings=random_e, columns=columns)
    for name, state in states.items():
        for key, val in state.items(): saved[name+'__'+key] = val
    buffer = io.BytesIO(); np.savez_compressed(buffer, **saved)
    payload['heads_and_embeddings.npz'] = buffer.getvalue()
    cosine = float((np.square(e.sum(0)).sum() - np.square(e).sum()) / max(len(e)*(len(e)-1), 1))
    report = dict(identity=identity, checkpoint_receipt=receipt, mean_pairwise_projected_cosine=cosine,
        rows=records, intervals=intervals, fits=fits,
        train_rows=len(tr), train_patients=int(data['metadata'].iloc[tr].PATIENT_ID.nunique()),
        eligible_train_features=len(columns), description_preview=[description(n) for n in vocab[:20]],
        status='completed_full_validation', elapsed_seconds=time.monotonic()-started,
        payload_hashes={k:hashlib.sha256(v).hexdigest() for k,v in payload.items()},
        limitations=['BiomedBERT biomedical text transfer, NOT original structured Med-BERT.',
            'Frozen description embeddings and fitted logistic heads; encoder is not fine-tuned.',
            'Description quality and pooling may limit benefit; feature names are not encounter ICD sequences.',
            'Matched arms share sample and fixed hyperparameters; old references use full TRAIN.',
            'Single seed; 200-draw pilot intervals omit training uncertainty. Validation is development data.',
            'Original archive contains TEST, but no TEST rows are transformed or scored.',
            'Production superiority is not evaluated. Runtime deadline is cooperative, including downloads.'])
    payload['report.json'] = wf.json_bytes(report)
    budget.require()
    wh.save_artifacts(table, payload)
    return table, report
