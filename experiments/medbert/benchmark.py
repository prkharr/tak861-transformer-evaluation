"""Bounded Med-BERT-inspired benchmark on frozen monthly claims tensors.

This is a local masked-code pretraining adaptation, not the unavailable official
pretrained Med-BERT weights. Monthly feature groups are not ordered encounters.
No TEST tensors are tokenized, fitted, scored, or used to select a checkpoint.
"""
import copy
import hashlib
import io
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

DEFAULTS = dict(seed=42, max_train_patients=2000, max_train_rows=25000,
                max_length=64, codes_per_month=5, d_model=64, heads=4, depth=2,
                ff=128, dropout=.2, batch_size=256, learning_rate=.0003,
                weight_decay=.001, pretrain_steps=200, epochs=3,
                finetune_steps_per_epoch=120, patience=2, threads=4,
                bootstrap_draws=200, total_seconds=3300, save_reserve=120,
                device='auto', code_prefixes=['DX__', 'PX__', 'RX__'])
SPECIAL = ['[PAD]', '[CLS]', '[MASK]', '[EMPTY]', '[UNK]']
MODEL_NAME = 'MEDBERT_MONTHLY_MLM'
CONTROL_NAME = 'MONTHLY_TOKEN_SCRATCH'


class Budget:
    """Cooperative deadline. Blocking warehouse calls cannot be interrupted."""
    def __init__(self, seconds, started=None):
        self.started = time.monotonic() if started is None else started
        self.seconds = float(seconds)

    def remaining(self):
        return self.seconds - (time.monotonic() - self.started)

    def require(self, reserve=0):
        if self.remaining() <= reserve:
            raise TimeoutError('Runtime budget exhausted; no incomplete validation result is reported')


def validate_config(cfg):
    if (not isinstance(cfg['code_prefixes'], list) or not cfg['code_prefixes']
            or any(p not in ('DX__', 'PX__', 'RX__') for p in cfg['code_prefixes'])):
        raise ValueError('Use explicit diagnosis/procedure/prescription count prefixes only')
    for name in ('max_train_patients', 'max_train_rows', 'max_length', 'codes_per_month',
                 'd_model', 'heads', 'depth', 'ff', 'batch_size', 'pretrain_steps',
                 'epochs', 'finetune_steps_per_epoch', 'patience', 'threads', 'bootstrap_draws'):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError('Positive integer required: ' + name)
    if cfg['max_length'] < 13 or cfg['max_length'] > 256 or cfg['d_model'] % cfg['heads']:
        raise ValueError('Use length 13..256 and a hidden width divisible by heads')
    if cfg['codes_per_month'] > (cfg['max_length'] - 1) // 12:
        raise ValueError('Length must retain the declared per-month capacity for all twelve months')
    if cfg['bootstrap_draws'] < 20 or not 0 < cfg['total_seconds'] <= 3600:
        raise ValueError('Use at least 20 bootstrap draws and at most one hour')
    if not 0 <= cfg['dropout'] < 1 or cfg['learning_rate'] <= 0 or cfg['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')


def validate_data(data):
    meta, x, mask = data['metadata'], data['X_count'], data['count_mask']
    contract = data['manifest']['count_contract']
    names = contract['feature_names']
    if (contract.get('orientation') != '0=newest' or contract.get('n_steps') != 12
            or contract.get('date_policy') != 'month_end_history'):
        raise ValueError('Require the verified twelve-month newest-first count contract')
    if x.shape != (len(meta), 12, len(names)) or mask.shape != (len(meta), 12) or mask.dtype != bool:
        raise ValueError('Count shape, dictionary, or explicit mask differs')
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError('Count dictionary must contain unique names')
    if meta.duplicated(['PATIENT_ID', 'END_DT']).any() or meta.groupby('PATIENT_ID').SPLIT.nunique().max() != 1:
        raise ValueError('Duplicate snapshots or patients crossing splits')
    if not np.array_equal(meta.RESP.to_numpy(), data['y']):
        raise ValueError('Labels differ from frozen metadata')
    for split in ('train', 'validation', 'test'):
        if not np.array_equal(data['indices'][split], np.flatnonzero(meta.SPLIT.eq(split))):
            raise ValueError('Frozen split index mismatch')
    for split in ('train', 'validation'):
        if set(np.unique(data['y'][data['indices'][split]])) != {0, 1}:
            raise ValueError('Both classes required in TRAIN and VALIDATION')
    return list(names)


def train_subset(meta, cfg):
    """Label-independent patient sampling, then label-independent row cap."""
    ti = np.flatnonzero(meta.SPLIT.eq('train'))
    patients = meta.PATIENT_ID.to_numpy()
    unique = np.unique(patients[ti])
    ranked = sorted(unique, key=lambda p: hashlib.sha256((str(cfg['seed']) + ':' + p).encode()).digest())
    chosen = set(ranked[:cfg['max_train_patients']])
    rows = ti[np.fromiter((p in chosen for p in patients[ti]), bool, count=len(ti))]
    if len(rows) > cfg['max_train_rows']:
        rows = np.sort(np.random.default_rng(cfg['seed']).choice(rows, cfg['max_train_rows'], replace=False))
    return rows


def fit_vocabulary(data, train_rows, cfg, budget):
    names = data['manifest']['count_contract']['feature_names']
    eligible = np.array([i for i, n in enumerate(names)
        if n.upper().startswith(tuple(cfg['code_prefixes']))], dtype=int)
    if not len(eligible):
        raise ValueError('No coded count channels match the explicit prefixes')
    seen = np.zeros(len(eligible), dtype=bool)
    for start in range(0, len(train_rows), 256):
        budget.require(cfg['save_reserve'])
        rows = train_rows[start:start+256]
        if not data['metadata'].iloc[rows].SPLIT.eq('train').all():
            raise ValueError('Vocabulary fitting requires TRAIN rows only')
        block = np.asarray(data['X_count'][rows])[:, :, eligible]
        valid = data['count_mask'][rows]
        seen |= ((block > 0) & valid[:, :, None]).any(axis=(0, 1))
    if not seen.any():
        raise ValueError('No observed codes in the TRAIN subset')
    return eligible[seen]


def tokenize(data, indices, cfg, budget, vocabulary_columns):
    """Keep up to K active groups per month by count, ties by frozen dictionary.

    Visit/month IDs retain recency. Code order within a month has no clinical
    meaning. Presence plus a clipped log2 count bucket replaces repeated codes.
    Empty observed months and unavailable months have different representations.
    """
    n, length = len(indices), cfg['max_length']
    names = data['manifest']['count_contract']['feature_names']
    eligible_columns = np.array([i for i, name in enumerate(names)
        if name.upper().startswith(tuple(cfg['code_prefixes']))], dtype=int)
    if not len(eligible_columns):
        raise ValueError('No coded count channels match the explicit prefixes')
    tokens = np.zeros((n, length), dtype=np.int64)
    months = np.zeros_like(tokens)
    frequency = np.zeros_like(tokens)
    if data['metadata'].SPLIT.to_numpy()[indices].tolist().count('test'):
        raise ValueError('TEST tokenization is disabled')
    known_map = {int(column): j + len(SPECIAL) for j, column in enumerate(vocabulary_columns)}
    active = dropped = empty = unknown = 0
    for i, row in enumerate(indices):
        if i % 256 == 0:
            budget.require(cfg['save_reserve'])
        values = np.asarray(data['X_count'][row])[:, eligible_columns]
        valid = data['count_mask'][row]
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError('Require finite nonnegative integer counts')
        tokens[i, 0] = 1
        cursor = 1
        for month in range(11, -1, -1):
            if not valid[month]:
                continue
            found = np.flatnonzero(values[month] > 0)
            active += len(found)
            unknown += sum(int(eligible_columns[c]) not in known_map for c in found)
            if not len(found):
                tokens[i, cursor], months[i, cursor] = 3, month + 1
                cursor += 1
                empty += 1
                continue
            order = np.lexsort((found, -values[month, found]))
            kept = found[order[:cfg['codes_per_month']]]
            kept = np.sort(kept)  # Dictionary order, never a claimed encounter order.
            dropped += len(found) - len(kept)
            size = len(kept)
            tokens[i, cursor:cursor+size] = [known_map.get(int(eligible_columns[c]), 4) for c in kept]
            months[i, cursor:cursor+size] = month + 1
            frequency[i, cursor:cursor+size] = np.minimum(7, 1 + np.floor(np.log2(values[month, kept]))).astype(int)
            cursor += size
    return (tokens, months, frequency), dict(rows=n, active_code_months=active,
        discarded_code_months=dropped, discarded_fraction=dropped / max(1, active),
        unknown_code_months=unknown, empty_observed_months=empty)


class MonthlyEncoder(nn.Module):
    def __init__(self, vocabulary_size, cfg):
        super().__init__()
        d = cfg['d_model']
        self.code = nn.Embedding(vocabulary_size, d, padding_idx=0)
        self.month = nn.Embedding(13, d, padding_idx=0)
        self.frequency = nn.Embedding(8, d, padding_idx=0)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg['dropout'])
        layer = nn.TransformerEncoderLayer(d, cfg['heads'], cfg['ff'], cfg['dropout'],
                                            batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, cfg['depth'], enable_nested_tensor=False)
        self.classifier = nn.Linear(d, 1)
        self.mlm = nn.Linear(d, vocabulary_size)

    def forward(self, ids, months, frequencies, mlm=False):
        x = self.dropout(self.norm(self.code(ids) + self.month(months) + self.frequency(frequencies)))
        h = self.encoder(x, src_key_padding_mask=ids.eq(0))
        if mlm:
            return self.mlm(h)
        # Mean pooling matches the original structured-code adaptation more
        # closely than introducing artificial within-month position order.
        valid = ids.ge(3).unsqueeze(-1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.classifier(pooled).squeeze(-1)


def batch_arrays(arrays, idx, device):
    return tuple(torch.as_tensor(a[idx], dtype=torch.long, device=device) for a in arrays)


def predict(model, arrays, cfg, budget, device):
    model.eval()
    out = np.empty(len(arrays[0]), dtype=np.float64)
    with torch.no_grad():
        for begin in range(0, len(out), cfg['batch_size']):
            budget.require(cfg['save_reserve'])
            end = min(len(out), begin + cfg['batch_size'])
            out[begin:end] = torch.sigmoid(model(*batch_arrays(arrays, slice(begin, end), device))).cpu().numpy()
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite predictions')
    return out


def mask_codes(ids, frequencies, vocabulary_size, rng):
    eligible = ids.ge(len(SPECIAL))
    selected = eligible & (torch.rand(ids.shape, generator=rng, device=ids.device) < .15)
    if not selected.any() and eligible.any():
        selected[tuple(torch.nonzero(eligible)[0])] = True
    labels = ids.clone()
    labels[~selected] = -100
    replacement = torch.rand(ids.shape, generator=rng, device=ids.device)
    corrupted, freq = ids.clone(), frequencies.clone()
    corrupted[selected & (replacement < .8)] = 2
    random_positions = selected & (replacement >= .8) & (replacement < .9)
    random_ids = torch.randint(len(SPECIAL), vocabulary_size, ids.shape, generator=rng, device=ids.device)
    corrupted[random_positions] = random_ids[random_positions]
    freq[selected] = 0  # Do not expose the masked token's count bucket.
    return corrupted, freq, labels


def pretrain(model, arrays, cfg, budget, device, stop_at):
    started = time.monotonic()
    mlm_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith('classifier.')]
    opt = torch.optim.AdamW(mlm_parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(cfg['seed'])
    torch_rng = torch.Generator(device=device).manual_seed(cfg['seed'])
    losses = []
    model.train()
    for step in range(cfg['pretrain_steps']):
        if time.monotonic() >= stop_at:
            break
        budget.require(cfg['save_reserve'])
        rows = rng.integers(0, len(arrays[0]), size=min(cfg['batch_size'], len(arrays[0])))
        ids, months, freq = batch_arrays(arrays, rows, device)
        ids, freq, labels = mask_codes(ids, freq, model.code.num_embeddings, torch_rng)
        if not labels.ne(-100).any():
            continue
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(ids, months, freq, mlm=True).flatten(0, 1), labels.flatten(), ignore_index=-100)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError('No masked-code pretraining update completed; cannot label this arm pretrained')
    return dict(steps=len(losses), requested_steps=cfg['pretrain_steps'],
        budget_truncated=len(losses) < cfg['pretrain_steps'], seconds=time.monotonic() - started,
        first_loss=losses[0], last_loss=losses[-1], loss_curve=losses, fit_scope='TRAIN only')


def finetune(model, train_arrays, val_arrays, yt, yv, weights, cfg, budget, device, mdl, stop_at, eval_seconds):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(cfg['seed'])
    best, best_state, best_scores, stale, history = -math.inf, None, None, 0, []
    for epoch in range(cfg['epochs']):
        model.train()
        order = rng.permutation(len(yt))
        losses = []
        for start in range(0, min(len(yt), cfg['finetune_steps_per_epoch'] * cfg['batch_size']), cfg['batch_size']):
            if time.monotonic() + eval_seconds > stop_at:
                break
            budget.require(cfg['save_reserve'])
            rows = order[start:start+cfg['batch_size']]
            opt.zero_grad(set_to_none=True)
            logits = model(*batch_arrays(train_arrays, rows, device))
            labels = torch.as_tensor(yt[rows], dtype=torch.float32, device=device)
            weight = torch.as_tensor(weights[rows], dtype=torch.float32, device=device)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weight).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            break
        scores = predict(model, val_arrays, cfg, budget, device)
        lift = mdl.lift_at_fraction(yv, scores, .1)
        requested = min(math.ceil(len(yt) / cfg['batch_size']), cfg['finetune_steps_per_epoch'])
        history.append(dict(epoch=epoch + 1, steps=len(losses), requested_steps=requested,
            budget_truncated=len(losses) < requested, training_loss=float(np.mean(losses)), validation_lift10=lift))
        if lift > best:
            best, best_scores, stale = lift, scores.copy(), 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg['patience'] or time.monotonic() + eval_seconds >= stop_at:
            break
    if best_state is None:
        raise TimeoutError('No trained checkpoint completed full validation within the runtime budget')
    model.load_state_dict(best_state)
    return best_scores, history


def frozen_benchmarks(data, prep, protocol, wf, wh, prefix):
    lock = json.loads(wh.read_artifacts(prefix + '_LOCK', {'lock.json'})['lock.json'])
    if (lock['protocol_hash'] != wf.digest(protocol) or lock['preparation_hash'] != wf.digest(prep)
            or lock['input_manifest_hash'] != wf.digest(data['manifest'])):
        raise ValueError('Frozen benchmark and current input/preparation identities differ')
    results = wf.read_runs(protocol, prefix)
    scores = {}
    for name, runs in results.items():
        if [r['payload_hashes'] for r in runs] != lock['models'][name]['predictor_hashes']:
            raise ValueError('Frozen predictor digest differs: ' + name)
        scores[name] = np.mean([r['val_scores'] for r in runs], axis=0)
    table = prefix + '_FRESH_LGBM_STATIC_V1_VALIDATION_LIFT'
    if wh.table_exists(table):
        blob = wh.read_artifacts(table, {'report.json', 'baseline.npy'})
        report = json.loads(blob['report.json'])
        identity = report['identity']
        value = wf.bytes_array(blob['baseline.npy'])
        if (identity['split'] != 'validation' or identity['baseline']['input_hash'] != wf.digest(data['manifest'])
                or identity['baseline']['lock_hash'] != wf.digest(lock)
                or identity['predictions']['LIGHTGBM_TRAIN_ONLY_STATIC'] != wf.array_sha256(value)):
            raise ValueError('Frozen static benchmark provenance or predictions differ')
        scores['LIGHTGBM_TRAIN_ONLY_STATIC'] = value
    for value in scores.values():
        if value.shape != (len(data['indices']['validation']),) or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError('Invalid frozen benchmark vector')
    return scores, wf.digest(lock)


def run(data, prep, protocol, source_prefix, implementation_sha256, cfg, wf, mdl, cmp, wh, started=None):
    cfg = dict(DEFAULTS, **cfg)
    validate_config(cfg)
    budget = Budget(cfg['total_seconds'], started)
    names = validate_data(data)
    omitted_names = [n for n in names if not n.upper().startswith(tuple(cfg['code_prefixes']))]
    train_rows = train_subset(data['metadata'], cfg)
    val_rows = data['indices']['validation']
    yt, yv = np.asarray(data['y'][train_rows]), np.asarray(data['y'][val_rows])
    if set(np.unique(yt)) != {0, 1}:
        raise ValueError('Label-independent TRAIN subset has one class; increase the prespecified patient cap')
    vocabulary_columns = fit_vocabulary(data, train_rows, cfg, budget)
    names = [names[i] for i in vocabulary_columns]
    existing, lock_hash = frozen_benchmarks(data, prep, protocol, wf, wh, source_prefix)
    identity = dict(version=1, implementation_sha256=implementation_sha256, configuration=cfg,
        input_manifest_hash=wf.digest(data['manifest']), frozen_lock_hash=lock_hash,
        train_rows_sha256=wf.array_sha256(train_rows), validation_rows_sha256=wf.array_sha256(val_rows),
        vocabulary_sha256=wf.digest(SPECIAL + names), count_contract_hash=wf.digest(data['manifest']['count_contract']),
        torch_version=torch.__version__,
        benchmark_predictions={k: wf.array_sha256(v) for k, v in existing.items()})
    table = source_prefix + '_MEDBERT_MONTHLY_V1_' + wf.digest(identity)[:12].upper()
    if wh.table_exists(table):
        saved = wh.read_artifacts(table, {'report.json'})
        report = json.loads(saved['report.json'])
        if report['identity'] != identity:
            raise ValueError('Saved benchmark identity changed')
        payload_names = set(report['payload_hashes'])
        required = {MODEL_NAME + '.npy', CONTROL_NAME + '.npy', 'models.pt', 'vocabulary.json', 'pretraining.json'}
        if not required.issubset(payload_names) or any(not isinstance(name, str) or
                not name or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-' for c in name)
                or name == 'report.json' for name in payload_names):
            raise ValueError('Saved benchmark payload manifest is invalid')
        saved.update(wh.read_artifacts(table, payload_names))
        for name, digest in report['payload_hashes'].items():
            if hashlib.sha256(saved[name]).hexdigest() != digest:
                raise ValueError('Saved benchmark payload checksum mismatch')
        return table, report
    torch.set_num_threads(cfg['threads'])
    device = 'cuda' if cfg['device'] == 'auto' and torch.cuda.is_available() else ('cpu' if cfg['device'] == 'auto' else cfg['device'])
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Configured CUDA device is unavailable')
    torch.manual_seed(cfg['seed'])
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(cfg['seed'])
    train_arrays, train_tokens = tokenize(data, train_rows, cfg, budget, vocabulary_columns)
    val_arrays, val_tokens = tokenize(data, val_rows, cfg, budget, vocabulary_columns)
    base = MonthlyEncoder(len(names) + len(SPECIAL), cfg).to(device)
    initial = copy.deepcopy(base.state_dict())
    # One bounded inference probe estimates the cost of complete validation.
    small = tuple(a[:min(len(a), 4 * cfg['batch_size'])] for a in val_arrays)
    begin = time.monotonic()
    predict(base, small, cfg, budget, device)
    probe_seconds = time.monotonic() - begin
    eval_seconds = max(2., 2. * probe_seconds * len(yv) / len(small[0]))
    available = budget.remaining() - cfg['save_reserve'] - max(120., .1 * cfg['total_seconds'])
    if available < 4 * eval_seconds + 30:
        raise TimeoutError('Preflight cannot reserve two complete validation passes and fitting. Use a faster cluster or smaller prespecified architecture')
    weights = mdl.sample_weights(data['metadata'].PATIENT_ID.to_numpy()[train_rows], 'patient')
    weights *= np.where(yt == 1, mdl.class_weight(yt, weights, 'sqrt'), 1.)
    weights /= weights.mean()
    start_fit = time.monotonic()
    arm_seconds = available / 2
    scores, histories, states, fit_seconds = {}, {}, {}, {}
    scratch_scores, histories[CONTROL_NAME] = finetune(base, train_arrays, val_arrays, yt, yv, weights,
        cfg, budget, device, mdl, start_fit + arm_seconds, eval_seconds)
    fit_seconds[CONTROL_NAME] = time.monotonic() - start_fit
    scores[CONTROL_NAME] = scratch_scores
    states[CONTROL_NAME] = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    base.load_state_dict(initial)
    torch.manual_seed(cfg['seed'])
    pretrain_end = min(time.monotonic() + arm_seconds * .25, start_fit + available - eval_seconds - 20)
    mlm = pretrain(base, train_arrays, cfg, budget, device, pretrain_end)
    fine_start = time.monotonic()
    scores[MODEL_NAME], histories[MODEL_NAME] = finetune(base, train_arrays, val_arrays, yt, yv, weights,
        cfg, budget, device, mdl, start_fit + available, eval_seconds)
    fit_seconds[MODEL_NAME] = time.monotonic() - fine_start
    states[MODEL_NAME] = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    rows = []
    for name, probabilities in dict(existing, **scores).items():
        metrics = cmp.binary_metrics(yv, probabilities, top_fractions=(.1, .2, .3))
        row = dict(Model=name, Split='validation', Snapshots=len(yv), Patients=data['metadata'].iloc[val_rows].PATIENT_ID.nunique(),
            Positives=int(yv.sum()), Average_precision=metrics['average_precision'], ROC_AUC=metrics['roc_auc'],
            Log_loss=metrics['log_loss'], Training_scope='bounded TRAIN subset' if name in scores else 'existing full TRAIN protocol')
        for value in metrics['topk']:
            key = str(int(value['fraction'] * 100))
            row['Lift@' + key + '%'], row['Captured@' + key + '%'] = value['lift'], value['selected_positives']
        rows.append(row)
    intervals = {}
    for reference in (CONTROL_NAME, 'LIGHTGBM_TRAIN_ONLY_STATIC', 'NLP_FUSION'):
        if reference in dict(existing, **scores):
            budget.require(cfg['save_reserve'] + 10)
            intervals[reference] = wf.paired_lift_bootstrap(yv, scores[MODEL_NAME], dict(existing, **scores)[reference],
                data['metadata'].PATIENT_ID.to_numpy()[val_rows], n_bootstrap=cfg['bootstrap_draws'], seed=cfg['seed'])
    buffer = io.BytesIO()
    torch.save(dict(states=states, vocabulary=SPECIAL + names, configuration=cfg), buffer)
    payload = {'models.pt': buffer.getvalue(), 'vocabulary.json': wf.json_bytes(SPECIAL + names),
        'pretraining.json': wf.json_bytes(mlm), **{n + '.npy': wf.array_bytes(s) for n, s in scores.items()}}
    fine_receipts = {name: dict(seed=cfg['seed'], seconds=fit_seconds[name], epochs_completed=len(history),
        epochs_requested=cfg['epochs'], total_steps=sum(h['steps'] for h in history),
        partial_epoch=any(h['budget_truncated'] for h in history),
        selected_epoch=max(history, key=lambda h: h['validation_lift10'])['epoch']) for name, history in histories.items()}
    report = dict(identity=identity, rows=rows, intervals=intervals, history=histories, fine_tuning=fine_receipts,
        benchmark_status={n: ('available' if n in existing else 'no saved benchmark artifact')
            for n in ('COUNT', 'STATIC_FUSION', 'NLP_FUSION', 'LIGHTGBM_TRAIN_ONLY_STATIC')},
        mlm=mlm, vocabulary=SPECIAL + names, tokenization=dict(train=train_tokens, validation=val_tokens,
            eligible_channels=len(names), omitted_channels=omitted_names),
        train_snapshots=len(train_rows), train_patients=int(data['metadata'].iloc[train_rows].PATIENT_ID.nunique()),
        device=device, parameters=sum(p.numel() for p in base.parameters()), estimated_validation_seconds=eval_seconds,
        elapsed_before_persistence=time.monotonic() - budget.started, runtime_budget_seconds=cfg['total_seconds'],
        test_prediction_performed=False, status='completed_full_validation',
        payload_hashes={k: hashlib.sha256(v).hexdigest() for k, v in payload.items()},
        limitations=['Med-BERT-inspired local masked-code pretraining. Official pretrained weights are not used.',
            'Monthly claims feature groups replace encounter-level ICD tokens. No length-of-stay auxiliary objective is available.',
            'Fixed per-month token truncation is reported. Count magnitudes use clipped log2 buckets.',
            'Single seed and bounded TRAIN subset. Existing benchmarks use their original larger training budgets.',
            'Vocabulary uses only observed TRAIN-subset codes. Unseen validation codes map to UNK.',
            'Time-limited fits may be undertrained. A weak pilot score does not reject the full Med-BERT approach.',
            'Validation selects checkpoints. Intervals condition on fitted predictions and selection, and omit retraining uncertainty.',
            'Runtime is cooperatively bounded. Warehouse reads, writes and single device kernels can exceed the deadline.',
            'Patient snapshots are the prediction unit. TEST and production comparison remain disabled.'])
    payload['report.json'] = wf.json_bytes(report)
    budget.require()
    wh.save_artifacts(table, payload)
    return table, report
