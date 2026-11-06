"""Small, immutable handoff packages for independent cached-feature analysis."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.automl_sequence import inputs
from experiments.challenger import time_lgbm


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def _verify_files(root, receipt):
    for name, digest in receipt['files'].items():
        if Path(name).name != name or file_hash(Path(root) / name) != digest:
            raise ValueError('Handoff content changed: ' + name)


def _verify_count_receipt(root, receipt):
    contract = receipt.get('identity', {}).get('contract', {})
    identity = receipt.get('identity', {})
    channels = identity.get('channel_names', [])
    indices = identity.get('channel_indices', [])
    if (receipt.get('verified') is not True
            or contract.get('orientation') != '0=newest'
            or contract.get('date_policy') != 'month_end_history'
            or len(channels) != 2 or len(set(channels)) != 2
            or len(indices) != 2 or len(set(indices)) != 2
            or any(type(i) is not int or i < 0 for i in indices)
            or set(receipt.get('files', {})) != {'counts.npy', 'count_mask.npy', 'count_metadata.csv'}):
        raise ValueError('Count handoff receipt or temporal contract is invalid')
    _verify_files(root, receipt)


def freeze_count_channels(counts, root, exact=None, row_block=1024):
    """Copy only the two prespecified channels; never copy the full count tensor."""
    if type(row_block) is not int or not 1 <= row_block <= 8192:
        raise ValueError('row_block must be between 1 and 8192')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    meta = inputs.normalize_snapshots(counts['metadata'])
    contract = counts['contract']
    if contract.get('orientation') != '0=newest' or contract.get('date_policy') != 'month_end_history':
        raise ValueError('Count date grid must be verified before the handoff')
    cols = time_lgbm.channels(counts['features'], exact or {'study': '', 'diagnosis': ''})
    x, mask = counts['X'], counts['valid']
    if x.shape != (len(meta), 12, len(counts['features'])) or mask.shape != (len(meta), 12) or mask.dtype != bool:
        raise ValueError('Count shape or mask differs from its dictionary/population')
    meta_hash = inputs.frame_hash(meta)
    if contract.get('metadata_sha256') != meta_hash:
        raise ValueError('Count metadata differs from its saved contract')
    identity = dict(metadata_sha256=meta_hash, contract=contract,
                    channel_names=[counts['features'][c] for c in cols], channel_indices=cols)
    receipt_path = root / 'counts.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['identity'] != identity:
            raise ValueError('Use a different handoff directory for a different count source')
        _verify_count_receipt(root, receipt)
        return str(receipt_path)
    outputs = [('counts.npy', (len(meta), 12, 2), np.float32), ('count_mask.npy', mask.shape, bool)]
    for name, shape, dtype in outputs:
        temp = root / (name + '.partial')
        target = np.lib.format.open_memmap(temp, mode='w+', dtype=dtype, shape=shape)
        try:
            for start in range(0, len(meta), row_block):
                stop = min(start + row_block, len(meta))
                block = x[start:stop, :, cols] if name == 'counts.npy' else mask[start:stop]
                if name == 'counts.npy' and (not np.isfinite(block).all() or (block < 0).any()
                        or (block > np.finfo(np.float32).max).any()):
                    raise ValueError('Invalid monthly count values')
                target[start:stop] = block
            target.flush()
        finally:
            target._mmap.close()
        os.replace(temp, root / name)
    meta.to_csv(root / 'count_metadata.csv.partial', index=False, date_format='%Y-%m-%d')
    os.replace(root / 'count_metadata.csv.partial', root / 'count_metadata.csv')
    receipt = dict(identity=identity, shape=[len(meta), 12, 2], verified=True,
                   files={name: file_hash(root / name) for name in ('counts.npy', 'count_mask.npy', 'count_metadata.csv')},
                   source_contract_status='Inherited count contract; copied subset independently hashed',
                   source_availability='Retrospective event-date bounds; ingestion-time availability unresolved')
    atomic_json(receipt_path, receipt)
    return str(receipt_path)


def timing_group(receipt_path, metadata):
    path = Path(receipt_path)
    receipt = json.loads(path.read_text(encoding='utf-8'))
    _verify_count_receipt(path.parent, receipt)
    parent = inputs.normalize_snapshots(pd.read_csv(path.parent / 'count_metadata.csv', dtype={'PATIENT_ID': str}))
    meta = inputs.normalize_snapshots(metadata)
    if inputs.frame_hash(parent) != receipt['identity']['metadata_sha256']:
        raise ValueError('Count metadata fingerprint changed')
    lookup = pd.MultiIndex.from_frame(parent[['PATIENT_ID', 'END_DT']])
    idx = lookup.get_indexer(pd.MultiIndex.from_frame(meta[['PATIENT_ID', 'END_DT']]))
    if (idx < 0).any() or not np.array_equal(parent.iloc[idx].RESP.to_numpy(), meta.RESP.to_numpy()) or not np.array_equal(parent.iloc[idx].SPLIT.to_numpy(), meta.SPLIT.to_numpy()):
        raise ValueError('Timing keys, labels or split assignments do not match')
    x = np.load(path.parent / 'counts.npy', mmap_mode='r', allow_pickle=False)
    mask = None
    try:
        mask = np.load(path.parent / 'count_mask.npy', mmap_mode='r', allow_pickle=False)
        if (receipt.get('shape') != [len(parent), 12, 2] or x.shape != (len(parent), 12, 2)
                or x.dtype != np.float32 or mask.shape != (len(parent), 12) or mask.dtype != bool):
            raise ValueError('Count subset shape or mask differs from its receipt')
        values = time_lgbm.temporal_columns(x, mask, idx, [0, 1])
    finally:
        x._mmap.close()
        if mask is not None:
            mask._mmap.close()
    return dict(values=values, metadata=meta, names=list(time_lgbm.FEATURE_NAMES),
                provenance=dict(verified=True, fit_scope='fixed_formulas', identity=receipt['identity'],
                                receipt_sha256=file_hash(path), incomplete_history='all nine values missing'))


def _nlp_request_identity(parent, row_block=4096):
    """Hash the original ordered document list without a full twelve-step grid.

    Patient-complete batches reuse the shared date construction and preserve
    deduplication across overlapping snapshots. The streamed JSON bytes have
    exactly the same meaning as inputs.stable_hash over the original list.
    """
    if type(row_block) is not int or row_block < 1:
        raise ValueError('NLP verification row block must be positive')
    ordered = parent.sort_values(['PATIENT_ID', 'END_DT'], kind='stable').reset_index(drop=True)
    patients = ordered.PATIENT_ID.to_numpy()
    boundaries = np.r_[np.flatnonzero(patients[1:] != patients[:-1]) + 1, len(ordered)]
    membership = ordered.groupby('PATIENT_ID').SPLIT.first().to_dict()
    digest = hashlib.sha256(b'[')
    total = train_count = start = 0
    while start < len(ordered):
        position = np.searchsorted(boundaries, start + row_block, side='right') - 1
        end = int(boundaries[position]) if position >= 0 and boundaries[position] > start else int(boundaries[np.searchsorted(boundaries, start, side='right')])
        requests, grid = inputs.make_requests(ordered.iloc[start:end], date_policy='month_end_history')
        del grid
        for patient, date in requests.itertuples(index=False, name=None):
            split = membership[patient]
            record = [total, patient, date.strftime('%Y-%m-%d'), split]
            if total:
                digest.update(b',')
            digest.update(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False, default=str).encode())
            train_count += int(split == 'train')
            total += 1
        start = end
    digest.update(b']')
    return train_count, digest.hexdigest()


def nlp_group(package, metadata):
    """Validate the existing TRAIN-fitted encoder, then summarize only pilot rows."""
    if package.get('_payload_verified') is not True:
        raise ValueError('Load NLP through the existing payload-verifying package reader first')
    parent = inputs.normalize_snapshots(package['metadata'])
    meta = inputs.normalize_snapshots(metadata)
    manifest = package['manifest']
    names = list(package['feature_names'])
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('NLP vocabulary must be nonempty and unique')
    if manifest.get('fit_scope') != 'train_only' or manifest.get('orientation') != '0=newest':
        raise ValueError('NLP fitting scope or temporal orientation is not verified')
    if manifest.get('metadata_hash') != inputs.frame_hash(parent) or names != manifest.get('feature_names'):
        raise ValueError('NLP parent metadata/vocabulary mismatch')
    train_hash = inputs.train_patient_hash(parent)
    train_document_count, request_hash = _nlp_request_identity(parent)
    for evidence in (manifest, manifest.get('fitted_state', {}), manifest.get('receipt', {})):
        if evidence.get('fit_scope', 'train_only' if evidence is manifest else None) != 'train_only' or evidence.get('train_patients_sha256') != train_hash or evidence.get('train_document_count') != train_document_count:
            raise ValueError('NLP TRAIN fitting provenance is incomplete')
    fitted = manifest['fitted_state']
    if fitted.get('terms') != names or fitted.get('term_order_sha256') != inputs.stable_hash(names):
        raise ValueError('NLP fitted term order differs')
    idf = np.asarray(fitted.get('idf', []), dtype=np.float64)
    frequency = np.asarray(fitted.get('document_frequency', []), dtype=np.float64)
    if (fitted.get('schema') != 'monthly_tfidf_v1' or fitted.get('training_documents') != train_document_count
            or idf.shape != (len(names),) or frequency.shape != idf.shape
            or not np.isfinite(idf).all() or not np.isfinite(frequency).all()
            or (frequency < 0).any() or (frequency > train_document_count).any()
            or (frequency != np.floor(frequency)).any()
            or not np.allclose(idf, np.log((1+train_document_count)/(1+frequency))+1, rtol=1e-12, atol=1e-12)):
        raise ValueError('NLP fitted IDF does not match its TRAIN document frequencies')
    if fitted.get('extraction_identity', {}).get('requests') != request_hash:
        raise ValueError('NLP source document identity differs')
    lookup = pd.MultiIndex.from_frame(parent[['PATIENT_ID', 'END_DT']])
    idx = lookup.get_indexer(pd.MultiIndex.from_frame(meta[['PATIENT_ID', 'END_DT']]))
    if (idx < 0).any() or not np.array_equal(parent.iloc[idx].RESP.to_numpy(), meta.RESP.to_numpy()) or not np.array_equal(parent.iloc[idx].SPLIT.to_numpy(), meta.SPLIT.to_numpy()):
        raise ValueError('NLP keys, labels or split assignments do not match')
    x, mask = package['X_sequence'], package['mask']
    if x.shape != (len(parent), 12, len(names)) or mask.shape != (len(parent), 12) or mask.dtype != bool:
        raise ValueError('NLP sequence shape/mask differs')
    values = np.empty((len(meta), len(names) + 1), dtype=np.float32)
    for start in range(0, len(idx), 256):
        rows = idx[start:start+256]
        block, observed = x[rows], mask[rows]
        if not np.isfinite(block).all() or (block < 0).any() or (block[~observed] != 0).any():
            raise ValueError('NLP values or inactive documents are invalid')
        means = block.mean(axis=1, dtype=np.float64)
        if not np.isfinite(means).all() or (means > np.finfo(np.float32).max).any():
            raise ValueError('NLP mean cannot be represented as finite float32 values')
        values[start:start+len(rows), :-1] = means
        values[start:start+len(rows), -1] = observed.mean(axis=1)
    return dict(values=values, metadata=meta, names=['NLP_MEAN_' + n for n in names] + ['NLP_ACTIVE_DOCUMENT_FRACTION'],
                provenance=dict(verified=True, fit_scope='train_only', fitted_on='full_parent_train',
                    train_patients_sha256=train_hash, manifest_sha256=inputs.stable_hash(manifest),
                    definition='Mean over all twelve positions, inactive documents zero; activity fraction separate'))


def save_handoff(root, cache_manifest, count_receipt, metadata, config, implementation, run_prefix):
    """Write only noncredential configuration; no source checkpoint is changed."""
    root = Path(root)
    def reject_credentials(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if any(term in str(key).lower() for term in ('password', 'secret', 'token', 'sf_options')):
                    raise ValueError('Credentials cannot be included in the handoff')
                reject_credentials(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                reject_credentials(item)
    reject_credentials(config)
    meta = inputs.normalize_snapshots(metadata)
    cache_path, count_path = Path(cache_manifest), Path(count_receipt)
    cached = json.loads(cache_path.read_text(encoding='utf-8'))
    count = json.loads(count_path.read_text(encoding='utf-8'))
    meta_hash = inputs.frame_hash(meta)
    if (config.get('raw_automl', {}).get('request_steps') != 1
            or cached.get('request_steps') != 1
            or cached.get('parent_metadata_sha256') != meta_hash
            or cached.get('identity', {}).get('run_identity') != run_prefix
            or count.get('identity', {}).get('metadata_sha256') != meta_hash):
        raise ValueError('Handoff source identity, metadata or summary request contract differs')
    _verify_count_receipt(count_path.parent, count)
    root.mkdir(parents=True, exist_ok=True)
    meta.to_csv(root / 'parent_metadata.csv.partial', index=False, date_format='%Y-%m-%d')
    os.replace(root / 'parent_metadata.csv.partial', root / 'parent_metadata.csv')
    receipt = dict(schema=1, cache_manifest=str(cache_path.resolve()), count_receipt=str(count_path.resolve()),
        cache_manifest_sha256=file_hash(cache_path), count_receipt_sha256=file_hash(count_path),
        parent_metadata=str((root / 'parent_metadata.csv').resolve()), metadata_sha256=meta_hash,
        metadata_file_sha256=file_hash(root / 'parent_metadata.csv'), config=config, implementation=implementation,
        run_prefix=run_prefix, nlp_table=run_prefix+'_NLP_INPUTS',
        extraction_resume='Resume the original extraction with its existing configuration and checkpoints')
    atomic_json(root / 'handoff.json', receipt)
    return str(root / 'handoff.json')
