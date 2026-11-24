"""Standalone Databricks notebook with embedded, checked source."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'archive/notebooks/expanded cohort/10_pretrained_biomedbert_tensor_transfer.ipynb'

def cell(source):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source.splitlines(True))

def build():
    book = json.loads((ROOT / 'archive/notebooks/expanded cohort/09_medbert_monthly_benchmark.ipynb').read_text())
    cells = copy.deepcopy(book['cells'][:3])
    setup = ''.join(cells[0]['source'])
    setup = setup[setup.index('import os, tempfile'):]
    intro = '''# Requires PyTorch >=2.6, transformers 4.57.x, huggingface_hub, sklearn, pandas, numpy, matplotlib.
# Install dependencies in your approved runtime BEFORE running all cells.
# Example, only if needed: %pip install "transformers==4.57.6" "huggingface_hub>=0.34,<1"
# Restart Python after installations, then run this notebook from the beginning.
import time, importlib.util
TRANSFER_STARTED = time.monotonic()
LOCAL_CHECKPOINT = None  # Default: download public Microsoft checkpoint. Optional verified local mirror.
TRANSFER_SETTINGS = dict(total_seconds=3300, max_train_patients=2000, max_train_rows=25000,
                         bootstrap_draws=200, device='auto', threads=4)
required = ['torch','transformers','huggingface_hub','sklearn','numpy','pandas','matplotlib']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing: raise RuntimeError('Install before starting the timed run: ' + ', '.join(missing))
import torch
from packaging.version import Version
if Version(torch.__version__.split('+')[0]) < Version('2.6'):
    raise RuntimeError('Use an approved runtime with PyTorch >=2.6 for loading the published .bin checkpoint.')
print('PRETRAINED BIOMEDBERT TEXT TRANSFER — NOT ORIGINAL MED-BERT')
print('Original Med-BERT checkpoint unavailable: https://github.com/ZhiGroup/Med-BERT#sharing-pre-trained-model')
print('This demo downloads Microsoft BiomedBERT weights, encodes dictionary descriptions locally, and fits logistic heads.')
print('No patient data is sent to a model service. TEST scoring is disabled. No production uplift is claimed.')
print('55-minute cooperative target including download/load; blocking operations may exceed it.')
'''
    cells[0] = cell(intro + setup)
    source = (ROOT / 'experiments/medbert/pretrained_transfer.py').read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    cells.append(cell('TRANSFER_SOURCE = '+repr(source)+'\nTRANSFER_SHA256 = '+repr(digest)+'''
if hashlib.sha256(TRANSFER_SOURCE.encode()).hexdigest() != TRANSFER_SHA256:
    raise ValueError('Transfer implementation checksum mismatch')
transfer = types.ModuleType('pretrained_biomedical_transfer')
exec(compile(TRANSFER_SOURCE, 'pretrained_biomedical_transfer', 'exec'), transfer.__dict__)
'''))
    cells.append(cell('''
transfer_data = None
try:
    transfer_data = load_stage()
    prep, protocol = preparation()
    transfer_table, transfer_report = transfer.run(transfer_data, prep, protocol,
        RUN_PREFIX, TRANSFER_SHA256, TRANSFER_SETTINGS, medbert, wf, wh, cmp, mdl,
        TRANSFER_STARTED, LOCAL_CHECKPOINT)
finally:
    if transfer_data is not None: wf.release_inputs(transfer_data)
'''))
    cells.append(cell('''
print('Saved:', transfer_table)
print('Model provenance:', transfer_report['checkpoint_receipt'])
print('Mean pairwise semantic cosine after TRAIN-vocabulary centering:', transfer_report['mean_pairwise_projected_cosine'])
print('TRAIN:', transfer_report['train_rows'], 'snapshots;', transfer_report['train_patients'], 'patients')
print('Eligible TRAIN features:', transfer_report['eligible_train_features'])
print('Description examples:', transfer_report['description_preview'])
print('Fit convergence:', transfer_report['fits'])
print('Elapsed:', round(time.monotonic()-TRANSFER_STARTED, 1), 'seconds')
display(pd.DataFrame(transfer_report['rows']).sort_values('Lift@10%', ascending=False))
for reference, interval in transfer_report['intervals'].items():
    print('COUNT_PLUS_BIOMEDBERT_LR minus', reference, '; pilot patient-cluster bootstrap')
    display(pd.DataFrame(interval['lift_by_fraction'].values()))
for limitation in transfer_report['limitations']: print(limitation)
import matplotlib.pyplot as plt
ranked = pd.DataFrame(transfer_report['rows']).sort_values('Lift@10%')
fig, ax = plt.subplots(figsize=(11,5))
ax.barh(ranked.Model, ranked['Lift@10%'])
ax.axvline(1, color='gray', linestyle=':')
ax.set_xlabel('Validation lift at top 10%')
ax.set_title('Pretrained biomedical text transfer: matched controls and contextual references')
fig.tight_layout(); display(fig); plt.close(fig)
'''))
    for i,c in enumerate(cells):
        ast.parse(''.join(c['source']))
        c.update(id='transfer-'+str(i), outputs=[], execution_count=None)
    book.update(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                     'language_info':{'name':'python'}})
    DEST.write_text(json.dumps(book, indent=1)+'\n', encoding='utf-8')
    return DEST

if __name__ == '__main__': print(build())
