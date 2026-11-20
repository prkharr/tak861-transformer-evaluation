"""Build independent Databricks deliveries with existing frozen input loader."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'notebooks/expanded cohort'

def code(s):
    ast.parse(s)
    return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True))

def md(s):
    return dict(cell_type='markdown',metadata={},source=s.splitlines(True))

def build():
    parent=json.loads((DEST/'02_patient_level_split_AutoML.ipynb').read_text(encoding='utf-8-sig'))
    source=(Path(__file__).with_name('pipeline.py')).read_text()
    timing=(ROOT/'experiments/challenger/time_lgbm.py').read_text()
    pilot=(ROOT/'experiments/medbert/benchmark.py').read_text()
    combined=dict(challenger=source,timing=timing,pilot=pilot)
    digest=hashlib.sha256(json.dumps(combined,sort_keys=True).encode()).hexdigest()
    embed=code('CHALLENGER_SOURCES='+repr(combined)+'\nCHALLENGER_SHA='+repr(digest)+'''
if hashlib.sha256(json.dumps(CHALLENGER_SOURCES,sort_keys=True).encode()).hexdigest()!=CHALLENGER_SHA:
    raise ValueError('Challenger embedded source checksum mismatch')
challenger_modules={}
for name,source in CHALLENGER_SOURCES.items():
    module=types.ModuleType(name)
    exec(compile(source,name,'exec'),module.__dict__)
    challenger_modules[name]=module
challenger,timing,pilot=[challenger_modules[n] for n in ('challenger','timing','pilot')]
CHALLENGER_SETTINGS=dict(seconds=28800,device='auto',threads=8,bootstrap=2000)
EXACT_CHANNELS={'study':'','diagnosis':''}  # Auto-resolve only an unambiguous dictionary match.
''')
    common=[md('''# Evidence-based development challenger

Reuse the completed `R20261030_NEW_SUMMARY` inputs. Fill **only the seven blank sf_options values** with the same connection used before. Keep the parent RUN_ID and CONFIG unchanged. No cohort rebuilding or production scoring occurs.

Requires the existing approved runtime with numpy, pandas, scikit-learn, torch, lightgbm and matplotlib. A GPU accelerates neural tuning; CPU is supported. Private tensors and predictions remain in the configured private warehouse.

Use classic Spark compute with JVM access so each saved artifact is verified before atomic publication. Shared/serverless Spark Connect without JVM access is rejected. GPU compute is strongly recommended for the full grid; CPU completion may require multiple overnight resumptions.

Saved NLP TF-IDF uses all TRAIN documents without labels. Internal OOF estimates are conditional on this fixed representation; they are not a fully inductive NLP vocabulary refit. All supervised feature selection and tuning occur inside TRAIN folds.
'''),*copy.deepcopy(parent['cells'][:2]),embed]
    audit=code('''data=None
try:
    data=load_stage()
    prep,protocol=preparation()
    audit=challenger.audit(data,prep,protocol,wf,timing,EXACT_CHANNELS)
    if not wh._has_artifact_jvm():
        raise RuntimeError('Classic Spark compute with JVM access is required for atomic artifact publication')
    print(audit['status'])
    display(pd.DataFrame(audit['population']))
    print('Timing channels:',audit['timing_channels'])
    for item in audit['limitations']: print(item)
finally:
    if data is not None: wf.release_inputs(data)
''')
    evidence=json.loads((ROOT/'docs/presentations/progress_20261120/evidence.json').read_text())
    nb11=common+[md('''## Useful observations informing this run

* NLP fusion: validation Lift@10% 2.954 versus static LightGBM 2.533; development result, not production superiority.
* Timing: matched fixed-budget lift 2.060 to 2.445; still below the separate early-stopped 2.533 control. Tune both under one protocol.
* Three seeds favor NLP fusion; selected train-validation gaps are approximately 7 to 11%.
* Local masked pretraining 1.367 trailed its matched scratch control 1.735. Genuine BiomedBERT has an implementation check but no clinical benchmark yet; it is not enabled in this challenger.
* TRAIN-fold feature reduction helped a logistic proxy. Feature selection is repeated inside training partitions rather than reusing the full-TRAIN selected features.
* Original 49-feature refit already exists at approximately 2.52 on the original population. Production verification is deferred.
* A blend is a hypothesis about complementary errors, not an established gain. Its weight includes both parent endpoints and is selected from TRAIN OOF predictions.
'''),code('EVIDENCE='+repr(evidence)+'\nprint(json.dumps(EVIDENCE,indent=2))\n'),audit]
    nb12=common+[md('''## Overnight tuning, patient folds and resumable stages

Three outer patient folds; each contains an inner 80/20 patient split. Static and time-aware LightGBM each use 12 candidates; NLP fusion uses 8. Inner stopping selects rounds/epochs. Outer holdout labels never stop training. Final configurations are selected by mean inner lift and refitted with median selected rounds over all TRAIN, for seeds 42/142/242.

The eight-hour budget starts after input loading and is cooperative. Completed trials, feature preparation and final seed fits are persisted with checksums. If time expires, rerun unchanged to resume; increasing `seconds` alone preserves completed stages. A partially finished search does **not** produce a complete report. Warehouse transfers and feature selection may overrun the deadline. No one-hour completion claim applies to this comprehensive run.

OOF blend fitting uses seed 42 and each fold's locally selected model. Final predictions average three seeds using the configuration chosen by mean inner lift. This transfer of blend weights is a development hypothesis. The neural challenger uses the existing architecture with a simplified fixed-learning-rate AdamW schedule.

Copy the printed `CHALLENGER_REPORT_TABLE` into notebook 13 after completion.
'''),code('''data=None
try:
    data=load_stage()
    prep,protocol=preparation()
    CHALLENGER_REPORT_TABLE,challenger_report=challenger.run(data,prep,protocol,
        RUN_PREFIX,CHALLENGER_SHA,CHALLENGER_SETTINGS,EXACT_CHANNELS,mdl,wf,wh,timing,pilot)
    print('CHALLENGER_REPORT_TABLE =',repr(CHALLENGER_REPORT_TABLE))
    print(challenger_report['status'])
    display(pd.DataFrame(challenger_report['rows']).sort_values('Lift@10%',ascending=False))
finally:
    if data is not None: wf.release_inputs(data)
''')]
    nb13=common+[md('''## Report saved predictions only

Fill the exact report table printed by notebook 12 below. This notebook does not load tensors or retrain. Intervals compare challengers with the saved NLP incumbent on identical validation rows. The validation set has been used in prior development; intervals are conditional on fitted predictions.
'''),code('''CHALLENGER_REPORT_TABLE=''  # Paste notebook 12's exact private report table.
if not CHALLENGER_REPORT_TABLE or not CHALLENGER_REPORT_TABLE.startswith(RUN_PREFIX+'_CHALLENGER_') or not CHALLENGER_REPORT_TABLE.endswith('_COMPLETE'):
    raise ValueError('Paste the COMPLETE challenger table printed by notebook 12')
report=challenger.report_saved(CHALLENGER_REPORT_TABLE,wf,wh,cmp,draws=CHALLENGER_SETTINGS['bootstrap'])
display(pd.DataFrame(report['rows']).sort_values('Lift@10%',ascending=False))
display(pd.DataFrame(report['overlap_vs_nlp']))
display(pd.DataFrame([dict(Model=name,**record) for name,records in report['fit_records'].items() for record in records])[['Model','seed','rounds','train_lift','validation_lift','train_validation_gap','seconds']])
print('Frozen TRAIN selection:',report['selection'])
for name,result in report['paired_intervals_vs_frozen_nlp'].items():
    print(name,'minus saved NLP_FUSION; conditional paired patient bootstrap')
    display(pd.DataFrame(result['lift_by_fraction'].values()))
for comparison,result in report['additional_paired_lift_intervals'].items():
    print(comparison)
    display(pd.DataFrame(result['lift_by_fraction'].values()))
for limitation in report['limitations']: print(limitation)
import matplotlib.pyplot as plt
rows=pd.DataFrame(report['rows']).sort_values('Lift@10%')
fig,ax=plt.subplots(figsize=(11,6))
ax.barh(rows.Model,rows['Lift@10%']); ax.axvline(1,color='gray',linestyle=':')
ax.set_xlabel('Validation Lift@10% of patient snapshots'); ax.set_title('Development challenger: same validation rows')
fig.tight_layout(); display(fig); plt.close(fig)
report_table=CHALLENGER_REPORT_TABLE+'_B'+str(CHALLENGER_SETTINGS['bootstrap'])
payload=wf.json_bytes(report)
if wh.table_exists(report_table):
    if wh.read_artifacts(report_table,{'report.json'})['report.json']!=payload:
        raise ValueError('Saved reporting payload differs')
else: wh.save_artifacts(report_table,{'report.json':payload})
print('Saved report with intervals:',report_table)
''')]
    outputs=[]
    for name,cells in [('11_challenger_evidence_and_input_audit',nb11),('12_challenger_tuning_and_blend',nb12),('13_challenger_validation_report',nb13)]:
        for i,c in enumerate(cells):
            c['id']=name[:2]+'-'+str(i)
            if c['cell_type']=='code': ast.parse(''.join(c['source']))
        book=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        dest=DEST/(name+'.ipynb'); dest.write_text(json.dumps(book,indent=1)+'\n',encoding='utf8'); outputs.append(str(dest))
    return outputs

if __name__=='__main__': print('\n'.join(build()))
