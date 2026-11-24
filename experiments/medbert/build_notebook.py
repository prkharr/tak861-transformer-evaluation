"""Build a standalone benchmark without modifying the completed pipeline."""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

from experiments.automl_sequence.build_notebooks import code, notebook_source

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'archive/notebooks/expanded cohort/09_medbert_monthly_benchmark.ipynb'


def build(output_path=DEST):
    parent = json.loads((ROOT / 'notebooks/established_experiments/02_patient_level_split_AutoML.ipynb').read_text(encoding='utf-8-sig'))
    cells = copy.deepcopy(parent['cells'][:2])
    setup = '''import time, importlib.util
BENCHMARK_STARTED = time.monotonic()
MEDBERT_SETTINGS = {"total_seconds": 3300, "max_train_patients": 2000,
    "max_train_rows": 25000, "pretrain_steps": 200, "epochs": 3,
    "max_length": 64, "codes_per_month": 5, "device": "auto", "threads": 4,
    "bootstrap_draws": 200}
missing = [p for p in ("numpy", "pandas", "sklearn", "torch", "matplotlib") if importlib.util.find_spec(p) is None]
if missing:
    raise RuntimeError("Install required runtime packages before execution: " + ", ".join(missing))
print("Med-BERT-inspired monthly claims benchmark with local TRAIN-only masked-code pretraining")
print("Official pretrained Med-BERT weights are not used. Monthly groups are not encounter-level ICD tokens.")
print("Target: 55 minutes including input loading. Cooperative stops cannot interrupt warehouse I/O or device kernels.")
print("TEST remains disabled. Full validation is required for every reported arm.")
print("Method reference: Rasmy et al., 2021, https://doi.org/10.1038/s41746-021-00455-y")
'''
    cells[0]['source'] = (setup + ''.join(cells[0]['source'])).splitlines(keepends=True)
    source = notebook_source(Path(__file__).with_name('benchmark.py').read_text(encoding='utf-8'))
    digest = hashlib.sha256(source.encode()).hexdigest()
    cells.append(code('MEDBERT_SOURCE = ' + repr(source) + '\nMEDBERT_SHA256 = ' + repr(digest) + '''
if hashlib.sha256(MEDBERT_SOURCE.encode()).hexdigest() != MEDBERT_SHA256:
    raise ValueError("Benchmark implementation checksum mismatch")
medbert = types.ModuleType("monthly_masked_code_benchmark")
exec(compile(MEDBERT_SOURCE, "monthly_masked_code_benchmark", "exec"), medbert.__dict__)
'''))
    cells.append(code('''
medbert_data = None
try:
    medbert_data = load_stage()
    medbert_prep, medbert_protocol = preparation()
    medbert_table, medbert_report = medbert.run(medbert_data, medbert_prep, medbert_protocol,
        RUN_PREFIX, MEDBERT_SHA256, MEDBERT_SETTINGS, wf, mdl, cmp, wh, started=BENCHMARK_STARTED)
finally:
    if medbert_data is not None:
        wf.release_inputs(medbert_data)
'''))
    cells.append(code('''
print("Saved immutable benchmark:", medbert_table)
print("Status:", medbert_report["status"], "Device:", medbert_report["device"])
print("Elapsed including input loading and persistence:", round(time.monotonic() - BENCHMARK_STARTED, 1), "seconds")
print("TRAIN subset:", medbert_report["train_snapshots"], "snapshots;", medbert_report["train_patients"], "patients")
print("Masked-code pretraining:", medbert_report["mlm"])
print("Fine-tuning receipts:", medbert_report["fine_tuning"])
print("Frozen benchmark availability:", medbert_report["benchmark_status"])
print("Token coverage:", medbert_report["tokenization"])
display(pd.DataFrame(medbert_report["rows"]).sort_values("Lift@10%", ascending=False))
for reference, interval in medbert_report["intervals"].items():
    print(medbert.MODEL_NAME, "minus", reference, "paired patient bootstrap")
    print("Pilot interval draws requested:", interval["requested_draws"], "valid:", interval["valid_draws"],
          "confidence:", interval["confidence"])
    display(pd.DataFrame(interval["lift_by_fraction"].values()))
for name, history in medbert_report["history"].items():
    print(name)
    display(pd.DataFrame(history))
for limitation in medbert_report["limitations"]:
    print(limitation)
import matplotlib.pyplot as plt
ranked = pd.DataFrame(medbert_report["rows"]).sort_values("Lift@10%")
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(ranked.Model, ranked["Lift@10%"], color=["#008A87" if n == medbert.MODEL_NAME else "#31536B" for n in ranked.Model])
ax.axvline(1, color="gray", linestyle=":")
ax.set_xlabel("Validation lift at top 10% of patient snapshots")
ax.set_title("Development benchmark: same validation rows, different training scopes")
fig.tight_layout()
display(fig)
plt.close(fig)
'''))
    for i, cell in enumerate(cells):
        cell['id'] = 'medbert-' + str(i)
        ast.parse(''.join(cell['source']))
        cell['outputs'], cell['execution_count'] = [], None
    book = dict(parent, cells=cells, metadata=dict(parent['metadata'], medbert_implementation_sha256=digest))
    raw = json.dumps(book, ensure_ascii=False, indent=1) + '\n'
    if re.search(r'\b(codex|claude|github|git|client|brian)\b', raw, re.I):
        raise ValueError('Unexpected presentation wording in generated notebook')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(raw, encoding='utf-8', newline='\n')
    return output_path


if __name__ == '__main__':
    print(build())
