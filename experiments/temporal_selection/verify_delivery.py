"""Check standalone notebook structure, exact names, credentials and ZIP parity."""
from pathlib import Path
import re
import zipfile
import nbformat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / 'outputs' / 'Temporal_Feature_Selection'
names = ['01_tensor_initialization.ipynb', '02_patient_level_split.ipynb',
         '03_transformer_training.ipynb', '04_transformer_evaluation.ipynb']
assert sorted(p.name for p in OUT.glob('*.ipynb')) == names
for name in names:
    path = OUT / name
    source = path.read_text(encoding='utf-8')
    assert not re.search(r'brian|client|sfPassword|sfToken|BEGIN PRIVATE KEY', source, re.I)
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) == 8
    for cell in nb.cells:
        assert cell.cell_type == 'code' and cell.execution_count is None and not cell.outputs
        compile(cell.source, name, 'exec')
    assert path.read_bytes() == (ROOT / 'notebooks' / 'Temporal Feature Selection' / name).read_bytes()
with zipfile.ZipFile(OUT.parent / 'Temporal_Feature_Selection_4_Notebooks.zip') as archive:
    assert archive.testzip() is None
    assert set(archive.namelist()) == set(names + ['START_HERE.md', 'VALIDATION.md'])
    for name in archive.namelist():
        assert archive.read(name) == (OUT / name).read_bytes()
print('Verified four standalone notebooks, 32 compiled cells, clean outputs and matching ZIP/repository files.')
