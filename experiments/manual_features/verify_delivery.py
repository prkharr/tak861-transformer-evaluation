"""Verify generated notebook formatting, cell compilation and packaging."""
from pathlib import Path
import zipfile
import nbformat
import re

repo = Path(__file__).resolve().parents[2]
folder = repo.parent / "outputs" / "Manual_Features"
notebooks = sorted(folder.glob("*.ipynb"))
assert len(notebooks) == 4
assert {p.name for p in notebooks} == {
    "01_tensor_initialization.ipynb", "02_patient_level_split.ipynb",
    "03_transformer_training.ipynb", "04_transformer_evaluation.ipynb"}
total = 0
for path in notebooks:
    assert not re.search(r"brian|client", path.read_text(encoding="utf-8"), re.I)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) == 8
    for cell in notebook.cells:
        assert cell.cell_type == "code" and cell.execution_count is None and not cell.outputs
        compile(cell.source, str(path), "exec")
        assert not any(s in cell.source for s in ("sfPassword", "sfToken", "BEGIN PRIVATE KEY", "password ="))
    total += len(notebook.cells)
    mirror = repo / "notebooks" / "Manual Features" / path.name
    assert path.read_bytes() == mirror.read_bytes()
    print(path.name, path.stat().st_size, "bytes")
archive = folder.parent / "Manual_Features_4_Notebooks.zip"
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert set(z.namelist()) == {f.name for f in notebooks} | {"START_HERE.md", "VALIDATION.md"}
    for path in folder.iterdir():
        if path.name in z.namelist():
            assert path.read_bytes() == z.read(path.name)
            assert not re.search(r"brian|client", path.read_text(encoding="utf-8"), re.I)
print(f"Validated {len(notebooks)} notebooks, {total} code cells, matching mirrors and ZIP contents.")
